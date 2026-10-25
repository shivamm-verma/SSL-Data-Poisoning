"""
One script covers every (dataset x model x attack) combination in the
grid. Which combination runs is entirely decided by command-line args /
a YAML config — this file itself never changes per experiment, which is
the whole point: everyone (you and your friends) runs this exact same
script, pointed at a different config.

Usage (direct CLI):
    python src/run_experiment.py \
        --dataset gtsrb --model resnet50 --attack backdoor \
        --rate 0.01 --target-class 0 --epochs 40 --max-minutes 170

Usage (from a YAML config — see configs/experiments/*.yaml):
    python src/run_experiment.py --config configs/experiments/gtsrb_resnet50_backdoor.yaml

Every run is identified by a run_id encoding dataset/model/attack/rate,
so results from many different laptops never collide when collected
into one shared results/ folder.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry_datasets as data_reg
import registry_poison as poison_reg
from registry_models import build_model
from utils import set_seed, get_device, save_checkpoint, load_checkpoint, EpochTimer, append_history_row


DEFAULTS = {
    "attack": "none", "rate": 0.0, "target_class": None, "source_class": None,
    "trigger_size": 3, "trigger_position": "bottom_right", "trigger_color": [255, 255, 255],
    "data_root": "./data", "ckpt_dir": "./checkpoints", "log_dir": "./logs", "results_dir": "./results",
    "epochs": 40, "batch_size": 128, "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
    "num_workers": 2, "seed": 42, "no_amp": False, "max_minutes": None,
    "run_detectors": False, "knn_k": 10,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="YAML file; any CLI flag below overrides its values")

    # Every optional flag defaults to None here — NOT to its real default
    # (see DEFAULTS above) — so we can tell "user didn't pass this" apart
    # from "user explicitly passed the same value as the default". That
    # distinction is what makes config-file values actually apply: a flag
    # left at None falls through to the config value (if given) and then
    # to DEFAULTS; a flag the user DID pass always wins.
    p.add_argument("--dataset", choices=data_reg.DATASET_SPECS.keys(), default=None)
    p.add_argument("--model", choices=["resnet10", "resnet18", "resnet50", "vgg19", "densenet161"], default=None)
    p.add_argument("--attack", choices=["none", "label_flip", "backdoor", "clean_label"], default=None)
    p.add_argument("--rate", type=float, default=None)
    p.add_argument("--target-class", type=int, default=None)
    p.add_argument("--source-class", type=int, default=None)
    p.add_argument("--trigger-size", type=int, default=None)
    p.add_argument("--trigger-position", default=None)
    p.add_argument("--trigger-color", type=int, nargs=3, default=None)

    p.add_argument("--data-root", default=None)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--results-dir", default=None)

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--momentum", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-amp", action="store_true", default=None)
    p.add_argument("--max-minutes", type=float, default=None)

    p.add_argument("--run-detectors", action="store_true", default=None,
                    help="after training, extract ImageNet-pretrained embeddings and run the Phase 5 detector suite")
    p.add_argument("--knn-k", type=int, default=None)

    cli_args = p.parse_args()

    cfg = {}
    if cli_args.config:
        import yaml
        with open(cli_args.config) as f:
            cfg = {k.replace("-", "_"): v for k, v in (yaml.safe_load(f) or {}).items()}

    # Resolution order per field: explicit CLI value > config file value
    # > hardcoded default. This is the standard "None means unset"
    # pattern and — unlike a sentinel-value check — works correctly
    # regardless of what a field's real default happens to be.
    resolved = argparse.Namespace()
    for key, default in DEFAULTS.items():
        cli_value = getattr(cli_args, key, None)
        value = cli_value if cli_value is not None else cfg.get(key, default)
        setattr(resolved, key, value)
    resolved.config = cli_args.config
    resolved.dataset = cli_args.dataset if cli_args.dataset is not None else cfg.get("dataset")
    resolved.model = cli_args.model if cli_args.model is not None else cfg.get("model")

    args = resolved

    if args.dataset is None or args.model is None:
        raise SystemExit("--dataset and --model are required (directly or via --config)")
    return args


def rate_slug(rate):
    return str(rate).replace(".", "p")


@torch.no_grad()
def evaluate_clean(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, labels, _idx in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total, loss_sum / total


@torch.no_grad()
def evaluate_asr(model, loader, device):
    model.eval()
    hit, total = 0, 0
    for images, _orig_labels, target_classes in loader:
        images, target_classes = images.to(device), target_classes.to(device)
        preds = model(images).argmax(1)
        hit += (preds == target_classes).sum().item()
        total += images.size(0)
    return hit / total


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels, _idx, _is_poisoned in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)

    run_id = f"{args.dataset}_{args.model}_{args.attack}_{rate_slug(args.rate)}"
    print(f"[run] {run_id}")

    set_seed(args.seed)
    device = get_device()

    spec = data_reg.DATASET_SPECS[args.dataset]
    num_classes = data_reg.num_classes_for(args.dataset, args.data_root)
    in_channels, image_size = spec["in_channels"], spec["image_size"]
    train_tf, eval_tf = data_reg.build_transforms(args.dataset)

    raw_train = data_reg.get_raw_train(args.dataset, args.data_root)

    if args.attack == "none":
        poison_meta = poison_reg.empty_poison_metadata()
    else:
        labels = poison_reg.extract_labels(raw_train)
        if args.attack == "label_flip":
            poison_meta = poison_reg.generate_label_poison_metadata(
                labels, args.rate, args.seed, target_class=args.target_class, source_class=args.source_class)
        else:
            if args.target_class is None:
                raise ValueError(f"--target-class is required for attack={args.attack}")
            trigger_config = {"size": args.trigger_size, "position": args.trigger_position, "color": args.trigger_color}
            gen = poison_reg.GENERATORS[args.attack]
            poison_meta = gen(labels, args.rate, args.seed, target_class=args.target_class, trigger_config=trigger_config)
        poison_meta["poisoned_indices_set"] = set(poison_meta["poisoned_indices"])
        meta_path = os.path.join(args.log_dir, f"{run_id}_poison_meta.json")
        poison_reg.save_poison_metadata(meta_path, poison_meta)
        print(f"[poison] {poison_meta['attack_type']} rate={poison_meta['rate']} n_poisoned={poison_meta['n_poisoned']}")

    train_dataset = poison_reg.PoisonedDataset(raw_train, train_tf, poison_meta)
    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                               generator=g, persistent_workers=args.num_workers > 0)

    clean_test = data_reg.get_dataset(args.dataset, args.data_root, train=False, transform=eval_tf)
    clean_test_loader = DataLoader(clean_test, batch_size=256, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    asr_loader = None
    if args.attack in ("backdoor", "clean_label"):
        raw_test = data_reg.get_raw(args.dataset, args.data_root, train=False, transform=None)
        triggered_test = poison_reg.TriggeredTestSet(raw_test, eval_tf, poison_meta["trigger_config"], poison_meta["target_class"])
        asr_loader = DataLoader(triggered_test, batch_size=256, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, num_classes=num_classes, in_channels=in_channels, image_size=image_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                           weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    ckpt_path = os.path.join(args.ckpt_dir, f"{run_id}.pt")
    history_path = os.path.join(args.log_dir, f"{run_id}_history.jsonl")

    start_epoch, best_acc, history = 1, 0.0, []
    if os.path.exists(ckpt_path):
        start_epoch, best_acc, history = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, device)
        print(f"[resume] from epoch {start_epoch}, best_acc so far={best_acc:.4f}")

    timer = EpochTimer(args.max_minutes * 60 if args.max_minutes else None)
    if start_epoch <= args.epochs:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
            scheduler.step()
            clean_acc, clean_loss = evaluate_clean(model, clean_test_loader, device)
            asr = evaluate_asr(model, asr_loader, device) if asr_loader is not None else None

            row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                   "clean_test_acc": clean_acc, "clean_test_loss": clean_loss, "asr": asr,
                   "lr": scheduler.get_last_lr()[0], "epoch_time_sec": time.time() - t0}
            history.append(row); append_history_row(history_path, row)
            msg = f"[epoch {epoch}/{args.epochs}] train_acc={train_acc:.4f} clean_test_acc={clean_acc:.4f}"
            if asr is not None:
                msg += f" ASR={asr:.4f}"
            print(msg)

            if clean_acc > best_acc:
                best_acc = clean_acc
                save_checkpoint(os.path.join(args.ckpt_dir, f"{run_id}_best.pt"), model, optimizer, scheduler, epoch, best_acc, history, scaler)
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_acc, history, scaler)

            if timer.over_budget():
                print(f"[time budget] stopping after epoch {epoch}. Rerun the same command to resume."); break
    else:
        print("[done] checkpoint already at or past target epochs.")

    final_asr = history[-1]["asr"] if history else None
    result = {
        "run_id": run_id, "dataset": args.dataset, "model": args.model, "attack": args.attack,
        "rate": args.rate, "target_class": args.target_class,
        "best_clean_test_acc": best_acc, "final_asr": final_asr, "epochs_run": len(history),
    }

    if args.run_detectors:
        result["detectors"] = run_detector_suite(args, poison_meta, in_channels, device)

    result_path = os.path.join(args.results_dir, f"{run_id}_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[final] best_clean_test_acc={best_acc:.4f}" + (f" final_asr={final_asr:.4f}" if final_asr is not None else ""))
    print(f"[done] result saved to {result_path} — copy this file back to the shared results folder")


def run_detector_suite(args, poison_meta, in_channels, device):
    """Reuses the Phase 5 baseline detectors on frozen ImageNet-pretrained
    embeddings. Self-supervised (Phase 6) comparison is not wired in here
    yet — this covers the baseline side of the grid first."""
    from embeddings import build_embedding_transform, build_pretrained_encoder, extract_embeddings
    from detectors import DETECTORS, LABEL_AWARE_DETECTORS
    from evaluate import evaluate_scores

    if poison_meta["attack_type"] == "none":
        print("[detectors] skipped — attack=none, nothing to detect")
        return None

    raw_train = data_reg.get_raw_train(args.dataset, args.data_root)
    emb_transform = build_embedding_transform(in_channels=in_channels)
    dataset = poison_reg.PoisonedDataset(raw_train, emb_transform, poison_meta)

    encoder = build_pretrained_encoder(device)
    embeddings, labels, indices, is_poisoned = extract_embeddings(dataset, encoder, device)

    results = {}
    for name, fn in DETECTORS.items():
        scores = fn(embeddings, k=args.knn_k) if name == "knn" else fn(embeddings)
        results[name] = evaluate_scores(scores, is_poisoned)
    for name, fn in LABEL_AWARE_DETECTORS.items():
        scores = fn(embeddings, labels, k=args.knn_k)
        results[name] = evaluate_scores(scores, is_poisoned)

    for name, m in results.items():
        print(f"[detector: {name}] AUROC={m['auroc']:.4f} precision@k={m['precision_at_k']:.4f}")
    return results


if __name__ == "__main__":
    main()
