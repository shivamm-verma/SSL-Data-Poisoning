"""
Generates configs/experiments/*.yaml for the full grid: every model in
MODEL_NAMES x every rate in RATES, for the backdoor attack (the attack
type most worth comparing across architectures, since ASR + clean
accuracy together show whether a bigger/different model is harder or
easier to backdoor).

Run once after cloning:
    python src/generate_configs.py

Re-run any time to regenerate — it always overwrites, so hand-edits to
generated files won't survive a re-run; edit RATES/MODEL_NAMES/ATTACK
below instead, or copy a generated file to a new name if you want a
one-off variant.
"""
import os
import yaml

MODELS = ["resnet10", "resnet18", "resnet50", "vgg19", "densenet161"]
DATASETS = ["cifar10", "mnist", "gtsrb", "tinyimagenet", "imagenet_subset"]
RATES = [0.001, 0.005, 0.01]
ATTACK = "backdoor"
TARGET_CLASS = 0

EPOCHS_BY_MODEL = {
    "resnet10": 60,
    "resnet18": 60,
    "resnet50": 40,
    "vgg19": 40,
    "densenet161": 30,
}

RATE_TAGS = {
    0.001: "0_1pct",
    0.005: "0_5pct",
    0.01: "1pct",
}

OUT_DIR = "configs/experiments"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for dataset in DATASETS:
        for model in MODELS:
            for rate in RATES:
                fname = f"{dataset}_{model}_{ATTACK}_{RATE_TAGS[rate]}.yaml"
                path = os.path.join(OUT_DIR, fname)

                if os.path.exists(path):
                    print(f"skip (exists) {path}")
                    continue

                cfg = {
                    "dataset": dataset,
                    "model": model,
                    "attack": ATTACK,
                    "rate": rate,
                    "target_class": TARGET_CLASS,
                    "epochs": EPOCHS_BY_MODEL[model],
                }
                with open(path, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
                print(f"wrote {path}")


if __name__ == "__main__":
    main()
