"""
Generalized poisoning registry — the same three attack-generation
functions and one dataset wrapper work across all 5 datasets in the
grid. Two new things versus the original CIFAR-10-only version:

1. apply_trigger now handles both 1-channel (MNIST) and 3-channel
   images, since the original assumed 3-channel RGB.
2. clean_label poisoning is new: unlike backdoor (which relabels
   triggered samples TO the target class), clean-label poisoning only
   triggers samples that ALREADY belong to the target class — so the
   label is never actually falsified. This is the canonical
   "clean-label backdoor" definition (Turner et al.) and is what makes
   it stealthier than ordinary backdoor poisoning: manually reviewing
   labels reveals nothing wrong.
"""
import json
import random

import numpy as np
from torch.utils.data import Dataset

NUM_CLASSES_UNUSED = None  # num_classes is supplied by the caller per-dataset


def extract_labels(dataset):
    """Works across CIFAR10/MNIST (.targets), ImageFolder-style datasets
    used for tinyimagenet/imagenet_subset (.samples), and falls back to
    a full pass for anything else (e.g. GTSRB, which doesn't expose a
    stable public labels attribute across torchvision versions)."""
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        return list(targets.tolist()) if hasattr(targets, "tolist") else list(targets)
    if hasattr(dataset, "samples"):
        return [label for _, label in dataset.samples]
    return [dataset[i][1] for i in range(len(dataset))]


def generate_label_poison_metadata(labels, rate, seed, target_class=None, source_class=None):
    rng = random.Random(seed)
    n = len(labels)
    num_classes = len(set(labels))

    candidates = [i for i, l in enumerate(labels) if l == source_class] if source_class is not None else list(range(n))
    k = min(int(round(rate * n)), len(candidates))
    poisoned_indices = sorted(rng.sample(candidates, k))

    poison_map = {}
    for idx in poisoned_indices:
        orig = labels[idx]
        new_label = target_class if target_class is not None else rng.choice([c for c in range(num_classes) if c != orig])
        poison_map[idx] = {"orig_label": orig, "new_label": new_label, "trigger": False}

    return {
        "attack_type": "label_poison", "rate": rate, "seed": seed, "n_total": n,
        "n_poisoned": len(poisoned_indices), "target_class": target_class, "source_class": source_class,
        "poisoned_indices": poisoned_indices, "poison_map": poison_map,
    }


def generate_backdoor_poison_metadata(labels, rate, seed, target_class, trigger_config):
    """Trigger + relabel to target_class. Candidates are non-target
    samples (relabeling something already at the target class wastes
    poison budget)."""
    rng = random.Random(seed)
    n = len(labels)
    candidates = [i for i, l in enumerate(labels) if l != target_class]
    k = min(int(round(rate * n)), len(candidates))
    poisoned_indices = sorted(rng.sample(candidates, k))
    poison_map = {idx: {"orig_label": labels[idx], "new_label": target_class, "trigger": True} for idx in poisoned_indices}
    return {
        "attack_type": "backdoor", "rate": rate, "seed": seed, "n_total": n,
        "n_poisoned": len(poisoned_indices), "target_class": target_class, "source_class": None,
        "poisoned_indices": poisoned_indices, "poison_map": poison_map, "trigger_config": trigger_config,
    }


def generate_clean_label_poison_metadata(labels, rate, seed, target_class, trigger_config):
    """Trigger only, NO relabeling — candidates are samples that
    ALREADY belong to target_class, so the stored label was always
    correct. Stealthier than `backdoor`: a human or a label-consistency
    detector (like Phase 5's knn_label_agreement) sees nothing wrong."""
    rng = random.Random(seed)
    n = len(labels)
    candidates = [i for i, l in enumerate(labels) if l == target_class]
    k = min(int(round(rate * n)), len(candidates))
    poisoned_indices = sorted(rng.sample(candidates, k))
    poison_map = {idx: {"orig_label": target_class, "new_label": target_class, "trigger": True} for idx in poisoned_indices}
    return {
        "attack_type": "clean_label", "rate": rate, "seed": seed, "n_total": n,
        "n_poisoned": len(poisoned_indices), "target_class": target_class, "source_class": None,
        "poisoned_indices": poisoned_indices, "poison_map": poison_map, "trigger_config": trigger_config,
    }


def apply_trigger(image, trigger_config):
    """image: PIL.Image, 1 or 3 channels. Returns a NEW image — never
    mutates the input. `color` may be a 3-list (RGB); on a 1-channel
    image only the first value is used (as the single-channel intensity)."""
    arr = np.array(image).copy()
    size = trigger_config.get("size", 3)
    color = trigger_config.get("color", [255, 255, 255])
    position = trigger_config.get("position", "bottom_right")

    is_grayscale = (arr.ndim == 2)
    patch_value = color[0] if is_grayscale else color

    if arr.ndim == 2:
        h, w = arr.shape
    else:
        h, w, _ = arr.shape

    if position == "bottom_right":
        arr[h - size:h, w - size:w] = patch_value
    elif position == "bottom_left":
        arr[h - size:h, 0:size] = patch_value
    elif position == "top_right":
        arr[0:size, w - size:w] = patch_value
    elif position == "top_left":
        arr[0:size, 0:size] = patch_value
    else:
        raise ValueError(f"unknown trigger position: {position}")

    from PIL import Image
    mode = "L" if is_grayscale else "RGB"
    return Image.fromarray(arr, mode=mode)


def save_poison_metadata(path, meta):
    serializable = dict(meta)
    if isinstance(serializable.get("poisoned_indices_set"), set):
            serializable["poisoned_indices_set"] = sorted(
                serializable["poisoned_indices_set"]
            )
    serializable["poison_map"] = {str(k): v for k, v in meta["poison_map"].items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_poison_metadata(path):
    with open(path) as f:
        meta = json.load(f)
    meta["poison_map"] = {int(k): v for k, v in meta["poison_map"].items()}
    meta["poisoned_indices_set"] = set(meta["poisoned_indices"])
    return meta


class PoisonedDataset(Dataset):
    """Generic version of Phase 3's PoisonedCIFAR10 — works with ANY
    raw (image, label) dataset, not just CIFAR-10. `raw_dataset` must
    be constructed with transform=None so trigger stamping happens on
    original pixels before any augmentation/normalization. Passing an
    empty poison_meta (poisoned_indices=[]) makes this behave exactly
    like the unmodified clean dataset — used for `--attack none` runs
    so the training loop never needs a separate code path.
    """

    def __init__(self, raw_dataset, transform, poison_meta):
        self.raw = raw_dataset
        self.transform = transform
        self.meta = poison_meta

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        image, label = self.raw[index]
        is_poisoned = index in self.meta["poisoned_indices_set"]

        if is_poisoned:
            entry = self.meta["poison_map"][index]
            if entry.get("trigger"):
                image = apply_trigger(image, self.meta["trigger_config"])
            label = entry["new_label"]

        if self.transform:
            image = self.transform(image)

        return image, label, index, is_poisoned


def empty_poison_metadata():
    """Used for --attack none: no samples poisoned, but the dataset
    wrapper's interface (and the 4-tuple it returns) stays identical to
    every poisoned run, so the training/eval code needs no branching."""
    return {"attack_type": "none", "rate": 0.0, "poisoned_indices": [], "poison_map": {}, "poisoned_indices_set": set()}


class TriggeredTestSet(Dataset):
    """For attack-success-rate evaluation: applies the SAME trigger to
    every test image (excluding images already of target_class, since
    "does the trigger flip this to target_class" is only meaningful for
    images that weren't target_class to begin with)."""

    def __init__(self, raw_test_dataset, transform, trigger_config, target_class):
        self.samples = [(img, lbl) for img, lbl in raw_test_dataset if lbl != target_class]
        self.transform = transform
        self.trigger_config = trigger_config
        self.target_class = target_class

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, orig_label = self.samples[index]
        image = apply_trigger(image, self.trigger_config)
        if self.transform:
            image = self.transform(image)
        return image, orig_label, self.target_class


GENERATORS = {
    "label_flip": generate_label_poison_metadata,
    "backdoor": generate_backdoor_poison_metadata,
    "clean_label": generate_clean_label_poison_metadata,
}
