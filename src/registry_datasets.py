"""
Dataset registry. Every dataset is fetched through get_dataset(name, ...)
so run_experiment.py never has dataset-specific branching — adding a
6th dataset later means adding one entry here, not touching the runner.

Each dataset reports its own (num_classes, in_channels, native_image_size)
via DATASET_SPECS so the model registry can adapt architectures correctly
without the two registries needing to know about each other's internals.
"""
import os
import urllib.request
import zipfile

import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

DATASET_SPECS = {
    "cifar10":        {"num_classes": 10,  "in_channels": 3, "image_size": 32},
    "mnist":          {"num_classes": 10,  "in_channels": 1, "image_size": 28},
    "gtsrb":          {"num_classes": 43,  "in_channels": 3, "image_size": 48},
    "tinyimagenet":   {"num_classes": 200, "in_channels": 3, "image_size": 64},
    # Real ImageNet-1k is not auto-downloadable (license-gated) and not
    # realistically trainable from scratch on free-tier Colab. This slot
    # expects a local ImageFolder-formatted directory the user supplies
    # via --data-root — e.g. a pre-downloaded subset such as Imagenette.
    "imagenet_subset": {"num_classes": None, "in_channels": 3, "image_size": 224},
}


def build_mean_std(name):
    """Per-dataset normalization stats. MNIST is grayscale so it gets a
    single-channel mean/std; everything else uses standard 3-channel
    stats (ImageNet stats for the two ImageNet-family entries, since
    that's what any pretrained-embedding comparison will assume)."""
    if name == "mnist":
        return (0.1307,), (0.3081,)
    if name == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    if name == "gtsrb":
        return (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)
    # tinyimagenet / imagenet_subset
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(name):
    spec = DATASET_SPECS[name]
    size = spec["image_size"]
    mean, std = build_mean_std(name)

    train_tf = T.Compose([
        T.Resize((size, size)) if name in ("gtsrb", "imagenet_subset") else T.Lambda(lambda x: x),
        T.RandomCrop(size, padding=max(2, size // 8)),
        T.RandomHorizontalFlip() if name != "mnist" and name != "gtsrb" else T.Lambda(lambda x: x),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.Resize((size, size)) if name in ("gtsrb", "imagenet_subset") else T.Lambda(lambda x: x),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, eval_tf


class IndexedWrapper(Dataset):
    """Makes any torchvision dataset return (image, label, index) —
    same convention used throughout the project so poisoning code
    doesn't need to change per-dataset."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return image, label, index


def _ensure_tinyimagenet(data_root):
    """Tiny-ImageNet-200 has no torchvision built-in — download once
    (~240MB) and reuse the extracted folder on later calls."""
    target_dir = os.path.join(data_root, "tiny-imagenet-200")
    if os.path.isdir(target_dir):
        return target_dir
    os.makedirs(data_root, exist_ok=True)
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    print(f"[data] downloading Tiny-ImageNet-200 from {url} (one-time, ~240MB)...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_root)
    return target_dir


def get_raw(name, data_root, train, transform=None, download=True):
    """Returns the raw underlying dataset (image, label) for the
    requested split. `name` must be a key in DATASET_SPECS."""
    if name == "cifar10":
        base = torchvision.datasets.CIFAR10(root=data_root, train=train, download=download, transform=transform)
    elif name == "mnist":
        base = torchvision.datasets.MNIST(root=data_root, train=train, download=download, transform=transform)
    elif name == "gtsrb":
        split = "train" if train else "test"
        base = torchvision.datasets.GTSRB(root=data_root, split=split, download=download, transform=transform)
    elif name == "tinyimagenet":
        root = _ensure_tinyimagenet(data_root)
        # Tiny-ImageNet ships train/ as class-subfolders (ImageFolder-
        # compatible) but val/ as a flat folder with a separate
        # annotations file — using train-only split here for both
        # phases keeps this simple; swap in a proper val loader if you
        # need a strict held-out test set later.
        split_dir = os.path.join(root, "train")
        base = torchvision.datasets.ImageFolder(split_dir, transform=transform)
    elif name == "imagenet_subset":
        split_dir = os.path.join(data_root, "train" if train else "val")
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"imagenet_subset expects an ImageFolder-formatted dataset at "
                f"{split_dir} (train/<class>/*.jpg, val/<class>/*.jpg) — this is "
                f"not auto-downloaded. Point --data-root at a local subset such "
                f"as Imagenette."
            )
        base = torchvision.datasets.ImageFolder(split_dir, transform=transform)
    else:
        raise ValueError(f"unknown dataset: {name}")

    return base


def get_raw_train(name, data_root, download=True):
    """Returns the raw (image, label) train dataset with NO transform —
    what the poisoning registry needs so trigger-stamping happens on
    original pixels before any augmentation/normalization is applied."""
    return get_raw(name, data_root, train=True, transform=None, download=download)


def get_dataset(name, data_root, train, transform, download=True):
    """Returns an IndexedWrapper dataset for the requested split."""
    return IndexedWrapper(get_raw(name, data_root, train, transform, download))


def num_classes_for(name, data_root=None):
    """imagenet_subset's class count depends on the folder actually
    supplied, so it's resolved at runtime rather than hardcoded."""
    spec = DATASET_SPECS[name]
    if spec["num_classes"] is not None:
        return spec["num_classes"]
    train_dir = os.path.join(data_root, "train")
    return len(os.listdir(train_dir))
