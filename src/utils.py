"""
Reproducibility + checkpointing utilities.

Free-tier notebook sessions (Colab/Kaggle) get disconnected without warning.
Every function here exists to make sure a killed session costs you at most
one epoch of work, not the whole run.
"""
import os
import random
import json
import time
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Seed every RNG that affects training so runs are reproducible.

    Note: full bit-for-bit determinism on GPU also needs
    torch.use_deterministic_algorithms(True) and CUBLAS_WORKSPACE_CONFIG,
    which slow training down noticeably. For this project we seed
    everything but do NOT force full determinism by default — that
    tradeoff is controlled by `deterministic` below.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, history, scaler=None):
    """Atomic-ish checkpoint save: write to a temp file then rename, so a
    session that dies mid-write never corrupts the last good checkpoint."""
    tmp_path = path + ".tmp"
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_acc": best_acc,
        "history": history,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # atomic on the same filesystem


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location=None):
    """Returns (start_epoch, best_acc, history). start_epoch is the epoch
    to resume FROM (i.e. the next one to run), not the last completed one."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["epoch"] + 1, ckpt["best_acc"], ckpt.get("history", [])


class EpochTimer:
    """Tracks elapsed wall-clock time so training can stop gracefully
    before a Colab/Kaggle session time limit hits, rather than getting
    killed mid-epoch with nothing saved."""

    def __init__(self, budget_seconds: float | None):
        self.budget_seconds = budget_seconds
        self.start = time.time()

    def over_budget(self) -> bool:
        if self.budget_seconds is None:
            return False
        return (time.time() - self.start) >= self.budget_seconds


def append_history_row(history_path, row: dict):
    """Append one epoch's metrics as a JSON line — cheap, crash-safe log
    you can tail/plot without waiting for training to finish."""
    with open(history_path, "a") as f:
        f.write(json.dumps(row) + "\n")
