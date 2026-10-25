"""
Scores a detector's continuous suspicion scores against the ground-truth
is_poisoned flags from the poisoning metadata (Phase 3). This is the
same evaluation harness Phase 6's self-supervised detector, and the
Phase 7-9 low-rate/unseen-attack/false-positive experiments, will all
reuse — get it right once, here.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_scores(scores, is_poisoned):
    """scores: [N] float, higher = more suspicious.
    is_poisoned: [N] bool ground truth (never shown to the detector).

    Returns AUROC, Average Precision (~area under PR curve), and
    precision@k / recall@k where k = the true number of poisoned
    samples — i.e. "if you flagged exactly as many samples as are
    actually poisoned, what fraction would you get right?". This is a
    realistic operating point: a real scanner doesn't know the true
    poison count in advance, but reporting @k against the known ground
    truth is the standard way benchmarks compare detectors fairly.
    """
    is_poisoned = np.asarray(is_poisoned).astype(bool)
    n_poisoned = int(is_poisoned.sum())

    auroc = roc_auc_score(is_poisoned, scores) if 0 < n_poisoned < len(is_poisoned) else float("nan")
    ap = average_precision_score(is_poisoned, scores) if 0 < n_poisoned < len(is_poisoned) else float("nan")

    order = np.argsort(-scores)  # descending suspicion
    top_k_idx = order[:n_poisoned]
    true_positives_at_k = int(is_poisoned[top_k_idx].sum())
    precision_at_k = true_positives_at_k / n_poisoned if n_poisoned > 0 else float("nan")
    recall_at_k = precision_at_k  # by construction when k == n_poisoned

    false_positive_rate_at_k = (n_poisoned - true_positives_at_k) / max(1, (len(is_poisoned) - n_poisoned))

    return {
        "auroc": float(auroc),
        "average_precision": float(ap),
        "precision_at_k": float(precision_at_k),
        "recall_at_k": float(recall_at_k),
        "false_positive_rate_at_k": float(false_positive_rate_at_k),
        "n_poisoned": n_poisoned,
        "n_total": len(is_poisoned),
    }
