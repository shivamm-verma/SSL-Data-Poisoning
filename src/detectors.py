"""
Phase 5 baseline detectors. Each function takes an [N, D] embedding
matrix and returns an [N] array of continuous suspicion scores — higher
= more suspicious. Keeping every detector's output on that same
"higher = more suspicious" scale is what makes them directly comparable
with the same evaluation code (evaluate.py), and with Phase 6's
self-supervised detector later.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans


def knn_distance_score(embeddings, k=10):
    """Distance to the k-th nearest neighbor. Poisoned samples (mislabeled
    or trigger-injected) are expected to sit further from their "true"
    local neighborhood in feature space than typical clean samples."""
    nn_model = NearestNeighbors(n_neighbors=k + 1)  # +1: a point is its own nearest neighbor
    nn_model.fit(embeddings)
    distances, _ = nn_model.kneighbors(embeddings)
    return distances[:, -1]  # distance to the k-th neighbor, excluding self


def isolation_forest_score(embeddings, n_estimators=200, seed=42):
    """IsolationForest's score_samples is HIGHER for normal points and
    LOWER (more negative) for anomalies — we negate it so this function's
    output follows the same "higher = more suspicious" convention as the
    other detectors."""
    clf = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    clf.fit(embeddings)
    return -clf.score_samples(embeddings)


def kmeans_distance_score(embeddings, n_clusters=10, seed=42):
    """Clustering baseline: fit KMeans (n_clusters defaults to num
    classes — the assumption being clean data roughly clusters by
    class), score = distance to the nearest cluster centroid. A point
    far from every centroid doesn't fit any learned cluster well."""
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(embeddings)
    dists_to_all_centroids = km.transform(embeddings)  # [N, n_clusters]
    return dists_to_all_centroids.min(axis=1)


def knn_label_agreement_score(embeddings, labels, k=10):
    """The detector KNN-distance can't be: instead of asking "is this
    image visually weird", ask "do this image's nearest visual
    neighbors agree with the label it was ASSIGNED". A label-flipped
    sample looks visually normal (low distance score) but sits in a
    neighborhood of same-looking images with a DIFFERENT label — that
    mismatch is exactly what label poisoning is. score = fraction of
    the k nearest neighbors whose label disagrees with this sample's
    own (possibly poisoned) label; higher = more suspicious."""
    labels = np.asarray(labels)
    nn_model = NearestNeighbors(n_neighbors=k + 1)
    nn_model.fit(embeddings)
    _, neighbor_idx = nn_model.kneighbors(embeddings)
    neighbor_idx = neighbor_idx[:, 1:]  # drop self (always the 1st neighbor at distance 0)

    neighbor_labels = labels[neighbor_idx]  # [N, k]
    own_label = labels[:, None]  # [N, 1]
    disagreement = (neighbor_labels != own_label).mean(axis=1)  # [N]
    return disagreement


DETECTORS = {
    "knn": knn_distance_score,
    "isolation_forest": isolation_forest_score,
    "kmeans": kmeans_distance_score,
}

# Detectors that need the (possibly poisoned) labels, not just embeddings —
# handled separately in run_baseline_detectors.py since their call signature differs.
LABEL_AWARE_DETECTORS = {
    "knn_label_agreement": knn_label_agreement_score,
}
