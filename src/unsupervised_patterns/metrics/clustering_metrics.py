"""Cluster validation metrics: Silhouette score, Davies-Bouldin index, and Elbow detection."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from unsupervised_patterns.base import check_array


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute the mean Silhouette Coefficient of all samples.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    labels : array-like of shape (n_samples,)
        Predicted cluster labels for each sample.
    """
    X_arr = check_array(X)
    y = np.asarray(labels)

    unique_labels = np.unique(y)
    n_labels = len(unique_labels)

    if n_labels < 2 or n_labels >= len(X_arr):
        raise ValueError(f"Number of distinct labels ({n_labels}) must be in [2, n_samples - 1].")

    n_samples = len(X_arr)
    # Pairwise Euclidean distance matrix
    diff = X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    s_values = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        c_i = y[i]
        same_cluster_mask = (y == c_i)
        same_cluster_mask[i] = False

        # Intra-cluster distance a(i)
        if np.any(same_cluster_mask):
            a_i = np.mean(dist[i, same_cluster_mask])
        else:
            a_i = 0.0

        # Inter-cluster distance b(i)
        b_candidates = []
        for other_c in unique_labels:
            if other_c == c_i:
                continue
            other_mask = (y == other_c)
            b_candidates.append(np.mean(dist[i, other_mask]))

        b_i = min(b_candidates) if b_candidates else 0.0

        denom = max(a_i, b_i)
        s_values[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return float(np.mean(s_values))


def davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute the Davies-Bouldin index.

    The score is defined as the average similarity measure of each cluster with its
    most similar cluster, where similarity is the ratio of within-cluster distances to
    between-cluster distances. Minimum score is zero, with lower values indicating better clustering.
    """
    X_arr = check_array(X)
    y = np.asarray(labels)

    unique_labels = np.unique(y)
    n_labels = len(unique_labels)

    if n_labels < 2:
        raise ValueError(f"Number of clusters must be >= 2 (got {n_labels}).")

    centroids = np.empty((n_labels, X_arr.shape[1]), dtype=np.float64)
    dispersions = np.empty(n_labels, dtype=np.float64)

    for idx, c in enumerate(unique_labels):
        cluster_points = X_arr[y == c]
        centroid = np.mean(cluster_points, axis=0)
        centroids[idx] = centroid
        dispersions[idx] = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))

    # Centroid distance matrix
    centroid_diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    centroid_dist = np.linalg.norm(centroid_diff, axis=-1)

    r_values = np.empty(n_labels, dtype=np.float64)

    for i in range(n_labels):
        r_ij = []
        for j in range(n_labels):
            if i == j:
                continue
            dist_ij = centroid_dist[i, j]
            if dist_ij == 0.0:
                r_ij.append(0.0)
            else:
                r_ij.append((dispersions[i] + dispersions[j]) / dist_ij)
        r_values[i] = max(r_ij) if r_ij else 0.0

    return float(np.mean(r_values))


def find_elbow_point(
    inertias: Sequence[float],
    ks: Optional[Sequence[int]] = None,
) -> int:
    """Detect the optimal elbow point on an inertia curve using maximum orthogonal distance to the chord."""
    y = np.asarray(inertias, dtype=np.float64)
    x = np.arange(len(y)) if ks is None else np.asarray(ks, dtype=np.float64)

    if len(y) < 3:
        return int(x[0])

    # Endpoints of the secant line
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    # Direction vector of the line
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len == 0.0:
        return int(x[0])

    unit_vec = line_vec / line_len

    # Distance from each point (x_i, y_i) to line
    pts = np.column_stack([x, y])
    vec_to_pts = pts - p1
    # Projection length
    proj_len = np.dot(vec_to_pts, unit_vec)
    proj_pts = p1 + proj_len[:, np.newaxis] * unit_vec
    distances = np.linalg.norm(pts - proj_pts, axis=1)

    best_idx = int(np.argmax(distances))
    return int(x[best_idx])
