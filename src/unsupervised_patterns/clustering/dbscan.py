"""Density-Based Spatial Clustering of Applications with Noise (DBSCAN)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from unsupervised_patterns.base import BaseEstimator, ClusterMixin, check_array


class DBSCAN(BaseEstimator, ClusterMixin):
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Finds core samples of high density and expands clusters from them. Good for data
    which contains clusters of similar density and arbitrary geometries.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered as a core point.
    metric : str, default='euclidean'
        The metric to use when calculating distance between instances in a feature array.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        self.labels_: Optional[np.ndarray] = None
        self.core_sample_indices_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> DBSCAN:
        """Perform DBSCAN clustering from features."""
        X_arr = check_array(X)
        n_samples = len(X_arr)

        # Pairwise Euclidean distance matrix
        # (N, 1, D) - (1, N, D) -> (N, N, D) -> norm
        diff = X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)

        # Neighborhood sets
        neighbors = [np.where(dist_matrix[i] <= self.eps)[0] for i in range(n_samples)]

        # Identify core points
        is_core = np.array([len(nbrs) >= self.min_samples for nbrs in neighbors], dtype=bool)
        self.core_sample_indices_ = np.where(is_core)[0]
        self.components_ = X_arr[self.core_sample_indices_].copy()

        labels = np.full(n_samples, -1, dtype=np.int64)
        visited = np.zeros(n_samples, dtype=bool)
        current_cluster = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            if not is_core[i]:
                # Mark as noise (can later become border point of a cluster)
                continue

            # Start expanding new cluster
            labels[i] = current_cluster
            # Queue of neighbors to explore
            queue = list(neighbors[i])

            idx = 0
            while idx < len(queue):
                neighbor_point = queue[idx]
                idx += 1

                if not visited[neighbor_point]:
                    visited[neighbor_point] = True
                    if is_core[neighbor_point]:
                        for q in neighbors[neighbor_point]:
                            if q not in queue:
                                queue.append(q)

                if labels[neighbor_point] == -1:
                    labels[neighbor_point] = current_cluster

            current_cluster += 1

        self.labels_ = labels
        return self
