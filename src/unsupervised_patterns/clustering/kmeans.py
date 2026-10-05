"""K-Means clustering with K-Means++ initialization and Lloyd's iterative algorithm."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from unsupervised_patterns.base import BaseEstimator, ClusterMixin, check_array


def _euclidean_distances_sq(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute squared pairwise Euclidean distance matrix between X (N, D) and Y (K, D)."""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>
    x2 = np.sum(X ** 2, axis=1, keepdims=True)
    y2 = np.sum(Y ** 2, axis=1, keepdims=True).T
    dist_sq = x2 + y2 - 2.0 * (X @ Y.T)
    return np.maximum(dist_sq, 0.0)


def _init_kmeans_plus_plus(X: np.ndarray, n_clusters: int, rng: np.random.RandomState) -> np.ndarray:
    """Initialize cluster centers using the K-Means++ D^2 distance-weighted sampling algorithm."""
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=np.float64)

    # Pick the first center uniformly at random
    initial_idx = rng.randint(0, n_samples)
    centers[0] = X[initial_idx]

    # Closest squared distance from each point to current centers
    closest_dist_sq = _euclidean_distances_sq(X, centers[:1]).ravel()

    for c in range(1, n_clusters):
        # Choose next center with probability proportional to D(x)^2
        total_dist_sq = np.sum(closest_dist_sq)
        if total_dist_sq == 0.0:
            probs = np.full(n_samples, 1.0 / n_samples)
        else:
            probs = closest_dist_sq / total_dist_sq

        next_idx = rng.choice(n_samples, p=probs)
        centers[c] = X[next_idx]

        # Update closest distances
        new_dist_sq = _euclidean_distances_sq(X, centers[c : c + 1]).ravel()
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

    return centers


class KMeans(BaseEstimator, ClusterMixin):
    """K-Means Clustering.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters to form and centroids to generate.
    init : {'k-means++', 'random'}, default='k-means++'
        Centroid initialization method.
    n_init : int, default=10
        Number of times the K-Means algorithm will be run with different centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations for a single run.
    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference in the cluster centers.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Literal["k-means++", "random"] = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = float("inf")
        self.n_iter_: int = 0

    def _fit_single_run(
        self, X: np.ndarray, rng: np.random.RandomState
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        n_samples = len(X)

        if self.init == "k-means++":
            centers = _init_kmeans_plus_plus(X, self.n_clusters, rng)
        elif self.init == "random":
            idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
            centers = X[idx].copy()
        else:
            raise ValueError(f"Unknown init '{self.init}'. Choose 'k-means++' or 'random'.")

        labels = np.zeros(n_samples, dtype=np.int64)

        for it in range(self.max_iter):
            # Assignment step
            dist_sq = _euclidean_distances_sq(X, centers)
            new_labels = np.argmin(dist_sq, axis=1)

            # Update step
            new_centers = np.empty_like(centers)
            for k in range(self.n_clusters):
                mask = new_labels == k
                if np.any(mask):
                    new_centers[k] = np.mean(X[mask], axis=0)
                else:
                    # Handle empty cluster: reinitialize with farthest point
                    farthest_idx = np.argmax(np.min(dist_sq, axis=1))
                    new_centers[k] = X[farthest_idx]

            # Check convergence
            center_shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            labels = new_labels

            if center_shift <= self.tol:
                break

        # Compute final inertia (within-cluster sum of squared errors)
        final_dist_sq = _euclidean_distances_sq(X, centers)
        inertia = float(np.sum(np.min(final_dist_sq, axis=1)))

        return centers, labels, inertia, it + 1

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> KMeans:
        """Compute K-Means clustering."""
        X_arr = check_array(X)
        if len(X_arr) < self.n_clusters:
            raise ValueError(f"n_samples={len(X_arr)} must be >= n_clusters={self.n_clusters}.")

        rng = np.random.RandomState(self.random_state)
        best_inertia = float("inf")
        best_centers = None
        best_labels = None
        best_n_iter = 0

        for _ in range(self.n_init):
            centers, labels, inertia, n_iter = self._fit_single_run(X_arr, rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the closest cluster for each sample in X."""
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans instance is not fitted yet.")
        X_arr = check_array(X)
        dist_sq = _euclidean_distances_sq(X_arr, self.cluster_centers_)
        return np.argmin(dist_sq, axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to a cluster-distance space."""
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans instance is not fitted yet.")
        X_arr = check_array(X)
        return np.sqrt(_euclidean_distances_sq(X_arr, self.cluster_centers_))
