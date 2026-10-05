"""Principal Component Analysis (PCA) implemented via both SVD and Eigendecomposition."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from unsupervised_patterns.base import BaseEstimator, TransformerMixin, check_array


def _flip_svd(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resolve SVD sign ambiguity by ensuring the largest absolute element of each column is positive."""
    max_abs_cols = np.argmax(np.abs(v), axis=1)
    signs = np.sign(v[range(v.shape[0]), max_abs_cols])
    u *= signs
    v *= signs[:, np.newaxis]
    return u, v


class PCA(BaseEstimator, TransformerMixin):
    """Principal Component Analysis (PCA).

    Linear dimensionality reduction using Singular Value Decomposition (SVD)
    or Eigendecomposition of the sample covariance matrix.

    Parameters
    ----------
    n_components : int or float, default=None
        Number of principal components to keep. If None, all components are kept.
        If float in (0, 1), selects the minimum number of components such that the
        cumulative explained variance ratio exceeds this threshold.
    method : {'svd', 'eigen'}, default='svd'
        Mathematical decomposition engine.
    whiten : bool, default=False
        When True, components are divided by singular values to ensure uncorrelated outputs with unit variance.
    """

    def __init__(
        self,
        n_components: Optional[int | float] = None,
        method: Literal["svd", "eigen"] = "svd",
        whiten: bool = False,
    ):
        self.n_components = n_components
        self.method = method
        self.whiten = whiten

        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.n_components_: int = 0
        self.n_samples_: int = 0
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> PCA:
        """Fit the model with X."""
        X_arr = check_array(X)
        n_samples, n_features = X_arr.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        # Center the data
        self.mean_ = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean_

        if self.method == "svd":
            # Singular Value Decomposition of X_centered: X = U * S * Vt
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            U, Vt = _flip_svd(U, Vt)

            explained_variance = (S ** 2) / (n_samples - 1)
            total_var = np.sum(explained_variance)
            explained_variance_ratio = explained_variance / total_var

            components = Vt
            singular_values = S

        elif self.method == "eigen":
            # Sample covariance matrix C = (1 / (N - 1)) * Xc^T * Xc
            cov = (X_centered.T @ X_centered) / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort eigenvalues and eigenvectors in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[idx], 0.0)
            components = eigenvectors[:, idx].T

            # Sign flip
            max_abs_cols = np.argmax(np.abs(components), axis=1)
            signs = np.sign(components[range(components.shape[0]), max_abs_cols])
            components *= signs[:, np.newaxis]

            explained_variance = eigenvalues
            total_var = np.sum(explained_variance)
            explained_variance_ratio = explained_variance / total_var if total_var > 0 else np.zeros_like(explained_variance)
            singular_values = np.sqrt(explained_variance * (n_samples - 1))

        else:
            raise ValueError(f"Unknown method '{self.method}'. Choose 'svd' or 'eigen'.")

        # Determine number of components
        if self.n_components is None:
            k = min(n_samples, n_features)
        elif isinstance(self.n_components, float) and 0.0 < self.n_components < 1.0:
            cum_ratio = np.cumsum(explained_variance_ratio)
            k = int(np.searchsorted(cum_ratio, self.n_components)) + 1
        elif isinstance(self.n_components, int):
            k = min(self.n_components, min(n_samples, n_features))
        else:
            raise ValueError(f"Invalid n_components value: {self.n_components}")

        self.n_components_ = k
        self.components_ = components[:k]
        self.explained_variance_ = explained_variance[:k]
        self.explained_variance_ratio_ = explained_variance_ratio[:k]
        self.singular_values_ = singular_values[:k]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to X."""
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA model is not fitted.")
        X_arr = check_array(X)
        X_centered = X_arr - self.mean_
        X_projected = X_centered @ self.components_.T

        if self.whiten:
            scale = np.sqrt(self.explained_variance_)
            scale[scale == 0] = 1.0
            X_projected /= scale

        return X_projected

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data back to its original space."""
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA model is not fitted.")
        X_arr = check_array(X)

        if self.whiten:
            scale = np.sqrt(self.explained_variance_)
            X_recon = (X_arr * scale) @ self.components_ + self.mean_
        else:
            X_recon = X_arr @ self.components_ + self.mean_

        return X_recon

    def scree_data(self) -> dict[str, np.ndarray]:
        """Return eigenvalues and cumulative explained variance ratio for scree plotting."""
        if self.explained_variance_ratio_ is None:
            raise RuntimeError("PCA model is not fitted.")
        return {
            "component": np.arange(1, self.n_components_ + 1),
            "explained_variance": self.explained_variance_,
            "explained_variance_ratio": self.explained_variance_ratio_,
            "cumulative_ratio": np.cumsum(self.explained_variance_ratio_),
        }
