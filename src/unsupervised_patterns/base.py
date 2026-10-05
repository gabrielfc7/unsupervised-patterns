"""Base classes and array checking utilities for unsupervised patterns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEstimator(ABC):
    """Abstract base estimator."""

    def get_params(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.endswith("_") and not k.startswith("_")}

    def set_params(self, **params: Any) -> BaseEstimator:
        for k, v in params.items():
            setattr(self, k, v)
        return self


class TransformerMixin:
    """Mixin for transformer estimators."""

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)  # type: ignore[attr-defined]


class ClusterMixin:
    """Mixin for clustering algorithms."""

    def fit_predict(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        self.fit(X, y)  # type: ignore[attr-defined]
        if hasattr(self, "labels_") and self.labels_ is not None:
            return self.labels_
        return self.predict(X)  # type: ignore[attr-defined]


def check_array(X: Any, ensure_2d: bool = True) -> np.ndarray:
    """Validate numeric input array."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Input array must not be empty.")
    if ensure_2d and arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif ensure_2d and arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Input array contains NaN or Inf.")
    return np.ascontiguousarray(arr)
