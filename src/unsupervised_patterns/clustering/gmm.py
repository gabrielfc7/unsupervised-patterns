"""Gaussian Mixture Models (GMM) with full Expectation-Maximization (EM) derivation."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy.special import logsumexp

from unsupervised_patterns.base import BaseEstimator, ClusterMixin, check_array
from unsupervised_patterns.clustering.kmeans import KMeans


def _log_multivariate_normal_density(
    X: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    cov_type: str = "full",
    reg_covar: float = 1e-6,
) -> np.ndarray:
    """Compute log Gaussian probability density log N(X | means[k], cov[k]) for each component k."""
    n_samples, n_features = X.shape
    n_components = len(means)
    log_probs = np.empty((n_samples, n_components), dtype=np.float64)

    const = -0.5 * n_features * np.log(2.0 * np.pi)

    for k in range(n_components):
        diff = X - means[k]
        if cov_type == "full":
            cov = covariances[k] + reg_covar * np.eye(n_features)
            # Cholesky decomposition for stable determinant and solve
            try:
                L = np.linalg.cholesky(cov)
                # Solve L * y = diff.T -> y has shape (n_features, n_samples)
                y = np.linalg.solve(L, diff.T)
                mahalanobis_sq = np.sum(y ** 2, axis=0)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                sign, log_det = np.linalg.slogdet(cov)
                inv_cov = np.linalg.pinv(cov)
                mahalanobis_sq = np.sum((diff @ inv_cov) * diff, axis=1)

            log_probs[:, k] = const - 0.5 * log_det - 0.5 * mahalanobis_sq

        elif cov_type == "diag":
            cov_diag = covariances[k] + reg_covar
            log_det = np.sum(np.log(cov_diag))
            mahalanobis_sq = np.sum((diff ** 2) / cov_diag, axis=1)
            log_probs[:, k] = const - 0.5 * log_det - 0.5 * mahalanobis_sq

    return log_probs


class GaussianMixture(BaseEstimator, ClusterMixin):
    """Gaussian Mixture Model.

    Representation of a normal distribution mixture with Expectation-Maximization (EM) parameter estimation.

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.
    covariance_type : {'full', 'diag'}, default='full'
        String describing the type of covariance parameters to use.
    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.
    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance matrices.
    max_iter : int, default=100
        The number of EM iterations to perform.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: Literal["full", "diag"] = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state

        self.weights_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float = float("-inf")

    def _estimate_weighted_log_prob(self, X: np.ndarray) -> np.ndarray:
        """Estimate log P(X, Z) = log pi_k + log N(X | mu_k, Sigma_k)."""
        log_prob = _log_multivariate_normal_density(
            X, self.means_, self.covariances_, self.covariance_type, self.reg_covar
        )
        return log_prob + np.log(self.weights_)

    def _e_step(self, X: np.ndarray) -> tuple[float, np.ndarray]:
        """Expectation step: compute log-likelihood and posterior responsibilities."""
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return float(np.mean(log_prob_norm)), log_resp

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray) -> None:
        """Maximization step: update weights, means, and covariances."""
        n_samples, n_features = X.shape
        resp = np.exp(log_resp)

        nk = np.sum(resp, axis=0) + 10.0 * np.finfo(np.float64).eps
        self.weights_ = nk / n_samples
        self.means_ = (resp.T @ X) / nk[:, np.newaxis]

        if self.covariance_type == "full":
            self.covariances_ = np.empty((self.n_components, n_features, n_features), dtype=np.float64)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                weighted_diff = diff * resp[:, k : k + 1]
                self.covariances_[k] = (weighted_diff.T @ diff) / nk[k]
                self.covariances_[k].flat[:: n_features + 1] += self.reg_covar

        elif self.covariance_type == "diag":
            self.covariances_ = np.empty((self.n_components, n_features), dtype=np.float64)
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covariances_[k] = np.sum(resp[:, k : k + 1] * (diff ** 2), axis=0) / nk[k] + self.reg_covar

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> GaussianMixture:
        """Estimate model parameters with the EM algorithm."""
        X_arr = check_array(X)
        n_samples, n_features = X_arr.shape

        if n_samples < self.n_components:
            raise ValueError(f"n_samples={n_samples} must be >= n_components={self.n_components}")

        rng = np.random.RandomState(self.random_state)

        # Initialize parameters via K-Means
        kmeans = KMeans(n_clusters=self.n_components, n_init=1, random_state=self.random_state)
        init_labels = kmeans.fit_predict(X_arr)
        self.means_ = kmeans.cluster_centers_.copy()

        # Initial weights and covariances
        self.weights_ = np.bincount(init_labels, minlength=self.n_components) / n_samples

        if self.covariance_type == "full":
            self.covariances_ = np.empty((self.n_components, n_features, n_features), dtype=np.float64)
            for k in range(self.n_components):
                mask = init_labels == k
                if np.sum(mask) > 1:
                    self.covariances_[k] = np.cov(X_arr[mask], rowvar=False) + self.reg_covar * np.eye(n_features)
                else:
                    self.covariances_[k] = np.eye(n_features) * (1.0 + self.reg_covar)
        else:
            self.covariances_ = np.empty((self.n_components, n_features), dtype=np.float64)
            for k in range(self.n_components):
                mask = init_labels == k
                if np.sum(mask) > 1:
                    self.covariances_[k] = np.var(X_arr[mask], axis=0) + self.reg_covar
                else:
                    self.covariances_[k] = np.ones(n_features) * (1.0 + self.reg_covar)

        prev_lower_bound = float("-inf")
        self.converged_ = False

        for it in range(self.max_iter):
            # E-step
            lower_bound, log_resp = self._e_step(X_arr)

            # M-step
            self._m_step(X_arr, log_resp)

            change = lower_bound - prev_lower_bound
            if it > 0 and 0.0 <= change < self.tol:
                self.converged_ = True
                self.n_iter_ = it + 1
                self.lower_bound_ = lower_bound
                break

            prev_lower_bound = lower_bound
        else:
            self.n_iter_ = self.max_iter
            self.lower_bound_ = prev_lower_bound

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior probability of each component given the data."""
        X_arr = check_array(X)
        _, log_resp = self._e_step(X_arr)
        return np.exp(log_resp)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the labels for the data samples in X using trained model."""
        return np.argmax(self.predict_proba(X), axis=1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute the log-likelihood of each sample."""
        X_arr = check_array(X)
        weighted_log_prob = self._estimate_weighted_log_prob(X_arr)
        return logsumexp(weighted_log_prob, axis=1)

    def score(self, X: np.ndarray) -> float:
        """Compute the per-sample average log-likelihood of the given data X."""
        return float(np.mean(self.score_samples(X)))
