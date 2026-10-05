"""Tests for Principal Component Analysis asserting parity with scikit-learn."""

import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

from unsupervised_patterns.decomposition import PCA


def test_pca_svd_parity_with_sklearn(iris_data):
    X, _ = iris_data

    # Scikit-learn PCA
    sk_pca = SklearnPCA(n_components=3, random_state=42)
    sk_trans = sk_pca.fit_transform(X)

    # Scratch PCA (SVD)
    scratch_pca = PCA(n_components=3, method="svd")
    scratch_trans = scratch_pca.fit_transform(X)

    # Check explained variance parity
    np.testing.assert_allclose(scratch_pca.explained_variance_, sk_pca.explained_variance_, rtol=1e-5)
    np.testing.assert_allclose(scratch_pca.explained_variance_ratio_, sk_pca.explained_variance_ratio_, rtol=1e-5)

    # Check projection parity (up to sign flip, but _flip_svd aligns signs)
    np.testing.assert_allclose(np.abs(scratch_trans), np.abs(sk_trans), atol=1e-5)

    # Check reconstruction
    X_recon = scratch_pca.inverse_transform(scratch_trans)
    assert np.mean((X - X_recon) ** 2) < 0.1


def test_pca_eigen_parity_with_svd(wine_data):
    X, _ = wine_data

    pca_svd = PCA(n_components=4, method="svd").fit(X)
    pca_eigen = PCA(n_components=4, method="eigen").fit(X)

    # Eigenvalues and SVD singular squared values must match to 1e-4
    np.testing.assert_allclose(pca_svd.explained_variance_, pca_eigen.explained_variance_, rtol=1e-4)
    np.testing.assert_allclose(pca_svd.explained_variance_ratio_, pca_eigen.explained_variance_ratio_, rtol=1e-4)


def test_pca_scree_data(wine_data):
    X, _ = wine_data
    pca = PCA(n_components=5).fit(X)
    scree = pca.scree_data()

    assert len(scree["component"]) == 5
    assert scree["cumulative_ratio"][-1] > 0.70
    assert np.all(np.diff(scree["cumulative_ratio"]) >= 0)
