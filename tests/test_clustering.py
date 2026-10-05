"""Tests for KMeans, GMM, and DBSCAN clustering algorithms."""

import numpy as np
import pytest
from sklearn.cluster import DBSCAN as SklearnDBSCAN, KMeans as SklearnKMeans
from sklearn.mixture import GaussianMixture as SklearnGMM

from unsupervised_patterns.clustering import DBSCAN, GaussianMixture, KMeans


def test_kmeans_convergence_and_parity(blobs_data):
    X, _ = blobs_data
    # 4 distinct blob clusters
    scratch_km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
    scratch_km.fit(X)

    sk_km = SklearnKMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
    sk_km.fit(X)

    # Inertias should be nearly identical (within 2% due to random seed in initialization)
    assert abs(scratch_km.inertia_ - sk_km.inertia_) / sk_km.inertia_ < 0.02
    assert scratch_km.cluster_centers_.shape == (4, X.shape[1])
    assert len(np.unique(scratch_km.labels_)) == 4


def test_gmm_em_convergence(blobs_data):
    X, _ = blobs_data
    gmm = GaussianMixture(n_components=4, covariance_type="full", max_iter=80, random_state=42)
    gmm.fit(X)

    assert gmm.converged_
    assert gmm.weights_.shape == (4,)
    np.testing.assert_allclose(np.sum(gmm.weights_), 1.0, atol=1e-5)

    probs = gmm.predict_proba(X)
    assert probs.shape == (len(X), 4)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(len(X)), atol=1e-5)

    preds = gmm.predict(X)
    assert len(np.unique(preds)) == 4


def test_dbscan_parity(moons_data):
    X, _ = moons_data

    # Scikit-learn DBSCAN
    sk_db = SklearnDBSCAN(eps=0.2, min_samples=5)
    sk_labels = sk_db.fit_predict(X)

    # Scratch DBSCAN
    scratch_db = DBSCAN(eps=0.2, min_samples=5)
    scratch_labels = scratch_db.fit_predict(X)

    # Compare core samples and number of clusters found
    assert len(scratch_db.core_sample_indices_) == len(sk_db.core_sample_indices_)
    np.testing.assert_array_equal(scratch_db.core_sample_indices_, sk_db.core_sample_indices_)

    # Cluster label counts (excluding permutation of cluster IDs)
    n_scratch_clusters = len(set(scratch_labels) - {-1})
    n_sk_clusters = len(set(sk_labels) - {-1})
    assert n_scratch_clusters == n_sk_clusters
