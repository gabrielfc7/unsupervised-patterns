"""Tests for cluster validation metrics asserting exact parity with scikit-learn."""

import numpy as np
import pytest
from sklearn.metrics import (
    davies_bouldin_score as sk_davies_bouldin_score,
    silhouette_score as sk_silhouette_score,
)

from unsupervised_patterns.metrics import (
    davies_bouldin_score,
    find_elbow_point,
    silhouette_score,
)


def test_silhouette_score_parity(blobs_data):
    X, labels = blobs_data
    # Compare against scikit-learn
    scratch_sil = silhouette_score(X, labels)
    sk_sil = sk_silhouette_score(X, labels)

    assert pytest.approx(scratch_sil, abs=1e-7) == sk_sil
    assert scratch_sil > 0.50


def test_davies_bouldin_score_parity(blobs_data):
    X, labels = blobs_data
    scratch_db = davies_bouldin_score(X, labels)
    sk_db = sk_davies_bouldin_score(X, labels)

    assert pytest.approx(scratch_db, abs=1e-7) == sk_db
    assert scratch_db < 1.0


def test_find_elbow_point():
    # Synthetic inertia curve with clear elbow at k=3
    ks = [1, 2, 3, 4, 5, 6, 7]
    inertias = [1000.0, 500.0, 150.0, 110.0, 90.0, 75.0, 60.0]

    elbow_k = find_elbow_point(inertias, ks)
    assert elbow_k == 3
