"""Test fixtures for unsupervised patterns test suite."""

import pytest
from sklearn.datasets import load_iris, load_wine, make_blobs, make_moons


@pytest.fixture
def blobs_data():
    X, y = make_blobs(n_samples=250, centers=4, n_features=5, cluster_std=0.8, random_state=42)
    return X, y


@pytest.fixture
def moons_data():
    X, y = make_moons(n_samples=200, noise=0.08, random_state=42)
    return X, y


@pytest.fixture
def iris_data():
    data = load_iris()
    return data.data, data.target


@pytest.fixture
def wine_data():
    data = load_wine()
    return data.data, data.target
