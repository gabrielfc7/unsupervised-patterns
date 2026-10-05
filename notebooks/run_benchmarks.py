"""Benchmark and demonstration script for unsupervised-patterns."""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from sklearn.datasets import load_iris, load_wine, make_blobs, make_moons
from sklearn.decomposition import PCA as SkPCA
from sklearn.cluster import KMeans as SkKM, DBSCAN as SkDB
from sklearn.metrics import silhouette_score as sk_sil, davies_bouldin_score as sk_db

from unsupervised_patterns.decomposition import PCA
from unsupervised_patterns.clustering import KMeans, GaussianMixture, DBSCAN
from unsupervised_patterns.metrics import silhouette_score, davies_bouldin_score, find_elbow_point


def main():
    print("=" * 80)
    print("           UNSUPERVISED-PATTERNS DEMONSTRATION & BENCHMARKS")
    print("=" * 80)

    # 1. PCA Decomposition Parity on Wine Dataset
    wine = load_wine()
    X_wine = wine.data
    pca_scratch = PCA(n_components=3, method="svd").fit(X_wine)
    pca_sk = SkPCA(n_components=3).fit(X_wine)

    print("\n### 1. PCA Explained Variance Ratio Comparison (Wine Dataset):")
    for i in range(3):
        print(f"  Component {i+1}: Scratch = {pca_scratch.explained_variance_ratio_[i]:.4f} | Sklearn = {pca_sk.explained_variance_ratio_[i]:.4f}")

    # 2. KMeans with KMeans++ vs Sklearn
    X_blobs, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
    km_scratch = KMeans(n_clusters=4, init="k-means++", random_state=42).fit(X_blobs)
    km_sk = SkKM(n_clusters=4, init="k-means++", random_state=42).fit(X_blobs)

    print("\n### 2. K-Means Clustering Parity (Blobs):")
    print(f"  Scratch Inertia: {km_scratch.inertia_:.2f}")
    print(f"  Sklearn Inertia: {km_sk.inertia_:.2f}")

    # 3. Validation Metrics
    sil_scratch = silhouette_score(X_blobs, km_scratch.labels_)
    sil_sk = sk_sil(X_blobs, km_scratch.labels_)
    db_scratch = davies_bouldin_score(X_blobs, km_scratch.labels_)
    db_sk = sk_db(X_blobs, km_scratch.labels_)

    print("\n### 3. Validation Metrics Parity:")
    print(f"  Silhouette Score:     Scratch = {sil_scratch:.6f} | Sklearn = {sil_sk:.6f}")
    print(f"  Davies-Bouldin Score: Scratch = {db_scratch:.6f} | Sklearn = {db_sk:.6f}")

    # 4. GMM with Expectation-Maximization
    gmm = GaussianMixture(n_components=4, covariance_type="full", max_iter=80, random_state=42).fit(X_blobs)
    print("\n### 4. Gaussian Mixture Model (EM Convergence):")
    print(f"  Converged: {gmm.converged_} in {gmm.n_iter_} iterations")
    print(f"  Component Weights: {np.round(gmm.weights_, 3)}")

    # 5. DBSCAN on Non-Linear Moons Dataset
    X_moons, _ = make_moons(n_samples=250, noise=0.08, random_state=42)
    db_scratch = DBSCAN(eps=0.2, min_samples=5).fit(X_moons)
    db_sk = SkDB(eps=0.2, min_samples=5).fit(X_moons)

    n_clusters_scratch = len(set(db_scratch.labels_) - {-1})
    n_clusters_sk = len(set(db_sk.labels_) - {-1})
    print("\n### 5. DBSCAN on Complex Manifold (Moons Dataset):")
    print(f"  Identified Clusters: Scratch = {n_clusters_scratch} | Sklearn = {n_clusters_sk}")
    print(f"  Core Samples Count:  Scratch = {len(db_scratch.core_sample_indices_)} | Sklearn = {len(db_sk.core_sample_indices_)}")

    # 6. Elbow Point Detection
    inertias = [1000.0, 520.0, 160.0, 120.0, 95.0, 75.0]
    elbow_k = find_elbow_point(inertias, ks=[1, 2, 3, 4, 5, 6])
    print(f"\n### 6. Automated Elbow Point: Optimal k = {elbow_k}")
    print("=" * 80)


if __name__ == "__main__":
    main()
