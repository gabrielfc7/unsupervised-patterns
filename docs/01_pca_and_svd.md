# Mathematical Foundations: Principal Component Analysis (PCA) & SVD

This document details the mathematical derivation of dimensionality reduction implemented in `unsupervised_patterns.decomposition`.

---

## 1. Dimensionality Reduction Objective

Given a centered data matrix $X_c \in \mathbb{R}^{N \times D}$ where each feature has zero empirical mean ($\frac{1}{N} \sum_{i=1}^N x_i = \mathbf{0}$), PCA seeks an orthonormal projection matrix $W \in \mathbb{R}^{D \times K}$ ($K \ll D$, $W^T W = I_K$) that maximizes the variance of the projected data $Z = X_c W$:

$$\max_{W} \text{Tr}\left( \frac{1}{N - 1} Z^T Z \right) = \max_W \text{Tr}\left( W^T \left( \frac{1}{N - 1} X_c^T X_c \right) W \right) \quad \text{s.t. } W^T W = I_K$$

---

## 2. Derivation via Eigendecomposition

The sample covariance matrix $C \in \mathbb{R}^{D \times D}$ is symmetric positive semi-definite:

$$C = \frac{1}{N - 1} X_c^T X_c$$

By the Spectral Theorem, $C$ admits an orthogonal spectral decomposition:

$$C = V \Lambda V^T$$

where:
* $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_D)$ with sorted eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$.
* $V = [v_1, v_2, \dots, v_D]$ are the orthonormal eigenvectors (principal direction vectors).

The first $K$ columns of $V$ form the optimal projection subspace. The explained variance along component $k$ is precisely eigenvalue $\lambda_k$.

---

## 3. Derivation via Singular Value Decomposition (SVD)

Rather than explicitly forming the $D \times D$ covariance matrix (which requires $O(N D^2)$ computation and memory), SVD decomposes $X_c$ directly:

$$X_c = U \Sigma V^T$$

where:
* $U \in \mathbb{R}^{N \times K}$ has orthonormal columns ($U^T U = I_K$).
* $\Sigma = \text{diag}(s_1, s_2, \dots, s_K)$ contains singular values sorted in descending order ($s_1 \ge s_2 \ge \dots \ge 0$).
* $V \in \mathbb{R}^{D \times K}$ has orthonormal columns ($V^T V = I_K$).

### 3.1 Exact Parity Between SVD and Eigendecomposition

Substituting SVD into the covariance matrix:

$$C = \frac{1}{N - 1} X_c^T X_c = \frac{1}{N - 1} (V \Sigma U^T) (U \Sigma V^T) = V \left( \frac{\Sigma^2}{N - 1} \right) V^T$$

Hence:
$$\lambda_k = \frac{s_k^2}{N - 1}$$

The right singular vectors $V$ are identical to the eigenvectors of covariance matrix $C$.

---

## 4. Variance Metrics & Whitening

### 4.1 Explained Variance Ratio

$$\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^D \lambda_j} = \frac{s_k^2}{\sum_{j=1}^D s_j^2}$$

### 4.2 Whitening

Whitening rescales projected components to possess spherical identity covariance:

$$\tilde{z}_{ik} = \frac{z_{ik}}{\sqrt{\lambda_k}} = \frac{z_{ik} \sqrt{N - 1}}{s_k}$$
