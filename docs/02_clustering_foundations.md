# Mathematical Foundations: Clustering (K-Means, GMM, DBSCAN)

This document formalizes the mathematical mechanisms of clustering algorithms implemented in `unsupervised_patterns.clustering`.

---

## 1. K-Means Clustering & K-Means++

### 1.1 Objective (Inertia)

Given samples $\{x_1, \dots, x_N\}$, K-Means minimizes the Within-Cluster Sum of Squares (WCSS / Inertia):

$$J(\mu) = \sum_{i=1}^N \min_{k \in \{1, \dots, K\}} \|x_i - \mu_k\|^2$$

### 1.2 Lloyd's Algorithm Iteration

1. **Assignment step:**
   $$c_i = \arg\min_k \|x_i - \mu_k\|^2$$
2. **Update step:**
   $$\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i, \quad \text{where } C_k = \{ i \mid c_i = k \}$$

### 1.3 K-Means++ Initialization

To guarantee an $O(\log K)$ approximation bound against the optimal clustering:
1. Choose initial center $\mu_1$ uniformly at random from dataset $X$.
2. Compute distance from each sample to closest existing center: $D(x) = \min_{j < k} \|x - \mu_j\|$.
3. Sample next center $\mu_k$ with probability proportional to squared distance:

$$P(x) = \frac{D(x)^2}{\sum_{x' \in X} D(x')^2}$$

---

## 2. Gaussian Mixture Models & Expectation-Maximization (EM)

A GMM models the data distribution as a linear superposition of $K$ Gaussian densities:

$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k), \quad \text{with } \sum_{k=1}^K \pi_k = 1, \; \pi_k \ge 0$$

### 2.1 E-Step (Posterior Responsibilities)

By Bayes' Theorem, the posterior probability (responsibility) that component $k$ generated sample $x_i$:

$$\gamma_{ik} \equiv P(z_i = k \mid x_i) = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$

### 2.2 M-Step (Parameter Updates)

Maximizing the expected complete-data log-likelihood with respect to parameters yields closed-form updates:

$$N_k = \sum_{i=1}^N \gamma_{ik}$$

$$\pi_k = \frac{N_k}{N}$$

$$\mu_k = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} x_i$$

$$\Sigma_k = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T$$

---

## 3. DBSCAN (Density-Based Spatial Clustering)

DBSCAN locates clusters as contiguous regions of high point density separated by low-density regions.

### 3.1 Density Definitions

* **$\epsilon$-Neighborhood:** $N_\epsilon(p) = \{ q \in X \mid \|p - q\| \le \epsilon \}$
* **Core Point:** Sample $p$ is a core point if $|N_\epsilon(p)| \ge \text{min\_samples}$.
* **Direct Density-Reachable:** Sample $q$ is directly density-reachable from $p$ if $p$ is a core point and $q \in N_\epsilon(p)$.
* **Density-Connected:** Points $p$ and $q$ are density-connected if there exists an intermediate point $o$ such that both $p$ and $q$ are density-reachable from $o$.
* **Noise / Outliers:** Points that are neither core points nor border points are assigned label $-1$.
