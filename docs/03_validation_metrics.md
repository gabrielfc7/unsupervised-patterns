# Mathematical Foundations: Cluster Evaluation Metrics

This document formalizes clustering validation metrics implemented in `unsupervised_patterns.metrics`.

---

## 1. Silhouette Coefficient

Evaluates both intra-cluster cohesion and inter-cluster separation without requiring ground truth labels.

For sample $i \in C_I$:

1. **Mean Intra-Cluster Distance $a(i)$:**
   $$a(i) = \frac{1}{|C_I| - 1} \sum_{j \in C_I, j \ne i} \|x_i - x_j\|$$

2. **Mean Nearest-Cluster Distance $b(i)$:**
   $$b(i) = \min_{J \ne I} \frac{1}{|C_J|} \sum_{j \in C_J} \|x_i - x_j\|$$

3. **Silhouette Value $s(i)$:**
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]$$

* $s(i) \approx +1$: Sample is well clustered and far from neighboring clusters.
* $s(i) \approx 0$: Sample is on or very close to the decision boundary between two clusters.
* $s(i) < 0$: Sample is likely assigned to the wrong cluster.

---

## 2. Davies-Bouldin Index

Measures the average similarity between each cluster and its most similar rival.

1. **Cluster Dispersion $s_i$:**
   $$s_i = \frac{1}{|C_i|} \sum_{x \in C_i} \|x - \mu_i\|$$

2. **Cluster Similarity Ratio $R_{ij}$:**
   $$R_{ij} = \frac{s_i + s_j}{\|\mu_i - \mu_j\|}$$

3. **Davies-Bouldin Score ($DB$):**
   $$DB = \frac{1}{K} \sum_{i=1}^K \max_{j \ne i} R_{ij}$$

Lower values indicate better clustering (higher between-cluster separation relative to within-cluster spread).

---

## 3. Elbow-Point Detection via Orthogonal Distance to Secant

Given the inertia curve $(k, J_k)$ for $k \in [k_{\min}, k_{\max}]$:

Let line $\mathcal{L}$ connect endpoints $(k_1, J_1)$ and $(k_M, J_M)$. The optimal number of clusters $k^*$ maximizes the perpendicular Euclidean distance from point $(k, J_k)$ to secant line $\mathcal{L}$.
