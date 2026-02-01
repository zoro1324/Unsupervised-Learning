# K-Means Clustering

> An unsupervised machine learning algorithm for grouping data into $k$ clusters by minimizing within-cluster variance.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Use Cases](#use-cases)
3. [Input & Output](#input--output)
4. [Data Preprocessing](#data-preprocessing)
5. [Algorithm Workflow](#algorithm-workflow)
6. [Hyperparameters](#hyperparameters)
7. [Assumptions](#assumptions)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Pros & Cons](#pros--cons)
10. [Implementation Example](#implementation-example)

---

## Overview

| Attribute  | Description              |
|------------|--------------------------|
| **Type**   | Unsupervised Learning    |
| **Task**   | Clustering               |
| **Library**| `sklearn.cluster.KMeans` |

K-Means partitions data into $k$ clusters by iteratively assigning each point to the nearest centroid and updating centroids to minimize the within-cluster sum of squares (inertia).

---

## Use Cases

| Scenario                           | Suitability                      |
|------------------------------------|----------------------------------|
| Customer segmentation              | ✅ Excellent                     |
| Document/image grouping            | ✅ Excellent                     |
| Market basket exploration          | ✅ Good                          |
| Large-scale datasets               | ✅ Good                          |
| Non-spherical clusters             | ⚠️ Consider other algorithms     |

---

## Input & Output

| Component      | Description                                                       |
|----------------|-------------------------------------------------------------------|
| **Input (X)**  | Numerical feature matrix *(categorical features must be encoded)* |
| **Output**     | Cluster labels and centroid positions                             |

---

## Data Preprocessing

| Preprocessing Step   | Required   | Notes                                             |
|-----------------------|------------|---------------------------------------------------|
| Feature Scaling      | ✅ Yes     | Strongly recommended (distance-based algorithm)  |
| Missing Value Handling | ✅ Yes     | Impute or remove missing values                   |
| Categorical Encoding   | ✅ Yes     | Use `OneHotEncoder` or similar                    |
| Outlier Treatment      | ⚠️ Optional| Can distort centroids and cluster assignments     |

---

## Algorithm Workflow

```text
┌─────────────────────────────────────────────────────────────┐
│  1. Initialize $k$ centroids (random or k-means++)          │
│                           ↓                                 │
│  2. Assign each point to the nearest centroid               │
│                           ↓                                 │
│  3. Update centroids as the mean of assigned points         │
│                           ↓                                 │
│  4. Repeat until assignments stabilize or max iterations    │
└─────────────────────────────────────────────────────────────┘
```

---

## Hyperparameters

| Parameter        | Type | Default     | Description                          |
|------------------|------|-------------|--------------------------------------|
| `n_clusters`     | int  | 8           | Number of clusters to form           |
| `init`           | str  | "k-means++" | Initialization strategy              |
| `max_iter`       | int  | 300         | Maximum number of iterations         |
| `n_init`         | int  | 10          | Number of initializations to try     |
| `random_state`   | int  | None        | Reproducibility seed                 |

---

## Assumptions

- Clusters are roughly spherical and similar in size
- Distances are meaningful in the feature space
- Features are on comparable scales

---

## Evaluation Metrics

| Metric               | Formula                                           | Interpretation                        |
|-----------------------|----------------------------------------------------|------------------------------------------|
| **Inertia (WCSS)**   | $\sum_{i=1}^{n} \lVert x_i - \mu_{c_i} \rVert^2$  | Within-cluster variance (lower is better) |
| **Silhouette**       | $\frac{b - a}{\max(a, b)}$                        | Cluster separation (higher is better)     |
| **Calinski-Harabasz**| $\frac{\text{between-cluster}}{\text{within-cluster}}$ | Higher indicates better-defined clusters  |

---

## Pros & Cons

| ✅ Advantages                 | ❌ Disadvantages                        |
|---------------------------------|---------------------------------------------|
| Simple and fast               | Requires choosing $k$                     |
| Scales to large datasets      | Sensitive to initialization               |
| Easy to interpret             | Struggles with non-spherical clusters     |
| Works well with clear separation | Sensitive to outliers and feature scaling |

---

## Implementation Example

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model initialization
model = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)

# Training
labels = model.fit_predict(X_scaled)

# Evaluation
print(f"Inertia: {model.inertia_:.2f}")
print(f"Silhouette: {silhouette_score(X_scaled, labels):.4f}")
```

---

**📚 Related:** DBSCAN | Hierarchical Clustering | Gaussian Mixture Models
