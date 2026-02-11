# DBSCAN Clustering

> A density-based unsupervised machine learning algorithm for grouping data into clusters based on density, identifying noise points.

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
| **Library**| `sklearn.cluster.DBSCAN` |

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups together points that are closely packed, marking points in low-density regions as outliers.

---

## Use Cases

| Scenario                           | Suitability                      |
|------------------------------------|----------------------------------|
| Arbitrary shaped clusters          | ✅ Excellent                     |
| Noise and outlier detection        | ✅ Excellent                     |
| Spatial data analysis              | ✅ Excellent                     |
| Customer segmentation with noise   | ✅ Good                          |
| Large-scale datasets               | ✅ Good                          |
| Spherical clusters                 | ⚠️ May over-segment              |

---

## Input & Output

| Component      | Description                                                       |
|----------------|-------------------------------------------------------------------|
| **Input (X)**  | Numerical feature matrix *(categorical features must be encoded)* |
| **Output**     | Cluster labels (-1 for noise) and core sample indices             |

---

## Data Preprocessing

| Preprocessing Step      | Required     | Notes                                         |
|------------------------|--------------|-----------------------------------------------|
| Feature Scaling        | ✅ Yes       | Strongly recommended (distance-based algorithm)|
| Missing Value Handling | ✅ Yes       | Impute or remove missing values                |
| Categorical Encoding   | ✅ Yes       | Use `OneHotEncoder` or similar                 |
| Outlier Treatment      | ❌ No        | Algorithm handles outliers naturally           |

---

## Algorithm Workflow

```text
┌─────────────────────────────────────────────────────────────┐
│  1. For each point, find neighbors within eps distance      │
│                           ↓                                 │
│  2. Classify as core if min_samples neighbors, border otherwise │
│                           ↓                                 │
│  3. Form clusters by connecting core points                 │
│                           ↓                                 │
│  4. Assign noise points (-1) to unclustered points          │
└─────────────────────────────────────────────────────────────┘
```

---

## Hyperparameters

| Parameter        | Type | Default     | Description                          |
|------------------|------|-------------|--------------------------------------|
| `eps`            | float| 0.5         | Maximum distance between points      |
| `min_samples`    | int  | 5           | Minimum samples in neighborhood      |
| `metric`         | str  | "euclidean" | Distance metric                      |
| `algorithm`      | str  | "auto"      | Algorithm for nearest neighbors      |

---

## Assumptions

- Dense regions separated by sparse regions
- Density within clusters is roughly uniform
- Distance metric is meaningful

---

## Evaluation Metrics

| Metric               | Formula                                           | Interpretation                        |
|-----------------------|----------------------------------------------------|------------------------------------------|
| **Silhouette**       | $\frac{b - a}{\max(a, b)}$                        | Cluster separation (higher is better)     |
| **Calinski-Harabasz**| $\frac{\text{between-cluster}}{\text{within-cluster}}$ | Higher indicates better-defined clusters  |
| **Adjusted Rand Index**| Agreement with ground truth                       | Higher indicates better clustering        |

---

## Pros & Cons

| ✅ Advantages                 | ❌ Disadvantages                        |
|---------------------------------|---------------------------------------------|
| Handles arbitrary shapes      | Sensitive to eps and min_samples          |
| Automatically detects noise   | Struggles with varying densities          |
| No need to specify k          | Cannot cluster sparse regions             |
| Robust to outliers            | Computationally intensive for large datasets|

---

## Implementation Example

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model initialization
model = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')

# Training
labels = model.fit_predict(X_scaled)

# Number of clusters (excluding noise)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Number of clusters: {n_clusters}")

# Evaluation (excluding noise)
if n_clusters > 1:
    print(f"Silhouette: {silhouette_score(X_scaled[labels != -1], labels[labels != -1]):.4f}")
```

---

**📚 Related:** K-Means | Hierarchical Clustering | OPTICS
