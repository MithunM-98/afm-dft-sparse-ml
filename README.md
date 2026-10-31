# Sparse Profile Reconstruction with Machine Learning

This project explores how to reconstruct a high-dimensional response curve from a small subset of strategically selected observations.

The dataset contains 45-dimensional energy profiles generated from computational simulations. Because evaluating every point in a profile can be expensive, the ML objective is:

> **Can we reconstruct the complete 45-point profile using only a small number of informative measurements?**

## Highlights

- Reduced the required input from **45 points to 6 selected points**.
- Used **PCA** to identify strong low-dimensional structure in the response profiles.
- Trained a **multi-output Ridge regression** model to predict the complete 45-dimensional profile.
- Used **greedy forward feature selection with cross-validation** to identify the most informative input positions.
- Compared learned feature selection against an **evenly spaced baseline**.
- Evaluated robustness across **10 independent train/test splits**.
- Performed **spatial holdout validation** to test generalization to unseen regions.
- Preserved complete profiles during train/test splitting to avoid data leakage.

For the initial validation system, the first 3 principal components explained approximately **99.78% of the standardized variance**.

Using only 6 selected observations:

- selected-feature Ridge MAE: **0.00623 ± 0.00103**
- evenly spaced Ridge MAE: **0.04719 ± 0.00479**
- average relative MAE improvement: **86.74 ± 2.19%**

Spatial holdout validation achieved:

- selected-feature MAE: **0.00742 ± 0.00244**
- evenly spaced MAE: **0.04676 ± 0.01420**

The six selected observations correspond to **13.3% of the original input positions**, reducing the number of sampled positions by **86.7%**.

## ML problem formulation

Each sample is represented by a full response vector

\[
\mathbf{y} =
[y_1, y_2, \ldots, y_{45}]
\]

and only a subset of entries is observed:

\[
\mathbf{x} =
[y_{i_1}, y_{i_2}, \ldots, y_{i_k}].
\]

The model learns the mapping

\[
\mathbf{x} \rightarrow \mathbf{y},
\]

where \(k \ll 45\).

This combines three common machine-learning problems:

- **dimensionality reduction**
- **multi-output regression**
- **feature / measurement selection**

## Why this matters

Many scientific and engineering workflows contain expensive measurements or simulations sampled over a grid, time axis, frequency range, or operating condition.

If the response is highly correlated and low-dimensional, it may be possible to observe only a small subset of positions and reconstruct the rest with machine learning.

This project treats the selection of those observations as an experimental-design problem rather than simply using uniformly spaced samples.

## Domain context

The data in this project come from computational chemistry simulations of tip–sample interaction energies used in probe-particle atomic force microscopy.

Each profile contains interaction energies evaluated at 45 vertical separations from 1.6 Å to 6.0 Å.

The chemistry provides the application domain, but the ML workflow itself is general:

```text
expensive high-dimensional profile
            ↓
dimensionality analysis
            ↓
sparse feature selection
            ↓
multi-output regression
            ↓
robust validation
            ↓
downstream physical evaluation
```
