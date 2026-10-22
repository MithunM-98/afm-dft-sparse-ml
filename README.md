# Sparse reconstruction of DFT tip–sample interaction profiles for PP-AFM

This project investigates whether complete DFT tip–sample interaction-energy profiles can be reconstructed from a small number of strategically selected tip heights.

The underlying calculations contain interaction energies sampled at 45 vertical tip–sample separations,

\[
z = 1.6, 1.7,........, 5.9, 6.0\ \mathrm{\AA},
\]

for many non-uniformly distributed lateral sampling positions \(p_i=(x_i,y_i)\). The original interaction-energy data are used to build a continuous 3D potential-energy landscape for probe-particle AFM (PP-AFM) simulations.

The ML question explored here is:

> How many DFT tip heights are needed to reconstruct the full 45-point interaction-energy profile \(E(z)\), and which heights are the most informative?

## Main result

For the representative `OO-h-NH3_N` sample–tip system:

- 151 lateral interaction profiles were available.
- Each profile contained 45 DFT energies from 1.6 to 6.0 Å.
- PCA showed that the first **3 principal components explain 99.78%** of the standardized profile variance.
- A multi-output Ridge regression model was trained to reconstruct the full 45-point profile from a sparse subset of energies.
- The informative tip heights were chosen using greedy forward feature selection performed only on the training data.

Using **6 selected tip heights** instead of all 45:

- 10 random train/test splits:
  - selected-height Ridge MAE: **0.00623 ± 0.00103**
  - evenly spaced Ridge MAE: **0.04719 ± 0.00479**
  - average MAE reduction: **86.74 ± 2.19%**
- 5 spatial holdout folds:
  - selected-height Ridge MAE: **0.00742 ± 0.00244**
  - evenly spaced Ridge MAE: **0.04676 ± 0.01420**

For one representative split, the six selected heights were:

```text
1.6, 1.8, 2.2, 2.4, 3.6, 5.0 Å
```

These six heights correspond to **13.3% of the original 45 vertical sampling points**, i.e. **86.7% fewer sampled tip heights**.

This should not be interpreted as an 86.7% reduction in DFT wall time, because computational cost and convergence need not scale exactly with the number of sampled geometries.

## Why this is useful

The interaction profiles are smooth and strongly correlated across tip height. The dimensionality analysis shows that their variation is highly structured, which makes sparse reconstruction possible.

The selected heights also follow a physically sensible pattern: most measurements are concentrated in the rapidly varying short-range part of the interaction, while a smaller number of longer-range points anchor the smooth asymptotic region.

The long-term goal is to determine whether sparse DFT sampling can preserve the interaction potential accurately enough for downstream PP-AFM image simulation.

## Current validated scope

The strongest validation currently covers:

1. unseen lateral profiles from the same sample–tip system,
2. repeated random train/test splits,
3. spatially separated holdout regions.

The preprocessing pipeline has successfully parsed **132 sample–tip datasets**, but cross-system chemical generalization has not yet been established. Results should therefore not yet be interpreted as universal across unseen molecules or tips.

## Data format

### Interaction-energy file

Each `SAMPLE-TIP.dat` file contains:

```text
p000 z00 ENERGY
p000 z01 ENERGY
...
p150 z44 ENERGY
```

where:

```text
z00 = 1.6 Å
z01 = 1.7 Å
...
z44 = 6.0 Å
```

### Lateral coordinates

Each sample has a corresponding:

```text
SAMPLE_points_clean.txt
```

with columns:

```text
type x y
```

The row order defines the lateral index:

```text
row 0 -> p000
row 1 -> p001
...
```

The number of lateral points can differ between samples.

## Workflow

```text
Raw DFT energies + lateral coordinates
                |
                v
        Data preparation
                |
                v
      Np x 45 E(z) matrix
                |
        +-------+-------+
        |               |
        v               v
       PCA       Sparse reconstruction
                        |
                        v
                 Ridge regression
                        |
                        v
               z-height selection
                        |
             +----------+----------+
             |                     |
             v                     v
      random robustness      spatial holdout
             |
             v
     reconstructed .dat
             |
             v
        PP-AFM simulation
```

## Repository structure

```text
afm-dft-sparse-ml/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── 01_prepare_data.py
│   ├── 02_sparse_reconstruction.py
│   ├── 03_select_z_heights.py
│   ├── 04_robustness.py
│   ├── 05_spatial_holdout.py
│   ├── 07_export_reconstructed_dat.py
│   └── 08_run_afm_from_reconstructed.py
│
├── data/
│   └── example/
│       └── README.md
│
├── figures/
│   ├── pca_explained_variance.png
│   ├── selected_vs_even_mae.png
│   ├── z_selection_frequency.png
│   └── spatial_holdout_mae.png
│
└── results/
    └── summary_tables/
```

Large raw DFT datasets, generated force fields, reconstructed datasets, and PP-AFM arrays are intentionally not tracked by Git.

## Methods

### 1. Data preparation

The raw `pXXX zXX energy` files are converted into:

- a long table containing `(p, x, y, z, energy)`,
- an \(N_p \times 45\) energy matrix,
- validated coordinate and metadata files.

Complete vertical profiles are kept together during train/test splitting to prevent row-level leakage.

### 2. PCA

PCA is applied to standardized interaction profiles to characterize their effective dimensionality.

For the representative `OO-h_NH3_N` system, 3 components explain approximately 99.78% of the standardized profile variance.

### 3. Sparse reconstruction

The input to the regression model is the DFT energy at a small number of selected heights:

\[
[E(z_{i_1}), E(z_{i_2}), \ldots, E(z_{i_k})]
\]

and the target is the complete 45-point profile:

\[
[E(z_0), E(z_1), \ldots, E(z_{44})].
\]

A multi-output Ridge regression model is used as a deliberately simple and interpretable baseline.

### 4. Data-driven height selection

Greedy forward feature selection chooses tip heights using cross-validation on the training set only.

At each step:

1. each remaining candidate height is tested,
2. Ridge reconstruction error is evaluated by cross-validation,
3. the best candidate is added,
4. the procedure repeats until the desired number of heights is selected.

The held-out test profiles are not used during height selection.

### 5. Robustness validation

The complete selection-and-evaluation procedure is repeated over multiple independent train/test splits.

### 6. Spatial holdout

The \((x,y)\) sampling plane is divided into compact spatial regions using geometry only. One complete region is held out at a time to test reconstruction away from training locations.

### 7. Reconstructed energy files

The model can export reconstructed interaction energies in the original format:

```text
pXXX zXX ENERGY
```

At selected heights, the original DFT values are retained exactly. Only the hidden heights are reconstructed.

For unbiased comparison, out-of-fold predictions can be generated so that a profile is not reconstructed by a model trained on that same profile.

## PP-AFM validation

The reconstructed `.dat` files can be passed through the same kriging + PP-AFM simulation workflow as the original DFT data.

The intended downstream comparison is:

```text
Original DFT .dat                 Sparse DFT + ML .dat
       |                                  |
       v                                  v
    Kriging                            Kriging
       |                                  |
       v                                  v
     GridFF                              GridFF
       |                                  |
       v                                  v
   PP-AFM image                      PP-AFM image
             \                        /
              \------ comparison ----/
```

This downstream image-level validation is important because a small energy MAE does not automatically guarantee that simulated AFM contrast is preserved.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The minimal ML analysis requires NumPy, pandas, SciPy, Matplotlib, and scikit-learn.

PP-AFM image generation additionally requires the existing `pyProbeParticle` environment and the local interpolation utilities used by the research workflow. Those dependencies are not included in the minimal `requirements.txt`.

## Reproducibility

A typical analysis sequence is:

```bash
python src/01_prepare_data.py --energy-file OO-h-NH3_N.dat
python src/02_sparse_reconstruction.py
python src/03_select_z_heights.py
python src/04_robustness.py
python src/05_spatial_holdout.py
python src/07_export_reconstructed_dat.py --dataset OO-h-NH3_N
```

Before publishing, make sure the scripts use command-line arguments or project-relative paths rather than local absolute paths.

## Data availability

The full DFT dataset is not included in this repository.

If the research data are unpublished or group-owned, only an approved example, anonymized subset, or synthetic dataset should be committed. The repository should contain enough information to describe the expected input format without distributing restricted data.

## Limitations

- The strongest current results are for reconstruction within one sample–tip system.
- Cross-system generalization to unseen chemistry remains to be tested.
- The selected height set may depend on the interaction-energy distribution and tip/sample chemistry.
- Reconstruction error in energy space does not directly guarantee identical PP-AFM images.
- Reduction in sampled heights should not be equated directly with the same percentage reduction in DFT wall time.

## Next steps

- Evaluate sparse reconstruction across multiple sample–tip combinations.
- Test whether one common height set transfers across different chemical systems.
- Compare original-DFT and reconstructed-energy PP-AFM images quantitatively.
- Investigate uncertainty-aware or active-learning selection of new DFT heights.
