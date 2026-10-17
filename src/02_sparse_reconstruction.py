#!/usr/bin/env python3

"""
Sparse reconstruction of DFT tip-sample interaction-energy profiles.

Question
--------
How many DFT tip heights are needed to reconstruct the complete E(z) curve?

Methods
-------
1. PCHIP interpolation
2. PCA low-rank reconstruction
3. Ridge multi-output regression

Important
---------
The train/test split is performed by complete lateral p profiles.
A test p point is never present in training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(
    "/home/mithun/work/essential_data/AFM_ML"
)

DATASET_NAME = "OO-h-NH3_N"

DATA_DIR = (
    PROJECT_DIR
    / "ml_prepared"
    / DATASET_NAME
)

ENERGY_MATRIX_FILE = (
    DATA_DIR
    / "energy_matrix.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results_sparse_reconstruction"
    / DATASET_NAME
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Physical z grid
N_Z = 45
Z = 1.6 + 0.1 * np.arange(N_Z)

# Reproducible train/test split
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Number of DFT heights supplied to the reconstruction
K_VALUES = [
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    15,
    20
]

# Number of PCA components used for sparse reconstruction
N_PCA_COMPONENTS = 3


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("Sparse DFT profile reconstruction")
print("=" * 70)

print(f"\nReading:\n{ENERGY_MATRIX_FILE}")

energy_df = pd.read_csv(
    ENERGY_MATRIX_FILE,
    index_col=0
)

E = energy_df.to_numpy(dtype=float)

p_labels = energy_df.index.to_numpy()

print("\nDataset:")
print(f"    profiles : {E.shape[0]}")
print(f"    z values : {E.shape[1]}")
print(f"    shape    : {E.shape}")


if E.shape[1] != N_Z:
    raise ValueError(
        f"Expected {N_Z} z values, "
        f"but found {E.shape[1]}"
    )

if np.isnan(E).any():
    raise ValueError(
        "NaN values found in the energy matrix."
    )


# ============================================================
# Train/test split
# ============================================================

indices = np.arange(len(E))

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

E_train = E[train_idx]
E_test = E[test_idx]

p_train = p_labels[train_idx]
p_test = p_labels[test_idx]

print("\nTrain/test split:")
print(f"    training profiles : {len(E_train)}")
print(f"    test profiles     : {len(E_test)}")

print("\nExample held-out p points:")
print("   ", p_test[:10])


# Save split so that every future experiment uses the same split

pd.DataFrame(
    {
        "p": p_train,
        "split": "train"
    }
).to_csv(
    OUTPUT_DIR / "train_profiles.csv",
    index=False
)

pd.DataFrame(
    {
        "p": p_test,
        "split": "test"
    }
).to_csv(
    OUTPUT_DIR / "test_profiles.csv",
    index=False
)


# ============================================================
# PCA analysis
# ============================================================

# IMPORTANT:
# Fit scaler only on training data.
#
# Each z column has a very different energy scale.
# Standardization prevents the short-distance region from
# completely dominating the PCA.

profile_scaler = StandardScaler()

E_train_scaled = profile_scaler.fit_transform(
    E_train
)

E_test_scaled = profile_scaler.transform(
    E_test
)

max_components = min(
    E_train_scaled.shape[0],
    E_train_scaled.shape[1]
)

pca_full = PCA(
    n_components=max_components
)

pca_full.fit(
    E_train_scaled
)

explained = pca_full.explained_variance_ratio_

cumulative_explained = np.cumsum(
    explained
)


# ============================================================
# Print PCA information
# ============================================================

print("\nPCA cumulative explained variance:")

for n in [1, 2, 3, 4, 5, 8, 10]:

    if n <= len(cumulative_explained):

        print(
            f"    {n:2d} components : "
            f"{100*cumulative_explained[n-1]:8.4f} %"
        )


# ============================================================
# PCA explained variance plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

n_show = min(
    15,
    len(cumulative_explained)
)

ax.plot(
    np.arange(1, n_show + 1),
    100 * cumulative_explained[:n_show],
    marker="o"
)

ax.axhline(
    95,
    linestyle="--",
    linewidth=1
)

ax.axhline(
    99,
    linestyle="--",
    linewidth=1
)

ax.set_xlabel(
    "Number of PCA components"
)

ax.set_ylabel(
    "Cumulative explained variance (%)"
)

ax.set_title(
    f"{DATASET_NAME}\nPCA of DFT interaction profiles"
)

ax.set_xticks(
    np.arange(1, n_show + 1)
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "pca_explained_variance.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# PCA model for sparse reconstruction
# ============================================================

n_pca = min(
    N_PCA_COMPONENTS,
    E_train_scaled.shape[0],
    E_train_scaled.shape[1]
)

pca = PCA(
    n_components=n_pca
)

pca.fit(
    E_train_scaled
)


# ============================================================
# Utility functions
# ============================================================

def choose_evenly_spaced_indices(k):
    """
    Select k approximately evenly spaced z values.

    Examples:
        k=3:
            z = 1.6, ~3.8, 6.0

        k=6:
            approximately evenly spread across 1.6--6.0 Å
    """

    indices = np.linspace(
        0,
        N_Z - 1,
        k
    )

    indices = np.round(
        indices
    ).astype(int)

    # Safety against duplicated indices
    indices = np.unique(
        indices
    )

    return indices


def hidden_indices(observed):
    """
    Return z indices that were NOT supplied to the model.
    """

    mask = np.ones(
        N_Z,
        dtype=bool
    )

    mask[observed] = False

    return np.where(mask)[0]


def pchip_reconstruct(
    curves,
    observed_idx
):
    """
    Reconstruct every curve independently using PCHIP.

    This is our non-ML baseline.
    """

    predictions = np.zeros_like(
        curves
    )

    z_observed = Z[observed_idx]

    for i, curve in enumerate(curves):

        interpolator = PchipInterpolator(
            z_observed,
            curve[observed_idx],
            extrapolate=True
        )

        predictions[i] = interpolator(Z)

    return predictions


def ridge_reconstruct(
    train_curves,
    test_curves,
    observed_idx
):
    """
    Multi-output Ridge regression.

    Input:
        energies at selected z positions

    Output:
        full 45-point E(z) profile
    """

    X_train = train_curves[
        :,
        observed_idx
    ]

    X_test = test_curves[
        :,
        observed_idx
    ]

    Y_train = train_curves

    # Scale only model inputs.
    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(
        X_train
    )

    X_test_scaled = input_scaler.transform(
        X_test
    )

    alphas = np.logspace(
        -6,
        4,
        40
    )

    model = RidgeCV(
        alphas=alphas
    )

    model.fit(
        X_train_scaled,
        Y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    return predictions, model.alpha_


def pca_sparse_reconstruct(
    test_curves,
    observed_idx
):
    """
    Estimate PCA coefficients using only the observed z values.

    PCA itself was fitted only to the training profiles.
    """

    predictions = np.zeros_like(
        test_curves
    )

    components = pca.components_

    pca_mean = pca.mean_

    # Matrix connecting PCA coefficients to observed z values.
    #
    # shape:
    #   n_observed x n_components

    A = components[
        :,
        observed_idx
    ].T

    for i, curve in enumerate(test_curves):

        # Convert observed physical energies into the same
        # standardized representation used to fit PCA.

        observed_scaled = (
            curve[observed_idx]
            - profile_scaler.mean_[observed_idx]
        ) / profile_scaler.scale_[observed_idx]

        b = (
            observed_scaled
            - pca_mean[observed_idx]
        )

        # Least-squares estimate of PCA scores
        scores, _, _, _ = np.linalg.lstsq(
            A,
            b,
            rcond=None
        )

        reconstructed_scaled = (
            pca_mean
            + scores @ components
        )

        reconstructed = (
            reconstructed_scaled
            * profile_scaler.scale_
            + profile_scaler.mean_
        )

        predictions[i] = reconstructed

    return predictions


def calculate_metrics(
    truth,
    prediction,
    hidden_idx
):
    """
    Metrics are evaluated ONLY on z values hidden from
    the reconstruction method.
    """

    y_true = truth[
        :,
        hidden_idx
    ].ravel()

    y_pred = prediction[
        :,
        hidden_idx
    ].ravel()

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    return mae, rmse


# ============================================================
# Run sparse reconstruction experiments
# ============================================================

results = []

prediction_store = {}

print("\n" + "=" * 70)
print("Sparse reconstruction experiments")
print("=" * 70)


for k in K_VALUES:

    observed_idx = choose_evenly_spaced_indices(
        k
    )

    hidden_idx = hidden_indices(
        observed_idx
    )

    actual_k = len(
        observed_idx
    )

    print("\n" + "-" * 70)

    print(
        f"Observed heights: {actual_k}"
    )

    print(
        "z =",
        np.round(
            Z[observed_idx],
            2
        )
    )

    # --------------------------------------------------------
    # 1. PCHIP
    # --------------------------------------------------------

    pchip_prediction = pchip_reconstruct(
        E_test,
        observed_idx
    )

    pchip_mae, pchip_rmse = (
        calculate_metrics(
            E_test,
            pchip_prediction,
            hidden_idx
        )
    )

    # --------------------------------------------------------
    # 2. Ridge ML
    # --------------------------------------------------------

    ridge_prediction, best_alpha = (
        ridge_reconstruct(
            E_train,
            E_test,
            observed_idx
        )
    )

    ridge_mae, ridge_rmse = (
        calculate_metrics(
            E_test,
            ridge_prediction,
            hidden_idx
        )
    )

    # --------------------------------------------------------
    # 3. PCA
    # --------------------------------------------------------

    pca_prediction = (
        pca_sparse_reconstruct(
            E_test,
            observed_idx
        )
    )

    pca_mae, pca_rmse = (
        calculate_metrics(
            E_test,
            pca_prediction,
            hidden_idx
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.append(
        {
            "n_observed": actual_k,

            "pchip_mae": pchip_mae,
            "pchip_rmse": pchip_rmse,

            "pca_mae": pca_mae,
            "pca_rmse": pca_rmse,

            "ridge_mae": ridge_mae,
            "ridge_rmse": ridge_rmse,

            "ridge_alpha": best_alpha,

            "observed_z": ",".join(
                f"{x:.1f}"
                for x in Z[observed_idx]
            )
        }
    )

    prediction_store[actual_k] = {
        "observed_idx": observed_idx,
        "pchip": pchip_prediction,
        "pca": pca_prediction,
        "ridge": ridge_prediction
    }

    print(
        f"PCHIP : MAE={pchip_mae:.6f}   "
        f"RMSE={pchip_rmse:.6f}"
    )

    print(
        f"PCA   : MAE={pca_mae:.6f}   "
        f"RMSE={pca_rmse:.6f}"
    )

    print(
        f"Ridge : MAE={ridge_mae:.6f}   "
        f"RMSE={ridge_rmse:.6f}   "
        f"alpha={best_alpha:.3e}"
    )


# ============================================================
# Save metric table
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_DIR
    / "reconstruction_metrics.csv",
    index=False
)

print("\n\nResults:")
print(
    results_df[
        [
            "n_observed",
            "pchip_mae",
            "pca_mae",
            "ridge_mae"
        ]
    ].to_string(index=False)
)


# ============================================================
# Plot MAE versus number of supplied DFT heights
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.plot(
    results_df["n_observed"],
    results_df["pchip_mae"],
    marker="o",
    label="PCHIP"
)

ax.plot(
    results_df["n_observed"],
    results_df["pca_mae"],
    marker="o",
    label="PCA"
)

ax.plot(
    results_df["n_observed"],
    results_df["ridge_mae"],
    marker="o",
    label="Ridge"
)

ax.set_xlabel(
    "Number of supplied DFT tip heights"
)

ax.set_ylabel(
    "MAE on hidden z values"
)

ax.set_title(
    f"{DATASET_NAME}\n"
    "Sparse reconstruction of DFT interaction profiles"
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "mae_vs_number_of_heights.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# Reconstruction plots
# ============================================================

# Six observed z positions is a nice first visualization.
plot_k = 6

if plot_k not in prediction_store:

    plot_k = sorted(
        prediction_store.keys()
    )[0]

experiment = prediction_store[
    plot_k
]

observed_idx = experiment[
    "observed_idx"
]

# Plot first 6 held-out profiles
n_examples = min(
    6,
    len(E_test)
)


for i in range(n_examples):

    truth = E_test[i]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    # Ground truth
    ax.plot(
        Z,
        truth,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Full DFT"
    )

    # PCHIP
    ax.plot(
        Z,
        experiment["pchip"][i],
        linestyle="--",
        linewidth=1.5,
        label="PCHIP"
    )

    # PCA
    ax.plot(
        Z,
        experiment["pca"][i],
        linestyle="-.",
        linewidth=1.5,
        label="PCA"
    )

    # Ridge
    ax.plot(
        Z,
        experiment["ridge"][i],
        linestyle=":",
        linewidth=2,
        label="Ridge ML"
    )

    # Supplied DFT points
    ax.scatter(
        Z[observed_idx],
        truth[observed_idx],
        s=55,
        zorder=10,
        label="DFT values supplied"
    )

    ax.set_xlabel(
        "Tip-sample distance z (Å)"
    )

    ax.set_ylabel(
        "Interaction energy"
    )

    ax.set_title(
        f"{DATASET_NAME} — {p_test[i]}\n"
        f"Reconstruction from {plot_k} DFT heights"
    )

    ax.legend(
        frameon=False,
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / (
            f"reconstruction_"
            f"{p_test[i]}_"
            f"k{plot_k}.png"
        ),
        dpi=300
    )

    plt.close(fig)


print("\n" + "=" * 70)

print("Analysis completed.")

print(
    "\nResults saved in:\n"
    f"{OUTPUT_DIR}"
)

print("\nImportant files:")

print(
    "    pca_explained_variance.png"
)

print(
    "    reconstruction_metrics.csv"
)

print(
    "    mae_vs_number_of_heights.png"
)

print(
    "    reconstruction_pXXX_k6.png"
)

print("=" * 70)
