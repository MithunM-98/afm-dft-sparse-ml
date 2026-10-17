#!/usr/bin/env python3

"""
Data-driven selection of informative DFT tip heights.

Goal
----
Select z positions that allow a Ridge regression model to reconstruct
the complete 45-point E(z) profile using as few DFT calculations as
possible.

IMPORTANT
---------
The fixed test set is NEVER used to select z positions.

Selection is performed using cross-validation only on the training
profiles produced by 02_sparse_reconstruction.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
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

BASELINE_DIR = (
    PROJECT_DIR
    / "results_sparse_reconstruction"
    / DATASET_NAME
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results_z_selection"
    / DATASET_NAME
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ENERGY_MATRIX_FILE = (
    DATA_DIR
    / "energy_matrix.csv"
)

TRAIN_FILE = (
    BASELINE_DIR
    / "train_profiles.csv"
)

TEST_FILE = (
    BASELINE_DIR
    / "test_profiles.csv"
)


# ------------------------------------------------------------
# z grid
# ------------------------------------------------------------

N_Z = 45

Z = 1.6 + 0.1 * np.arange(N_Z)


# Select up to this many heights
MAX_SELECTED = 10

# We will report these subset sizes
REPORT_K = [
    3,
    4,
    5,
    6,
    8,
    10
]


# ------------------------------------------------------------
# Cross-validation used ONLY inside training set
# ------------------------------------------------------------

N_CV_FOLDS = 5
CV_RANDOM_STATE = 123


# Ridge regularization strengths
ALPHAS = np.logspace(
    -6,
    4,
    40
)


# ============================================================
# Load data
# ============================================================

print("=" * 72)
print("Data-driven selection of DFT tip heights")
print("=" * 72)

energy_df = pd.read_csv(
    ENERGY_MATRIX_FILE,
    index_col=0
)

train_df = pd.read_csv(
    TRAIN_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)

E = energy_df.to_numpy(
    dtype=float
)

labels = energy_df.index.astype(
    str
).to_numpy()


# ------------------------------------------------------------
# Recover exact same train/test split from experiment 02
# ------------------------------------------------------------

train_labels = set(
    train_df["p"].astype(str)
)

test_labels = set(
    test_df["p"].astype(str)
)

train_mask = np.array(
    [
        p in train_labels
        for p in labels
    ]
)

test_mask = np.array(
    [
        p in test_labels
        for p in labels
    ]
)

E_train = E[train_mask]
E_test = E[test_mask]

p_train = labels[train_mask]
p_test = labels[test_mask]


print(
    f"\nTraining profiles : {len(E_train)}"
)

print(
    f"Test profiles     : {len(E_test)}"
)

print(
    f"z values          : {E.shape[1]}"
)


if len(E_train) + len(E_test) != len(E):
    raise ValueError(
        "Train/test split does not cover complete dataset."
    )


# ============================================================
# Helper functions
# ============================================================

def hidden_indices(observed_idx):

    mask = np.ones(
        N_Z,
        dtype=bool
    )

    mask[observed_idx] = False

    return np.where(mask)[0]


def evenly_spaced_indices(k):

    idx = np.linspace(
        0,
        N_Z - 1,
        k
    )

    idx = np.round(
        idx
    ).astype(int)

    return np.unique(idx)


# ============================================================
# Cross-validation score used for feature selection
# ============================================================

def cv_reconstruction_mae(
    curves,
    observed_idx
):
    """
    Cross-validated reconstruction MAE using only training profiles.

    For each CV fold:

      selected energies -> Ridge -> complete E(z)

    Error is evaluated only at hidden z positions.
    """

    observed_idx = np.array(
        sorted(observed_idx),
        dtype=int
    )

    hidden_idx = hidden_indices(
        observed_idx
    )

    kfold = KFold(
        n_splits=N_CV_FOLDS,
        shuffle=True,
        random_state=CV_RANDOM_STATE
    )

    fold_errors = []

    for fit_idx, val_idx in kfold.split(curves):

        train_curves = curves[
            fit_idx
        ]

        val_curves = curves[
            val_idx
        ]

        X_train = train_curves[
            :,
            observed_idx
        ]

        X_val = val_curves[
            :,
            observed_idx
        ]

        Y_train = train_curves

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_val_scaled = scaler.transform(
            X_val
        )

        model = RidgeCV(
            alphas=ALPHAS
        )

        model.fit(
            X_train_scaled,
            Y_train
        )

        prediction = model.predict(
            X_val_scaled
        )

        error = mean_absolute_error(
            val_curves[
                :,
                hidden_idx
            ].ravel(),
            prediction[
                :,
                hidden_idx
            ].ravel()
        )

        fold_errors.append(
            error
        )

    return float(
        np.mean(fold_errors)
    )


# ============================================================
# Greedy forward selection
# ============================================================

selected = []

selection_history = []


print("\n" + "=" * 72)
print("Greedy forward selection")
print("=" * 72)


for step in range(
    1,
    MAX_SELECTED + 1
):

    remaining = [
        i
        for i in range(N_Z)
        if i not in selected
    ]

    candidate_results = []

    print(
        f"\nSelecting height {step}/{MAX_SELECTED}"
    )

    for candidate in remaining:

        trial = sorted(
            selected + [candidate]
        )

        score = cv_reconstruction_mae(
            E_train,
            trial
        )

        candidate_results.append(
            (
                score,
                candidate
            )
        )

    candidate_results.sort(
        key=lambda x: x[0]
    )

    best_score, best_index = (
        candidate_results[0]
    )

    selected.append(
        best_index
    )

    selected = sorted(
        selected
    )

    selection_history.append(
        {
            "step": step,
            "selected_index": best_index,
            "selected_z": Z[best_index],
            "cv_mae": best_score,
            "all_selected_indices":
                ",".join(
                    str(i)
                    for i in selected
                ),
            "all_selected_z":
                ",".join(
                    f"{Z[i]:.1f}"
                    for i in selected
                )
        }
    )

    print(
        f"Selected z{best_index:02d} = "
        f"{Z[best_index]:.1f} Å"
    )

    print(
        f"CV hidden-point MAE = "
        f"{best_score:.6f}"
    )

    print(
        "Current selected heights:",
        np.round(
            Z[selected],
            2
        )
    )


selection_df = pd.DataFrame(
    selection_history
)

selection_df.to_csv(
    OUTPUT_DIR
    / "selection_history.csv",
    index=False
)


# ============================================================
# Fit/evaluate Ridge on untouched test set
# ============================================================

def evaluate_ridge(
    observed_idx
):

    observed_idx = np.array(
        sorted(observed_idx),
        dtype=int
    )

    hidden_idx = hidden_indices(
        observed_idx
    )

    X_train = E_train[
        :,
        observed_idx
    ]

    X_test = E_test[
        :,
        observed_idx
    ]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = RidgeCV(
        alphas=ALPHAS
    )

    model.fit(
        X_train_scaled,
        E_train
    )

    prediction = model.predict(
        X_test_scaled
    )

    truth_hidden = E_test[
        :,
        hidden_idx
    ].ravel()

    pred_hidden = prediction[
        :,
        hidden_idx
    ].ravel()

    mae = mean_absolute_error(
        truth_hidden,
        pred_hidden
    )

    rmse = np.sqrt(
        mean_squared_error(
            truth_hidden,
            pred_hidden
        )
    )

    return {
        "prediction": prediction,
        "mae": mae,
        "rmse": rmse,
        "alpha": model.alpha_
    }


# ============================================================
# Compare selected versus evenly spaced heights
# ============================================================

comparison_results = []

prediction_store = {}


print("\n" + "=" * 72)
print("Final evaluation on untouched test profiles")
print("=" * 72)


# Order in which features were selected
selection_order = [
    row["selected_index"]
    for row in selection_history
]


for k in REPORT_K:

    # --------------------------------------------------------
    # Data-driven selection
    # --------------------------------------------------------

    optimized_idx = sorted(
        selection_order[:k]
    )

    optimized = evaluate_ridge(
        optimized_idx
    )

    # --------------------------------------------------------
    # Evenly spaced baseline
    # --------------------------------------------------------

    even_idx = evenly_spaced_indices(
        k
    )

    even = evaluate_ridge(
        even_idx
    )

    comparison_results.append(
        {
            "n_observed": k,

            "selected_mae":
                optimized["mae"],

            "selected_rmse":
                optimized["rmse"],

            "even_mae":
                even["mae"],

            "even_rmse":
                even["rmse"],

            "selected_alpha":
                optimized["alpha"],

            "even_alpha":
                even["alpha"],

            "selected_z":
                ",".join(
                    f"{Z[i]:.1f}"
                    for i in optimized_idx
                ),

            "even_z":
                ",".join(
                    f"{Z[i]:.1f}"
                    for i in even_idx
                )
        }
    )

    prediction_store[k] = {
        "selected_idx": optimized_idx,
        "selected_prediction":
            optimized["prediction"],

        "even_idx": even_idx,
        "even_prediction":
            even["prediction"]
    }

    print("\n" + "-" * 72)

    print(
        f"k = {k}"
    )

    print(
        "Selected z:",
        np.round(
            Z[optimized_idx],
            2
        )
    )

    print(
        f"Selected Ridge MAE = "
        f"{optimized['mae']:.6f}"
    )

    print(
        "Even z:",
        np.round(
            Z[even_idx],
            2
        )
    )

    print(
        f"Even Ridge MAE     = "
        f"{even['mae']:.6f}"
    )


comparison_df = pd.DataFrame(
    comparison_results
)

comparison_df.to_csv(
    OUTPUT_DIR
    / "selected_vs_even.csv",
    index=False
)


# ============================================================
# Print summary table
# ============================================================

print("\n\nSummary:")
print(
    comparison_df[
        [
            "n_observed",
            "selected_mae",
            "even_mae",
            "selected_z"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Plot: selected vs evenly spaced
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.plot(
    comparison_df[
        "n_observed"
    ],
    comparison_df[
        "even_mae"
    ],
    marker="o",
    label="Evenly spaced heights"
)

ax.plot(
    comparison_df[
        "n_observed"
    ],
    comparison_df[
        "selected_mae"
    ],
    marker="o",
    label="Data-selected heights"
)

ax.set_xlabel(
    "Number of supplied DFT tip heights"
)

ax.set_ylabel(
    "MAE on hidden z values"
)

ax.set_title(
    f"{DATASET_NAME}\n"
    "Data-driven selection of DFT heights"
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "selected_vs_even_mae.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# Plot selected height sequence
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 3)
)

for rank, idx in enumerate(
    selection_order,
    start=1
):

    ax.scatter(
        Z[idx],
        0,
        s=100
    )

    ax.text(
        Z[idx],
        0.04,
        str(rank),
        ha="center",
        va="bottom",
        fontsize=9
    )

ax.set_xlim(
    Z.min() - 0.1,
    Z.max() + 0.1
)

ax.set_ylim(
    -0.15,
    0.25
)

ax.set_yticks([])

ax.set_xlabel(
    "Tip-sample distance z (Å)"
)

ax.set_title(
    "Order in which informative z heights were selected"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "selected_z_positions.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# Plot reconstruction examples for k = 6
# ============================================================

PLOT_K = 6

experiment = prediction_store[
    PLOT_K
]

selected_idx = experiment[
    "selected_idx"
]

even_idx = experiment[
    "even_idx"
]

selected_prediction = experiment[
    "selected_prediction"
]

even_prediction = experiment[
    "even_prediction"
]


n_examples = min(
    6,
    len(E_test)
)


for i in range(n_examples):

    truth = E_test[i]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        Z,
        truth,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Full DFT"
    )

    ax.plot(
        Z,
        even_prediction[i],
        linestyle="--",
        linewidth=1.5,
        label="Ridge: evenly spaced"
    )

    ax.plot(
        Z,
        selected_prediction[i],
        linestyle="-.",
        linewidth=1.8,
        label="Ridge: selected heights"
    )

    ax.scatter(
        Z[selected_idx],
        truth[selected_idx],
        s=55,
        zorder=10,
        label="Selected DFT values"
    )

    ax.set_xlabel(
        "Tip-sample distance z (Å)"
    )

    ax.set_ylabel(
        "Interaction energy"
    )

    ax.set_title(
        f"{DATASET_NAME} — {p_test[i]}\n"
        f"Reconstruction from {PLOT_K} DFT heights"
    )

    ax.legend(
        frameon=False,
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / f"optimized_reconstruction_{p_test[i]}_k{PLOT_K}.png",
        dpi=300
    )

    plt.close(fig)


print("\n" + "=" * 72)

print("Completed.")

print(
    "\nResults saved to:\n"
    f"{OUTPUT_DIR}"
)

print("\nImportant outputs:")

print(
    "    selection_history.csv"
)

print(
    "    selected_vs_even.csv"
)

print(
    "    selected_z_positions.png"
)

print(
    "    selected_vs_even_mae.png"
)

print(
    "    optimized_reconstruction_pXXX_k6.png"
)

print("=" * 72)
