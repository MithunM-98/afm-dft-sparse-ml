#!/usr/bin/env python3

"""
Robustness test for sparse DFT profile reconstruction.

For several independent train/test splits:

1. Split complete p-profiles into training and test sets.
2. Use ONLY the training profiles to select 6 informative z heights.
3. Train a Ridge model using those 6 heights.
4. Evaluate reconstruction on untouched test profiles.
5. Compare against 6 evenly spaced z heights.
6. Measure how frequently each z height is selected.

Dataset:
    OO-h-NH3_N

Input:
    ml_prepared/OO-h-NH3_N/energy_matrix.csv

Output:
    results_robustness/OO-h-NH3_N/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
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

ENERGY_MATRIX_FILE = (
    PROJECT_DIR
    / "ml_prepared"
    / DATASET_NAME
    / "energy_matrix.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results_robustness"
    / DATASET_NAME
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# DFT z grid
# ------------------------------------------------------------

N_Z = 45

Z = (
    1.6
    + 0.1 * np.arange(N_Z)
)


# ------------------------------------------------------------
# Main experiment settings
# ------------------------------------------------------------

N_SELECTED = 6

TEST_SIZE = 0.20

# 10 independent outer train/test splits
OUTER_RANDOM_STATES = list(range(10))


# ------------------------------------------------------------
# Cross-validation used INSIDE each training set
# ------------------------------------------------------------

N_CV_FOLDS = 5
CV_RANDOM_STATE = 123


# Ridge regularization values
ALPHAS = np.logspace(
    -6,
    4,
    30
)


# ============================================================
# Load energy matrix
# ============================================================

print("=" * 75)
print("Robustness analysis")
print("=" * 75)

print(
    f"\nReading:\n{ENERGY_MATRIX_FILE}"
)

energy_df = pd.read_csv(
    ENERGY_MATRIX_FILE,
    index_col=0
)

E = energy_df.to_numpy(
    dtype=float
)

p_labels = energy_df.index.astype(
    str
).to_numpy()


print(
    f"\nNumber of profiles : {E.shape[0]}"
)

print(
    f"Number of z values : {E.shape[1]}"
)


if E.shape[1] != N_Z:

    raise ValueError(
        f"Expected {N_Z} z values, "
        f"found {E.shape[1]}"
    )


if np.isnan(E).any():

    raise ValueError(
        "NaN values found in energy matrix."
    )


# ============================================================
# Utility functions
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
# Inner CV score
# ============================================================

def cv_reconstruction_mae(
    curves,
    observed_idx
):
    """
    Estimate reconstruction error using only training curves.

    observed_idx:
        z positions supplied to Ridge

    Error is measured only on the hidden z values.
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

        fit_curves = curves[
            fit_idx
        ]

        val_curves = curves[
            val_idx
        ]

        X_fit = fit_curves[
            :,
            observed_idx
        ]

        X_val = val_curves[
            :,
            observed_idx
        ]

        # Target = complete 45-point curve
        Y_fit = fit_curves

        scaler = StandardScaler()

        X_fit_scaled = scaler.fit_transform(
            X_fit
        )

        X_val_scaled = scaler.transform(
            X_val
        )

        model = RidgeCV(
            alphas=ALPHAS
        )

        model.fit(
            X_fit_scaled,
            Y_fit
        )

        prediction = model.predict(
            X_val_scaled
        )

        mae = mean_absolute_error(
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
            mae
        )

    return float(
        np.mean(fold_errors)
    )


# ============================================================
# Greedy selection
# ============================================================

def select_z_heights(
    train_curves,
    n_selected
):
    """
    Greedy forward feature selection.

    At every step, test every remaining z height and choose
    the one giving the lowest CV reconstruction MAE.
    """

    selected = []

    history = []

    for step in range(
        1,
        n_selected + 1
    ):

        candidates = [
            i
            for i in range(N_Z)
            if i not in selected
        ]

        candidate_scores = []

        for candidate in candidates:

            trial = sorted(
                selected
                + [candidate]
            )

            score = cv_reconstruction_mae(
                train_curves,
                trial
            )

            candidate_scores.append(
                (
                    score,
                    candidate
                )
            )

        candidate_scores.sort(
            key=lambda item: item[0]
        )

        best_score, best_candidate = (
            candidate_scores[0]
        )

        selected.append(
            best_candidate
        )

        # Keep selection ORDER separately.
        # Do not sort here because we want to know which
        # height was selected first, second, etc.

        history.append(
            {
                "step": step,
                "z_index": best_candidate,
                "z": Z[best_candidate],
                "cv_mae": best_score
            }
        )

    return selected, history


# ============================================================
# Final Ridge evaluation
# ============================================================

def evaluate_ridge(
    E_train,
    E_test,
    observed_idx
):
    """
    Train on all outer-training profiles and evaluate
    once on untouched outer-test profiles.
    """

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

    prediction_hidden = prediction[
        :,
        hidden_idx
    ].ravel()

    mae = mean_absolute_error(
        truth_hidden,
        prediction_hidden
    )

    rmse = np.sqrt(
        mean_squared_error(
            truth_hidden,
            prediction_hidden
        )
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "alpha": model.alpha_,
        "prediction": prediction
    }


# ============================================================
# Experiment
# ============================================================

all_indices = np.arange(
    len(E)
)

results = []

selection_records = []


# Fixed six-height evenly spaced baseline
even_idx = evenly_spaced_indices(
    N_SELECTED
)


print(
    "\nEvenly spaced reference heights:"
)

print(
    np.round(
        Z[even_idx],
        2
    )
)


print(
    "\n" + "=" * 75
)

print(
    f"Running {len(OUTER_RANDOM_STATES)} "
    f"independent train/test splits"
)

print(
    "=" * 75
)


for run_number, random_state in enumerate(
    OUTER_RANDOM_STATES,
    start=1
):

    print(
        f"\nRUN {run_number}/"
        f"{len(OUTER_RANDOM_STATES)}"
    )

    print(
        f"Outer random state = "
        f"{random_state}"
    )

    print("-" * 75)


    # --------------------------------------------------------
    # Outer train/test split
    # --------------------------------------------------------

    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=random_state
    )

    E_train = E[
        train_idx
    ]

    E_test = E[
        test_idx
    ]


    # --------------------------------------------------------
    # Select six heights using TRAINING DATA ONLY
    # --------------------------------------------------------

    selected_order, history = (
        select_z_heights(
            E_train,
            N_SELECTED
        )
    )

    selected_sorted = sorted(
        selected_order
    )


    print(
        "Selected heights:"
    )

    print(
        np.round(
            Z[selected_sorted],
            2
        )
    )


    # --------------------------------------------------------
    # Evaluate selected heights
    # --------------------------------------------------------

    selected_result = evaluate_ridge(
        E_train,
        E_test,
        selected_sorted
    )


    # --------------------------------------------------------
    # Evaluate evenly spaced heights
    # --------------------------------------------------------

    even_result = evaluate_ridge(
        E_train,
        E_test,
        even_idx
    )


    improvement_percent = (
        (
            even_result["mae"]
            - selected_result["mae"]
        )
        / even_result["mae"]
        * 100.0
    )


    print(
        f"Selected MAE : "
        f"{selected_result['mae']:.6f}"
    )

    print(
        f"Even MAE     : "
        f"{even_result['mae']:.6f}"
    )

    print(
        f"Improvement  : "
        f"{improvement_percent:.2f} %"
    )


    # --------------------------------------------------------
    # Save outer result
    # --------------------------------------------------------

    results.append(
        {
            "run": run_number,
            "random_state": random_state,

            "n_train": len(train_idx),
            "n_test": len(test_idx),

            "selected_mae":
                selected_result["mae"],

            "selected_rmse":
                selected_result["rmse"],

            "even_mae":
                even_result["mae"],

            "even_rmse":
                even_result["rmse"],

            "improvement_percent":
                improvement_percent,

            "selected_alpha":
                selected_result["alpha"],

            "even_alpha":
                even_result["alpha"],

            "selected_z":
                ",".join(
                    f"{Z[i]:.1f}"
                    for i in selected_sorted
                ),

            "selection_order":
                ",".join(
                    f"{Z[i]:.1f}"
                    for i in selected_order
                )
        }
    )


    # --------------------------------------------------------
    # Record every selected height and its selection rank
    # --------------------------------------------------------

    for rank, idx in enumerate(
        selected_order,
        start=1
    ):

        selection_records.append(
            {
                "run": run_number,
                "random_state":
                    random_state,
                "selection_rank":
                    rank,
                "z_index":
                    idx,
                "z":
                    Z[idx]
            }
        )


# ============================================================
# Save raw results
# ============================================================

results_df = pd.DataFrame(
    results
)

selection_df = pd.DataFrame(
    selection_records
)


results_df.to_csv(
    OUTPUT_DIR
    / "robustness_results.csv",
    index=False
)

selection_df.to_csv(
    OUTPUT_DIR
    / "all_selected_heights.csv",
    index=False
)


# ============================================================
# Selection frequency
# ============================================================

frequency = (
    selection_df
    .groupby(
        ["z_index", "z"]
    )
    .size()
    .reset_index(
        name="count"
    )
)

frequency["frequency_percent"] = (
    100.0
    * frequency["count"]
    / len(OUTER_RANDOM_STATES)
)

frequency = frequency.sort_values(
    [
        "count",
        "z"
    ],
    ascending=[
        False,
        True
    ]
)


frequency.to_csv(
    OUTPUT_DIR
    / "selection_frequency.csv",
    index=False
)


# ============================================================
# Summary statistics
# ============================================================

selected_mean = (
    results_df[
        "selected_mae"
    ].mean()
)

selected_std = (
    results_df[
        "selected_mae"
    ].std()
)

even_mean = (
    results_df[
        "even_mae"
    ].mean()
)

even_std = (
    results_df[
        "even_mae"
    ].std()
)

improvement_mean = (
    results_df[
        "improvement_percent"
    ].mean()
)

improvement_std = (
    results_df[
        "improvement_percent"
    ].std()
)


print(
    "\n" + "=" * 75
)

print(
    "ROBUSTNESS SUMMARY"
)

print(
    "=" * 75
)


print(
    "\nSelected-height Ridge:"
)

print(
    f"    MAE = "
    f"{selected_mean:.6f} "
    f"± {selected_std:.6f}"
)


print(
    "\nEvenly spaced Ridge:"
)

print(
    f"    MAE = "
    f"{even_mean:.6f} "
    f"± {even_std:.6f}"
)


print(
    "\nRelative MAE improvement:"
)

print(
    f"    {improvement_mean:.2f} "
    f"± {improvement_std:.2f} %"
)


print(
    "\nMost frequently selected heights:"
)

print(
    frequency[
        [
            "z",
            "count",
            "frequency_percent"
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# Save summary file
# ============================================================

summary = pd.DataFrame(
    {
        "metric": [
            "selected_mae_mean",
            "selected_mae_std",
            "even_mae_mean",
            "even_mae_std",
            "improvement_percent_mean",
            "improvement_percent_std"
        ],

        "value": [
            selected_mean,
            selected_std,
            even_mean,
            even_std,
            improvement_mean,
            improvement_std
        ]
    }
)

summary.to_csv(
    OUTPUT_DIR
    / "summary_statistics.csv",
    index=False
)


# ============================================================
# Plot 1: MAE for every random split
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    results_df["run"],
    results_df["even_mae"],
    marker="o",
    label="Evenly spaced"
)

ax.plot(
    results_df["run"],
    results_df["selected_mae"],
    marker="o",
    label="Data-selected"
)

ax.set_xlabel(
    "Train/test split"
)

ax.set_ylabel(
    "Test MAE on hidden z values"
)

ax.set_title(
    f"{DATASET_NAME}\n"
    "Robustness across independent train/test splits"
)

ax.set_xticks(
    results_df["run"]
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "mae_across_splits.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# Plot 2: selected-height frequency
# ============================================================

# Include all 45 positions, including those never selected.

frequency_full = pd.DataFrame(
    {
        "z_index":
            np.arange(N_Z),

        "z":
            Z
    }
)

frequency_full = frequency_full.merge(
    frequency[
        [
            "z_index",
            "count"
        ]
    ],
    on="z_index",
    how="left"
)

frequency_full["count"] = (
    frequency_full["count"]
    .fillna(0)
    .astype(int)
)


fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.bar(
    frequency_full["z"],
    frequency_full["count"],
    width=0.075
)

ax.set_xlabel(
    "Tip-sample distance z (Å)"
)

ax.set_ylabel(
    "Number of splits selecting this height"
)

ax.set_title(
    f"{DATASET_NAME}\n"
    f"Selection frequency for {N_SELECTED}-height reconstruction"
)

ax.set_xticks(
    np.arange(
        1.6,
        6.01,
        0.4
    )
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "z_selection_frequency.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# Plot 3: selected vs even MAE scatter
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 6)
)

ax.scatter(
    results_df["even_mae"],
    results_df["selected_mae"],
    s=60
)

limit = max(
    results_df["even_mae"].max(),
    results_df["selected_mae"].max()
)

ax.plot(
    [0, limit],
    [0, limit],
    linestyle="--",
    linewidth=1
)

ax.set_xlabel(
    "Evenly spaced test MAE"
)

ax.set_ylabel(
    "Selected-height test MAE"
)

ax.set_title(
    "Selected versus evenly spaced heights"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "selected_vs_even_scatter.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# Finish
# ============================================================

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nImportant files:"
)

print(
    "    robustness_results.csv"
)

print(
    "    summary_statistics.csv"
)

print(
    "    selection_frequency.csv"
)

print(
    "    mae_across_splits.png"
)

print(
    "    z_selection_frequency.png"
)

print(
    "    selected_vs_even_scatter.png"
)

print(
    "\nDone."
)
