#!/usr/bin/env python3

"""
Spatial holdout validation for sparse DFT E(z) reconstruction.

Instead of randomly holding out individual lateral p points,
divide the (x,y) sampling plane into spatial regions.

For each spatial fold:

    1. Hold out one complete spatial region.
    2. Select 6 informative z heights using TRAINING regions only.
    3. Train Ridge using those six energies.
    4. Reconstruct full 45-point E(z) profiles in the unseen region.
    5. Compare with six evenly spaced heights.

This is a harder generalization test than a random p-profile split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
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

ENERGY_FILE = (
    DATA_DIR
    / "energy_matrix.csv"
)

COORDINATE_FILE = (
    DATA_DIR
    / "coordinates.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results_spatial_holdout"
    / DATASET_NAME
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# z grid
# ============================================================

N_Z = 45

Z = (
    1.6
    + 0.1 * np.arange(N_Z)
)


# ============================================================
# Experiment settings
# ============================================================

N_SPATIAL_REGIONS = 5

N_SELECTED = 6

N_CV_FOLDS = 5

CV_RANDOM_STATE = 123

KMEANS_RANDOM_STATE = 42

ALPHAS = np.logspace(
    -6,
    4,
    30
)


# ============================================================
# Load data
# ============================================================

print("=" * 75)
print("Spatial holdout validation")
print("=" * 75)


energy_df = pd.read_csv(
    ENERGY_FILE,
    index_col=0
)

coordinates = pd.read_csv(
    COORDINATE_FILE
)


E = energy_df.to_numpy(
    dtype=float
)

p_labels = energy_df.index.astype(
    str
).to_numpy()


# ------------------------------------------------------------
# Ensure coordinate order matches energy matrix order
# ------------------------------------------------------------

coordinates = coordinates.copy()

coordinates["p"] = coordinates["p"].astype(
    str
)

coordinate_lookup = (
    coordinates
    .set_index("p")
    .loc[p_labels]
    .reset_index()
)


XY = coordinate_lookup[
    ["x", "y"]
].to_numpy(
    dtype=float
)


print(
    f"\nProfiles : {len(E)}"
)

print(
    f"z values : {E.shape[1]}"
)

print(
    f"x range  : "
    f"{XY[:,0].min():.2f} to "
    f"{XY[:,0].max():.2f} Å"
)

print(
    f"y range  : "
    f"{XY[:,1].min():.2f} to "
    f"{XY[:,1].max():.2f} Å"
)


if E.shape[0] != XY.shape[0]:
    raise ValueError(
        "Energy and coordinate tables have different "
        "numbers of p points."
    )


# ============================================================
# Create spatial regions
# ============================================================

# Important:
# Clustering uses ONLY x,y geometry, never energies.

kmeans = KMeans(
    n_clusters=N_SPATIAL_REGIONS,
    random_state=KMEANS_RANDOM_STATE,
    n_init=20
)

region = kmeans.fit_predict(
    XY
)

coordinate_lookup[
    "spatial_region"
] = region


coordinate_lookup.to_csv(
    OUTPUT_DIR
    / "spatial_regions.csv",
    index=False
)


print(
    "\nSpatial region sizes:"
)

for r in range(N_SPATIAL_REGIONS):

    print(
        f"    Region {r}: "
        f"{np.sum(region == r)} profiles"
    )


# ============================================================
# Plot spatial regions
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 8)
)

for r in range(N_SPATIAL_REGIONS):

    mask = region == r

    ax.scatter(
        XY[mask, 0],
        XY[mask, 1],
        s=40,
        label=f"Region {r}"
    )


ax.set_xlabel(
    "x (Å)"
)

ax.set_ylabel(
    "y (Å)"
)

ax.set_title(
    f"{DATASET_NAME}\nSpatial validation regions"
)

ax.set_aspect(
    "equal",
    adjustable="box"
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "spatial_regions.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# Helpers
# ============================================================

def hidden_indices(
    observed_idx
):

    mask = np.ones(
        N_Z,
        dtype=bool
    )

    mask[
        observed_idx
    ] = False

    return np.where(
        mask
    )[0]


def evenly_spaced_indices(
    k
):

    idx = np.linspace(
        0,
        N_Z - 1,
        k
    )

    idx = np.round(
        idx
    ).astype(int)

    return np.unique(
        idx
    )


# ============================================================
# Inner cross-validation
# ============================================================

def cv_reconstruction_mae(
    curves,
    observed_idx
):

    observed_idx = np.array(
        sorted(observed_idx),
        dtype=int
    )

    hidden_idx = hidden_indices(
        observed_idx
    )

    n_splits = min(
        N_CV_FOLDS,
        len(curves)
    )

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=CV_RANDOM_STATE
    )

    errors = []


    for fit_idx, val_idx in cv.split(curves):

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
            fit_curves
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

        errors.append(
            mae
        )


    return float(
        np.mean(errors)
    )


# ============================================================
# Greedy z-height selection
# ============================================================

def select_z_heights(
    train_curves
):

    selected = []

    history = []


    for step in range(
        1,
        N_SELECTED + 1
    ):

        remaining = [
            i
            for i in range(N_Z)
            if i not in selected
        ]

        scores = []


        for candidate in remaining:

            trial = sorted(
                selected
                + [candidate]
            )

            score = cv_reconstruction_mae(
                train_curves,
                trial
            )

            scores.append(
                (
                    score,
                    candidate
                )
            )


        scores.sort(
            key=lambda x: x[0]
        )

        best_score, best_candidate = (
            scores[0]
        )

        selected.append(
            best_candidate
        )

        history.append(
            {
                "step":
                    step,

                "z_index":
                    best_candidate,

                "z":
                    Z[
                        best_candidate
                    ],

                "cv_mae":
                    best_score
            }
        )


    return selected, history


# ============================================================
# Final Ridge model
# ============================================================

def evaluate_ridge(
    train_curves,
    test_curves,
    observed_idx
):

    observed_idx = np.array(
        sorted(observed_idx),
        dtype=int
    )

    hidden_idx = hidden_indices(
        observed_idx
    )


    X_train = train_curves[
        :,
        observed_idx
    ]

    X_test = test_curves[
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
        train_curves
    )


    prediction = model.predict(
        X_test_scaled
    )


    truth_hidden = test_curves[
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
        "mae":
            mae,

        "rmse":
            rmse,

        "prediction":
            prediction,

        "alpha":
            model.alpha_
    }


# ============================================================
# Run spatial folds
# ============================================================

even_idx = evenly_spaced_indices(
    N_SELECTED
)


results = []

selection_records = []


print(
    "\nEvenly spaced six-height baseline:"
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
    "Spatial holdout folds"
)

print(
    "=" * 75
)


for test_region in range(
    N_SPATIAL_REGIONS
):

    print(
        f"\nHolding out spatial region "
        f"{test_region}"
    )

    print(
        "-" * 75
    )


    test_mask = (
        region == test_region
    )

    train_mask = (
        region != test_region
    )


    E_train = E[
        train_mask
    ]

    E_test = E[
        test_mask
    ]


    print(
        f"Training profiles : "
        f"{len(E_train)}"
    )

    print(
        f"Test profiles     : "
        f"{len(E_test)}"
    )


    # --------------------------------------------------------
    # Select z values using training spatial regions only
    # --------------------------------------------------------

    selected_order, history = (
        select_z_heights(
            E_train
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
    # Final test
    # --------------------------------------------------------

    selected_result = evaluate_ridge(
        E_train,
        E_test,
        selected_sorted
    )


    even_result = evaluate_ridge(
        E_train,
        E_test,
        even_idx
    )


    improvement = (
        (
            even_result[
                "mae"
            ]
            -
            selected_result[
                "mae"
            ]
        )
        /
        even_result[
            "mae"
        ]
        * 100
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
        f"{improvement:.2f} %"
    )


    results.append(
        {
            "test_region":
                test_region,

            "n_train":
                len(E_train),

            "n_test":
                len(E_test),

            "selected_mae":
                selected_result[
                    "mae"
                ],

            "selected_rmse":
                selected_result[
                    "rmse"
                ],

            "even_mae":
                even_result[
                    "mae"
                ],

            "even_rmse":
                even_result[
                    "rmse"
                ],

            "improvement_percent":
                improvement,

            "selected_z":
                ",".join(
                    f"{Z[i]:.1f}"
                    for i
                    in selected_sorted
                )
        }
    )


    for rank, idx in enumerate(
        selected_order,
        start=1
    ):

        selection_records.append(
            {
                "test_region":
                    test_region,

                "rank":
                    rank,

                "z_index":
                    idx,

                "z":
                    Z[idx]
            }
        )


# ============================================================
# Results
# ============================================================

results_df = pd.DataFrame(
    results
)

selection_df = pd.DataFrame(
    selection_records
)


results_df.to_csv(
    OUTPUT_DIR
    / "spatial_holdout_results.csv",
    index=False
)

selection_df.to_csv(
    OUTPUT_DIR
    / "spatial_selection_history.csv",
    index=False
)


print(
    "\n" + "=" * 75
)

print(
    "SPATIAL HOLDOUT SUMMARY"
)

print(
    "=" * 75
)


print(
    "\nPer-region results:\n"
)

print(
    results_df[
        [
            "test_region",
            "n_test",
            "selected_mae",
            "even_mae",
            "improvement_percent",
            "selected_z"
        ]
    ].to_string(
        index=False
    )
)


selected_mean = results_df[
    "selected_mae"
].mean()

selected_std = results_df[
    "selected_mae"
].std()

even_mean = results_df[
    "even_mae"
].mean()

even_std = results_df[
    "even_mae"
].std()


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
    .sort_values(
        [
            "count",
            "z"
        ],
        ascending=[
            False,
            True
        ]
    )
)


frequency.to_csv(
    OUTPUT_DIR
    / "spatial_selection_frequency.csv",
    index=False
)


print(
    "\nMost frequently selected heights:"
)

print(
    frequency.head(
        15
    ).to_string(
        index=False
    )
)


# ============================================================
# Plot comparison
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)


x = np.arange(
    len(results_df)
)


ax.plot(
    x,
    results_df[
        "even_mae"
    ],
    marker="o",
    label="Evenly spaced"
)

ax.plot(
    x,
    results_df[
        "selected_mae"
    ],
    marker="o",
    label="Data-selected"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        f"Region {r}"
        for r in results_df[
            "test_region"
        ]
    ]
)


ax.set_xlabel(
    "Held-out spatial region"
)

ax.set_ylabel(
    "MAE on hidden z values"
)

ax.set_title(
    f"{DATASET_NAME}\n"
    "Spatial holdout validation"
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "spatial_holdout_mae.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# Finished
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
    "    spatial_regions.png"
)

print(
    "    spatial_holdout_results.csv"
)

print(
    "    spatial_selection_frequency.csv"
)

print(
    "    spatial_holdout_mae.png"
)

print(
    "\nDone."
)
