#!/usr/bin/env python3

"""
Export ML-reconstructed interaction energies in the SAME format
as the original SAMPLE-TIP.dat files.

Example output:

    result_new/OO-h-NH3_N.dat

with lines

    p000 z00 energy
    p000 z01 energy
    ...
    p150 z44 energy

The selected z heights retain the ORIGINAL DFT values.
All other z heights are reconstructed with Ridge regression.

For honest comparison with DFT, predictions are generated
out-of-fold: a p-profile is never used to train the model
that reconstructs that same profile.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(
    "/home/mithun/work/essential_data/AFM_ML"
)

PREPARED_DIR = (
    PROJECT_DIR
    / "ml_prepared"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "result_new"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# z grid
# ============================================================

N_Z = 45

Z_START = 1.6
DZ = 0.1

Z = (
    Z_START
    + DZ * np.arange(N_Z)
)


# ============================================================
# Six sparse DFT heights
# ============================================================

# These are the six heights selected from the previous experiment.
#
# The energy at these heights will remain EXACTLY equal to DFT.
# The remaining 39 heights will be ML reconstructed.

OBSERVED_Z = np.array(
    [
        1.6,
        1.8,
        2.2,
        2.4,
        3.6,
        5.0
    ]
)


# ============================================================
# Model configuration
# ============================================================

N_FOLDS = 5

RANDOM_STATE = 42

ALPHAS = np.logspace(
    -6,
    4,
    40
)


# ============================================================
# Helper: convert physical z to z index
# ============================================================

def z_to_index(z):
    """
    Convert physical z in Angstrom to zXX index.

    Example:
        1.6 -> 0
        1.8 -> 2
        2.2 -> 6
    """

    index = int(
        round(
            (z - Z_START) / DZ
        )
    )

    if index < 0 or index >= N_Z:
        raise ValueError(
            f"z={z} Å is outside the "
            f"{Z_START:.1f}-{Z[-1]:.1f} Å grid."
        )

    reconstructed_z = (
        Z_START
        + index * DZ
    )

    if not np.isclose(
        reconstructed_z,
        z,
        atol=1e-8
    ):
        raise ValueError(
            f"z={z} is not on the 0.1 Å grid."
        )

    return index


OBSERVED_IDX = np.array(
    [
        z_to_index(z)
        for z in OBSERVED_Z
    ],
    dtype=int
)


HIDDEN_IDX = np.array(
    [
        i
        for i in range(N_Z)
        if i not in OBSERVED_IDX
    ],
    dtype=int
)


# ============================================================
# Reconstruct one system
# ============================================================

def reconstruct_dataset(dataset_name):

    dataset_dir = (
        PREPARED_DIR
        / dataset_name
    )

    matrix_file = (
        dataset_dir
        / "energy_matrix.csv"
    )

    if not matrix_file.exists():
        raise FileNotFoundError(
            f"Missing:\n{matrix_file}"
        )

    print("\n" + "=" * 75)
    print(f"Dataset: {dataset_name}")
    print("=" * 75)

    # --------------------------------------------------------
    # Load complete DFT profiles
    # --------------------------------------------------------

    energy_df = pd.read_csv(
        matrix_file,
        index_col=0
    )

    E = energy_df.to_numpy(
        dtype=float
    )

    p_labels = energy_df.index.astype(
        str
    ).to_numpy()


    if E.shape[1] != N_Z:
        raise ValueError(
            f"{dataset_name}: expected "
            f"{N_Z} z values but found "
            f"{E.shape[1]}"
        )


    if np.isnan(E).any():
        raise ValueError(
            f"{dataset_name}: NaNs found."
        )


    n_profiles = len(E)

    print(
        f"Profiles : {n_profiles}"
    )

    print(
        "Observed DFT heights:",
        OBSERVED_Z
    )

    print(
        "Observed z indices:",
        OBSERVED_IDX
    )


    # --------------------------------------------------------
    # Array for out-of-fold predictions
    # --------------------------------------------------------

    reconstructed = np.full_like(
        E,
        np.nan,
        dtype=float
    )


    # Track which fold predicted each profile
    fold_number = np.full(
        n_profiles,
        -1,
        dtype=int
    )


    # --------------------------------------------------------
    # Cross-validation over complete p profiles
    # --------------------------------------------------------

    n_splits = min(
        N_FOLDS,
        n_profiles
    )

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )


    for fold, (train_idx, test_idx) in enumerate(
        cv.split(E),
        start=1
    ):

        E_train = E[
            train_idx
        ]

        E_test = E[
            test_idx
        ]


        # --------------------------------------------
        # Input = only six DFT energies
        # --------------------------------------------

        X_train = E_train[
            :,
            OBSERVED_IDX
        ]

        X_test = E_test[
            :,
            OBSERVED_IDX
        ]


        # --------------------------------------------
        # Target = complete 45-point profile
        # --------------------------------------------

        Y_train = E_train


        # --------------------------------------------
        # Standardize model inputs
        # --------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )


        # --------------------------------------------
        # Ridge
        # --------------------------------------------

        model = RidgeCV(
            alphas=ALPHAS
        )

        model.fit(
            X_train_scaled,
            Y_train
        )


        prediction = model.predict(
            X_test_scaled
        )


        # --------------------------------------------
        # IMPORTANT:
        #
        # At the six calculated heights, use the
        # ORIGINAL DFT value rather than ML prediction.
        #
        # Only hidden heights are ML reconstructed.
        # --------------------------------------------

        prediction[
            :,
            OBSERVED_IDX
        ] = E_test[
            :,
            OBSERVED_IDX
        ]


        reconstructed[
            test_idx
        ] = prediction

        fold_number[
            test_idx
        ] = fold


        print(
            f"Fold {fold}/{n_splits}: "
            f"train={len(train_idx)}, "
            f"test={len(test_idx)}, "
            f"alpha={model.alpha_:.3e}"
        )


    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if np.isnan(
        reconstructed
    ).any():

        raise RuntimeError(
            f"{dataset_name}: some profiles "
            f"were never reconstructed."
        )


    # Observed values must be identical to DFT
    max_observed_difference = np.max(
        np.abs(
            reconstructed[
                :,
                OBSERVED_IDX
            ]
            -
            E[
                :,
                OBSERVED_IDX
            ]
        )
    )


    if max_observed_difference > 1e-12:

        raise RuntimeError(
            "Observed DFT values were modified."
        )


    # --------------------------------------------------------
    # Metrics ONLY on reconstructed/hidden points
    # --------------------------------------------------------

    dft_hidden = E[
        :,
        HIDDEN_IDX
    ].ravel()

    ml_hidden = reconstructed[
        :,
        HIDDEN_IDX
    ].ravel()


    mae = mean_absolute_error(
        dft_hidden,
        ml_hidden
    )

    rmse = np.sqrt(
        mean_squared_error(
            dft_hidden,
            ml_hidden
        )
    )


    print(
        f"\nHidden-point MAE  : "
        f"{mae:.8f}"
    )

    print(
        f"Hidden-point RMSE : "
        f"{rmse:.8f}"
    )


    # ========================================================
    # Write SAMPLE-TIP.dat style file
    # ========================================================

    output_file = (
        OUTPUT_DIR
        / f"{dataset_name}.dat"
    )


    with open(
        output_file,
        "w"
    ) as handle:

        for p_i, p_label in enumerate(
            p_labels
        ):

            for z_index in range(
                N_Z
            ):

                z_label = (
                    f"z{z_index:02d}"
                )

                energy = reconstructed[
                    p_i,
                    z_index
                ]

                handle.write(
                    f"{p_label} "
                    f"{z_label} "
                    f"{energy:.12f}\n"
                )


    # ========================================================
    # Save detailed comparison CSV
    # ========================================================

    rows = []


    for p_i, p_label in enumerate(
        p_labels
    ):

        for z_index in range(
            N_Z
        ):

            dft_energy = E[
                p_i,
                z_index
            ]

            ml_energy = reconstructed[
                p_i,
                z_index
            ]

            rows.append(
                {
                    "p":
                        p_label,

                    "z_label":
                        f"z{z_index:02d}",

                    "z":
                        Z[z_index],

                    "dft_energy":
                        dft_energy,

                    "reconstructed_energy":
                        ml_energy,

                    "error":
                        ml_energy
                        - dft_energy,

                    "absolute_error":
                        abs(
                            ml_energy
                            - dft_energy
                        ),

                    "is_observed_dft":
                        z_index
                        in OBSERVED_IDX,

                    "fold":
                        fold_number[
                            p_i
                        ]
                }
            )


    comparison_df = pd.DataFrame(
        rows
    )


    comparison_file = (
        OUTPUT_DIR
        / f"{dataset_name}_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_file,
        index=False
    )


    print(
        f"\nReconstructed .dat:\n"
        f"    {output_file}"
    )

    print(
        f"\nComparison table:\n"
        f"    {comparison_file}"
    )


    return {
        "dataset":
            dataset_name,

        "n_profiles":
            n_profiles,

        "mae":
            mae,

        "rmse":
            rmse,

        "output_file":
            str(output_file)
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--dataset",
        type=str,
        help=(
            "Process one prepared dataset, "
            "for example OO-h-NH3_N"
        )
    )

    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Process every dataset under ml_prepared/"
        )
    )


    args = parser.parse_args()


    if args.dataset:

        reconstruct_dataset(
            args.dataset
        )

        return


    # ========================================================
    # Process all systems
    # ========================================================

    dataset_dirs = sorted(
        [
            path
            for path in PREPARED_DIR.iterdir()
            if path.is_dir()
            and (
                path
                / "energy_matrix.csv"
            ).exists()
        ]
    )


    print(
        f"\nFound {len(dataset_dirs)} "
        "prepared datasets."
    )


    all_results = []

    failures = []


    for dataset_dir in dataset_dirs:

        dataset_name = (
            dataset_dir.name
        )

        try:

            result = reconstruct_dataset(
                dataset_name
            )

            all_results.append(
                result
            )

        except Exception as exc:

            print(
                f"\nFAILED: "
                f"{dataset_name}"
            )

            print(
                exc
            )

            failures.append(
                {
                    "dataset":
                        dataset_name,

                    "error":
                        str(exc)
                }
            )


    # ========================================================
    # Overall summary
    # ========================================================

    if all_results:

        summary_df = pd.DataFrame(
            all_results
        )

        summary_df.to_csv(
            OUTPUT_DIR
            / "reconstruction_summary.csv",
            index=False
        )


        print(
            "\n" + "=" * 75
        )

        print(
            "ALL-SYSTEM SUMMARY"
        )

        print(
            "=" * 75
        )

        print(
            f"Successful : "
            f"{len(all_results)}"
        )

        print(
            f"Failed     : "
            f"{len(failures)}"
        )

        print(
            f"\nMean MAE across systems : "
            f"{summary_df['mae'].mean():.8f}"
        )

        print(
            f"Median MAE across systems : "
            f"{summary_df['mae'].median():.8f}"
        )


    if failures:

        pd.DataFrame(
            failures
        ).to_csv(
            OUTPUT_DIR
            / "failed_datasets.csv",
            index=False
        )


if __name__ == "__main__":
    main()
