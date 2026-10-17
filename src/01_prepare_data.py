#!/usr/bin/env python3

"""
Prepare DFT tip-sample interaction-energy data for ML.

Expected directory structure
----------------------------

/home/mithun/work/essential_data/AFM_SCAN/12-endgroups/
│
├── results/
│   ├── OO-h-NH3_N.dat
│   ├── ...
│
├── point_all/
│   ├── OO-h_points_clean.txt
│   ├── ...
│
└── ml_prepared/
    └── created automatically


Energy file format
------------------

p000 z00  energy
p000 z01  energy
...
p001 z00  energy
...


Point file format
-----------------

type x y
C       x y
O       x y
center  x y
bond    x y
cp      x y
grid    x y
...


Physical z mapping
------------------

z00 -> 1.6 Å
z01 -> 1.7 Å
...
z44 -> 6.0 Å
"""

from pathlib import Path
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(
    "/home/mithun/work/essential_data/AFM_SCAN/12-endgroups"
)
BASEOUT_DIR = Path(
    "/home/mithun/work/essential_data/AFM_ML"
)

RESULTS_DIR = BASE_DIR / "results"
POINTS_DIR = BASE_DIR / "point_all"

# Keep processed ML data separate from raw DFT files
OUTPUT_DIR = BASEOUT_DIR / "ml_prepared"


# z00 = 1.6 Å
# z01 = 1.7 Å
# ...
# z44 = 6.0 Å

Z_START = 1.6
DZ = 0.1
N_Z = 45


# ============================================================
# Helper functions
# ============================================================

def read_points(points_file):
    """
    Read SAMPLE_points_clean.txt.

    Row number defines p index:

        row 0 -> p000
        row 1 -> p001
        ...
    """

    points = pd.read_csv(
        points_file,
        sep=r"\s+",
        comment="#"
    )

    required_columns = {"type", "x", "y"}

    if not required_columns.issubset(points.columns):
        raise ValueError(
            f"\nPoints file {points_file} does not contain "
            f"the required columns {required_columns}.\n"
            f"Found columns: {list(points.columns)}"
        )

    points = points[["type", "x", "y"]].copy()

    # Ensure coordinates are numeric
    points["x"] = pd.to_numeric(points["x"], errors="raise")
    points["y"] = pd.to_numeric(points["y"], errors="raise")

    # Assign p-number according to row order
    points["p_index"] = np.arange(len(points), dtype=int)

    points["p"] = points["p_index"].map(
        lambda i: f"p{i:03d}"
    )

    return points


def read_energies(energy_file):
    """
    Read SAMPLE-TIP.dat.

    Expected columns:

        pXXX zXX energy
    """

    energies = pd.read_csv(
        energy_file,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 2],
        names=["p", "z_label", "energy"]
    )

    if len(energies) == 0:
        raise ValueError(
            f"Energy file is empty: {energy_file}"
        )

    energies["p"] = energies["p"].astype(str)
    energies["z_label"] = energies["z_label"].astype(str)

    # --------------------------------------------------------
    # Extract numerical p index
    # --------------------------------------------------------

    p_number = energies["p"].str.extract(
        r"^p(\d+)$",
        expand=False
    )

    if p_number.isna().any():
        bad = energies.loc[
            p_number.isna(), "p"
        ].unique()

        raise ValueError(
            f"Could not understand some p labels:\n{bad[:10]}"
        )

    energies["p_index"] = p_number.astype(int)

    # --------------------------------------------------------
    # Extract numerical z index
    # --------------------------------------------------------

    z_number = energies["z_label"].str.extract(
        r"^z(\d+)$",
        expand=False
    )

    if z_number.isna().any():
        bad = energies.loc[
            z_number.isna(), "z_label"
        ].unique()

        raise ValueError(
            f"Could not understand some z labels:\n{bad[:10]}"
        )

    energies["z_index"] = z_number.astype(int)

    # Energy must be numeric
    energies["energy"] = pd.to_numeric(
        energies["energy"],
        errors="raise"
    )

    # Physical tip-sample distance
    energies["z"] = (
        Z_START
        + DZ * energies["z_index"]
    )

    return energies


# ============================================================
# Matching SAMPLE-TIP.dat to SAMPLE_points_clean.txt
# ============================================================

def find_matching_points_file(energy_file):
    """
    Automatically identify the point file belonging to an
    energy file.

    Example:

        energy:
            OO-h-NH3_N.dat

        points:
            OO-h_points_clean.txt

    Since sample names themselves may contain '-' characters,
    we identify the longest matching sample prefix.
    """

    energy_stem = energy_file.stem

    point_files = list(
        POINTS_DIR.glob("*_points_clean.txt")
    )

    matches = []

    for point_file in point_files:

        sample = point_file.name.replace(
            "_points_clean.txt",
            ""
        )

        if (
            energy_stem.startswith(sample + "-")
            or energy_stem.startswith(sample + "_")
            or energy_stem == sample
        ):
            matches.append(
                (len(sample), sample, point_file)
            )

    if not matches:
        raise FileNotFoundError(
            "\nCould not automatically find a points file for:\n"
            f"    {energy_file}\n\n"
            f"Looked inside:\n"
            f"    {POINTS_DIR}\n"
        )

    # Choose longest matching sample name.
    # This avoids problems if names overlap.
    matches.sort(reverse=True)

    _, sample_name, points_file = matches[0]

    return sample_name, points_file


# ============================================================
# Validation
# ============================================================

def validate_dataset(energies, points):
    """
    Perform strict consistency checks before ML.
    """

    print("\nChecking dataset...")

    n_points = len(points)

    # --------------------------------------------------------
    # Check duplicated p,z entries
    # --------------------------------------------------------

    duplicates = energies.duplicated(
        subset=["p_index", "z_index"]
    )

    if duplicates.any():

        duplicated_rows = energies.loc[
            duplicates,
            ["p", "z_label", "energy"]
        ]

        raise ValueError(
            "\nDuplicate (p,z) combinations found:\n"
            f"{duplicated_rows.head(20)}"
        )

    # --------------------------------------------------------
    # Check p indices
    # --------------------------------------------------------

    energy_p_indices = set(
        energies["p_index"].unique()
    )

    expected_p_indices = set(
        range(n_points)
    )

    missing_p = sorted(
        expected_p_indices - energy_p_indices
    )

    extra_p = sorted(
        energy_p_indices - expected_p_indices
    )

    if missing_p:
        raise ValueError(
            "\nSome coordinate points have no energies:\n"
            f"{missing_p[:20]}"
        )

    if extra_p:
        raise ValueError(
            "\nEnergy file contains p indices larger than "
            "the coordinate file:\n"
            f"{extra_p[:20]}"
        )

    # --------------------------------------------------------
    # Check z indices
    # --------------------------------------------------------

    expected_z = set(range(N_Z))

    bad_profiles = []

    for p_index, group in energies.groupby("p_index"):

        z_values = set(group["z_index"])

        if z_values != expected_z:

            missing_z = sorted(
                expected_z - z_values
            )

            extra_z = sorted(
                z_values - expected_z
            )

            bad_profiles.append(
                (
                    p_index,
                    missing_z,
                    extra_z
                )
            )

    if bad_profiles:

        print("\nIncomplete z profiles found:")

        for p_index, missing_z, extra_z in bad_profiles[:10]:

            print(
                f"p{p_index:03d}: "
                f"missing={missing_z}, "
                f"extra={extra_z}"
            )

        raise ValueError(
            "\nDataset does not contain exactly "
            f"{N_Z} z-values for every p point."
        )

    print("Dataset validation PASSED.")


# ============================================================
# Construct ML dataframe
# ============================================================

def build_dataframe(energies, points):
    """
    Attach x,y coordinates and sampling-point type
    to every DFT energy value.
    """

    coordinate_table = points[
        [
            "p_index",
            "type",
            "x",
            "y"
        ]
    ]

    df = energies.merge(
        coordinate_table,
        on="p_index",
        how="left",
        validate="many_to_one"
    )

    df = df[
        [
            "p",
            "p_index",
            "type",
            "x",
            "y",
            "z_label",
            "z_index",
            "z",
            "energy"
        ]
    ]

    df = df.sort_values(
        ["p_index", "z_index"]
    ).reset_index(drop=True)

    return df


# ============================================================
# Construct Np x 45 energy matrix
# ============================================================

def build_energy_matrix(df, n_points):
    """
    Each row = one lateral position.

    Each column = one z-height.

                    z00  z01 ... z44

        p000         E    E       E
        p001         E    E       E
        ...
    """

    matrix = df.pivot(
        index="p_index",
        columns="z_index",
        values="energy"
    )

    matrix = matrix.reindex(
        index=range(n_points),
        columns=range(N_Z)
    )

    matrix.index = [
        f"p{i:03d}"
        for i in matrix.index
    ]

    matrix.columns = [
        f"z{i:02d}"
        for i in matrix.columns
    ]

    return matrix


# ============================================================
# Plot 1: lateral sampling geometry
# ============================================================

def plot_sampling_geometry(points, output_file, title):
    """
    Show x,y positions of all p sampling points.
    """

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    for point_type, group in points.groupby("type"):

        ax.scatter(
            group["x"],
            group["y"],
            s=30,
            label=point_type,
            alpha=0.8
        )

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")

    ax.set_title(
        f"{title}\nNon-uniform lateral DFT sampling"
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.legend(
        fontsize=8,
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# Plot 2: all E(z) profiles
# ============================================================

def plot_all_profiles(df, output_file, title):
    """
    Plot E(z) for every p position.
    """

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for _, group in df.groupby("p_index"):

        group = group.sort_values("z")

        ax.plot(
            group["z"],
            group["energy"],
            linewidth=0.7,
            alpha=0.30
        )

    ax.set_xlabel(
        "Tip-sample distance z (Å)"
    )

    ax.set_ylabel(
        "Interaction energy"
    )

    ax.set_title(
        f"{title}\nAll DFT interaction-energy profiles"
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# Plot 3: representative E(z) curves
# ============================================================

def plot_representative_profiles(
    df,
    n_points,
    output_file,
    title,
    n_examples=9
):
    """
    Plot a small set of representative p profiles.
    """

    n_examples = min(
        n_examples,
        n_points
    )

    selected = np.linspace(
        0,
        n_points - 1,
        n_examples,
        dtype=int
    )

    selected = np.unique(selected)

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for p_index in selected:

        group = df[
            df["p_index"] == p_index
        ].sort_values("z")

        ax.plot(
            group["z"],
            group["energy"],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=f"p{p_index:03d}"
        )

    ax.set_xlabel(
        "Tip-sample distance z (Å)"
    )

    ax.set_ylabel(
        "Interaction energy"
    )

    ax.set_title(
        f"{title}\nRepresentative DFT profiles"
    )

    ax.legend(
        fontsize=8,
        frameon=False,
        ncol=2
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# Process one SAMPLE-TIP file
# ============================================================

def process_file(
    energy_file,
    points_file=None
):
    """
    Process a single SAMPLE-TIP.dat file.
    """

    energy_file = Path(energy_file).resolve()

    # --------------------------------------------------------
    # Find corresponding coordinate file
    # --------------------------------------------------------

    if points_file is None:

        sample_name, points_file = (
            find_matching_points_file(
                energy_file
            )
        )

    else:

        points_file = Path(
            points_file
        ).resolve()

        sample_name = (
            points_file.name.replace(
                "_points_clean.txt",
                ""
            )
        )

    # Tip name is whatever remains after sample name
    remainder = energy_file.stem[
        len(sample_name):
    ].lstrip("-_")

    tip_name = (
        remainder
        if remainder
        else "unknown-tip"
    )

    dataset_name = energy_file.stem

    print("\n" + "=" * 70)

    print(
        f"Processing dataset: {dataset_name}"
    )

    print(
        f"Sample: {sample_name}"
    )

    print(
        f"Tip: {tip_name}"
    )

    print(
        f"Energy file:\n    {energy_file}"
    )

    print(
        f"Points file:\n    {points_file}"
    )

    # --------------------------------------------------------
    # Read files
    # --------------------------------------------------------

    points = read_points(
        points_file
    )

    energies = read_energies(
        energy_file
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset(
        energies,
        points
    )

    # --------------------------------------------------------
    # Build long dataframe
    # --------------------------------------------------------

    df = build_dataframe(
        energies,
        points
    )

    # --------------------------------------------------------
    # Build Np x 45 matrix
    # --------------------------------------------------------

    energy_matrix = build_energy_matrix(
        df,
        len(points)
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    dataset_output = (
        OUTPUT_DIR
        / dataset_name
    )

    dataset_output.mkdir(
        parents=True,
        exist_ok=True
    )

    figures_dir = (
        dataset_output
        / "figures"
    )

    figures_dir.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save clean data
    # --------------------------------------------------------

    df.to_csv(
        dataset_output
        / "long_table.csv",
        index=False
    )

    energy_matrix.to_csv(
        dataset_output
        / "energy_matrix.csv"
    )

    points.to_csv(
        dataset_output
        / "coordinates.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    summary = {
        "dataset": dataset_name,
        "sample": sample_name,
        "tip": tip_name,
        "energy_file": str(energy_file),
        "points_file": str(points_file),
        "number_of_lateral_points": int(
            len(points)
        ),
        "number_of_z_values": int(
            df["z_index"].nunique()
        ),
        "number_of_energy_values": int(
            len(df)
        ),
        "z_min_angstrom": float(
            df["z"].min()
        ),
        "z_max_angstrom": float(
            df["z"].max()
        ),
        "z_spacing_angstrom": DZ
    }

    with open(
        dataset_output
        / "metadata.json",
        "w"
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=4
        )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_sampling_geometry(
        points,
        figures_dir
        / "sampling_geometry.png",
        dataset_name
    )

    plot_all_profiles(
        df,
        figures_dir
        / "all_energy_profiles.png",
        dataset_name
    )

    plot_representative_profiles(
        df,
        len(points),
        figures_dir
        / "representative_profiles.png",
        dataset_name
    )

    # --------------------------------------------------------
    # Print useful information
    # --------------------------------------------------------

    counts = df.groupby(
        "p"
    ).size()

    print("\nDataset summary")
    print("-" * 70)

    print(
        f"Lateral positions : "
        f"{df['p'].nunique()}"
    )

    print(
        f"z values          : "
        f"{df['z'].nunique()}"
    )

    print(
        f"Energy values     : "
        f"{len(df)}"
    )

    print(
        f"z range           : "
        f"{df['z'].min():.1f} "
        f"to "
        f"{df['z'].max():.1f} Å"
    )

    print(
        f"Values per p      : "
        f"{counts.min()} "
        f"to "
        f"{counts.max()}"
    )

    print(
        f"Energy matrix     : "
        f"{energy_matrix.shape}"
    )

    print(
        "\nOutput written to:\n"
        f"    {dataset_output}"
    )

    print("\nCreated files:")

    print(
        "    long_table.csv"
    )

    print(
        "    energy_matrix.csv"
    )

    print(
        "    coordinates.csv"
    )

    print(
        "    metadata.json"
    )

    print(
        "    figures/sampling_geometry.png"
    )

    print(
        "    figures/all_energy_profiles.png"
    )

    print(
        "    figures/representative_profiles.png"
    )

    print("=" * 70)

    return summary


# ============================================================
# Command-line interface
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare DFT AFM interaction-energy "
            "data for ML."
        )
    )

    parser.add_argument(
        "--energy-file",
        type=str,
        default=None,
        help=(
            "Process one .dat file. "
            "May be an absolute path or a filename "
            "inside the results directory. "
            "If omitted, all .dat files are processed."
        )
    )

    parser.add_argument(
        "--points-file",
        type=str,
        default=None,
        help=(
            "Optional explicit coordinate file. "
            "Normally unnecessary because it is "
            "matched automatically."
        )
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process ONE energy file
    # --------------------------------------------------------

    if args.energy_file:

        energy_file = Path(
            args.energy_file
        )

        if not energy_file.is_absolute():

            energy_file = (
                RESULTS_DIR
                / energy_file
            )

        if not energy_file.exists():

            sys.exit(
                f"\nEnergy file does not exist:\n"
                f"{energy_file}"
            )

        points_file = None

        if args.points_file:

            points_file = Path(
                args.points_file
            )

            if not points_file.is_absolute():

                points_file = (
                    POINTS_DIR
                    / points_file
                )

            if not points_file.exists():

                sys.exit(
                    f"\nPoints file does not exist:\n"
                    f"{points_file}"
                )

        process_file(
            energy_file,
            points_file
        )

    # --------------------------------------------------------
    # Process ALL .dat files
    # --------------------------------------------------------

    else:

        energy_files = sorted(
            RESULTS_DIR.glob("*.dat")
        )

        if not energy_files:

            sys.exit(
                "\nNo .dat files found in:\n"
                f"{RESULTS_DIR}"
            )

        print(
            f"\nFound {len(energy_files)} "
            f".dat files."
        )

        successful = []
        failed = []

        for energy_file in energy_files:

            try:

                summary = process_file(
                    energy_file
                )

                successful.append(
                    summary["dataset"]
                )

            except Exception as exc:

                print(
                    "\nERROR while processing:"
                )

                print(
                    f"    {energy_file.name}"
                )

                print(
                    f"    {exc}"
                )

                failed.append(
                    (
                        energy_file.name,
                        str(exc)
                    )
                )

        print("\n\nFINAL SUMMARY")
        print("=" * 70)

        print(
            f"Successful: {len(successful)}"
        )

        print(
            f"Failed    : {len(failed)}"
        )

        if failed:

            print(
                "\nFailed datasets:"
            )

            for filename, error in failed:

                print(
                    f"\n{filename}\n"
                    f"    {error}"
                )


if __name__ == "__main__":
    main()
