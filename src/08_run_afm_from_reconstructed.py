#!/usr/bin/env python3

"""
Generate PP-AFM images from ML-reconstructed SAMPLE-TIP.dat files.

Input energies:
    /home/mithun/work/essential_data/AFM_ML/result_new/*.dat

Coordinates:
    /home/mithun/work/essential_data/AFM_SCAN/12-endgroups/point_all/
        SAMPLE_points_clean.txt

Output:
    /home/mithun/work/essential_data/AFM_ML/afm_from_ml/
"""

from pathlib import Path
import argparse
import subprocess
import sys
import shutil


# ============================================================
# Defaults
# ============================================================

PROJECT_DIR = Path(
    "/home/mithun/work/essential_data/AFM_ML"
)

DEFAULT_RESULTS_DIR = (
    PROJECT_DIR
    / "result_new"
)

DEFAULT_POINTS_DIR = Path(
    "/home/mithun/work/essential_data/AFM_SCAN/12-endgroups/point_all"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "afm_from_ml"
)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--results_dir",
    type=Path,
    default=DEFAULT_RESULTS_DIR
)

parser.add_argument(
    "--points_dir",
    type=Path,
    default=DEFAULT_POINTS_DIR
)

parser.add_argument(
    "--out_dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR
)

parser.add_argument(
    "--relax_script",
    type=Path,
    required=True,
    help="Full path to test_relax_kriging_crop.py"
)

parser.add_argument(
    "--only_dataset_prefix",
    type=str,
    default="",
    help="For example: OO-h"
)

parser.add_argument(
    "--only_tip",
    type=str,
    default="",
    help="For example: NH3_N"
)

parser.add_argument(
    "--rebuild_gridff",
    action="store_true",
    help="Delete cached gridFF.npy/lvec.npy before running"
)

parser.add_argument(
    "--fast",
    action="store_true",
    help=(
        "Run only klat=0.2 without rigid scan "
        "and disable heavy comparison figures"
    )
)

args = parser.parse_args()


RESULTS_DIR = args.results_dir.resolve()
POINTS_DIR = args.points_dir.resolve()
OUTPUT_DIR = args.out_dir.resolve()
RELAX_SCRIPT = args.relax_script.resolve()


# ============================================================
# Checks
# ============================================================

if not RESULTS_DIR.exists():
    sys.exit(
        f"Results directory does not exist:\n{RESULTS_DIR}"
    )

if not POINTS_DIR.exists():
    sys.exit(
        f"Points directory does not exist:\n{POINTS_DIR}"
    )

if not RELAX_SCRIPT.exists():
    sys.exit(
        f"Relaxation script does not exist:\n{RELAX_SCRIPT}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Find samples
# ============================================================

points_files = sorted(
    POINTS_DIR.glob(
        "*_points_clean.txt"
    )
)

if not points_files:
    sys.exit(
        f"No *_points_clean.txt files found in:\n{POINTS_DIR}"
    )


print(
    f"\nFound {len(points_files)} point files."
)

print(
    f"ML energies:\n    {RESULTS_DIR}"
)

print(
    f"AFM outputs:\n    {OUTPUT_DIR}"
)

print(
    f"Relaxation script:\n    {RELAX_SCRIPT}"
)


# ============================================================
# Run
# ============================================================

n_run = 0
n_failed = 0
n_skipped = 0


for points_file in points_files:

    sample = points_file.name.replace(
        "_points_clean.txt",
        ""
    )

    if (
        args.only_dataset_prefix
        and not sample.startswith(
            args.only_dataset_prefix
        )
    ):
        continue


    # --------------------------------------------------------
    # Find reconstructed tip variants for this sample
    # --------------------------------------------------------

    zscan_files = sorted(
        RESULTS_DIR.glob(
            f"{sample}-*.dat"
        )
    )


    if not zscan_files:

        print(
            f"[skip] {sample}: "
            "no reconstructed .dat files"
        )

        n_skipped += 1
        continue


    print(
        f"\n[{sample}] "
        f"{len(zscan_files)} tip variants"
    )


    for zscan_file in zscan_files:

        # Everything after SAMPLE-
        tip_variant = (
            zscan_file.stem[
                len(sample) + 1:
            ]
        )


        if (
            args.only_tip
            and tip_variant
            != args.only_tip
        ):
            continue


        # ----------------------------------------------------
        # Separate output/cache for ML data
        # ----------------------------------------------------

        out_dir = (
            OUTPUT_DIR
            / sample
            / f"{sample}_{tip_variant}"
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # IMPORTANT:
        # Keep cache here, separately from original DFT runs.
        cache_dir = out_dir


        # ----------------------------------------------------
        # Optionally remove stale ML cache
        # ----------------------------------------------------

        if args.rebuild_gridff:

            for filename in [
                "gridFF.npy",
                "lvec.npy"
            ]:

                path = (
                    cache_dir
                    / filename
                )

                if path.exists():

                    print(
                        f"[cache] removing {path}"
                    )

                    path.unlink()


        # ----------------------------------------------------
        # Command
        # ----------------------------------------------------

        cmd = [
            sys.executable,
            str(RELAX_SCRIPT),

            "--points_file",
            str(points_file),

            "--zscan_file",
            str(zscan_file),

            "--out_dir",
            str(out_dir),

            "--gridff_cache_dir",
            str(cache_dir),

            "--save_outputs",
            "0",

            "--outfz_cmap",
            "gray",

            "--tip_name",
            tip_variant,
        ]


        # ----------------------------------------------------
        # Fast mode
        # ----------------------------------------------------

        if args.fast:

            cmd.extend(
                [
                    "--klat_values",
                    "0.2",

                    "--include_rigid",
                    "0",

                    "--plot_pppos",
                    "0",

                    "--plot_comparison",
                    "0",

                    "--plot_gridff",
                    "0",

                    "--plot_stiffness_comparison",
                    "0",

                    "--plot_outfz",
                    "1",
                ]
            )

        else:

            # These reproduce the behavior of your
            # existing batch runner more closely.
            cmd.extend(
                [
                    "--plot_comparison",
                    "1",

                    "--plot_pppos",
                    "0",

                    "--plot_gridff",
                    "0",
                ]
            )


        print("\n" + "=" * 75)

        print(
            f"[run] sample = {sample}"
        )

        print(
            f"[run] tip    = {tip_variant}"
        )

        print(
            f"[run] energy = {zscan_file}"
        )

        print(
            f"[run] output = {out_dir}"
        )

        print("=" * 75)


        try:

            subprocess.run(
                cmd,
                check=True
            )

            n_run += 1

            print(
                f"[done] "
                f"{sample}-{tip_variant}"
            )


        except subprocess.CalledProcessError as exc:

            n_failed += 1

            print(
                f"[FAILED] "
                f"{sample}-{tip_variant}: "
                f"{exc}"
            )


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 75)

print("FINAL SUMMARY")

print("=" * 75)

print(
    f"Completed : {n_run}"
)

print(
    f"Failed    : {n_failed}"
)

print(
    f"Skipped   : {n_skipped}"
)

print(
    f"\nResults:\n{OUTPUT_DIR}"
)
