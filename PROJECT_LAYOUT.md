# Suggested project layout

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

Suggested mapping:

```text
prepare_dft_data.py          -> src/01_prepare_data.py
02_sparse_reconstruction.py  -> src/02_sparse_reconstruction.py
03_select_z_heights.py       -> src/03_select_z_heights.py
04_robustness.py             -> src/04_robustness.py
05_spatial_holdout.py        -> src/05_spatial_holdout.py
07_export_reconstructed_dat.py
                             -> src/07_export_reconstructed_dat.py
run_all_ml_afm.py            -> src/08_run_afm_from_reconstructed.py
```

Keep large/private datasets outside the repository. Commit only approved example data and small summary CSV files.
