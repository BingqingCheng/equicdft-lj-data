# Equi-cDFT Lennard-Jones study

## Current revised-paper release

Use **[releases/lj-paper-v2](releases/lj-paper-v2/README.md)** for the revised
paper: the normalized-radial **C4 model without stability training**, the full
11,219/1,255/2,055 training/validation/held-out corpus, fine-grid MD reference
fields, and updated paired-grid figure data and scripts. Large datasets are
available as [versioned release assets](https://github.com/BingqingCheng/equicdft-lj-data/releases/tag/lj-paper-v2).

The material below and the original root-level model, scripts and archives
are the **historical C3 release**. They are preserved for reproducibility;
its configurations and field counts do not describe the revised C4 fit.

## Historical C3 release

This repository contains the files needed to reproduce the paper
**Equivariant learning of classical density functionals for fluids**. The
Equi-cDFT library itself is maintained at
[`BingqingCheng/equicdft`](https://github.com/BingqingCheng/equicdft).

## Contents

- `model/`: production model and its configuration.
- `manifests/split_manifest.json`: exact training, validation, and test split.
- `training/`: training and split-generation scripts.
- `evaluation/`: scripts that run the model tests and applications, analyze
  the outputs, and generate the figures; reference simulation inputs are kept
  beside the corresponding application scripts.
- `reference_data/`: compressed held-out and reference datasets used for the
  reported benchmarks and applications.
- `source_data/`: numerical data underlying the paper figures.

The training and validation corpus is not included. The released model and the
compressed evaluation datasets are sufficient to recompute the reported model
benchmarks and applications.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements pin the Equi-cDFT code revision used for the production
model.

## Training data and split

The fit uses 10,957 canonical equilibrium density fields on periodic
three-dimensional grids: 8,427 for training, 938 for validation, and 1,592 for
testing. Temperatures `T=0.7`, `1.1`, and `1.5` are held out completely.

Each EXTXYZ frame stores the density and external potential on the same regular
grid. Numerical values use reduced Lennard-Jones units with
`epsilon = sigma = m = k_B = 1`; the pair potential is truncated and shifted
at `r_c = 2.5 sigma`.

The exact split is recorded in `manifests/split_manifest.json`. Only the 1,592
held-out fields are distributed here; the training and validation fields are
not part of this repository.

## Reproduction

First extract the compressed reference datasets:

```bash
python unpack_reference_data.py
```

This creates the ignored working directory `data/` using the paths expected by
the evaluation scripts. Training code and the exact production configuration
are retained for transparency, but reproducing the fit itself requires the
unreleased training corpus.

The model tests are run by:

- `evaluation/benchmarks/run_heldout_nvt.py`;
- `evaluation/benchmarks/run_large_nvt.py`;
- `evaluation/benchmarks/run_gcmc_mu.py`.

The thermodynamic and interfacial calculations are run by the `compute_*` and
`solve_*` scripts in `evaluation/bulk_response/`. The two-colloid reference
scan, fixed-particle-number solves and dense force curve are run by the
`prepare_*`, `summarize_*` and `solve_*` scripts in
`evaluation/colloid_bridge/`. The paired GCMC campaigns and learned adsorption
isotherm are run by the corresponding `prepare_*`, `postprocess_*`,
`analyze_*` and `solve_*` scripts in `evaluation/gyroid_adsorption/`.
The common fixed-particle-number density solves, including the NVT gyroid
structural calculation, are run by
`evaluation/solve_fixed_n_applications.py`.

Figures can then be regenerated from the resulting or checked-in source data.
For example:

```bash
python evaluation/benchmarks/plot_combined_benchmarks.py
```

Generated outputs are written to `output/` and are not tracked.

## License

Data and the trained model are CC BY 4.0. Scripts are MIT licensed; see
`LICENSE`.
