# Revised-paper LJ data release (20 November 2026)

For **Equivariant learning of a transferable three-dimensional classical
density functional**, this is the current reference release. The repository
root outside `releases/lj-paper-v2` preserves the historical C3 study and must
not be mixed with the C4 configuration below.

## Model and data

- Frozen model: `fit-beta-p3nu2-rbf1normtrain-c4`, **without stability training**.
  The unchanged checkpoint is [model/model.pt](model/model.pt), SHA-256
  `6edbebb18192ca7f6a311a88c3236d61c3c5d285c739b9f533b4dc1b6e7137ca`.
- Native spacing `h=0.5 sigma`; spherical physical cutoff `2 sigma`; Cartesian
  degree 3, product order 2; one unit-normalized, trainable Gaussian radial
  channel (initial exponent 0.125, learned exponent 0.3247145414352417).
  Local-density and CACE readouts are added. See
  [configuration](model/run_config.json) and [architecture](model/model_metadata.json).
- **11,219 training + 1,255 validation + 2,055 held-out fields** are included
  in the release assets, across 174 unchanged EXTXYZ source files. The exact
  per-source validation indices and held-out membership are in the
  [split manifest](manifests/split_manifest.json). Temperatures 0.7, 1.1 and
  1.5 are excluded from training/validation.
- The paper's originally assigned benchmark pool has 1,592 small-system
  fields; the extra 463 held-out fields are distributed too, not relabeled as
  plotted paper tests. The selected revised benchmark displays **1,590 small,
  89 large and 66 GCMC fields**, with three disclosed post-hoc exclusions and
  named extended Euler runs. Full baseline records and unsuccessful outcomes
  are retained in the snapshot. The frozen training split is unchanged.
- The independent fine-grid reference contains **80 L=6 sigma fields** at
  `h=0.125,0.25,0.5 sigma`, with eight MD replicas per field, replica uncertainty
  arrays and exact averaging records. The main grid-transfer figure uses the
  disclosed post-hoc 69-field density domain; all 80 references and all-field
  metrics remain available. The `h=1` comparison uses nested block averaging,
  not an independently simulated dataset.

## Download and extract

Run from this directory. Large immutable datasets are GitHub **release assets**,
not ordinary Git history:

```sh
gh release download lj-paper-v2 --repo BingqingCheng/equicdft-lj-data --dir reference_data
python unpack.py
```

Alternatively download the three assets from the
[release page](https://github.com/BingqingCheng/equicdft-lj-data/releases/tag/lj-paper-v2)
into `reference_data/`. `unpack.py` checks their SHA-256 hashes before extraction
and refuses to overwrite extracted files. The manifest
[release_assets.json](manifests/release_assets.json) records sizes and hashes.

The extracted `data/study/cDFT` tree retains the original project-relative
paths; its sibling `cDFT-electrolyte` holds the few historical bulk-reference
tables used by the frozen plotter. These names are layout conventions, not
additional software dependencies. No trajectories or per-step MD dumps are
included. The colloid figure payload includes the plotted density slices,
force/free-energy data and uncertainty records, not every full 3D solve.

## Environment and a small example

Use a fresh environment, separate from the original C3 release:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python evaluation/evaluate_field.py
python evaluation/plot_benchmark.py
```

The example is the repository's single `T=1.5, N=208, L=8 sigma` regression
field, now evaluated with **C4**, not the older model bundled with that field.
It returns excess free energy and its first functional derivative and reports
canonical gauge-aligned forward error. The original trusted whole-object
checkpoint is loaded through `equicdft.conversion.load_converted_model`;
never use that unpickling path for an untrusted checkpoint.

Training used library commit `8832a1b41d28f6da839ecf4cde12fd128edea08d`;
the retained revision evaluations used
`45d0b47b40b74f6a5e3190b5d81b4fb55bee0134`. Clean Git exports of both are
included in [code/](code/), so reproduction does not depend on the current
development branch. Exact historical training package/CUDA versions were not
recorded; byte-identical retraining across hardware is not promised.

## Training and figures

[training/train.py](training/train.py) is the **unchanged driver retrieved
from the completed C4 fit on BRC**, not a later development script. The retained
configuration selects epoch 200 by minimum validation loss. See
[training/README.md](training/README.md) for the command. Full training and
fine-grid/application solves should run on a suitably provisioned machine,
not a memory-limited laptop.

[source_data/](source_data/) provides directly browsable numerical data for
the current main figures: bulk thermodynamics/interfaces, sharper colloids
(`w_c=0.25 sigma`), sharper gyroid walls (`w=0.2`), learning curves, the combined
benchmark and grid transfer. [evaluation/README.md](evaluation/README.md) maps
them to the frozen scripts and PDFs in `paper-snapshot_001.zip`, including
supporting harder/softer-wall results and negative outcomes. Existing root
reference archives remain the source for unchanged original large-NVT/GCMC
test inputs.

Off-native evaluation preserves physical support and radial width, rescales
Cartesian moments, and applies the retained training-grid center convention
**inside the scalar functional**. It is not density interpolation or a new
fit. The frozen helper's original diagnostic wording is historical; this
release records its subsequent adoption for the revised paper, not a change
to the library default. Chemical-potential raw/calibrated columns and shared
bulk reference offsets are retained as recorded; no independent pore gauge
is fitted. Gauge conventions and masks must be preserved when comparing
published metrics.

Data/model: CC BY 4.0; scripts: MIT, as in the repository [license](../../LICENSE).
