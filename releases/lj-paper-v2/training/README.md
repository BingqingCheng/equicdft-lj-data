# Reproduce the C4 fit

The unmodified `train.py` was copied from the completed BRC fit's `training.py`.
Extract `code/equicdft-training-8832a1b.tar.gz` into `code/training/` and use a
dedicated environment with that source and its dependencies installed. Do not
use the evaluation pin or the repository-root historical requirements for training.
The full corpus must first be downloaded/extracted as in the release README.

Run on a GPU/HPC machine, from the release directory:

```sh
export EQUICDFT_REPOSITORY="$PWD/code/training"
export EQUICDFT_COMMIT=8832a1b41d28f6da839ecf4cde12fd128edea08d
python training/train.py \
  --manifest manifests/split_manifest.json \
  --general-directory data/study/cDFT/training_data_nvt \
  --dimensional-directory data/study/cDFT/training_data_nvt_1D2D \
  --mixed-directory data/study/cDFT/training_data_nvt_mixed \
  --extended-directory data/study/cDFT/training_data_nvt_extended \
  --mixed-more-directory data/study/cDFT/training_data_nvt_mixed_more \
  --output-directory output/retraining \
  --fit-name fit-beta-p3nu2-rbf1normtrain-c4 \
  --max-power 3 --cutoff-grid 4 --radial-basis gaussian \
  --radial-exponents 0.125 --trainable-radial-exponents \
  --coordinate-scaling none --free-energy-mode beta \
  --epochs 200 --batch-size 2 --seed 1 --early-stopping-patience 200
```

The driver retains the production loss, density mask, optimizer, scheduler,
resume behavior and checkpoint-selection rule. There is no Fourier-stability
loss. [history.csv](history.csv) is the retained production history, not a new
retraining run. Hardware and unrecorded training-environment differences may
change the trajectory; use the released checkpoint to reproduce paper predictions.
