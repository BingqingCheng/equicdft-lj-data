# Revised-paper results

First extract the three release assets. Let
`P=data/study/cDFT/NatComm-review/2026-11-19-c4-production-update`.
The snapshot preserves data, original scripts, figure PDFs, masks, gauges,
uncertainties and failure records. Some BRC runners deliberately retain their
original absolute paths/CUDA guards; they are execution records, not laptop
entry points. The following are lightweight plotting entry points from the
release directory (they do not run models):

```sh
P=data/study/cDFT/NatComm-review/2026-11-19-c4-production-update
python "$P/plot_bulk_panels.py"
python "$P/assemble_bulk_figure.py"
python "$P/dimer/plot_dimer.py"
python evaluation/plot_benchmark.py
python "$P/grid-transfer/plot_figure.py"
python data/study/cDFT/benchmark-plot-scripts/cace_cubic_data_efficiency.py
python data/study/cDFT/LJ-applications-soft/05-gyroid-pore/render_c4_gyroid_fine_revision_v3.py \
  "$P/gyroid/rendering" output/gyroid/output/pdf
```

| Main figure | Direct source data | Frozen PDF within `$P` |
|---|---|---|
| 2: bulk/phase/interfaces | `source_data/fig02_bulk/`, plus snapshot interface profiles | `figure_bulk_response/figure_bulk_response_c4_two_grid.pdf` |
| 3: colloids | `source_data/fig03_colloid/`, snapshot `dimer/density/` | `dimer/two_colloid_C4_wc025_paired_grids.pdf` |
| 4: gyroid | `source_data/fig04_gyroid/`, snapshot `gyroid/rendering/output/structural_comparison.npz` | `gyroid/fine-grid-main-parity-v3/output/pdf/gyroid_application_C4_w020_fine025_parity.pdf` |
| 5: learning curve | `source_data/fig05_learning_curve/`, saved histories/configs under the snapshot training directory | snapshot `train-nvt-morecomprehensive-brc/data_efficiency_c4_comparison_20261119/cace_cubic_data_efficiency.pdf` (relative to `data/study/cDFT`) |
| 6: held-out benchmark | `source_data/fig06_benchmarks/` | `benchmark/three-case-exclusion-small-euler-recovered-v2/figure5_three_excluded_small_euler_recovered_notitle_r2_3dp.pdf` |
| 7: grid transfer | `source_data/fig07_grid_transfer/` | `grid-transfer/output/pdf/equicdft_c4_grid_transfer.pdf` |

Historical filenames containing “Fig5” do not imply current manuscript numbering.
The plotting cache is display-only; the accompanying full-data metrics are not
recomputed from sampled plot points. Axis clipping in the benchmark is a display
choice, not an exclusion from metric aggregation.

The gyroid includes pre-rendered isosurface PNGs to avoid a memory-intensive
3D rerender. Its final composition script is
`LJ-applications-soft/05-gyroid-pore/render_c4_gyroid_fine_revision_v3.py`;
see its `--help` and the frozen rendering completion record. Complete fields
and unsmoothed parity arrays remain in the snapshot.

The independent grid-transfer reference is under
`data/study/cDFT/LJ_inputs_veryfine_gaussians_0125/reference/`. Each field has
three MD density/potential grids, `field.json`, replica statistics and a
hash-bearing aggregate validation record. Its 80 fields are separate from the
174-file training corpus. Both the selected 69-field figure and complete
80-field histograms/metrics are retained; the selection is post hoc and is not
a claim of universal grid independence or universal accuracy at high density.

The snapshot's `inference-timing-l8-v2/` contains the measured CPU/GPU timings.
Its h=0.25 fixture repeats the h=0.5 voxels **for timing only**; unlike the L6
references, it is not fine-grid MD or an accuracy benchmark. Finite 200-update
nonconverged timing solves are explicitly retained.
