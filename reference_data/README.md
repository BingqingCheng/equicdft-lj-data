# Reference-data archives

These archives contain the evaluation inputs needed to reproduce the reported
results with the released model. They do not contain the training or validation
corpus. Every archive stores paths beginning with `data/` and can be extracted
from the repository root with:

```bash
python unpack_reference_data.py
```

The archives contain:

- `heldout_nvt_*.zip`: 1,592 held-out NVT density fields in the four data
  collections used by the forward and inverse benchmarks;
- `large_nvt.zip`: 90 larger-system NVT fields;
- `gcmc_benchmark.zip`: the 66 GCMC fields shown in the chemical-potential
  benchmark, together with their selection table;
- `bulk_response.zip`: three reference structure factors, the reference EOS
  summary, and four liquid--vapor slab density fields;
- `colloid_bridge.zip`: ten colloid reference density fields, external-field
  definitions, and the direct-force summary;
- `gyroid_adsorption.zip`: the fixed-N gyroid structural reference and eleven
  confined GCMC density fields used for the adsorption calculation.

Numerical source data underlying the final figures are tracked separately in
`source_data/`.
