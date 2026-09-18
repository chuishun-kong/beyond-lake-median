# TSS metadata and eligibility

This analysis summarizes the local GLORIA release and the 400–750 nm TSS
archive used by the nested analyses. It describes data selection, measurement
metadata, and eligible target units without fitting models.

Run from the repository root after downloading GLORIA:

```powershell
python experiments/tss_metadata_evidence/run_tss_metadata_evidence.py
```

Inputs are `GLORIA_meta_and_lab.csv`, `GLORIA_Rrs.csv`, the GLORIA method
dictionary workbook, and `data/processed/eligible_tasks.csv`. Canonical groups
use `Site_name + Country + date-day + 3-decimal latitude/longitude`, as
implemented in `src/lake_insitu/gloria.py`.

`results/` contains data-flow, eligibility, source-composition, unit/log,
method-dictionary, missingness, and checksum tables. The file
`candidate_target_eligibility.csv` covers all 26 pre-diversity candidates.
The generated report is written to `results/analysis_summary.md`.

The recorded `n_unique_campaigns_dataset_id` is a Dataset_ID proxy, not a
direct field-campaign identifier. Targets are GLORIA lake-labelled sites.
These metadata describe archive heterogeneity and do not identify causal
effects of laboratory methods or providers on transfer performance.
