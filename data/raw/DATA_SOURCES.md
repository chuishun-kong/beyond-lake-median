# Data sources

This project uses the public GLORIA dataset. **It is not bundled in this
repository** — download it separately and place it under `data/raw/` before
reproducing the analyses.

## GLORIA (primary benchmark)

| | |
|---|---|
| Paper | Loisel et al. 2023, *Scientific Data* 10, 100 |
| Paper DOI | https://doi.org/10.1038/s41597-023-01973-y |
| Data DOI | https://doi.org/10.1594/PANGAEA.948492 |
| Place at | `data/raw/gloria/` (unzip the downloaded `GLORIA-2022.zip`) |

Key files used by the code:

- `GLORIA_meta_and_lab.csv`
- `GLORIA_Rrs.csv`
- `GLORIA_Rrs_mean.csv`
- `GLORIA_qc_flags.csv`

GLORIA contains 7,572 hyperspectral in situ Rrs records with water-quality
labels; this study uses the 400–750 nm range and the TSS label.
