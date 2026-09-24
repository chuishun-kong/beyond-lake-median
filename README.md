# Matched-Control Evaluation of Few-Label Calibration of Machine-Learning Models for Total Suspended Solids Retrieval from In Situ Hyperspectral Reflectance

Code and derived results accompanying the manuscript.

## Overview

This study evaluates few-label calibration against predictors using the same
support labels and query observations, with budgets of 1, 3, 5, and 10 labels
and a source-only reference. It uses 400–750 nm in situ hyperspectral
reflectance from GLORIA under a nested, target-query-isolated protocol.
Matched controls distinguish improvements relative to a label-only baseline
from improvements relative to simpler source-model corrections. These are
predictive comparisons, not a causal decomposition of spectral information.

The repository includes analysis code, configurations, tests, and derived
results. Raw datasets must be downloaded separately. The
[experiment guide](experiments/README.md) maps analyses to scripts and tables
and describes which results can be reproduced from the included files.

## Installation

Python 3.10 or later is required.

```bash
pip install -r requirements.txt
```

## Data

| Dataset | Use | Download | Local directory |
|---|---|---|---|
| GLORIA | Primary TSS benchmark | [PANGAEA](https://doi.org/10.1594/PANGAEA.948492) | `data/raw/gloria/` |
| China lake TSM | Supporting stress test | [Zenodo](https://doi.org/10.5281/zenodo.13777017) | `data/raw/china_lakes/` |

See [data sources](data/raw/DATA_SOURCES.md) for the expected filenames.
The China analysis crosses optical, geographic, and label-definition
boundaries; its TSM results are reported separately from GLORIA TSS.

## Repository structure

```text
src/lake_insitu/    data loading, models, metrics, sampling, and selection
experiments/       analysis scripts, configurations, and derived results
tests/             model-fitting smoke tests and aggregation checks
data/processed/    small derived tables used by the analyses
data/raw/          download instructions and local data directories
```

## Reproducing results

The following commands use included results and do not require raw data:

```bash
python experiments/h1_absolute_loss/render_supplementary_table.py
python experiments/h2_common_query_evidence/render_supplementary_tables.py
python experiments/ladder_statistics/run_ladder_statistics.py
python experiments/h1_sensitivity/recompute_h1_sensitivity.py
python experiments/capacity_boundary/validate_h3_archived.py
python -m pytest tests/ -v
```

Model fitting requires the raw data. From the repository root, in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python experiments/capacity_boundary/run_nested_capacity_boundary.py
python experiments/active_selection/run_h2_true_nested.py
```

On Linux or macOS, use `export PYTHONPATH=src` before the same Python commands.

The included episode-level results support H1/H2 reaggregation and the
matched-control ladder. H3 includes the archived paired metrics, target summaries,
and self-check record. The validation command checks finite values, complete
100-draw/24-target coverage, and reproduces the summaries without fitting models.
Reconstructing query predictions still requires raw spectra and model fitting. Repeating
the Jasień identity sensitivity or the China stress test also requires raw data.

Table S1 is included as a derived CSV, but its separately seeded historical
bootstrap inputs are not bundled; its table-generation script is therefore
not standalone. The post hoc H1 three-learner multiplicity sensitivity
(Table S6E) is provided in [experiments/h1_sensitivity](experiments/h1_sensitivity/README.md),
including its full-precision 72 target values, original script and JSON outputs,
and NumPy 2.3.5 dependency pin. Its seed-0 outputs reproduce the manuscript;
seed-1/2 checks retain the same directional classification. This three-learner
sensitivity is separate from the ladder
[analysis protocol](experiments/ladder_statistics/analysis_protocol.yaml).

## License

Code is released under the [MIT License](LICENSE). Datasets retain their
providers' licenses. Derived result tables are supplied for reproducibility.
