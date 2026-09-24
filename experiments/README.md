# Experiments

Each directory contains analysis scripts and derived results under `results/`.
The table below identifies the main entry points. Run commands from the
repository root; download raw data as described in
[DATA_SOURCES.md](../data/raw/DATA_SOURCES.md) before fitting models or rebuilding
metadata.

## Analysis map

| Analysis | Directory | Main entry points | Included results |
|---|---|---|---|
| H1 calibration error and H3 within-target ordering | `capacity_boundary/` | `run_nested_capacity_boundary.py`, `run_h3_paired_source_audit_nested.py` | Repeat-level H1 and H3 paired metrics, target summaries, and full-precision summaries |
| Source-distance diagnostic | `capacity_boundary/` | `run_h4_reverify.py` | `h4_reverify_merged.csv` |
| H2 selector comparisons and Table S11 | `active_selection/` | `run_h2_true_nested.py`, `build_h2_nested_summaries.py`, `build_h2_mean_aggregation_sensitivity.py` | Own-pool, common-query, and mean-over-repetitions summaries |
| Matched controls M0–M4 | `ladder_statistics/` | `run_ladder_statistics.py` | `target_effects.csv`, `interval_table.csv` |
| H1 three-learner sensitivity, Table S6E | `h1_sensitivity/` | `recompute_h1_sensitivity.py` | 72 full-precision inputs and seed-0/1/2 JSON outputs |
| H1 bootstrap reconstruction and sensitivity | `h1_bootstrap_provenance/` | `run_h1_bootstrap_provenance.py` | `canonical_h1_reconstruction.csv` |
| Absolute-loss analysis, Table S6 | `h1_absolute_loss/` | `run_h1_absolute_loss.py`, `render_supplementary_table.py` | Repeat-level, target-level, and across-target effects; Markdown table |
| Common-query sensitivity, Tables S7–S10 | `h2_common_query_evidence/` | `run_h2_common_query_evidence.py`, `render_supplementary_tables.py` | Pairwise and four-way effects, coverage, threshold-50 sensitivity |
| Metadata, Tables S2–S5 | `tss_metadata_evidence/` | `run_tss_metadata_evidence.py`, `render_manuscript_tables.py` | Composition, eligibility, methods, missingness, and table sources |
| Eligibility and source composition | `source_domain_audit/` | `run_source_domain_audit.py`, `verify_eligibility_rule.py` | Source-composition, alias, method, and inner-fold exclusion tables |
| Jasień identity sensitivity | `jasien_identity_sensitivity/` | `run_jasien_identity_sensitivity.py` | Targeted-fold and hybrid-summary results |
| China TSM stress test | `china_stress_test/` | `run_china_stress_test.py` | Repeat-level and summary tables |

## Reproduction scope

H1 absolute-loss analysis, H1 bootstrap reconstruction, H2 summary aggregation,
common-query sensitivities, and ladder statistics use included CSVs without
refitting models. The table renderers likewise use included results.
Model-fitting entry points, metadata reconstruction, source-domain analysis,
and Jasień/China reruns require raw data. H3 predictions and fitted checkpoints
are not bundled. Stored paired metrics and self-check records are included;
`python experiments/capacity_boundary/validate_h3_archived.py` verifies coverage,
finite-value agreement, target medians, and bootstrap summaries without fitting
models. This is not an independent reconstruction of query predictions.

Table S1 is supplied at
`active_selection/results/table_s1_h2_direct_mae_vs_rer.csv`. Its exploratory,
separately seeded bootstrap inputs are not bundled, so the accompanying
table-generation script is not standalone. Table S6E can be regenerated with
`python experiments/h1_sensitivity/recompute_h1_sensitivity.py`; see its README
for the NumPy 2.3.5 pin, 72-value source mapping, aggregation, and exact-output
comparison. It uses only included derived inputs.

## Matched-control aggregation

The ladder was added after the primary H1-H3 analyses. The
[analysis protocol](ladder_statistics/analysis_protocol.yaml) is a reader-facing
transcription exported on 2026-09-24 from the protocol created on 2026-09-05
and revised before execution on 2026-09-06. It is not a preregistration or an
independently timestamped deposit. The six-interval D24/D34 family preceded
aggregation of these new contrasts; D14 had already been reported as Table S6B.

The ladder uses episode-level losses from
`capacity_boundary/results/calibration_direction_ablation_nested_per_rep.csv`
and `capacity_boundary/results/m1_label_only_per_rep_algebraic.csv`.
The latter recovers label-only loss as
`B = (C + r * 1e-8) / (1 - r)`, where `C` is calibrated log-MAE and `r` is
the stored fractional RER. A direct support-seed replay verified the
reconstruction to numerical precision.

Loss differences are paired within each episode, reduced to a median over
100 repetitions within each target, and then averaged equally over 24 targets.
Nominal intervals use 10,000 target-level bootstrap resamples with a fresh
`default_rng(0)` per cell. The D24/D34 family spans three learners and uses six
Bonferroni intervals with 100,000 resamples. D02/D23 are nominal-only diagnostic
contrasts. These comparisons do not establish a pure or causal decomposition
of spectral information.

## Shared inputs and run records

- `capacity_boundary/results/capacity_boundary_repeats_nested.csv` and
  `capacity_boundary_summary_nested.csv` support the H1 analyses.
- `capacity_boundary/results/nested_hyperparameters.csv` is also read by H2.
- `active_selection/results/true_nested_h2/h2_nested_per_repeat.csv` supports
  common-query reaggregation. Keep these input tables together.

Run manifests retain the input hashes and settings from their generating runs.
Historical run identifiers refer to those runs, not to additional execution
steps. Newly generated analysis summaries are written beside the result tables.
