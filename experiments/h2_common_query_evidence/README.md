# H2 common-query supporting analyses

The analysis reads the frozen true-nested H2 own-pool, pairwise common-query,
four-way common-query, attrition, per-repeat, and per-lake artifacts. It does
not fit or tune a model, regenerate supports, queries, or predictions, or
replace the primary own-remaining-pool H2 estimand.

Generate the supporting summaries and tables:

```powershell
python experiments/h2_common_query_evidence/run_h2_common_query_evidence.py
```

## Estimand boundary

The frozen four-way summary is an **all-24-lake available-repeat common-query
sensitivity**: unusable repeat rows were removed, but the persisted aggregator
did not apply its configured >=50 usable-repeat threshold before forming
lake-level medians. Thus the number of usable repeats per lake can differ.
At k=10, all 72 lake--learner units entered the frozen summary; 54 had at
least 50 usable repeats and 18 had fewer.

The script also derives a separate **post hoc >=50-usable-repeat restricted-pool
sensitivity** from the frozen repeat-level metrics. It is a different
usability-defined estimand and never replaces the frozen available-repeat
result.

Outputs are written under `results/`:

- `h2_pairwise_common_query_audited.csv` and
  `h2_fourway_common_query_audited.csv` retain the frozen common-query effects,
  normalize the manuscript sign (`MAE_random - MAE_selector`), and attach
  repeat-level provenance.
- `h2_fourway_available_repeat_audited.csv` and
  `h2_fourway_repeat_coverage.csv` document the frozen all-lake entry rule and
  unequal within-lake usable-repeat coverage.
- `h2_fourway_threshold50_sensitivity.csv` and
  `h2_fourway_available_vs_threshold50.csv` retain the post hoc thresholded
  estimand and its explicit comparison with the frozen result.
- `h2_query_attrition_audited.csv` distinguishes row-level usable-repeat
  attrition from frozen lake-entry coverage.
- `h2_cross_estimand_decision_matrix.csv` keeps the primary weighted own-pool
  result separate from the two supporting sensitivities.
- Tables S7--S10 are renderer-produced supporting tables only. S8 is frozen
  available-repeat four-way evidence; S9 is its coverage report; S10 is the
  post hoc threshold-50 sensitivity.

Intervals are nominal cell-wise lake-bootstrap intervals and are not
multiplicity-adjusted. The Jasień identity sensitivity remains a numerical
stability note: it does not change the primary H2 conclusion.
The generated reports `analysis_summary.md` and `fourway_aggregation.md`
are also written under `results/`.
