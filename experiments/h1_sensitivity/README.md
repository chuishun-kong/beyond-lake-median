# H1 three-learner multiplicity sensitivity (Table S6E)

From the repository root, using Python 3.12.3 for the recorded reproduction:

```bash
python -m pip install -r experiments/h1_sensitivity/requirements.txt
python experiments/h1_sensitivity/recompute_h1_sensitivity.py
```

Only NumPy 2.3.5 is needed. No raw observations, model fitting, network access,
or manuscript handoff directory is needed to run the script. The script and
two JSON files are the original manuscript sensitivity artifacts. It embeds
the full-precision input values and writes the two JSON files beside itself;
copy them elsewhere first if comparing output bytes.

The 72 inputs are the k=5 calibrated `median_rer_log` values in
`../capacity_boundary/results/capacity_boundary_summary_nested.csv`, in the
24-target order recorded in `h1_k5_public_input_values.json`. RER is stored as
a fraction and converted to percent before aggregation. All 72 values were
matched exactly to that CSV with round-trip float parsing on 2026-09-24.
They are target-wise medians of the 100 support-draw RERs, not reconstructed
from manuscript means or confidence limits. The inferential statistic is the
equal-weight mean over targets.

For each learner the script starts a fresh `default_rng(0)`, samples 100,000
sets of 24 target indices, and uses percentile limits `0.05/(2*3)` and
`1-0.05/(2*3)`. Table S6E corresponds to `post_hoc_100000_seed0`, not the
primary 10,000-resample block. Seeds 1 and 2 are Monte Carlo checks; they do
not replace seed 0. The three-learner sensitivity is post hoc, separate from
the primary nominal intervals and the ladder's six-interval family.

| Learner | Mean RER (%) | Table S6E interval (%) |
|---|---:|---|
| PLSR | 8.28 | [-5.26, 18.25] |
| XGBoost | 14.71 | [-2.41, 27.88] |
| MLP | 17.59 | [3.54, 28.75] |

The output records the NumPy version; use the pinned version for exact JSON
bytes. This reproduces the statistical aggregation of stored target effects,
not an independent query-prediction recomputation. Table S1 is a separate
derived table whose historical bootstrap inputs remain unavailable.
