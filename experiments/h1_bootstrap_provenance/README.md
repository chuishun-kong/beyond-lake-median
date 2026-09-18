# H1 bootstrap reconstruction and sensitivity

This directory reconstructs the H1 lake-level bootstrap
intervals and tests whether their qualitative interpretation is sensitive to
the random seed or to increasing the number of bootstrap resamples.

The historical execution record shows that each learner-budget-protocol cell
created a fresh `numpy.random.default_rng(0)` and used 10,000 percentile
bootstrap resamples of the 24 lake-level RER values. The canonical scheme is
therefore `fresh_seed0_per_cell`. The other three schemes are reconstruction
diagnostics only.

The stability analysis was predeclared before its results were inspected:
seeds 0--19 at B=10,000 for all 12 true-nested learner-budget cells, plus
B=100,000 with seed 0 for the three k=5 cells. These diagnostics do not replace
the frozen primary results.

The same script also provides a post hoc, bounded robustness sensitivity for
the primary k=5 cells. It reuses the 24 frozen target-unit median RER values and
adds a BCa interval for the mean, a 10% per-tail winsorized-mean percentile
interval (winsorization repeated within each bootstrap sample), and 24
delete-one-target-unit percentile intervals. No model is retrained, and these
outputs do not replace the prespecified percentile-bootstrap analysis.

Run from the repository root:

```powershell
python experiments/h1_bootstrap_provenance/run_h1_bootstrap_provenance.py
```

The script reads but never writes the three frozen capacity-boundary CSV files.
It verifies their hashes again before exit and writes only under this
experiment's `results/` directory.
