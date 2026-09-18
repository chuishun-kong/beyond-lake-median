# H1 absolute-loss supporting analysis

This analysis derives absolute log-MAE and repeat-paired loss differences from
the stored H1 metrics, without refitting models. It reconstructs label-only
loss from the fractional RER and calibrated loss before target-level aggregation.

```powershell
python experiments/h1_absolute_loss/run_h1_absolute_loss.py
```

CSV summaries and `analysis_summary.md` are written under `results/`.
The primary relative-error-reduction endpoint and this post hoc absolute-loss
endpoint can have different interval classifications because they weight
target-level error scales differently.

The input metric table contains source-only rows at k=0 on full-target query
sets, not on positive-budget query subsets. Same-query source-only context is
provided separately by `source_only_context.csv` and
`source_only_vs_label_only_k5_paired.csv`. The script
`build_source_only_context.py` rebuilds that context from the included
capacity-boundary and ablation results without fitting models.
