## Supplementary Table S10 — Post hoc >=50-usable-repeat restricted-pool sensitivity

**Table S10. Post hoc >=50-usable-repeat restricted-pool four-way sensitivity.** Using the frozen repeat-level four-way metrics only, a lake--learner unit is included when it has at least 50 usable repetitions. Delta-MAE is `MAE_random - MAE_selector`; positive values favor the selector. Each cell is the across-lake mean of lake-level median paired effects with a nominal, cell-wise 10,000-resample percentile lake-bootstrap interval and no multiplicity adjustment. This post hoc usability-defined estimand does not replace the frozen all-24-lake available-repeat sensitivity; unresolved does not establish equivalence.

| Learner | k | Selector | Included lakes | Excluded lakes | Mean delta-MAE | Median delta-MAE | 95% CI | Positive-lake fraction | Frozen available-repeat class | Threshold-50 class | Classification changed? |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|
| PLSR | 1 | Kennard--Stone | 24 | 0 | -0.110233 | -0.061432 | [-0.166498, -0.059592] | 25% | random_better | random_better | no |
| PLSR | 1 | Unweighted D-optimal | 24 | 0 | -0.413283 | -0.066730 | [-0.762953, -0.144841] | 21% | random_better | random_better | no |
| PLSR | 1 | Weighted D-optimal | 24 | 0 | -0.055926 | -0.010955 | [-0.105629, -0.014834] | 38% | random_better | random_better | no |
| PLSR | 3 | Kennard--Stone | 24 | 0 | -0.027428 | 0.002893 | [-0.066750, 0.004399] | 62% | unresolved | unresolved | no |
| PLSR | 3 | Unweighted D-optimal | 24 | 0 | -0.020881 | 0.003427 | [-0.068728, 0.014350] | 54% | unresolved | unresolved | no |
| PLSR | 3 | Weighted D-optimal | 24 | 0 | -0.030776 | 0.001848 | [-0.068077, -0.003024] | 54% | random_better | random_better | no |
| PLSR | 5 | Kennard--Stone | 24 | 0 | -0.010741 | -0.001556 | [-0.030428, 0.004899] | 50% | unresolved | unresolved | no |
| PLSR | 5 | Unweighted D-optimal | 24 | 0 | -0.014930 | -0.001644 | [-0.035863, 0.002058] | 50% | unresolved | unresolved | no |
| PLSR | 5 | Weighted D-optimal | 24 | 0 | -0.012946 | -0.002483 | [-0.027481, -0.000511] | 46% | random_better | random_better | no |
| PLSR | 10 | Kennard--Stone | 18 | 6 | -0.004000 | 0.004368 | [-0.016523, 0.006756] | 67% | unresolved | unresolved | no |
| PLSR | 10 | Unweighted D-optimal | 18 | 6 | -0.027055 | -0.008457 | [-0.049690, -0.007799] | 33% | random_better | random_better | no |
| PLSR | 10 | Weighted D-optimal | 18 | 6 | -0.019013 | -0.007589 | [-0.033053, -0.006767] | 33% | random_better | random_better | no |
| XGBoost | 1 | Kennard--Stone | 24 | 0 | -0.045058 | -0.007881 | [-0.090241, -0.009799] | 42% | random_better | random_better | no |
| XGBoost | 1 | Unweighted D-optimal | 24 | 0 | -0.076610 | -0.002530 | [-0.147836, -0.018163] | 46% | random_better | random_better | no |
| XGBoost | 1 | Weighted D-optimal | 24 | 0 | -0.055910 | -0.018183 | [-0.097484, -0.019071] | 33% | random_better | random_better | no |
| XGBoost | 3 | Kennard--Stone | 24 | 0 | -0.014326 | 0.001647 | [-0.035555, 0.003010] | 54% | unresolved | unresolved | no |
| XGBoost | 3 | Unweighted D-optimal | 24 | 0 | 0.003635 | 0.008760 | [-0.009926, 0.015173] | 67% | unresolved | unresolved | no |
| XGBoost | 3 | Weighted D-optimal | 24 | 0 | -0.024911 | 0.003917 | [-0.056552, 0.000508] | 54% | unresolved | unresolved | no |
| XGBoost | 5 | Kennard--Stone | 24 | 0 | 0.006142 | 0.012127 | [-0.004005, 0.015908] | 67% | unresolved | unresolved | no |
| XGBoost | 5 | Unweighted D-optimal | 24 | 0 | -0.004229 | 0.005524 | [-0.019618, 0.009787] | 58% | unresolved | unresolved | no |
| XGBoost | 5 | Weighted D-optimal | 24 | 0 | -0.007525 | 0.001391 | [-0.023651, 0.006024] | 62% | unresolved | unresolved | no |
| XGBoost | 10 | Kennard--Stone | 18 | 6 | -0.002494 | 0.003368 | [-0.011211, 0.005450] | 56% | unresolved | unresolved | no |
| XGBoost | 10 | Unweighted D-optimal | 18 | 6 | -0.010709 | -0.000862 | [-0.029398, 0.004655] | 44% | unresolved | unresolved | no |
| XGBoost | 10 | Weighted D-optimal | 18 | 6 | -0.007255 | 0.004631 | [-0.023496, 0.006307] | 61% | unresolved | unresolved | no |
| MLP | 1 | Kennard--Stone | 24 | 0 | -0.067780 | -0.036043 | [-0.127800, -0.022820] | 38% | random_better | random_better | no |
| MLP | 1 | Unweighted D-optimal | 24 | 0 | -0.325166 | -0.048530 | [-0.688888, -0.061483] | 42% | random_better | random_better | no |
| MLP | 1 | Weighted D-optimal | 24 | 0 | -0.039970 | -0.013086 | [-0.074485, -0.009167] | 46% | random_better | random_better | no |
| MLP | 3 | Kennard--Stone | 24 | 0 | -0.026897 | 0.008316 | [-0.060719, 0.000610] | 62% | unresolved | unresolved | no |
| MLP | 3 | Unweighted D-optimal | 24 | 0 | -0.003929 | 0.007570 | [-0.022547, 0.016241] | 54% | unresolved | unresolved | no |
| MLP | 3 | Weighted D-optimal | 24 | 0 | -0.019227 | 0.003821 | [-0.049068, 0.004192] | 58% | unresolved | unresolved | no |
| MLP | 5 | Kennard--Stone | 24 | 0 | -0.008864 | -0.000010 | [-0.029753, 0.005673] | 50% | unresolved | unresolved | no |
| MLP | 5 | Unweighted D-optimal | 24 | 0 | -0.004534 | 0.004438 | [-0.016710, 0.006051] | 54% | unresolved | unresolved | no |
| MLP | 5 | Weighted D-optimal | 24 | 0 | -0.012574 | -0.010370 | [-0.029786, 0.001441] | 46% | unresolved | unresolved | no |
| MLP | 10 | Kennard--Stone | 18 | 6 | -0.000762 | 0.003379 | [-0.012057, 0.007818] | 61% | unresolved | unresolved | no |
| MLP | 10 | Unweighted D-optimal | 18 | 6 | -0.018798 | -0.004456 | [-0.040390, -0.002680] | 33% | random_better | random_better | no |
| MLP | 10 | Weighted D-optimal | 18 | 6 | -0.009579 | 0.000592 | [-0.025201, 0.003719] | 50% | unresolved | unresolved | no |
