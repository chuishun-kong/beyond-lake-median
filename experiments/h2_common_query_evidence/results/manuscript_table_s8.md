## Supplementary Table S8 — Frozen all-24-lake available-repeat four-way common-query sensitivity

**Table S8. Available-repeat four-way common-query H2 sensitivity.** Every selector is compared with the same random support on repeat-specific $Q_{all}=\mathrm{pool}\setminus(S_R\cup S_{KS}\cup S_{unweighted}\cup S_{weighted})$. Delta-MAE is `MAE_random - MAE_selector`; positive values favor the selector. The frozen analysis retained every lake with available usable repetitions, so all 24 lakes entered each cell while within-lake usable-repeat coverage varied. Intervals are nominal cell-wise 10,000-resample lake-bootstrap intervals and are not multiplicity-adjusted. This is a secondary sensitivity; it is not a >=50-repeat lake-entry analysis, and unresolved does not establish equivalence. Table S9 gives coverage details.

| Learner | k | Selector | Mean delta-MAE | Median delta-MAE | 95% CI | Classification | Lakes entered | Usable repeats | Usable fraction | Min usable repeats/lake | Median | Max | Lakes >=50 | Lakes <50 |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PLSR | 1 | Kennard--Stone | -0.110233 | -0.061432 | [-0.166498, -0.059592] | random_better | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| PLSR | 1 | Unweighted D-optimal | -0.413283 | -0.066730 | [-0.762953, -0.144841] | random_better | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| PLSR | 1 | Weighted D-optimal | -0.055926 | -0.010955 | [-0.105629, -0.014834] | random_better | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| PLSR | 3 | Kennard--Stone | -0.027428 | 0.002893 | [-0.066750, 0.004399] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| PLSR | 3 | Unweighted D-optimal | -0.020881 | 0.003427 | [-0.068728, 0.014350] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| PLSR | 3 | Weighted D-optimal | -0.030776 | 0.001848 | [-0.068077, -0.003024] | random_better | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| PLSR | 5 | Kennard--Stone | -0.010741 | -0.001556 | [-0.030428, 0.004899] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| PLSR | 5 | Unweighted D-optimal | -0.014930 | -0.001644 | [-0.035863, 0.002058] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| PLSR | 5 | Weighted D-optimal | -0.012946 | -0.002483 | [-0.027481, -0.000511] | random_better | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| PLSR | 10 | Kennard--Stone | -0.002248 | 0.004368 | [-0.012104, 0.006014] | unresolved | 24 | 1803 | 75.13% | 3 | 100 | 100 | 18 | 6 |
| PLSR | 10 | Unweighted D-optimal | -0.023321 | -0.008457 | [-0.040739, -0.008329] | random_better | 24 | 1803 | 75.13% | 3 | 100 | 100 | 18 | 6 |
| PLSR | 10 | Weighted D-optimal | -0.014640 | -0.005009 | [-0.026261, -0.004819] | random_better | 24 | 1803 | 75.13% | 3 | 100 | 100 | 18 | 6 |
| XGBoost | 1 | Kennard--Stone | -0.045058 | -0.007881 | [-0.090241, -0.009799] | random_better | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| XGBoost | 1 | Unweighted D-optimal | -0.076610 | -0.002530 | [-0.147836, -0.018163] | random_better | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| XGBoost | 1 | Weighted D-optimal | -0.055910 | -0.018183 | [-0.097484, -0.019071] | random_better | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| XGBoost | 3 | Kennard--Stone | -0.014326 | 0.001647 | [-0.035555, 0.003010] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| XGBoost | 3 | Unweighted D-optimal | 0.003635 | 0.008760 | [-0.009926, 0.015173] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| XGBoost | 3 | Weighted D-optimal | -0.024911 | 0.003917 | [-0.056552, 0.000508] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| XGBoost | 5 | Kennard--Stone | 0.006142 | 0.012127 | [-0.004005, 0.015908] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| XGBoost | 5 | Unweighted D-optimal | -0.004229 | 0.005524 | [-0.019618, 0.009787] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| XGBoost | 5 | Weighted D-optimal | -0.007525 | 0.001391 | [-0.023651, 0.006024] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| XGBoost | 10 | Kennard--Stone | -0.005221 | 0.001072 | [-0.013389, 0.002157] | unresolved | 24 | 1819 | 75.79% | 1 | 100 | 100 | 18 | 6 |
| XGBoost | 10 | Unweighted D-optimal | -0.006674 | 0.000714 | [-0.021022, 0.005067] | unresolved | 24 | 1819 | 75.79% | 1 | 100 | 100 | 18 | 6 |
| XGBoost | 10 | Weighted D-optimal | -0.003669 | 0.006149 | [-0.016315, 0.007053] | unresolved | 24 | 1819 | 75.79% | 1 | 100 | 100 | 18 | 6 |
| MLP | 1 | Kennard--Stone | -0.067780 | -0.036043 | [-0.127800, -0.022820] | random_better | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| MLP | 1 | Unweighted D-optimal | -0.325166 | -0.048530 | [-0.688888, -0.061483] | random_better | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| MLP | 1 | Weighted D-optimal | -0.039970 | -0.013086 | [-0.074485, -0.009167] | random_better | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| MLP | 3 | Kennard--Stone | -0.026897 | 0.008316 | [-0.060719, 0.000610] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| MLP | 3 | Unweighted D-optimal | -0.003929 | 0.007570 | [-0.022547, 0.016241] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| MLP | 3 | Weighted D-optimal | -0.019227 | 0.003821 | [-0.049068, 0.004192] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| MLP | 5 | Kennard--Stone | -0.008864 | -0.000010 | [-0.029753, 0.005673] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| MLP | 5 | Unweighted D-optimal | -0.004534 | 0.004438 | [-0.016710, 0.006051] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| MLP | 5 | Weighted D-optimal | -0.012574 | -0.010370 | [-0.029786, 0.001441] | unresolved | 24 | 2400 | 100.00% | 100 | 100 | 100 | 24 | 0 |
| MLP | 10 | Kennard--Stone | 0.000438 | 0.003445 | [-0.009139, 0.008252] | unresolved | 24 | 1812 | 75.50% | 1 | 100 | 100 | 18 | 6 |
| MLP | 10 | Unweighted D-optimal | -0.014682 | -0.006078 | [-0.031096, -0.001999] | random_better | 24 | 1812 | 75.50% | 1 | 100 | 100 | 18 | 6 |
| MLP | 10 | Weighted D-optimal | -0.006524 | 0.000592 | [-0.018675, 0.003833] | unresolved | 24 | 1812 | 75.50% | 1 | 100 | 100 | 18 | 6 |
