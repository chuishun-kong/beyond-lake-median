## Supplementary Table S7 — H2 pairwise common-query sensitivity

**Table S7. Pairwise common-query H2 sensitivity.** Each selector is compared with the same random support on repeat-specific $Q_{sR}=\mathrm{pool}\setminus(S_s\cup S_R)$. Delta-MAE is `MAE_random - MAE_selector`; positive values favor the selector. Each cell is the across-lake mean of lake-level median paired effects; intervals are nominal, cell-wise 10,000-resample percentile bootstrap intervals over lakes and are not multiplicity-adjusted. This is a secondary sensitivity; unresolved does not establish equivalence. The primary own-pool classification applies only to weighted D-optimal.

| Learner | k | Selector | Mean delta-MAE | Median delta-MAE | 95% CI | Classification | Lakes | Repeat rows | Repeats/lake | Primary own-pool class |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---|
| PLSR | 1 | Kennard--Stone | -0.109775 | -0.062404 | [-0.165670, -0.058791] | random_better | 24 | 2400 | 100--100 | not_primary_comparison |
| PLSR | 1 | Unweighted D-optimal | -0.413209 | -0.069445 | [-0.763337, -0.144903] | random_better | 24 | 2400 | 100--100 | not_primary_comparison |
| PLSR | 1 | Weighted D-optimal | -0.054316 | -0.015491 | [-0.102407, -0.013970] | random_better | 24 | 2400 | 100--100 | random_better |
| PLSR | 3 | Kennard--Stone | -0.025269 | 0.005317 | [-0.063535, 0.005738] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| PLSR | 3 | Unweighted D-optimal | -0.019248 | 0.004541 | [-0.067029, 0.015794] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| PLSR | 3 | Weighted D-optimal | -0.029991 | 0.003001 | [-0.065783, -0.003440] | random_better | 24 | 2400 | 100--100 | unresolved |
| PLSR | 5 | Kennard--Stone | -0.004072 | 0.008232 | [-0.023282, 0.010325] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| PLSR | 5 | Unweighted D-optimal | -0.010068 | 0.006115 | [-0.030263, 0.006453] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| PLSR | 5 | Weighted D-optimal | -0.007913 | 0.004088 | [-0.024524, 0.007012] | unresolved | 24 | 2400 | 100--100 | unresolved |
| PLSR | 10 | Kennard--Stone | 0.001747 | 0.007511 | [-0.007388, 0.009618] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| PLSR | 10 | Unweighted D-optimal | -0.018341 | -0.003362 | [-0.036810, -0.002747] | random_better | 24 | 2400 | 100--100 | not_primary_comparison |
| PLSR | 10 | Weighted D-optimal | -0.008658 | 0.001002 | [-0.023078, 0.004111] | unresolved | 24 | 2400 | 100--100 | unresolved |
| XGBoost | 1 | Kennard--Stone | -0.044330 | -0.005216 | [-0.089284, -0.009166] | random_better | 24 | 2400 | 100--100 | not_primary_comparison |
| XGBoost | 1 | Unweighted D-optimal | -0.078325 | -0.002493 | [-0.151425, -0.019353] | random_better | 24 | 2400 | 100--100 | not_primary_comparison |
| XGBoost | 1 | Weighted D-optimal | -0.053620 | -0.018943 | [-0.094461, -0.017958] | random_better | 24 | 2400 | 100--100 | random_better |
| XGBoost | 3 | Kennard--Stone | -0.014155 | 0.002631 | [-0.035548, 0.003301] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| XGBoost | 3 | Unweighted D-optimal | 0.005317 | 0.009445 | [-0.008212, 0.016470] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| XGBoost | 3 | Weighted D-optimal | -0.023779 | 0.002919 | [-0.055543, 0.001374] | unresolved | 24 | 2400 | 100--100 | unresolved |
| XGBoost | 5 | Kennard--Stone | 0.008666 | 0.014574 | [-0.002104, 0.018825] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| XGBoost | 5 | Unweighted D-optimal | -0.003465 | 0.006568 | [-0.019196, 0.010724] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| XGBoost | 5 | Weighted D-optimal | -0.008254 | 0.003479 | [-0.025552, 0.006742] | unresolved | 24 | 2400 | 100--100 | unresolved |
| XGBoost | 10 | Kennard--Stone | -0.003048 | 0.002187 | [-0.014030, 0.006962] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| XGBoost | 10 | Unweighted D-optimal | -0.005486 | 0.002158 | [-0.019310, 0.005718] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| XGBoost | 10 | Weighted D-optimal | -0.007147 | 0.004308 | [-0.020775, 0.005123] | unresolved | 24 | 2400 | 100--100 | unresolved |
| MLP | 1 | Kennard--Stone | -0.067601 | -0.034577 | [-0.124747, -0.024266] | random_better | 24 | 2400 | 100--100 | not_primary_comparison |
| MLP | 1 | Unweighted D-optimal | -0.325478 | -0.047035 | [-0.689248, -0.061795] | random_better | 24 | 2400 | 100--100 | not_primary_comparison |
| MLP | 1 | Weighted D-optimal | -0.037518 | -0.013127 | [-0.071819, -0.007437] | random_better | 24 | 2400 | 100--100 | random_better |
| MLP | 3 | Kennard--Stone | -0.027805 | 0.009450 | [-0.061512, -0.000463] | random_better | 24 | 2400 | 100--100 | not_primary_comparison |
| MLP | 3 | Unweighted D-optimal | -0.003906 | 0.005922 | [-0.023683, 0.017700] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| MLP | 3 | Weighted D-optimal | -0.017863 | 0.006048 | [-0.048351, 0.006258] | unresolved | 24 | 2400 | 100--100 | unresolved |
| MLP | 5 | Kennard--Stone | -0.003806 | 0.008145 | [-0.024680, 0.010750] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| MLP | 5 | Unweighted D-optimal | -0.001459 | 0.009039 | [-0.014424, 0.009928] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| MLP | 5 | Weighted D-optimal | -0.006222 | 0.004576 | [-0.023519, 0.008488] | unresolved | 24 | 2400 | 100--100 | unresolved |
| MLP | 10 | Kennard--Stone | 0.000668 | 0.004554 | [-0.008760, 0.008662] | unresolved | 24 | 2400 | 100--100 | not_primary_comparison |
| MLP | 10 | Unweighted D-optimal | -0.012805 | 0.000615 | [-0.029069, -0.000340] | random_better | 24 | 2400 | 100--100 | not_primary_comparison |
| MLP | 10 | Weighted D-optimal | -0.004593 | 0.006535 | [-0.018048, 0.007399] | unresolved | 24 | 2400 | 100--100 | unresolved |
