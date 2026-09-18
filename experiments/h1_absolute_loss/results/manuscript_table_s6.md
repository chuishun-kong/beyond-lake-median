## Supplementary Table S6 — Post hoc supporting absolute-loss analysis

**Table S6. Post hoc supporting absolute-loss analysis for H1.** Values are dimensionless log10-ratio units. Label-only and calibrated values are across-lake summaries of lake-level medians; paired delta-MAE is the across-lake mean of lake-level medians of repeat-paired differences. Positive delta-MAE favors calibrated prediction. Intervals are 10,000-resample percentile-bootstrap intervals over lakes. This table does not replace the primary RER inference.

| Learner | k | Mean label-only | Median label-only | Mean calibrated | Median calibrated | Mean paired delta-MAE | Median paired delta-MAE | 95% CI on mean paired delta-MAE | Positive-lake fraction | Frozen RER class | Absolute delta-MAE class | Classification differs? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| PLSR | 1 | 0.250258 | 0.254437 | 0.240245 | 0.221692 | 0.011786 | 0.016415 | [-0.018856, 0.038333] | 71% | crosses_zero | crosses_zero | no |
| PLSR | 3 | 0.229327 | 0.233543 | 0.208772 | 0.197299 | 0.021554 | 0.018809 | [-0.000857, 0.042164] | 71% | crosses_zero | crosses_zero | no |
| PLSR | 5 | 0.223181 | 0.231050 | 0.200694 | 0.196407 | 0.026851 | 0.030491 | [0.008842, 0.044315] | 71% | crosses_zero | positive | yes |
| PLSR | 10 | 0.218282 | 0.228546 | 0.186962 | 0.174964 | 0.032972 | 0.030833 | [0.015533, 0.049956] | 79% | positive | positive | no |
| XGBoost | 1 | 0.250258 | 0.254437 | 0.196149 | 0.179130 | 0.052899 | 0.046016 | [0.024039, 0.080831] | 79% | positive | positive | no |
| XGBoost | 3 | 0.229327 | 0.233543 | 0.173167 | 0.163797 | 0.056720 | 0.053329 | [0.030661, 0.081165] | 92% | positive | positive | no |
| XGBoost | 5 | 0.223181 | 0.231050 | 0.178728 | 0.180567 | 0.045092 | 0.046070 | [0.020397, 0.069144] | 79% | positive | positive | no |
| XGBoost | 10 | 0.218282 | 0.228546 | 0.166274 | 0.157755 | 0.052283 | 0.042982 | [0.028923, 0.076206] | 83% | positive | positive | no |
| MLP | 1 | 0.250258 | 0.254437 | 0.202036 | 0.186161 | 0.047751 | 0.049393 | [0.014153, 0.077814] | 75% | crosses_zero | positive | yes |
| MLP | 3 | 0.229327 | 0.233543 | 0.179570 | 0.172618 | 0.050082 | 0.042318 | [0.018725, 0.077787] | 83% | crosses_zero | positive | yes |
| MLP | 5 | 0.223181 | 0.231050 | 0.177689 | 0.176252 | 0.048931 | 0.038343 | [0.026440, 0.071033] | 83% | positive | positive | no |
| MLP | 10 | 0.218282 | 0.228546 | 0.166599 | 0.154830 | 0.052070 | 0.043899 | [0.029684, 0.074863] | 79% | positive | positive | no |
