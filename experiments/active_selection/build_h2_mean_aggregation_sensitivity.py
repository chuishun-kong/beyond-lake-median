"""Post hoc mean-over-repetitions sensitivity for the true-nested H2 own-pool estimand.

The frozen primary analysis summarizes repeated-random loss within each target
unit by the median.  This script keeps the same supports, own-query losses,
target units, across-unit mean, and percentile bootstrap, but replaces the
within-target median with the mean.  It does not refit a model or replace the
primary 0/3/9 classification.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "true_nested_h2"
INPUT = RESULTS / "h2_nested_per_repeat.csv"
OUTPUT = RESULTS / "h2_nested_own_pool_mean_sensitivity.csv"
DIAGNOSTIC_OUTPUT = RESULTS / "h2_nested_random_loss_shape_diagnostic.csv"
LEARNERS = ("PLSR", "XGBoost", "MLP")
BUDGETS = (1, 3, 5, 10)
N_BOOT = 10_000


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def classify(lo: float, hi: float) -> str:
    if lo > 0:
        return "weighted-better"
    if hi < 0:
        return "random-better"
    return "unresolved"


def build_random_loss_shape_diagnostic(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize why mean and median random-arm losses differ."""
    random = frame[frame["method"].eq("repeated_random")].copy()
    rows = []
    for (lake, learner, budget), cell in random.groupby(
        ["lake", "learner", "k"], sort=True
    ):
        values = cell["mae_log"].to_numpy(dtype=float)
        if len(values) != 100:
            raise AssertionError(
                f"Expected 100 random-arm losses for {lake}, {learner}, k={budget}"
            )
        rows.append(
            {
                "lake": lake,
                "learner": learner,
                "k": int(budget),
                "n_repetitions": len(values),
                "random_mae_sample_skewness": float(skew(values, bias=True)),
                "random_mae_mean_minus_median": float(
                    values.mean() - np.median(values)
                ),
            }
        )

    per_target = pd.DataFrame(rows)
    summaries = []
    for learner in LEARNERS:
        for budget in BUDGETS:
            cell = per_target[
                per_target["learner"].eq(learner)
                & per_target["k"].eq(budget)
            ]
            if len(cell) != 24:
                raise AssertionError(f"Expected 24 target units for {learner}, k={budget}")
            skewness = cell["random_mae_sample_skewness"]
            gap = cell["random_mae_mean_minus_median"]
            summaries.append(
                {
                    "learner": learner,
                    "k": budget,
                    "n_target_units": len(cell),
                    "median_random_mae_sample_skewness": round(
                        float(skewness.median()), 6
                    ),
                    "q1_random_mae_sample_skewness": round(
                        float(skewness.quantile(0.25)), 6
                    ),
                    "q3_random_mae_sample_skewness": round(
                        float(skewness.quantile(0.75)), 6
                    ),
                    "n_positive_sample_skewness": int((skewness > 0).sum()),
                    "median_random_mae_mean_minus_median": round(
                        float(gap.median()), 6
                    ),
                    "q1_random_mae_mean_minus_median": round(
                        float(gap.quantile(0.25)), 6
                    ),
                    "q3_random_mae_mean_minus_median": round(
                        float(gap.quantile(0.75)), 6
                    ),
                    "n_positive_mean_minus_median": int((gap > 0).sum()),
                    "across_target_mean_delta_shift": round(float(gap.mean()), 6),
                    "skewness_definition": "scipy.stats.skew_bias_true",
                }
            )
    result = pd.DataFrame(summaries)
    result.to_csv(DIAGNOSTIC_OUTPUT, index=False)
    return result


def run() -> pd.DataFrame:
    frame = pd.read_csv(INPUT)
    frame = frame[
        frame["estimand"].eq("own_pool")
        & frame["method"].isin(
            ["repeated_random", "proposed_weighted_d_optimal"]
        )
    ].copy()

    grouped = frame.groupby(["lake", "learner", "k", "method"])
    counts = grouped.size()
    if not counts.eq(100).all():
        raise AssertionError("Every own-pool target/learner/budget/method cell must have 100 rows")

    deterministic = frame[frame["method"].eq("proposed_weighted_d_optimal")]
    if deterministic.groupby(["lake", "learner", "k"])["mae_log"].nunique().max() != 1:
        raise AssertionError("Weighted D-optimal loss must be deterministic within each cell")

    build_random_loss_shape_diagnostic(frame)

    means = grouped["mae_log"].mean().unstack("method").reset_index()
    means["delta_mae"] = (
        means["repeated_random"] - means["proposed_weighted_d_optimal"]
    )

    rows = []
    for learner in LEARNERS:
        for budget in BUDGETS:
            cell = means[
                means["learner"].eq(learner) & means["k"].eq(budget)
            ]
            if len(cell) != 24:
                raise AssertionError(f"Expected 24 target units for {learner}, k={budget}")
            values = cell["delta_mae"].to_numpy(dtype=float)
            lo, hi = bootstrap_mean_ci(values)
            rows.append(
                {
                    "learner": learner,
                    "k": budget,
                    "estimand": "own_pool_post_hoc_mean_sensitivity",
                    "within_target_aggregation": "mean_over_100_repetitions",
                    "delta_definition": "MAE_random - MAE_weighted",
                    "positive_means": "weighted_better_than_random",
                    "mean_delta_mae": round(float(values.mean()), 6),
                    "median_delta_mae": round(float(np.median(values)), 6),
                    "ci_lo": round(float(lo), 6),
                    "ci_hi": round(float(hi), 6),
                    "classification": classify(lo, hi),
                    "n_target_units": len(values),
                    "n_bootstrap": N_BOOT,
                    "multiplicity_adjusted": False,
                    "post_hoc": True,
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)
    return result


if __name__ == "__main__":
    output = run()
    print(output[["learner", "k", "mean_delta_mae", "ci_lo", "ci_hi", "classification"]].to_string(index=False))
    print(f"\nWrote {OUTPUT}")
    print(f"Wrote {DIAGNOSTIC_OUTPUT}")
