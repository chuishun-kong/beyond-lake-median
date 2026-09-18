"""Summarize archived source-only losses without refitting any learner.

The k=0 rows evaluate the full target-unit pool.  The k=5 and k=10 rows come
from the frozen calibration-direction ablation and use the same query sets as
the corresponding capacity-boundary calibration rows.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CAPACITY_RESULTS = ROOT / "experiments" / "capacity_boundary" / "results"
CAPACITY_REPEATS = CAPACITY_RESULTS / "capacity_boundary_repeats_nested.csv"
CAPACITY_SUMMARY = CAPACITY_RESULTS / "capacity_boundary_summary_nested.csv"
ABLATION_REPEATS = (
    CAPACITY_RESULTS / "calibration_direction_ablation_nested_per_rep.csv"
)
OUTPUT = Path(__file__).resolve().parent / "results" / "source_only_context.csv"
PAIRED_OUTPUT = (
    Path(__file__).resolve().parent
    / "results"
    / "source_only_vs_label_only_k5_paired.csv"
)
LEARNERS = ("PLSR", "XGBoost", "MLP")
RER_EPSILON = 1e-8
N_BOOT = 100_000
BOOTSTRAP_SEED = 0


def _paired_k5_summary(
    capacity_repeats: pd.DataFrame, source_same_query: pd.DataFrame
) -> pd.DataFrame:
    """Pair target-unit median source-only and label-only losses at k=5."""
    keys = ["lake", "learner", "k", "rep"]
    calibrated = capacity_repeats[
        capacity_repeats["method"].eq("calibrated")
        & capacity_repeats["k"].eq(5)
    ][keys + ["mae_log", "rer_log"]].copy()
    calibrated["label_only_log_mae"] = (
        calibrated["mae_log"] + calibrated["rer_log"] * RER_EPSILON
    ) / (1.0 - calibrated["rer_log"])

    source = source_same_query[source_same_query["k"].eq(5)][
        keys + ["mae_log"]
    ].rename(columns={"mae_log": "source_only_log_mae"})
    paired = source.merge(
        calibrated[keys + ["label_only_log_mae"]],
        on=keys,
        validate="one_to_one",
    )
    if len(paired) != 7_200:
        raise AssertionError("Expected 7,200 paired k=5 rows")

    per_target = (
        paired.groupby(["lake", "learner"], sort=True)
        .agg(
            target_median_source_only_log_mae=("source_only_log_mae", "median"),
            target_median_label_only_log_mae=("label_only_log_mae", "median"),
            repeat_count=("rep", "count"),
        )
        .reset_index()
    )
    if len(per_target) != 72 or not per_target["repeat_count"].eq(100).all():
        raise AssertionError("Expected 24 target units and 100 repeats per learner")
    per_target["source_minus_label_log_mae"] = (
        per_target["target_median_source_only_log_mae"]
        - per_target["target_median_label_only_log_mae"]
    )

    rows = []
    for learner in LEARNERS:
        cell = per_target[per_target["learner"].eq(learner)]
        differences = cell["source_minus_label_log_mae"].to_numpy(dtype=float)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws = differences[
            rng.integers(0, len(differences), size=(N_BOOT, len(differences)))
        ].mean(axis=1)
        ci_lo, ci_hi = np.percentile(draws, [2.5, 97.5])
        rows.append(
            {
                "learner": learner,
                "k": 5,
                "n_target_units": len(cell),
                "mean_source_only_log_mae": cell[
                    "target_median_source_only_log_mae"
                ].mean(),
                "mean_label_only_log_mae": cell[
                    "target_median_label_only_log_mae"
                ].mean(),
                "mean_source_minus_label_log_mae": differences.mean(),
                "median_source_minus_label_log_mae": np.median(differences),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_source_loss_higher": int((differences > 0).sum()),
                "bootstrap_n": N_BOOT,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "target_unit_statistic": "difference_of_within_target_medians",
            }
        )
    return pd.DataFrame(rows)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    capacity_repeats = pd.read_csv(CAPACITY_REPEATS)
    capacity_summary = pd.read_csv(CAPACITY_SUMMARY)
    ablation = pd.read_csv(ABLATION_REPEATS)

    keys = ["lake", "learner", "k", "rep"]
    calibrated = capacity_repeats[
        capacity_repeats["method"].eq("calibrated")
        & capacity_repeats["k"].isin([5, 10])
    ][keys + ["mae_log"]].rename(columns={"mae_log": "frozen_calibrated_mae"})
    full = ablation[ablation["variant"].eq("full")][keys + ["mae_log"]].rename(
        columns={"mae_log": "ablation_full_mae"}
    )
    check = calibrated.merge(full, on=keys, validate="one_to_one")
    if len(check) != 14_400:
        raise AssertionError("Expected 14,400 aligned k=5/k=10 calibrated rows")
    max_difference = (
        check["frozen_calibrated_mae"] - check["ablation_full_mae"]
    ).abs().max()
    if max_difference > 1e-12:
        raise AssertionError(
            f"Ablation query alignment failed; max calibrated MAE difference={max_difference}"
        )

    source_same_query = ablation[ablation["variant"].eq("source_only")]
    if len(source_same_query) != 14_400:
        raise AssertionError("Expected 14,400 archived same-query source-only rows")
    per_target = (
        source_same_query.groupby(["lake", "learner", "k"], sort=True)["mae_log"]
        .median()
        .rename("target_median_source_only_log_mae")
        .reset_index()
    )

    rows = []
    full_pool = capacity_summary[
        capacity_summary["method"].eq("source_only")
        & capacity_summary["k"].eq(0)
    ]
    for learner in LEARNERS:
        cell = full_pool[full_pool["learner"].eq(learner)]["median_mae_log"]
        if len(cell) != 24:
            raise AssertionError(f"Expected 24 full-pool target units for {learner}")
        rows.append(
            {
                "learner": learner,
                "k": 0,
                "query_scope": "full_target_pool_P_t",
                "n_target_units": len(cell),
                "mean_source_only_log_mae": round(float(cell.mean()), 6),
                "median_source_only_log_mae": round(float(cell.median()), 6),
                "source": "capacity_boundary_summary_nested.csv",
                "max_calibrated_alignment_difference": "not_applicable",
            }
        )
        for budget in (5, 10):
            cell = per_target[
                per_target["learner"].eq(learner)
                & per_target["k"].eq(budget)
            ]["target_median_source_only_log_mae"]
            if len(cell) != 24:
                raise AssertionError(
                    f"Expected 24 same-query target units for {learner}, k={budget}"
                )
            rows.append(
                {
                    "learner": learner,
                    "k": budget,
                    "query_scope": "same_positive_budget_query_Q_S",
                    "n_target_units": len(cell),
                    "mean_source_only_log_mae": round(float(cell.mean()), 6),
                    "median_source_only_log_mae": round(float(cell.median()), 6),
                    "source": "calibration_direction_ablation_nested_per_rep.csv",
                    "max_calibrated_alignment_difference": f"{max_difference:.3e}",
                }
            )

    result = pd.DataFrame(rows)
    paired_result = _paired_k5_summary(capacity_repeats, source_same_query)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    paired_result.to_csv(PAIRED_OUTPUT, index=False)
    return result, paired_result


if __name__ == "__main__":
    output, paired_output = run()
    print(output.to_string(index=False))
    print("\nPaired k=5 source-only versus label-only diagnostic")
    print(paired_output.to_string(index=False))
    print(f"\nWrote {OUTPUT}")
    print(f"Wrote {PAIRED_OUTPUT}")
