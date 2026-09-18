"""Phase 3 prerequisite: primary-protocol re-aggregation of the Phase 2 ablation.

WHY THIS EXISTS:
  The original nested summary (calibration_direction_ablation_nested_summary.csv)
  used MEAN-over-repeats as the per-lake aggregator. This is an ALTERNATIVE
  aggregation, NOT the protocol-frozen primary. The primary protocol is:
    Layer 1: per lake = MEDIAN over 100 repeats
    Layer 2: 24 lake-level values; inference = across-lake MEAN bootstrap CI
  Mean-over-repeats was retained as a sensitivity but must NOT be used to
  explain Phase 0's CI classifications. This script re-aggregates from the
  already-persisted per_rep.csv without refitting any model, producing the
  primary-protocol summary that Phase 3 will cite.

ALSO: reports a STANDARDIZED condition number (computed on centered,
unit-variance-scaled feature columns, excluding the intercept column) alongside
the raw condition number already in per_rep.csv. The raw condition number
conflates scale with collinearity; the standardized version isolates
collinearity per NIST's condition-index definition of multicollinearity
diagnosis. Neither is claimed as a causal-attribution proof -- both are
diagnostics of design-matrix column dependence, which makes coefficient-level
attribution unstable.

This script does NOT refit any base model, regenerate any RNG, or change any
frozen number. It is a pure pandas re-aggregation of one CSV.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PER_REP_CSV = RESULTS_DIR / "calibration_direction_ablation_nested_per_rep.csv"
N_BOOT = 10000


def mean_bootstrap_ci(values, n_boot=N_BOOT, seed=0):
    """Across-lake MEAN bootstrap CI -- the Sec 2.8 primary inference."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([np.mean(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def median_bootstrap_ci(values, n_boot=N_BOOT, seed=0):
    """Across-lake MEDIAN bootstrap CI -- post-hoc sensitivity only."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([np.median(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def variants_for_k(k):
    base = ["source_only", "intercept_only", "affine_only", "repr_only",
            "full", "full_minus_affine", "full_minus_repr0"]
    if k == 10:
        base.append("full_minus_repr1")
    return base


def run():
    per_rep = pd.read_csv(PER_REP_CSV)
    print(f"Loaded {len(per_rep)} per-rep rows from {PER_REP_CSV.name}")

    # ---- Layer 1 (primary): per lake = MEDIAN over 100 repeats ----
    per_lake_primary = per_rep.groupby(["lake", "learner", "k", "variant"]).agg(
        delta_spearman_lake=("delta_spearman", "median"),
        delta_r2_log_lake=("delta_anomaly_r2_logspace", "median"),
        mae_log_lake=("mae_log", "median"),
        rer_log_lake=("rer_log", "median"),
        # also record the alternative (mean-over-repeats) for sensitivity doc
        delta_spearman_lake_altmean=("delta_spearman", "mean"),
        delta_r2_log_lake_altmean=("delta_anomaly_r2_logspace", "mean"),
    ).reset_index()
    per_lake_primary.to_csv(
        RESULTS_DIR / "calibration_direction_ablation_per_lake_primary.csv", index=False)

    # ---- Layer 2 (primary): across-lake mean + bootstrap CI on mean ----
    primary_rows = []
    alt_rows = []  # alternative aggregation (mean-over-repeats) for sensitivity
    for learner in ["PLSR", "XGBoost", "MLP"]:
        for k in [5, 10]:
            for variant in variants_for_k(k):
                sub = per_lake_primary[
                    (per_lake_primary.learner == learner)
                    & (per_lake_primary.k == k)
                    & (per_lake_primary.variant == variant)]
                n = len(sub)
                # PRIMARY (median-over-repeats -> across-lake mean + CI)
                rho_vals = sub.delta_spearman_lake.to_numpy()
                r2_vals = sub.delta_r2_log_lake.to_numpy()
                rho_ci = mean_bootstrap_ci(rho_vals)
                r2_ci = mean_bootstrap_ci(r2_vals)
                rho_ci_med = median_bootstrap_ci(rho_vals)
                # ALTERNATIVE (mean-over-repeats -> across-lake mean + CI)
                rho_alt = sub.delta_spearman_lake_altmean.to_numpy()
                rho_alt_ci = mean_bootstrap_ci(rho_alt)

                primary_rows.append({
                    "learner": learner, "k": k, "variant": variant, "n_lakes": n,
                    # PRIMARY
                    "mean_delta_rho": round(sub.delta_spearman_lake.mean(), 4),
                    "mean_delta_rho_ci": (round(rho_ci[0], 4), round(rho_ci[1], 4)),
                    "median_delta_rho": round(sub.delta_spearman_lake.median(), 4),
                    "median_delta_rho_ci": (round(rho_ci_med[0], 4), round(rho_ci_med[1], 4)),
                    "mean_delta_r2_log": round(sub.delta_r2_log_lake.mean(), 4),
                    "mean_delta_r2_log_ci": (round(r2_ci[0], 4), round(r2_ci[1], 4)),
                    "mae_log_lake_median": round(sub.mae_log_lake.median(), 4),
                    "rer_log_lake_median": round(sub.rer_log_lake.median(), 4),
                })
                alt_rows.append({
                    "learner": learner, "k": k, "variant": variant, "n_lakes": n,
                    "aggregation": "mean_over_repeats",
                    "across_lake_mean_delta_rho": round(sub.delta_spearman_lake_altmean.mean(), 4),
                    "across_lake_mean_delta_rho_ci": (round(rho_alt_ci[0], 4), round(rho_alt_ci[1], 4)),
                    "note": "ALTERNATIVE aggregation; do NOT use for Phase 0 CI classification",
                })

    primary = pd.DataFrame(primary_rows)
    primary.to_csv(
        RESULTS_DIR / "calibration_direction_ablation_summary_primary.csv", index=False)
    alt = pd.DataFrame(alt_rows)
    alt.to_csv(
        RESULTS_DIR / "calibration_direction_ablation_summary_alt_sensitivity.csv", index=False)

    print("\n=== PRIMARY-PROTOCOL ablation summary (median-over-repeats -> across-lake mean + CI) ===")
    print("This is the version Phase 3 / Phase 0 CI classification must cite.")
    with pd.option_context("display.width", 320, "display.max_columns", 30):
        print(primary.to_string(index=False))

    print("\n=== ALTERNATIVE (mean-over-repeats) sensitivity -- DO NOT use for Phase 0 ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(alt.to_string(index=False))


if __name__ == "__main__":
    run()
