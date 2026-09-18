r"""Phase 4 post-processing: build the 5 persistent summary CSVs from
h2_nested_per_repeat.csv, with UNIFIED ΔMAE sign convention per the manuscript.

NO model rerun, NO bootstrap rerun beyond the summary aggregation. This script
reads the already-persisted per_repeat.csv and produces:
  1. h2_nested_per_lake.csv          -- per-lake median-over-repeats for all layers
  2. h2_nested_own_pool_summary.csv  -- 12-cell primary estimand + B/M decomposition
  3. h2_nested_pairwise_common_query.csv  -- 3 contrasts x 12 cells sensitivity
  4. h2_nested_fourway_common_query.csv   -- restricted-pool sensitivity
  5. h2_nested_query_attrition.csv  -- per-k attrition report

SIGN CONVENTION (unified to match manuscript / global-tuned H2):
  ΔMAE = MAE_random - MAE_weighted
  positive => weighted better; negative => random better; CI crosses 0 => unresolved.

The per_repeat.csv stores delta_mae as (mae_method - mae_random) for pairwise/
fourway rows (the convention used during the run). This script flips it to the
manuscript convention in the summaries and adds an explicit `delta_definition`
+ `positive_means` column to every summary to remove all ambiguity.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "true_nested_h2"
PER_REP = RESULTS_DIR / "h2_nested_per_repeat.csv"
N_BOOT = 10000

# Manuscript sign convention (frozen).
DELTA_DEFINITION = "MAE_random - MAE_method"
POSITIVE_MEANS = "method_better_than_random"


def mean_boot_ci(values, n_boot=N_BOOT, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([np.mean(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def classify(ci_lo, ci_hi):
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return "unavailable"
    if ci_hi < -1e-12:
        return "random_better"
    if ci_lo > 1e-12:
        return "method_better"
    return "unresolved"


def run():
    df = pd.read_csv(PER_REP)
    print(f"Loaded {len(df)} rows from {PER_REP.name}")
    print(f"  estimands: {df.estimand.value_counts().to_dict()}")

    # =====================================================================
    # 1. PER-LAKE (median over 100 repeats, all layers)
    # =====================================================================
    per_lake_rows = []

    # own_pool: per lake, per learner, per k, per method -> median mae_log, rer_log, label_mae_log
    op = df[df.estimand == "own_pool"]
    for (lake, L, k, method), g in op.groupby(["lake", "learner", "k", "method"]):
        per_lake_rows.append({
            "lake": lake, "learner": L, "k": int(k),
            "estimand": "own_pool", "method": method,
            "mae_log_median": g.mae_log.median(),
            "rer_log_median": g.rer_log.median(),
            "label_mae_log_median": g.label_mae_log.median(),
        })

    # pairwise: per lake -> median delta_mae (FLIPPED to manuscript sign)
    pw = df[df.estimand == "pairwise_common_query"].dropna(subset=["delta_mae"])
    for (lake, L, k, method), g in pw.groupby(["lake", "learner", "k", "method"]):
        per_lake_rows.append({
            "lake": lake, "learner": L, "k": int(k),
            "estimand": "pairwise_common_query", "method": method,
            "delta_mae_median_manuscript": -(g.delta_mae.median()),  # flip to MAE_random - MAE_method
            "mae_method_median": g.mae_log_method.median(),
            "mae_random_median": g.mae_log_random.median(),
            "query_size_median": g.query_size.median(),
        })

    # fourway: per lake -> median delta_mae (FLIPPED) on usable rows only
    fw = df[df.estimand == "fourway_common_query"]
    fw_usable = fw[fw.usable == True].dropna(subset=["delta_mae"])
    for (lake, L, k, method), g in fw_usable.groupby(["lake", "learner", "k", "method"]):
        per_lake_rows.append({
            "lake": lake, "learner": L, "k": int(k),
            "estimand": "fourway_common_query", "method": method,
            "delta_mae_median_manuscript": -(g.delta_mae.median()),
            "mae_method_median": g.mae_log_method.median(),
            "mae_random_median": g.mae_log_random.median(),
            "query_size_median": g.query_size.median(),
            "n_usable_repeats": len(g),
        })

    per_lake = pd.DataFrame(per_lake_rows)
    per_lake.to_csv(RESULTS_DIR / "h2_nested_per_lake.csv", index=False)
    print(f"  per_lake: {len(per_lake)} rows")

    # =====================================================================
    # 2. OWN-POOL SUMMARY (PRIMARY, 12 cells) + B/M decomposition
    # =====================================================================
    summary_rows = []
    for L in ["PLSR", "XGBoost", "MLP"]:
        for k in [1, 3, 5, 10]:
            rnd = per_lake[(per_lake.estimand == "own_pool") &
                           (per_lake.learner == L) & (per_lake.k == k) &
                           (per_lake.method == "repeated_random")]
            wtd = per_lake[(per_lake.estimand == "own_pool") &
                           (per_lake.learner == L) & (per_lake.k == k) &
                           (per_lake.method == "proposed_weighted_d_optimal")]
            # ΔMAE per lake = MAE_random - MAE_weighted (manuscript sign)
            merged = rnd[["lake", "mae_log_median"]].rename(
                columns={"mae_log_median": "M_random"}).merge(
                wtd[["lake", "mae_log_median", "label_mae_log_median", "rer_log_median"]].rename(
                    columns={"mae_log_median": "M_weighted",
                             "label_mae_log_median": "B_weighted",
                             "rer_log_median": "rer_weighted"}),
                on="lake")
            # B_random from random method's label_mae_log
            rnd_lbl = rnd[["lake", "label_mae_log_median", "rer_log_median"]].rename(
                columns={"label_mae_log_median": "B_random", "rer_log_median": "rer_random"})
            merged = merged.merge(rnd_lbl, on="lake")
            merged["delta_mae"] = merged.M_random - merged.M_weighted  # manuscript sign
            delta_vals = merged.delta_mae.values
            ci = mean_boot_ci(delta_vals)
            n_lakes = len(merged)
            n_method_better = int((merged.delta_mae > 1e-12).sum())
            n_random_better = int((merged.delta_mae < -1e-12).sum())
            summary_rows.append({
                "learner": L, "k": k,
                "estimand": "own_pool", "contrast": "weighted_vs_random",
                "query_scope": "own_remaining_pool",
                "delta_definition": DELTA_DEFINITION,
                "positive_means": POSITIVE_MEANS,
                "mean_delta_mae": round(merged.delta_mae.mean(), 6),
                "median_delta_mae": round(merged.delta_mae.median(), 6),
                "ci_lo": round(ci[0], 6), "ci_hi": round(ci[1], 6),
                "classification": classify(ci[0], ci[1]),
                "n_lakes": n_lakes,
                "n_method_better": n_method_better,
                "n_random_better": n_random_better,
                "multiplicity_adjusted": False,
                # B/M decomposition (medians across lakes)
                "B_random_median": round(merged.B_random.median(), 6),
                "B_weighted_median": round(merged.B_weighted.median(), 6),
                "M_random_median": round(merged.M_random.median(), 6),
                "M_weighted_median": round(merged.M_weighted.median(), 6),
                "rer_random_median": round(merged.rer_random.median(), 6),
                "rer_weighted_median": round(merged.rer_weighted.median(), 6),
            })
    own_pool_summary = pd.DataFrame(summary_rows)
    own_pool_summary.to_csv(RESULTS_DIR / "h2_nested_own_pool_summary.csv", index=False)
    print(f"\n=== OWN-POOL PRIMARY (manuscript sign: ΔMAE = MAE_random - MAE_weighted) ===")
    with pd.option_context("display.width", 240, "display.max_columns", 20):
        print(own_pool_summary[["learner", "k", "mean_delta_mae", "ci_lo", "ci_hi",
                                 "classification", "B_random_median", "B_weighted_median",
                                 "M_random_median", "M_weighted_median"]].to_string(index=False))

    # =====================================================================
    # 3. PAIRWISE common-query summary (3 contrasts x 12 cells)
    # =====================================================================
    pw_rows = []
    pw_pl = per_lake[per_lake.estimand == "pairwise_common_query"]
    for L in ["PLSR", "XGBoost", "MLP"]:
        for k in [1, 3, 5, 10]:
            for method in ["farthest_point_kennard_stone", "unweighted_d_optimal",
                           "proposed_weighted_d_optimal"]:
                sub = pw_pl[(pw_pl.learner == L) & (pw_pl.k == k) & (pw_pl.method == method)]
                if len(sub) == 0:
                    continue
                vals = sub.delta_mae_median_manuscript.values
                ci = mean_boot_ci(vals)
                pw_rows.append({
                    "learner": L, "k": k, "method": method,
                    "estimand": "pairwise_common_query",
                    "contrast": f"{method}_vs_random",
                    "query_scope": "Q_sR = pool \\ (S_method ∪ S_random)",
                    "delta_definition": DELTA_DEFINITION,
                    "positive_means": POSITIVE_MEANS,
                    "mean_delta_mae": round(np.mean(vals), 6),
                    "median_delta_mae": round(np.median(vals), 6),
                    "ci_lo": round(ci[0], 6), "ci_hi": round(ci[1], 6),
                    "classification": classify(ci[0], ci[1]),
                    "n_lakes": len(sub),
                    "multiplicity_adjusted": False,
                })
    pw_summary = pd.DataFrame(pw_rows)
    pw_summary.to_csv(RESULTS_DIR / "h2_nested_pairwise_common_query.csv", index=False)

    # =====================================================================
    # 4. FOUR-WAY common-query summary (restricted-pool sensitivity)
    # =====================================================================
    fw_rows = []
    fw_pl = per_lake[per_lake.estimand == "fourway_common_query"]
    for L in ["PLSR", "XGBoost", "MLP"]:
        for k in [1, 3, 5, 10]:
            for method in ["farthest_point_kennard_stone", "unweighted_d_optimal",
                           "proposed_weighted_d_optimal"]:
                sub = fw_pl[(fw_pl.learner == L) & (fw_pl.k == k) & (fw_pl.method == method)]
                if len(sub) == 0:
                    fw_rows.append({
                        "learner": L, "k": k, "method": method,
                        "estimand": "fourway_common_query",
                        "classification": "no_usable_units",
                        "n_lakes": 0, "multiplicity_adjusted": False,
                    })
                    continue
                vals = sub.delta_mae_median_manuscript.values
                ci = mean_boot_ci(vals)
                fw_rows.append({
                    "learner": L, "k": k, "method": method,
                    "estimand": "fourway_common_query",
                    "contrast": f"{method}_vs_random",
                    "query_scope": "Q_all = pool \\ (S_random ∪ S_KS ∪ S_unwD ∪ S_wD)",
                    "delta_definition": DELTA_DEFINITION,
                    "positive_means": POSITIVE_MEANS,
                    "mean_delta_mae": round(np.mean(vals), 6),
                    "median_delta_mae": round(np.median(vals), 6),
                    "ci_lo": round(ci[0], 6), "ci_hi": round(ci[1], 6),
                    "classification": classify(ci[0], ci[1]),
                    "n_lakes": len(sub),
                    "mean_n_usable_repeats_per_lake": round(sub.n_usable_repeats.mean(), 1),
                    "multiplicity_adjusted": False,
                })
    fw_summary = pd.DataFrame(fw_rows)
    fw_summary.to_csv(RESULTS_DIR / "h2_nested_fourway_common_query.csv", index=False)

    # =====================================================================
    # 5. ATTRITION report (per-k)
    # =====================================================================
    attr_rows = []
    fw_raw = df[df.estimand == "fourway_common_query"]
    for k in [1, 3, 5, 10]:
        sub = fw_raw[(fw_raw.k == k) & (fw_raw.method == "proposed_weighted_d_optimal")]
        # one row per (lake, learner, rep)
        unit = sub[["lake", "learner", "rep", "query_size", "usable"]].drop_duplicates(
            ["lake", "learner", "rep"])
        n_total = len(unit)
        n_usable = int(unit.usable.sum())
        # lake-learner units with >=50 usable repeats
        lake_usable = unit.groupby(["lake", "learner"]).usable.sum()
        n_ll_50 = int((lake_usable >= 50).sum())
        pool_per_lake = unit.groupby(["lake", "learner"]).query_size.max()  # proxy via max
        attr_rows.append({
            "k": k,
            "n_lake_learner_rep_units_total": n_total,
            "n_usable_units": n_usable,
            "frac_usable": round(n_usable / n_total, 4),
            "n_lake_learner_units_50plus_repeats": n_ll_50,
            "query_size_median": float(unit.query_size.median()),
            "query_size_min": int(unit.query_size.min()),
            "query_size_max": int(unit.query_size.max()),
            "restricted_pool_note": ("restricted-pool sensitivity (attrition > 0)"
                                     if n_usable < n_total else "full pool"),
        })
    attr = pd.DataFrame(attr_rows)
    attr.to_csv(RESULTS_DIR / "h2_nested_query_attrition.csv", index=False)

    print(f"\n=== ATTRITION ===")
    print(attr.to_string(index=False))

    # =====================================================================
    # Self-check: report the 12-cell own-pool numbers in manuscript sign
    # =====================================================================
    print(f"\n=== SELF-CHECK: k=1 three learners (manuscript sign, should be NEGATIVE) ===")
    k1 = own_pool_summary[own_pool_summary.k == 1]
    for _, r in k1.iterrows():
        print(f"  {r.learner}: ΔMAE={r.mean_delta_mae:+.4f} CI=[{r.ci_lo:+.4f},{r.ci_hi:+.4f}] "
              f"class={r.classification}")

    print(f"\nAll 5 summaries written to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
