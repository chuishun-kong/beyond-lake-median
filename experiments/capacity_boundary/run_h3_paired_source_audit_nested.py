"""H3 paired source-vs-calibrated audit under the TRUE-NESTED pipeline
(user-approved, 2026-07-13) -- companion to run_h3_paired_source_audit.py
(which audited the global-tuned pipeline).

Refits each outer lake's PLSR/XGBoost/MLP source model using THAT LAKE'S
OWN nested_hyperparameters.csv selection (fit on the 23 inner lakes, exactly
as run_nested_capacity_boundary.py did -- this script does not repeat the
expensive inner-23 grid search, only the cheap final refit), regenerates the
identical RNG-seeded support/query splits, and evaluates source-only on the
SAME query subset as the calibrated prediction, for k in {1,3,5,10}.

Adds a LOG-SPACE anomaly R2 alongside the existing original-space (mg/L)
version, per the correction that k=1's intercept-only shift is additive in
log space (so log-space anomaly R2 should be exactly invariant, like
Spearman) but multiplicative in original space (so original-space anomaly R2
is a legitimate but scale-sensitive secondary metric, not a gate to pass).
Neither version is merged into a single H3 headline judgment.

Self-check: recomputed 'calibrated' mae_log/rer_log/spearman/anomaly_r2 must
match capacity_boundary_repeats_nested.csv row-for-row within tight
floating-point tolerance before any paired difference is trusted.
"""
import ast
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lake_insitu import gloria, baselines, metrics, sampling  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FROZEN_REPEATS_CSV = RESULTS_DIR / "capacity_boundary_repeats_nested.csv"
HP_CSV = RESULTS_DIR / "nested_hyperparameters.csv"

RNG_SEED = 20260712
K_LEVELS = [0, 1, 3, 5, 10]
N_REPEATS = 100
MIN_GROUP = 20
REPR_PLS_COMPONENTS = 2
STRICT_24_EXCLUDE = {"Ba Be Lake", "Lake Constance"}
SELF_CHECK_TOL = 1e-9


def metric_self_check(actual, archived, metric):
    """Distinguish missingness from a finite numerical comparison."""
    if np.isinf(actual) or np.isinf(archived):
        return np.inf, False, "infinite"
    if np.isnan(actual) or np.isnan(archived):
        status = "both_nan" if np.isnan(actual) and np.isnan(archived) else "one_nan"
        return np.nan, False, status
    difference = abs(actual - archived)
    tolerance = SELF_CHECK_TOL
    if metric == "anomaly_r2":
        # Original-space squared errors can be ~1e7. The historical two
        # exceptions differ by one ULP; allow at most two representable steps.
        tolerance += 2 * abs(np.spacing(max(abs(actual), abs(archived))))
    return difference, difference <= tolerance, "finite"


def require_finite(values, context):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"{context}: {np.isnan(values).sum()} NaN, "
                         f"{np.isinf(values).sum()} infinite, {values.size} values")
    return values


def validate_complete_pairs(per_rep):
    keys = ["lake", "learner", "k", "rep"]
    if per_rep[keys].isna().any().any() or per_rep.duplicated(keys).any():
        raise ValueError("H3 episode keys are missing or duplicated")
    expected = pd.MultiIndex.from_product(
        [sorted(per_rep.lake.unique()), ["PLSR", "XGBoost", "MLP"],
         [1, 3, 5, 10], range(N_REPEATS)], names=keys)
    observed = pd.MultiIndex.from_frame(per_rep[keys])
    if (per_rep.lake.nunique() != 24 or len(expected.difference(observed))
            or len(observed.difference(expected))):
        raise ValueError("H3 requires 24 targets x 3 learners x 4 budgets x 100 draws")
    for col in per_rep.columns.difference(keys):
        require_finite(per_rep[col], f"H3 {col}")


def save_checked_per_rep(per_rep, detail_df):
    """Keep failed replays separate from the accepted result filenames."""
    if detail_df["n_failed_metrics"].sum():
        detail_df.to_csv(RESULTS_DIR / "h3_nested_self_check_detail_provisional.csv", index=False)
        per_rep.to_csv(RESULTS_DIR / "h3_paired_source_audit_nested_per_rep_provisional.csv", index=False)
        raise ValueError("H3 self-check failed; provisional files saved, accepted files unchanged")
    validate_complete_pairs(per_rep)
    detail_df.to_csv(RESULTS_DIR / "h3_nested_self_check_detail.csv", index=False)
    per_rep.to_csv(RESULTS_DIR / "h3_paired_source_audit_nested_per_rep.csv", index=False)


def log_mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def log_space_skill(y_true_log, y_pred_log):
    """Same demeaning logic as metrics.lake_demeaned_skill, but evaluated
    directly in log10(TSS) space -- where the k=1 intercept shift is
    additive, so this version IS invariant to it (unlike the original-space
    anomaly R2, which is not, because exp() turns an additive log-space
    shift into a multiplicative original-space one)."""
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    if len(y_true_log) < 3 or np.std(y_true_log) == 0:
        return np.nan
    yt_dm = y_true_log - y_true_log.mean()
    yp_dm = y_pred_log - y_pred_log.mean()
    ss_res = np.sum((yt_dm - yp_dm) ** 2)
    ss_tot = np.sum(yt_dm ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def z_features_for_k(k, affine_z, repr_scores):
    if k == 1:
        return None
    n_directions = {3: 1, 5: 2, 10: 3}[k]
    cols = [affine_z]
    if n_directions >= 2:
        cols.append(repr_scores[:, 0])
    if n_directions >= 3:
        cols.append(repr_scores[:, 1])
    return np.column_stack(cols)


def median_bootstrap_ci(values, n_boot=10000, seed=0):
    """Post-hoc sensitivity statistic: percentile bootstrap on the MEDIAN.

    The pre-specified primary inference is the
    across-lake MEAN; median-CI is retained only as a robustness sensitivity
    (post-hoc). See mean_bootstrap_ci for the primary inference.
    """
    values = require_finite(values, "median bootstrap")
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([np.median(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def mean_bootstrap_ci(values, n_boot=10000, seed=0):
    """PRIMARY inference: percentile bootstrap
    on the across-lake MEAN of the 24 lake-level (median-of-repeats) values.

    Protocol (frozen): each lake's value is the median over 100 repeats
    (Layer 1); the 24 lake-level values are then the inference sample
    (Layer 2); the bootstrap CI is constructed around the MEAN of those 24
    values. The median-of-24 is a robust descriptive statistic, NOT the
    primary estimand -- switching to median-CI after seeing results would be
    the "choose the favorable statistic" trap this project has hit before.
    """
    values = require_finite(values, "mean bootstrap")
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([np.mean(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run():
    meta = gloria.load_meta()
    rrs = gloria.load_rrs()
    band_cols = gloria.rrs_band_columns(rrs, lo=400, hi=750)
    groups = gloria.build_tss_groups(meta, rrs, band_cols)
    groups["log10_TSS"] = np.log10(groups["TSS"].clip(lower=1e-3))
    eligible = gloria.eligible_tss_lakes(groups, min_group=MIN_GROUP, lakes_reservoirs_only=True)
    lake_list = [l for l in eligible.index if l not in STRICT_24_EXCLUDE]

    hp = pd.read_csv(HP_CSV).set_index("outer_lake")
    frozen = pd.read_csv(FROZEN_REPEATS_CSV)
    frozen_cal = frozen[frozen.method == "calibrated"].set_index(
        ["lake", "learner", "k", "rep"])

    rows = []
    self_check_diffs = []
    self_check_detail = []  # (lake, learner, k, rep, which_metric, diff)

    for target_lake in lake_list:
        hp_row = hp.loc[target_lake]
        plsr_nc = int(hp_row["plsr_n_components"])
        xgb_params = ast.literal_eval(hp_row["xgb_params"])
        mlp_params = ast.literal_eval(hp_row["mlp_params"])
        mlp_params["hidden_layer_sizes"] = tuple(mlp_params["hidden_layer_sizes"])

        is_target = groups["Site_name"] == target_lake
        target_df = groups[is_target].reset_index(drop=True)
        source_df = groups[~is_target].reset_index(drop=True)
        X_source_raw = source_df[band_cols].to_numpy(dtype=float)
        y_source_log = source_df["log10_TSS"].to_numpy(dtype=float)
        X_target_raw = target_df[band_cols].to_numpy(dtype=float)
        y_target = target_df["TSS"].to_numpy(dtype=float)
        y_target_log = target_df["log10_TSS"].to_numpy(dtype=float)
        mu, sigma = X_source_raw.mean(axis=0), X_source_raw.std(axis=0)
        sigma[sigma == 0] = 1.0
        X_source = (X_source_raw - mu) / sigma
        X_target = (X_target_raw - mu) / sigma

        repr_pls = baselines.fit_plsr_source(X_source, y_source_log,
                                              n_components=REPR_PLS_COMPONENTS)
        repr_scores_target = repr_pls.transform(X_target)

        learners = {
            "PLSR": baselines.fit_plsr_source(X_source, y_source_log, n_components=plsr_nc),
            "XGBoost": XGBRegressor(random_state=0, n_jobs=-1, **xgb_params).fit(X_source, y_source_log),
            "MLP": MLPRegressor(activation="relu", learning_rate_init=1e-3, max_iter=2000,
                                 early_stopping=True, n_iter_no_change=20, random_state=0,
                                 **mlp_params).fit(X_source, y_source_log),
        }
        base_preds_target = {name: np.asarray(m.predict(X_target)).ravel()
                              for name, m in learners.items()}

        rng = np.random.default_rng(RNG_SEED + zlib.crc32(target_lake.encode()) % 10_000)

        for rep in range(N_REPEATS):
            for k in K_LEVELS:
                if k == 0:
                    continue
                support_idx, query_idx = sampling.random_support_query_split(
                    np.arange(len(target_df)), k=k, rng=rng)
                y_qry, y_qry_log = y_target[query_idx], y_target_log[query_idx]
                y_sup, y_sup_log = y_target[support_idx], y_target_log[support_idx]
                lbl = baselines.label_only_baselines(y_sup)
                best_label_mae_log = log_mae(
                    y_qry_log, np.full_like(y_qry_log, np.log10(lbl["label_median"])))

                for learner_name, base_pred_target_log in base_preds_target.items():
                    base_pred_qry_log = base_pred_target_log[query_idx]
                    base_pred_sup_log = base_pred_target_log[support_idx]
                    repr_sup = repr_scores_target[support_idx]
                    repr_qry = repr_scores_target[query_idx]

                    if k == 1:
                        b_l = baselines.intercept_only_calibration(base_pred_sup_log, y_sup_log)
                        cal_pred_qry_log = base_pred_qry_log + b_l
                    else:
                        z_sup = z_features_for_k(k, base_pred_sup_log, repr_sup)
                        z_qry = z_features_for_k(k, base_pred_qry_log, repr_qry)
                        resid_sup_log = y_sup_log - base_pred_sup_log
                        ridge = baselines.residual_ridge_calibration(z_sup, resid_sup_log, alpha=1.0)
                        cal_pred_qry_log = base_pred_qry_log + ridge.predict(z_qry).ravel()

                    cal_pred_qry_orig = baselines.safe_pow10(cal_pred_qry_log)
                    cal_mae_log = log_mae(y_qry_log, cal_pred_qry_log)
                    cal_rer_log = metrics.relative_error_reduction(best_label_mae_log, cal_mae_log)
                    cal_skill = metrics.lake_demeaned_skill(y_qry, cal_pred_qry_orig)

                    frozen_row = frozen_cal.loc[(target_lake, learner_name, k, rep)]
                    pairs = {
                        "mae_log": (cal_mae_log, frozen_row["mae_log"]),
                        "rer_log": (cal_rer_log, frozen_row["rer_log"]),
                        "spearman": (cal_skill["spearman"], frozen_row["demeaned_spearman"]),
                        "anomaly_r2": (cal_skill["anomaly_r2"], frozen_row["anomaly_r2"]),
                    }
                    checks = {name: metric_self_check(a, b, name)
                              for name, (a, b) in pairs.items()}
                    worst_metric = max(checks, key=lambda name:
                                       checks[name][0] if np.isfinite(checks[name][0]) else np.inf)
                    worst_diff = checks[worst_metric][0]
                    self_check_diffs.append(worst_diff)
                    self_check_detail.append((target_lake, learner_name, k, rep, worst_metric,
                                              worst_diff, sum(not c[1] for c in checks.values()),
                                              ";".join(f"{name}:{c[2]}" for name, c in checks.items()
                                                       if c[2] != "finite")))

                    src_pred_qry_orig = baselines.safe_pow10(base_pred_qry_log)
                    src_skill = metrics.lake_demeaned_skill(y_qry, src_pred_qry_orig)
                    cal_log_r2 = log_space_skill(y_qry_log, cal_pred_qry_log)
                    src_log_r2 = log_space_skill(y_qry_log, base_pred_qry_log)

                    rows.append({
                        "lake": target_lake, "learner": learner_name, "k": k, "rep": rep,
                        "calibrated_spearman": cal_skill["spearman"],
                        "source_spearman": src_skill["spearman"],
                        "delta_spearman": cal_skill["spearman"] - src_skill["spearman"],
                        "calibrated_anomaly_r2_origspace": cal_skill["anomaly_r2"],
                        "source_anomaly_r2_origspace": src_skill["anomaly_r2"],
                        "delta_anomaly_r2_origspace": cal_skill["anomaly_r2"] - src_skill["anomaly_r2"],
                        "calibrated_anomaly_r2_logspace": cal_log_r2,
                        "source_anomaly_r2_logspace": src_log_r2,
                        "delta_anomaly_r2_logspace": cal_log_r2 - src_log_r2,
                    })

        print(f"  {target_lake}: done")

    per_rep = pd.DataFrame(rows)
    detail_df = pd.DataFrame(self_check_detail,
                              columns=["lake", "learner", "k", "rep", "worst_metric", "diff_val",
                                       "n_failed_metrics", "nonfinite_metrics"])
    max_diff = detail_df["diff_val"].max()
    over_tol_mask = detail_df["diff_val"] > SELF_CHECK_TOL
    n_over_tol = int(over_tol_mask.sum())
    print(f"\n=== Self-check vs {FROZEN_REPEATS_CSV.name} (tolerance {SELF_CHECK_TOL:.0e}) ===")
    print(f"  max abs diff = {max_diff:.3e}; {n_over_tol}/{len(detail_df)} rows exceed tolerance")
    if n_over_tol:
        print(f"  which metric caused it, among rows over tolerance: "
              f"{detail_df.loc[over_tol_mask, 'worst_metric'].value_counts().to_dict()}")
        print("  worst 5 rows:")
        print(detail_df.sort_values('diff_val', ascending=False).head(5).to_string(index=False))
    print("  Original-space anomaly R2 additionally allows two floating-point steps; "
          "non-finite pairs never pass.")
    save_checked_per_rep(per_rep, detail_df)

    k1 = per_rep[per_rep.k == 1]
    print("=== k=1 identity check (log-space should be ~0 exactly; orig-space need not be) ===")
    print(f"  max|delta_spearman| = {k1.delta_spearman.abs().max():.3e}")
    print(f"  max|delta_anomaly_r2_logspace| = {k1.delta_anomaly_r2_logspace.abs().max():.3e}")
    print(f"  max|delta_anomaly_r2_origspace| = {k1.delta_anomaly_r2_origspace.abs().max():.3e}")

    per_lake = per_rep.groupby(["lake", "learner", "k"]).agg(
        calibrated_spearman=("calibrated_spearman", "median"),
        source_spearman=("source_spearman", "median"),
        delta_spearman=("delta_spearman", "median"),
        calibrated_anomaly_r2_logspace=("calibrated_anomaly_r2_logspace", "median"),
        source_anomaly_r2_logspace=("source_anomaly_r2_logspace", "median"),
        delta_anomaly_r2_logspace=("delta_anomaly_r2_logspace", "median"),
        delta_anomaly_r2_origspace=("delta_anomaly_r2_origspace", "median"),
    ).reset_index()
    per_lake.to_csv(RESULTS_DIR / "h3_paired_source_audit_nested_per_lake.csv", index=False)
    compute_summary_from_per_lake(per_lake)


def compute_summary_from_per_lake(per_lake):
    """Compute the dual-inference summary (mean-CI PRIMARY + median-CI
    SENSITIVITY) from the 24-lake x learner x k per-lake table.

    Factored out so Phase 0 (H3 mean-CI closure) can re-run ONLY this part
    from the already-persisted per_lake.csv without re-fitting any base
    model -- see bootstrap_only() entry point. The per-lake aggregation
    itself (median over 100 repeats per lake, the Layer-1 protocol) is
    frozen and untouched; only the Layer-2 inference statistic is extended
    to report mean as primary alongside the existing median as sensitivity.
    """
    print("\n=== Nested paired Delta-rho / Delta-anomaly-R2 (log-space & orig-space) ===")
    print("PRIMARY inference = across-lake MEAN bootstrap CI (per Sec 2.8 protocol).")
    print("SENSITIVITY = median bootstrap CI (post-hoc robustness only).\n")
    summary_rows = []
    fullprec_rows = []  # unrounded CI endpoints for audit traceability
    for learner in ["PLSR", "XGBoost", "MLP"]:
        for k in [1, 3, 5, 10]:
            sub = per_lake[(per_lake.learner == learner) & (per_lake.k == k)]
            if len(sub) != 24 or sub.lake.nunique() != 24 or sub.lake.isna().any():
                raise ValueError(f"{learner}, k={k}: expected 24 distinct targets")
            for col in ("source_spearman", "calibrated_spearman", "delta_spearman",
                        "delta_anomaly_r2_logspace", "delta_anomaly_r2_origspace"):
                require_finite(sub[col], f"{learner}, k={k}, {col}")
            n_pos = int((sub.delta_spearman > 1e-9).sum())
            n_neg = int((sub.delta_spearman < -1e-9).sum())
            n_zero = int((sub.delta_spearman.abs() <= 1e-9).sum())
            rho_arr = sub.delta_spearman.to_numpy()
            r2log_arr = sub.delta_anomaly_r2_logspace.to_numpy()
            r2orig_arr = sub.delta_anomaly_r2_origspace.to_numpy()

            # PRIMARY: mean bootstrap CI (Sec 2.8 protocol)
            mean_ci_rho = mean_bootstrap_ci(rho_arr)
            mean_ci_r2log = mean_bootstrap_ci(r2log_arr)
            mean_ci_r2orig = mean_bootstrap_ci(r2orig_arr)
            # SENSITIVITY: median bootstrap CI (post-hoc)
            med_ci_rho = median_bootstrap_ci(rho_arr)
            med_ci_r2log = median_bootstrap_ci(r2log_arr)
            med_ci_r2orig = median_bootstrap_ci(r2orig_arr)

            # Pre-frozen classification rule (gate #1 + #2):
            #   mean-CI fully <0 -> "negative"; fully >0 -> "improvement";
            #   crosses zero -> "unresolved" (MLP even with upper bound far
            #   from zero but lower bound crossing zero stays unresolved).
            #   k=1 is the intercept-only identity (delta_spearman ~ 0 by
            #   construction) and is NOT tested -- reported as "identity".
            if k == 1:
                classification = "identity"
            elif mean_ci_rho[1] < -1e-12:
                classification = "negative"
            elif mean_ci_rho[0] > 1e-12:
                classification = "improvement"
            else:
                classification = "unresolved"

            summary_rows.append({
                "learner": learner, "k": k,
                "source_rho_median": round(sub.source_spearman.median(), 4),
                "calibrated_rho_median": round(sub.calibrated_spearman.median(), 4),
                # PRIMARY (mean-based)
                "mean_delta_rho": round(sub.delta_spearman.mean(), 4),
                "mean_delta_rho_ci": (round(mean_ci_rho[0], 4), round(mean_ci_rho[1], 4)),
                "mean_delta_rho_classification": classification,
                # SENSITIVITY (median-based, post-hoc)
                "median_delta_rho": round(sub.delta_spearman.median(), 4),
                "median_delta_rho_ci": (round(med_ci_rho[0], 4), round(med_ci_rho[1], 4)),
                "n_lakes_pos": n_pos, "n_lakes_neg": n_neg, "n_lakes_zero": n_zero,
                # PRIMARY R2 (log-space, the H3 main metric)
                "mean_delta_R2_logspace": round(sub.delta_anomaly_r2_logspace.mean(), 4),
                "mean_delta_R2_logspace_ci": (round(mean_ci_r2log[0], 4), round(mean_ci_r2log[1], 4)),
                # SENSITIVITY R2 (log-space)
                "median_delta_R2_logspace": round(sub.delta_anomaly_r2_logspace.median(), 4),
                "median_delta_R2_logspace_ci": (round(med_ci_r2log[0], 4), round(med_ci_r2log[1], 4)),
                # original-space R2 (scale-sensitive secondary; NOT a gate)
                "mean_delta_R2_origspace": round(sub.delta_anomaly_r2_origspace.mean(), 4),
                "mean_delta_R2_origspace_ci": (round(mean_ci_r2orig[0], 4), round(mean_ci_r2orig[1], 4)),
                "median_delta_R2_origspace": round(sub.delta_anomaly_r2_origspace.median(), 4),
                "median_delta_R2_origspace_ci": (round(med_ci_r2orig[0], 4), round(med_ci_r2orig[1], 4)),
            })
            fullprec_rows.append({
                "learner": learner, "k": k,
                "mean_delta_rho": float(sub.delta_spearman.mean()),
                "mean_delta_rho_ci_lo": mean_ci_rho[0], "mean_delta_rho_ci_hi": mean_ci_rho[1],
                "median_delta_rho": float(sub.delta_spearman.median()),
                "median_delta_rho_ci_lo": med_ci_rho[0], "median_delta_rho_ci_hi": med_ci_rho[1],
                "mean_delta_R2_logspace": float(sub.delta_anomaly_r2_logspace.mean()),
                "mean_delta_R2_logspace_ci_lo": mean_ci_r2log[0], "mean_delta_R2_logspace_ci_hi": mean_ci_r2log[1],
                "mean_delta_R2_origspace": float(sub.delta_anomaly_r2_origspace.mean()),
                "mean_delta_R2_origspace_ci_lo": mean_ci_r2orig[0], "mean_delta_R2_origspace_ci_hi": mean_ci_r2orig[1],
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "h3_paired_source_audit_nested_summary.csv", index=False)
    fullprec = pd.DataFrame(fullprec_rows)
    fullprec.to_csv(RESULTS_DIR / "h3_paired_source_audit_nested_summary_fullprecision.csv", index=False)
    with pd.option_context("display.width", 320, "display.max_columns", 40):
        print(summary.to_string(index=False))


def bootstrap_only():
    """Phase 0 entry point: re-run ONLY the dual-inference bootstrap summary
    from the already-persisted per_lake.csv. No base-model refit, no RNG.
    Used to close the H3 mean-CI estimand gap (Sec 2.8 freezes across-lake
    MEAN as primary; the original summary used median bootstrap CI).
    """
    per_lake_path = RESULTS_DIR / "h3_paired_source_audit_nested_per_lake.csv"
    per_lake = pd.read_csv(per_lake_path)
    print(f"Loaded {len(per_lake)} per-lake rows from {per_lake_path.name}")
    compute_summary_from_per_lake(per_lake)


if __name__ == "__main__":
    if "--bootstrap-only" in sys.argv:
        bootstrap_only()
    else:
        run()
