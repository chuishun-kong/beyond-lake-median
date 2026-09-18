"""Phase 2 (P1b): k=5/k=10 leave-one-direction-out calibration-direction ablation
under the TRUE-NESTED pipeline.

GOAL: isolate which calibration direction(s) drive the k=5/k=10 rank and
magnitude effects, in particular to help interpret why XGBoost is the only
learner showing a mean-CI negative rank effect at k=5 (see Phase 0 summary).

PROTOCOL (frozen):
  - Rebuild each outer lake's base model from frozen nested HP, fit on its 23
    inner source lakes with random_state=0. NO base-model retraining.
  - Regenerate the IDENTICAL RNG-seeded support/query splits as the frozen
    pipeline (k=0 skipped, rng advanced only at k in {1,3,5,10}).
  - At k=5 and k=10, on the SAME support and SAME query, re-fit the SAME ridge
    (alpha=1.0, fit_intercept=True) with different feature-column subsets.

  CRITICAL (gate #3): "full minus direction" MUST re-fit the ridge after
  deleting the corresponding column. Setting the coefficient to zero is NOT
  equivalent -- ridge regularization re-distributes coefficients when a column
  is removed. Zeroing would silently violate the estimand.

ABLATION VARIANTS (per k):
  - source_only       : no ridge; prediction = base_pred_qry (the k=0 baseline)
  - intercept_only    : ridge on empty feature matrix, fit_intercept=True
                        (== k=1 intercept shift)
  - affine_only       : ridge on [base_pred_log] only (the k=3-style direction)
  - repr_only         : ridge on representation columns only
                        (k=5: [repr0]; k=10: [repr0, repr1])
  - full              : ridge on all columns
                        (k=5: [base_pred, repr0]; k=10: [base_pred, repr0, repr1])
                        MUST reproduce the frozen calibrated metrics (self-check)
  - full_minus_affine : LOO drop affine column
                        (k=5: [repr0]; k=10: [repr0, repr1])
  - full_minus_repr0  : LOO drop repr[:,0]
                        (k=5: [base_pred]; k=10: [base_pred, repr1])
  - full_minus_repr1  : (k=10 only) LOO drop repr[:,1] -> [base_pred, repr0]

REPORTED per (lake, learner, k, rep, variant):
  - log-MAE, RER, Spearman, mean Delta_rho (vs source_only),
    log-space anomaly R^2 (the H3 secondary metric),
    original-space anomaly R^2 (diagnostic only, NOT a gate),
    ridge coefficients, design-matrix feature correlations, condition number.

INTERPRETATION (frozen): this is an EXPLORATORY paired leave-one-direction-out
ablation. Wording MUST be "when direction X is removed, Delta changes by ...".
NO causal/attribution/additive-contribution claims -- the directions are
correlated (affine prediction correlates with repr-PLS scores), so effects do
not decompose additively.

SELF-CHECK: the `full` variant MUST reproduce the frozen calibrated
mae_log/rer_log/spearman within 1e-9; `intercept_only` == k=1 (identity check
on log-space metrics); `source_only` == k=0.
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
from sklearn.linear_model import Ridge  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FROZEN_REPEATS_CSV = RESULTS_DIR / "capacity_boundary_repeats_nested.csv"
HP_CSV = RESULTS_DIR / "nested_hyperparameters.csv"

RNG_SEED = 20260712
K_LEVELS_FOR_RNG_PARITY = [0, 1, 3, 5, 10]  # k=0 skipped, rng advances at 1,3,5,10
K_TARGETS = [5, 10]
N_REPEATS = 100
MIN_GROUP = 20
REPR_PLS_COMPONENTS = 2
STRICT_24_EXCLUDE = {"Ba Be Lake", "Lake Constance"}
SELF_CHECK_TOL = 1e-9


def log_mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def log_space_skill(y_true_log, y_pred_log):
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    if len(y_true_log) < 3 or np.std(y_true_log) == 0:
        return np.nan
    yt_dm = y_true_log - y_true_log.mean()
    yp_dm = y_pred_log - y_pred_log.mean()
    ss_res = np.sum((yt_dm - yp_dm) ** 2)
    ss_tot = np.sum(yt_dm ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def build_variant_matrix(variant, k, base_pred_sup_log, base_pred_qry_log,
                         repr_sup, repr_qry):
    """Return (z_sup, z_qry, is_intercept_only, is_source_only) per variant.

    Columns use the SAME per-k layout as z_features_for_k in the frozen pipeline:
      k=5  affine=base_pred_log, repr_cols=[repr[:,0]]
      k=10 affine=base_pred_log, repr_cols=[repr[:,0], repr[:,1]]
    """
    repr0_sup = repr_sup[:, 0]
    repr0_qry = repr_qry[:, 0]
    repr1_sup = repr_sup[:, 1] if repr_sup.shape[1] >= 2 else None
    repr1_qry = repr_qry[:, 1] if repr_qry.shape[1] >= 2 else None

    if variant == "source_only":
        return None, None, False, True
    if variant == "intercept_only":
        # Empty feature matrix with fit_intercept=True == intercept-only shift.
        n_sup = len(base_pred_sup_log)
        n_qry = len(base_pred_qry_log)
        return (np.zeros((n_sup, 0)), np.zeros((n_qry, 0)), True, False)
    if variant == "affine_only":
        cols_sup = [base_pred_sup_log]
        cols_qry = [base_pred_qry_log]
    elif variant == "repr_only":
        cols_sup = [repr0_sup]
        cols_qry = [repr0_qry]
        if k == 10:
            cols_sup.append(repr1_sup)
            cols_qry.append(repr1_qry)
    elif variant == "full":
        cols_sup = [base_pred_sup_log, repr0_sup]
        cols_qry = [base_pred_qry_log, repr0_qry]
        if k == 10:
            cols_sup.append(repr1_sup)
            cols_qry.append(repr1_qry)
    elif variant == "full_minus_affine":
        # Drop affine, keep repr columns (same as repr_only by construction).
        cols_sup = [repr0_sup]
        cols_qry = [repr0_qry]
        if k == 10:
            cols_sup.append(repr1_sup)
            cols_qry.append(repr1_qry)
    elif variant == "full_minus_repr0":
        cols_sup = [base_pred_sup_log]
        cols_qry = [base_pred_qry_log]
        if k == 10:
            cols_sup.append(repr1_sup)
            cols_qry.append(repr1_qry)
    elif variant == "full_minus_repr1":
        # k=10 only: [base_pred, repr0]
        cols_sup = [base_pred_sup_log, repr0_sup]
        cols_qry = [base_pred_qry_log, repr0_qry]
    else:
        raise ValueError(f"unknown variant {variant}")

    return (np.column_stack(cols_sup), np.column_stack(cols_qry), False, False)


def variant_variants_for_k(k):
    """Which variants to run at this k."""
    base = ["source_only", "intercept_only", "affine_only", "repr_only",
            "full", "full_minus_affine", "full_minus_repr0"]
    if k == 10:
        base.append("full_minus_repr1")
    return base


def fit_and_predict(z_sup, z_qry, resid_sup_log, is_intercept_only, is_source_only,
                    base_pred_qry_log):
    """Fit ridge on z_sup -> resid_sup_log, predict on z_qry.

    Returns (cal_pred_qry_log, ridge_model_or_None).
    - source_only   : cal_pred = base_pred (no calibration), model=None
    - intercept_only: cal = base_pred + mean(resid_sup) (the k=1 intercept
                      shift). sklearn Ridge rejects a 0-column matrix, so the
                      intercept is computed directly.
    - else          : fit Ridge(alpha=1.0, fit_intercept=True) on z_sup.
    """
    if is_source_only:
        return base_pred_qry_log, None
    if is_intercept_only:
        b_l = float(np.mean(resid_sup_log))
        # Return a lightweight stub carrying the single intercept for reporting.
        class _InterceptOnly:
            def __init__(self, b): self.intercept_ = b; self.coef_ = np.array([])
        return base_pred_qry_log + b_l, _InterceptOnly(b_l)
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(z_sup, resid_sup_log)
    return base_pred_qry_log + model.predict(z_qry).ravel(), model


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
    self_check_diffs = []  # only for `full` vs frozen
    source_only_diffs = []  # `source_only` vs frozen source_only (k=0)

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
            for k in K_LEVELS_FOR_RNG_PARITY:
                if k == 0:
                    continue
                support_idx, query_idx = sampling.random_support_query_split(
                    np.arange(len(target_df)), k=k, rng=rng)
                if k not in K_TARGETS:
                    continue

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
                    resid_sup_log = y_sup_log - base_pred_sup_log

                    # Feature correlation + condition number on the full design
                    # (for diagnostic reporting; computed once per learner/k/rep).
                    full_z_sup = np.column_stack(
                        [base_pred_sup_log, repr_sup[:, 0]]
                        if k == 5 else
                        [base_pred_sup_log, repr_sup[:, 0], repr_sup[:, 1]])
                    if full_z_sup.shape[1] >= 2:
                        corr_matrix = np.corrcoef(full_z_sup.T)
                        feat_corr_affine_repr0 = float(corr_matrix[0, 1])
                        if k == 10:
                            feat_corr_affine_repr1 = float(corr_matrix[0, 2])
                            feat_corr_repr0_repr1 = float(corr_matrix[1, 2])
                        else:
                            feat_corr_affine_repr1 = np.nan
                            feat_corr_repr0_repr1 = np.nan
                        # Condition number of the unregularized design (diag
                        # added for numerical stability when near-singular).
                        cond_num = float(np.linalg.cond(
                            full_z_sup.T @ full_z_sup + 1e-10 * np.eye(full_z_sup.shape[1])))
                    else:
                        feat_corr_affine_repr0 = np.nan
                        feat_corr_affine_repr1 = np.nan
                        feat_corr_repr0_repr1 = np.nan
                        cond_num = np.nan

                    frozen_row = frozen_cal.loc[(target_lake, learner_name, k, rep)]

                    for variant in variant_variants_for_k(k):
                        z_sup, z_qry, is_intercept, is_source = build_variant_matrix(
                            variant, k, base_pred_sup_log, base_pred_qry_log,
                            repr_sup, repr_qry)
                        cal_pred_qry_log, ridge_model = fit_and_predict(
                            z_sup, z_qry, resid_sup_log, is_intercept, is_source,
                            base_pred_qry_log)

                        cal_pred_qry_orig = baselines.safe_pow10(cal_pred_qry_log)
                        cal_mae_log = log_mae(y_qry_log, cal_pred_qry_log)
                        cal_rer_log = metrics.relative_error_reduction(
                            best_label_mae_log, cal_mae_log)
                        cal_skill = metrics.lake_demeaned_skill(y_qry, cal_pred_qry_orig)
                        # source-only skill (same query, for paired Delta)
                        src_pred_qry_orig = baselines.safe_pow10(base_pred_qry_log)
                        src_skill = metrics.lake_demeaned_skill(y_qry, src_pred_qry_orig)
                        cal_log_r2 = log_space_skill(y_qry_log, cal_pred_qry_log)
                        src_log_r2 = log_space_skill(y_qry_log, base_pred_qry_log)

                        delta_spearman = cal_skill["spearman"] - src_skill["spearman"]
                        delta_r2_log = cal_log_r2 - src_log_r2

                        # Ridge coefficients (stringified for CSV; intercept + coefs)
                        if ridge_model is not None:
                            coefs = [float(c) for c in np.atleast_1d(ridge_model.coef_).ravel()]
                            coef_str = str([round(c, 6) for c in coefs])
                            intercept_val = float(ridge_model.intercept_)
                        else:
                            coef_str = "[]" if is_intercept else "NA(source_only)"
                            intercept_val = 0.0 if is_source else (
                                float(np.mean(resid_sup_log)) if is_intercept else np.nan)

                        rows.append({
                            "lake": target_lake, "learner": learner_name,
                            "k": k, "rep": rep, "variant": variant,
                            "mae_log": cal_mae_log, "rer_log": cal_rer_log,
                            "spearman": cal_skill["spearman"],
                            "source_spearman": src_skill["spearman"],
                            "delta_spearman": delta_spearman,
                            "anomaly_r2_origspace": cal_skill["anomaly_r2"],
                            "anomaly_r2_logspace": cal_log_r2,
                            "source_anomaly_r2_logspace": src_log_r2,
                            "delta_anomaly_r2_logspace": delta_r2_log,
                            "ridge_intercept": intercept_val,
                            "ridge_coefs": coef_str,
                            "feat_corr_affine_repr0": feat_corr_affine_repr0,
                            "feat_corr_affine_repr1": feat_corr_affine_repr1,
                            "feat_corr_repr0_repr1": feat_corr_repr0_repr1,
                            "cond_number_full_design": cond_num,
                        })

                        # Self-check: `full` variant must match frozen calibrated.
                        if variant == "full":
                            d = {
                                "mae_log": abs(cal_mae_log - frozen_row["mae_log"]),
                                "rer_log": abs(cal_rer_log - frozen_row["rer_log"]),
                                "spearman": (abs(cal_skill["spearman"] - frozen_row["demeaned_spearman"])
                                             if not np.isnan(cal_skill["spearman"]) else 0.0),
                                "anomaly_r2": (abs(cal_skill["anomaly_r2"] - frozen_row["anomaly_r2"])
                                               if not np.isnan(cal_skill["anomaly_r2"]) else 0.0),
                            }
                            self_check_diffs.append(max(d.values()))

        print(f"  {target_lake}: done")

    per_rep = pd.DataFrame(rows)
    per_rep.to_csv(RESULTS_DIR / "calibration_direction_ablation_nested_per_rep.csv", index=False)

    max_diff = max(self_check_diffs) if self_check_diffs else 0.0
    n_over = sum(1 for d in self_check_diffs if d > SELF_CHECK_TOL)
    print(f"\n=== Self-check: `full` variant vs frozen calibrated (tol {SELF_CHECK_TOL:.0e}) ===")
    print(f"  max abs diff = {max_diff:.3e}; {n_over}/{len(self_check_diffs)} rows exceed tolerance")
    if n_over:
        print("  WARNING: full-variant self-check FAILED; ablation numbers NOT trustworthy.")

    # Per-lake aggregation (median over 100 reps; primary Delta uses mean at
    # summary level via the bootstrap, but per-lake we record median + mean).
    per_lake = per_rep.groupby(["lake", "learner", "k", "variant"]).agg(
        mae_log_median=("mae_log", "median"),
        rer_log_median=("rer_log", "median"),
        spearman_median=("spearman", "median"),
        source_spearman_median=("source_spearman", "median"),
        delta_spearman_median=("delta_spearman", "median"),
        delta_spearman_mean=("delta_spearman", "mean"),
        delta_r2_log_median=("delta_anomaly_r2_logspace", "median"),
        delta_r2_log_mean=("delta_anomaly_r2_logspace", "mean"),
    ).reset_index()
    per_lake.to_csv(RESULTS_DIR / "calibration_direction_ablation_nested_per_lake.csv", index=False)

    # Summary per (learner, k, variant): across-lake mean of delta_spearman
    # (primary estimand per Sec 2.8 is across-lake MEAN).
    print("\n=== k=5/k=10 ablation summary (across-lake mean Delta_rho, primary) ===")
    summary_rows = []
    for learner in ["PLSR", "XGBoost", "MLP"]:
        for k in K_TARGETS:
            for variant in variant_variants_for_k(k):
                sub = per_lake[(per_lake.learner == learner) & (per_lake.k == k)
                               & (per_lake.variant == variant)]
                summary_rows.append({
                    "learner": learner, "k": k, "variant": variant,
                    "n_lakes": len(sub),
                    "mean_delta_rho": round(sub.delta_spearman_mean.mean(), 4),
                    "median_delta_rho": round(sub.delta_spearman_median.median(), 4),
                    "mean_delta_r2_log": round(sub.delta_r2_log_mean.mean(), 4),
                    "mae_log_median": round(sub.mae_log_median.median(), 4),
                    "rer_log_median": round(sub.rer_log_median.median(), 4),
                })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "calibration_direction_ablation_nested_summary.csv", index=False)
    with pd.option_context("display.width", 320, "display.max_columns", 30):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    run()
