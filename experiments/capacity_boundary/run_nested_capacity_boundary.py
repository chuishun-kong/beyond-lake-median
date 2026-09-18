"""True-nested capacity_boundary rerun (user-approved, 2026-07-13).

Motivation: the current headline H1/H3 numbers (capacity_boundary_summary_
final.csv) select ONE global hyperparameter per learner via a leave-one-
lake-out sweep over ALL 24 eligible lakes -- including, for each lake's
eventual outer-fold evaluation, that same lake's own held-out error as one
of the 24 votes that chose the hyperparameter later applied to it. This is
the exact model-selection/performance-estimation conflation Cawley & Talbot
(JMLR 2010) describe: the fix is to reselect hyperparameters independently
within each outer fold, using the other eligible lakes as inner validation
units while fitting each inner model on the complete source archive after the
required outer/inner `Site_name` exclusions.

PROTOCOL FREEZE (recorded before any outer-lake result is computed; do not
edit any of the following after seeing a result):

- Candidate grids: PLSR n_components in {4, 8, 12, 16, 20} -- PROTOCOL
  AMENDMENT. The original 2026-07-11 PLSR sweep's exact candidate grid was
  not persisted as a reusable artifact (see run_nested_hp_dry_run.py's
  docstring); this grid is proposed fresh, in the same spirit (a modest
  range bracketing the previously-selected value of 16), and is frozen
  here before any outer-lake result exists. XGBoost grid (9 candidates)
  and MLP grid (5 candidates) are identical to run_hyperparam_sweep.py.
- Selection objective: median inner-held-out log-MAE of the UNCALIBRATED
  source model's own prediction -- IDENTICAL objective to the existing
  global-tuned pipeline. NOT changed to a post-calibration RER objective,
  even though the H3 paired-Delta-rho finding might argue a downstream-
  aligned objective would be more informative: changing the objective now,
  after seeing that finding, would itself be a result-driven protocol
  change. The objective/downstream-target mismatch (Section 4.1) remains a
  documented limitation, not something this rerun is designed to fix.
- Tie-breaking: lowest median log-MAE wins; on an exact tie, prefer the
  SIMPLEST candidate (fewest PLSR components; smallest n_estimators *
  max_depth for XGBoost; smallest architecture -- fewest total hidden
  units -- for MLP). Disclosed because ties are a free protocol choice
  this rerun must make explicit rather than leave to incidental sort
  order (in practice, ties at float precision are effectively impossible
  on this metric, but the rule is stated for completeness).
- Fit scope: standardization (mu, sigma) and the representation-PLS
  embedding are fit on the all-other-GLORIA source archive after exact
  outer-target `Site_name` exclusion. The 23 other eligible lakes are
  inner validation units; each inner model excludes both the outer target
  and the current inner-validation `Site_name` from that complete archive.
- Randomness reuse: support/query splits use the IDENTICAL RNG_SEED
  (20260712) and per-lake seed derivation (RNG_SEED + zlib.crc32(lake) %
  10000) as the existing global-tuned pipeline, so any numerical
  difference in H1/H3 results is attributable to the hyperparameter-
  selection change alone, not to a different random sequence. Base-learner
  fitting uses fixed random_state=0 (XGBoost/MLP), as before.
- Checkpointing: each outer lake's full result (selected hyperparameters +
  k=0/1/3/5/10 repeats) is written to
  results/nested/<lake>_repeats.csv and results/nested/<lake>_hp.json
  immediately after that lake finishes. Re-running this script skips any
  outer lake whose checkpoint file already exists, so an interruption
  loses at most the one outer lake in progress.
- Output isolation: final combined outputs are
  capacity_boundary_repeats_nested.csv / capacity_boundary_summary_nested.csv
  / nested_hyperparameters.csv. The existing *_final.csv files are never
  opened for writing and remain the development-stage/sensitivity
  reference (not overwritten, not deleted).

Usage: python experiments/capacity_boundary/run_nested_capacity_boundary.py
       (safe to re-run to resume after an interruption; run a final pass
       with --merge-only after all 24 lakes have checkpoints to rebuild
       the combined _nested.csv files without recomputing anything)
"""
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lake_insitu import gloria, baselines, metrics, sampling, models  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
NESTED_DIR = RESULTS_DIR / "nested"

RNG_SEED = 20260712
K_LEVELS = [0, 1, 3, 5, 10]
N_REPEATS = 100
MIN_GROUP = 20
REPR_PLS_COMPONENTS = 2
STRICT_24_EXCLUDE = {"Ba Be Lake", "Lake Constance"}

PLSR_GRID = [4, 8, 12, 16, 20]  # protocol amendment -- see module docstring
XGB_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": 0.05,
     "subsample": 0.8, "colsample_bytree": 0.8}
    for n in [100, 300, 500] for d in [3, 4, 6]
]
MLP_GRID = [
    {"hidden_layer_sizes": hl, "alpha": 1e-3}
    for hl in [(32,), (64,), (64, 32), (128, 64), (64, 64)]
]


def log_mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def select_plsr(per_lake_inner):
    scored = []
    for nc in PLSR_GRID:
        maes = []
        for lake, (X_r, y_r, X_h, y_h) in per_lake_inner.items():
            n_comp = min(nc, X_r.shape[0] - 1, X_r.shape[1])
            model = baselines.fit_plsr_source(X_r, y_r, n_components=max(1, n_comp))
            pred = np.asarray(model.predict(X_h)).ravel()
            maes.append(log_mae(y_h, pred))
        scored.append((float(np.median(maes)), nc, nc))  # (score, tie-break-key, param)
    scored.sort(key=lambda t: (t[0], t[1]))  # simplest (fewest components) breaks ties
    return scored[0][2]


def select_xgb(per_lake_inner):
    scored = []
    for params in XGB_GRID:
        maes = []
        for lake, (X_r, y_r, X_h, y_h) in per_lake_inner.items():
            model = XGBRegressor(random_state=0, n_jobs=-1, **params)
            model.fit(X_r, y_r)
            pred = model.predict(X_h)
            maes.append(log_mae(y_h, pred))
        complexity = params["n_estimators"] * params["max_depth"]
        scored.append((float(np.median(maes)), complexity, params))
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2]


def select_mlp(per_lake_inner):
    scored = []
    for params in MLP_GRID:
        maes = []
        for lake, (X_r, y_r, X_h, y_h) in per_lake_inner.items():
            model = MLPRegressor(activation="relu", learning_rate_init=1e-3,
                                  max_iter=2000, early_stopping=True,
                                  n_iter_no_change=20, random_state=0, **params)
            model.fit(X_r, y_r)
            pred = model.predict(X_h)
            maes.append(log_mae(y_h, pred))
        complexity = sum(params["hidden_layer_sizes"])
        scored.append((float(np.median(maes)), complexity, params))
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[0][2]


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


def process_outer_lake(target_lake, groups, band_cols, inner_lakes,
                       additional_source_exclusions=None):
    """Full nested pipeline for one outer target: inner-23 validation-based
    HP selection, final refit on all other GLORIA groups, then k=0/1/3/5/10
    evaluation on the target."""
    target_df, source_df = gloria.split_outer_target_source(
        groups, target_lake, additional_source_exclusions)

    X_source_raw = source_df[band_cols].to_numpy(dtype=float)
    y_source_log = source_df["log10_TSS"].to_numpy(dtype=float)
    X_target_raw = target_df[band_cols].to_numpy(dtype=float)
    y_target = target_df["TSS"].to_numpy(dtype=float)
    y_target_log = target_df["log10_TSS"].to_numpy(dtype=float)

    # Standardization is fit on all source groups after target-name exclusion.
    mu, sigma = X_source_raw.mean(axis=0), X_source_raw.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_source = (X_source_raw - mu) / sigma
    X_target = (X_target_raw - mu) / sigma

    # --- inner-23 LOLO hyperparameter selection (outer lake never touched) ---
    per_lake_inner = {}
    for lake in inner_lakes:
        is_l = source_df["Site_name"] == lake
        held_df = source_df[is_l].reset_index(drop=True)
        rest_df = source_df[~is_l].reset_index(drop=True)
        X_rest_raw = rest_df[band_cols].to_numpy(dtype=float)
        y_rest_log = rest_df["log10_TSS"].to_numpy(dtype=float)
        X_held_raw = held_df[band_cols].to_numpy(dtype=float)
        y_held_log = held_df["log10_TSS"].to_numpy(dtype=float)
        mu_i, sigma_i = X_rest_raw.mean(axis=0), X_rest_raw.std(axis=0)
        sigma_i[sigma_i == 0] = 1.0
        per_lake_inner[lake] = (
            (X_rest_raw - mu_i) / sigma_i, y_rest_log,
            (X_held_raw - mu_i) / sigma_i, y_held_log,
        )

    t_hp0 = time.time()
    plsr_nc = select_plsr(per_lake_inner)
    xgb_params = select_xgb(per_lake_inner)
    mlp_params = select_mlp(per_lake_inner)
    hp_selection_seconds = time.time() - t_hp0

    hp_record = {
        "outer_lake": target_lake,
        "plsr_n_components": plsr_nc,
        "xgb_params": xgb_params,
        "mlp_params": mlp_params,
        "hp_selection_seconds": hp_selection_seconds,
    }
    if additional_source_exclusions:
        hp_record["additional_source_exclusions"] = "|".join(
            sorted(set(additional_source_exclusions)))
        hp_record["source_group_count"] = len(source_df)

    # --- final refit on all other GLORIA source groups using outer-specific HP ---
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

    # --- k=0/1/3/5/10 evaluation, IDENTICAL seeding/logic to run_capacity_boundary.py ---
    rows = []
    rng = np.random.default_rng(RNG_SEED + zlib.crc32(target_lake.encode()) % 10_000)
    for rep in range(N_REPEATS):
        for k in K_LEVELS:
            if k == 0:
                support_idx = np.array([], dtype=int)
                query_idx = np.arange(len(target_df))
            else:
                support_idx, query_idx = sampling.random_support_query_split(
                    np.arange(len(target_df)), k=k, rng=rng)

            y_qry, y_qry_log = y_target[query_idx], y_target_log[query_idx]
            if k > 0:
                y_sup, y_sup_log = y_target[support_idx], y_target_log[support_idx]
                lbl = baselines.label_only_baselines(y_sup)
                best_label_mae_log = log_mae(
                    y_qry_log, np.full_like(y_qry_log, np.log10(lbl["label_median"])))
                best_label_mae_orig = metrics.mae(
                    y_qry, np.full_like(y_qry, lbl["label_median"]))
            else:
                best_label_mae_log = best_label_mae_orig = np.nan

            for learner_name, base_pred_target_log in base_preds_target.items():
                base_pred_qry_log = base_pred_target_log[query_idx]

                def record(pred_log, method):
                    pred_orig = baselines.safe_pow10(pred_log)
                    skill = metrics.lake_demeaned_skill(y_qry, pred_orig)
                    rows.append({
                        "lake": target_lake, "learner": learner_name,
                        "k": k, "rep": rep, "method": method,
                        "mae_log": log_mae(y_qry_log, pred_log),
                        "mae_orig": metrics.mae(y_qry, pred_orig),
                        "rer_log": (metrics.relative_error_reduction(
                            best_label_mae_log, log_mae(y_qry_log, pred_log))
                            if k > 0 else np.nan),
                        "demeaned_spearman": skill["spearman"],
                        "anomaly_r2": skill["anomaly_r2"],
                    })

                if k == 0:
                    record(base_pred_qry_log, "source_only")
                    continue

                base_pred_sup_log = base_pred_target_log[support_idx]
                repr_sup = repr_scores_target[support_idx]
                repr_qry = repr_scores_target[query_idx]

                if k == 1:
                    b_l = baselines.intercept_only_calibration(base_pred_sup_log, y_sup_log)
                    record(base_pred_qry_log + b_l, "calibrated")
                else:
                    z_sup = z_features_for_k(k, base_pred_sup_log, repr_sup)
                    z_qry = z_features_for_k(k, base_pred_qry_log, repr_qry)
                    resid_sup_log = y_sup_log - base_pred_sup_log
                    ridge = baselines.residual_ridge_calibration(z_sup, resid_sup_log, alpha=1.0)
                    record(base_pred_qry_log + ridge.predict(z_qry).ravel(), "calibrated")

    return pd.DataFrame(rows), hp_record


def run(merge_only=False):
    NESTED_DIR.mkdir(parents=True, exist_ok=True)

    meta = gloria.load_meta()
    rrs = gloria.load_rrs()
    band_cols = gloria.rrs_band_columns(rrs, lo=400, hi=750)
    groups = gloria.build_tss_groups(meta, rrs, band_cols)
    groups["log10_TSS"] = np.log10(groups["TSS"].clip(lower=1e-3))
    eligible = gloria.eligible_tss_lakes(groups, min_group=MIN_GROUP, lakes_reservoirs_only=True)
    lake_list = [l for l in eligible.index if l not in STRICT_24_EXCLUDE]
    print(f"Strict-24 lake pool: {len(lake_list)} lakes")

    if not merge_only:
        t0 = time.time()
        for li, target_lake in enumerate(lake_list):
            safe_name = target_lake.replace(" ", "_").replace("/", "_")
            repeats_ckpt = NESTED_DIR / f"{safe_name}_repeats.csv"
            hp_ckpt = NESTED_DIR / f"{safe_name}_hp.json"
            if repeats_ckpt.exists() and hp_ckpt.exists():
                print(f"  [{li+1}/{len(lake_list)}] {target_lake}: checkpoint exists, skipping")
                continue

            inner_lakes = [l for l in lake_list if l != target_lake]
            t_lake0 = time.time()
            repeats_df, hp_record = process_outer_lake(target_lake, groups, band_cols, inner_lakes)
            repeats_df.to_csv(repeats_ckpt, index=False)
            with open(hp_ckpt, "w") as f:
                json.dump(hp_record, f, indent=2)
            print(f"  [{li+1}/{len(lake_list)}] {target_lake}: done in "
                  f"{time.time()-t_lake0:.0f}s (total elapsed {time.time()-t0:.0f}s) "
                  f"-- PLSR nc={hp_record['plsr_n_components']}, "
                  f"XGB={hp_record['xgb_params']}, MLP={hp_record['mlp_params']}")

    # --- merge all per-lake checkpoints into combined outputs ---
    missing = [l for l in lake_list
               if not (NESTED_DIR / f"{l.replace(' ', '_').replace('/', '_')}_repeats.csv").exists()]
    if missing:
        print(f"\nWARNING: {len(missing)} outer lakes have no checkpoint yet: {missing}")
        print("Merge will only include completed lakes. Re-run this script to finish the rest.")

    all_repeats = []
    hp_rows = []
    for target_lake in lake_list:
        safe_name = target_lake.replace(" ", "_").replace("/", "_")
        repeats_ckpt = NESTED_DIR / f"{safe_name}_repeats.csv"
        hp_ckpt = NESTED_DIR / f"{safe_name}_hp.json"
        if repeats_ckpt.exists():
            all_repeats.append(pd.read_csv(repeats_ckpt))
        if hp_ckpt.exists():
            with open(hp_ckpt) as f:
                hp_rows.append(json.load(f))

    if not all_repeats:
        print("No checkpoints found yet; nothing to merge.")
        return

    repeats = pd.concat(all_repeats, ignore_index=True)
    repeats.to_csv(RESULTS_DIR / "capacity_boundary_repeats_nested.csv", index=False)
    summary = repeats.groupby(["lake", "learner", "k", "method"]).agg(
        median_rer_log=("rer_log", "median"),
        median_mae_log=("mae_log", "median"),
        median_demeaned_spearman=("demeaned_spearman", "median"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "capacity_boundary_summary_nested.csv", index=False)
    pd.DataFrame(hp_rows).to_csv(RESULTS_DIR / "nested_hyperparameters.csv", index=False)
    print(f"\nMerged {len(all_repeats)}/{len(lake_list)} outer lakes into "
          f"capacity_boundary_{{repeats,summary}}_nested.csv + nested_hyperparameters.csv")


if __name__ == "__main__":
    run(merge_only=("--merge-only" in sys.argv))
