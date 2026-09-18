r"""Phase 4: True-Nested H2 rerun (FROZEN protocol, see true_nested_h2_protocol.yaml).

Three evaluation layers (estimand hierarchy frozen pre-run):
  1. PRIMARY    : own-remaining-pool direct Delta-MAE, weighted-D vs random
  2. SENSITIVITY: pairwise common-query (weighted/unweighted/KS each vs random)
  3. RESTRICTED : four-way common-query (all 4 selectors on shared query)

k grid: {1,3,5,10} (k=0 has no support selection; source-only kept as reference
only, NOT in the 12-cell selector inference).

Base models: outer-specific true-nested (rebuilt from frozen nested_hyperparameters.csv
per lake, random_state=0). This matches the H1/H3 protocol (Cawley & Talbot 2010
selection-bias correction). No HP re-search, no ridge alpha change, no new selector,
no weight formula change.

Deterministic selectors (KS, unweighted-D, weighted-D) produce a SINGLE support per
lake-k; random uses 100 repeats. For the pairwise/four-way common-query layers, the
random selector's r-th repeat support is paired with the deterministic supports under
Q_sR(r) = pool \ (S_s union S_random(r)) and Q_all(r) = pool \ (S_random(r) union S_KS
union S_unweighted_D union S_weighted_D).

Output isolation: writes ONLY to results/true_nested_h2/, never overwrites old H2.
"""
import argparse
import ast
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lake_insitu import gloria, baselines, metrics, sampling, selection  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
NESTED_RESULTS = Path(__file__).resolve().parents[1] / "capacity_boundary" / "results"
OUTPUT_DIR = RESULTS_DIR / "true_nested_h2"
FROZEN_REPEATS_CSV = NESTED_RESULTS / "capacity_boundary_repeats_nested.csv"
HP_CSV = NESTED_RESULTS / "nested_hyperparameters.csv"

RNG_SEED = 20260712
K_LEVELS = [1, 3, 5, 10]
N_RANDOM_REPEATS = 100
MIN_GROUP = 20
REPR_PLS_COMPONENTS = 2
STRICT_24_EXCLUDE = {"Ba Be Lake", "Lake Constance"}
ALPHA_WEIGHT = 1.0
BETA_WEIGHT = 1.0
SELF_CHECK_TOL = 1e-9

# Four-way common-query availability thresholds (frozen)
FOURWAY_MIN_Q_ABS = 5
FOURWAY_MIN_Q_FRAC = 0.20
FOURWAY_MIN_USABLE_REPEATS = 50
SPEARMAN_MIN_Q = 10


def log_mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


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


def calibrate_and_eval(support_idx, query_idx, y_target, y_target_log,
                        base_pred_log, affine_z, repr_scores, k):
    """Return (mae_log, rer_log, label_mae_log) on the given query."""
    y_qry, y_qry_log = y_target[query_idx], y_target_log[query_idx]
    y_sup_log = y_target_log[support_idx]
    base_pred_sup_log = base_pred_log[support_idx]
    base_pred_qry_log = base_pred_log[query_idx]

    lbl = baselines.label_only_baselines(y_target[support_idx])
    label_mae_log = log_mae(
        y_qry_log, np.full_like(y_qry_log, np.log10(lbl["label_median"])))

    if k == 1:
        b_l = baselines.intercept_only_calibration(base_pred_sup_log, y_sup_log)
        pred_log = base_pred_qry_log + b_l
    else:
        z_sup = z_features_for_k(k, affine_z[support_idx], repr_scores[support_idx])
        z_qry = z_features_for_k(k, affine_z[query_idx], repr_scores[query_idx])
        resid_sup_log = y_sup_log - base_pred_sup_log
        ridge = baselines.residual_ridge_calibration(z_sup, resid_sup_log, alpha=1.0)
        pred_log = base_pred_qry_log + ridge.predict(z_qry).ravel()

    mae_log = log_mae(y_qry_log, pred_log)
    rer_log = metrics.relative_error_reduction(label_mae_log, mae_log)
    return mae_log, rer_log, label_mae_log


def build_base_learners(X_source, y_source_log, hp_row):
    """Rebuild the three base learners from frozen outer-specific HP."""
    plsr_nc = int(hp_row["plsr_n_components"])
    xgb_params = ast.literal_eval(hp_row["xgb_params"])
    mlp_params = ast.literal_eval(hp_row["mlp_params"])
    mlp_params["hidden_layer_sizes"] = tuple(mlp_params["hidden_layer_sizes"])
    return {
        "PLSR": baselines.fit_plsr_source(X_source, y_source_log, n_components=plsr_nc),
        "XGBoost": XGBRegressor(random_state=0, n_jobs=-1, **xgb_params).fit(X_source, y_source_log),
        "MLP": MLPRegressor(activation="relu", learning_rate_init=1e-3, max_iter=2000,
                            early_stopping=True, n_iter_no_change=20, random_state=0,
                            **mlp_params).fit(X_source, y_source_log),
    }


def process_lake(target_lake, groups, band_cols, hp, frozen_source,
                  learner_filter=None, k_filter=None,
                  max_reps=None, verbose=True,
                  additional_source_exclusions=None):
    """Process one outer lake through all three estimand layers.

    frozen_source: dict mapping learner_name -> frozen k=0/source_only row
                   (for base-model reconstruction self-check).
    Returns (rows, self_check_rows).
    """
    target_df, source_df = gloria.split_outer_target_source(
        groups, target_lake, additional_source_exclusions)
    X_source_raw = source_df[band_cols].to_numpy(dtype=float)
    y_source_log = source_df["log10_TSS"].to_numpy(dtype=float)
    X_target_raw = target_df[band_cols].to_numpy(dtype=float)
    y_target = target_df["TSS"].to_numpy(dtype=float)
    y_target_log = target_df["log10_TSS"].to_numpy(dtype=float)
    n_pool = len(target_df)
    pool_idx = np.arange(n_pool)

    mu, sigma = X_source_raw.mean(axis=0), X_source_raw.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_source = (X_source_raw - mu) / sigma
    X_target = (X_target_raw - mu) / sigma

    hp_row = hp.loc[target_lake]
    repr_pls = baselines.fit_plsr_source(X_source, y_source_log,
                                          n_components=REPR_PLS_COMPONENTS)
    repr_scores_target = repr_pls.transform(X_target)
    s_score = selection.support_score(repr_scores_target,
                                       repr_pls.transform(X_source), k=10)
    d_score = selection.target_density_score(repr_scores_target, k=10)
    eps = 1e-6
    weight = (1.0 / (s_score + eps)) ** ALPHA_WEIGHT * (1.0 / (d_score + eps)) ** BETA_WEIGHT

    learners = build_base_learners(X_source, y_source_log, hp_row)

    # Deterministic selector orderings (computed once per lake, learner-independent).
    ks_order = selection.farthest_point_ks_order(repr_scores_target)
    unweighted_order = selection.greedy_d_optimal_order(
        repr_scores_target, weights=None, max_k=max(K_LEVELS))
    weighted_order = selection.greedy_d_optimal_order(
        repr_scores_target, weights=weight, max_k=max(K_LEVELS))

    rng = np.random.default_rng(RNG_SEED + zlib.crc32(target_lake.encode()) % 10_000)

    learners_to_run = learner_filter or ["PLSR", "XGBoost", "MLP"]
    ks_to_run = k_filter or K_LEVELS
    reps_to_run = max_reps or N_RANDOM_REPEATS
    rows = []
    self_check_rows = []

    for learner_name in learners_to_run:
        base_pred_log = np.asarray(learners[learner_name].predict(X_target)).ravel()
        affine_z = base_pred_log

        # Self-check: source-only base-model prediction (no calibration, whole
        # lake as query) must reproduce the H1 nested k=0/source_only mae_log.
        # This verifies that the outer-specific HP + 23-lake rebuild + repr-PLS
        # reconstruction is correct. It does NOT verify random support identity
        # (H2 and H1 use DIFFERENT RNG loop structures -- H2 is
        # for-learner/for-k/for-rep, H1 is for-rep/for-k -- so H2 random support
        # is not expected to match H1 random support; only the base model must).
        source_mae_log = log_mae(y_target_log, base_pred_log)
        if learner_name in frozen_source:
            fr_src = frozen_source[learner_name]
            self_check_rows.append({
                "lake": target_lake, "learner": learner_name, "k": 0, "rep": -1,
                "mae_log_diff": abs(source_mae_log - fr_src["mae_log"]),
                "rer_log_diff": 0.0,  # rer is NaN for k=0; not checked
            })

        for k in ks_to_run:
            # Pre-compute the three deterministic supports for this k (single support each).
            det_supports = {
                "farthest_point_kennard_stone": ks_order[:k],
                "unweighted_d_optimal": unweighted_order[:k],
                "proposed_weighted_d_optimal": weighted_order[:k],
            }
            det_own_queries = {
                m: np.array([i for i in range(n_pool) if i not in set(s)])
                for m, s in det_supports.items()
            }

            for rep in range(reps_to_run):
                # --- random support for this repeat ---
                random_support, random_own_query = sampling.random_support_query_split(
                    pool_idx, k=k, rng=rng)

                # --- Layer A: own-remaining-pool (PRIMARY) ---
                # random own-pool
                mae_r, rer_r, lbl_r = calibrate_and_eval(
                    random_support, random_own_query, y_target, y_target_log,
                    base_pred_log, affine_z, repr_scores_target, k)
                rows.append({
                    "lake": target_lake, "learner": learner_name, "k": k,
                    "estimand": "own_pool", "rep": rep,
                    "method": "repeated_random", "query_size": len(random_own_query),
                    "mae_log": mae_r, "rer_log": rer_r, "label_mae_log": lbl_r,
                })

                # Self-check: random own-pool must reproduce frozen nested H1 values
                # (mae_log / rer_log; the frozen file stores calibrated method on the
                # same random own-query, since that IS the H1 evaluation).
                # NOTE: this check was REMOVED -- H2 and H1 use different RNG loop
                # structures (see comment above), so H2 random support does not match
                # H1 random support. Base-model reconstruction is verified separately
                # via the source-only check above.

                # deterministic selectors own-pool (single rep, but record under rep for joinability)
                for m, s in det_supports.items():
                    q = det_own_queries[m]
                    mae_d, rer_d, lbl_d = calibrate_and_eval(
                        s, q, y_target, y_target_log, base_pred_log, affine_z,
                        repr_scores_target, k)
                    rows.append({
                        "lake": target_lake, "learner": learner_name, "k": k,
                        "estimand": "own_pool", "rep": rep,
                        "method": m, "query_size": len(q),
                        "mae_log": mae_d, "rer_log": rer_d, "label_mae_log": lbl_d,
                    })

                # --- Layer B: pairwise common-query (SENSITIVITY) ---
                # Q_sR(r) = pool \ (S_s union S_random(r)); evaluate s AND random on it.
                random_set = set(random_support)
                for m, s in det_supports.items():
                    s_set = set(s)
                    pairwise_query = np.array(
                        [i for i in range(n_pool) if i not in s_set and i not in random_set])
                    if len(pairwise_query) == 0:
                        continue
                    # deterministic method s on the shared pairwise query
                    mae_s, rer_s, lbl_s = calibrate_and_eval(
                        s, pairwise_query, y_target, y_target_log, base_pred_log,
                        affine_z, repr_scores_target, k)
                    # random on the SAME shared pairwise query
                    mae_rand_pq, rer_rand_pq, lbl_rand_pq = calibrate_and_eval(
                        random_support, pairwise_query, y_target, y_target_log,
                        base_pred_log, affine_z, repr_scores_target, k)
                    rows.append({
                        "lake": target_lake, "learner": learner_name, "k": k,
                        "estimand": "pairwise_common_query", "rep": rep,
                        "method": m, "contrast": f"{m}_vs_random",
                        "query_size": len(pairwise_query),
                        "mae_log_method": mae_s, "mae_log_random": mae_rand_pq,
                        "delta_mae": mae_s - mae_rand_pq,
                        "rer_log_method": rer_s, "rer_log_random": rer_rand_pq,
                    })

                # --- Layer C: four-way common-query (RESTRICTED) ---
                # Q_all(r) = pool \ (S_random(r) union S_KS union S_D union S_W)
                all_excluded = random_set | set(ks_order[:k]) | set(unweighted_order[:k]) | set(weighted_order[:k])
                fourway_query = np.array(
                    [i for i in range(n_pool) if i not in all_excluded])
                qsize = len(fourway_query)
                usable = (qsize >= FOURWAY_MIN_Q_ABS
                          and qsize / n_pool >= FOURWAY_MIN_Q_FRAC)
                for m, s in det_supports.items():
                    if qsize == 0:
                        mae_m = rer_m = mae_rand = rer_rand = np.nan
                    else:
                        mae_m, rer_m, _ = calibrate_and_eval(
                            s, fourway_query, y_target, y_target_log,
                            base_pred_log, affine_z, repr_scores_target, k)
                        mae_rand, rer_rand, _ = calibrate_and_eval(
                            random_support, fourway_query, y_target, y_target_log,
                            base_pred_log, affine_z, repr_scores_target, k)
                    rows.append({
                        "lake": target_lake, "learner": learner_name, "k": k,
                        "estimand": "fourway_common_query", "rep": rep,
                        "method": m, "query_size": qsize,
                        "usable": bool(usable),
                        "mae_log_method": mae_m, "mae_log_random": mae_rand,
                        "delta_mae": (mae_m - mae_rand
                                      if not (np.isnan(mae_m) or np.isnan(mae_rand))
                                      else np.nan),
                        "rer_log_method": rer_m, "rer_log_random": rer_rand,
                    })

        if verbose:
            print(f"    {target_lake} / {learner_name}: done")

    return rows, self_check_rows


def run_full():
    meta = gloria.load_meta()
    rrs = gloria.load_rrs()
    band_cols = gloria.rrs_band_columns(rrs, lo=400, hi=750)
    groups = gloria.build_tss_groups(meta, rrs, band_cols)
    groups["log10_TSS"] = np.log10(groups["TSS"].clip(lower=1e-3))
    eligible = gloria.eligible_tss_lakes(groups, min_group=MIN_GROUP, lakes_reservoirs_only=True)
    lake_list = [l for l in eligible.index if l not in STRICT_24_EXCLUDE]

    hp = pd.read_csv(HP_CSV).set_index("outer_lake")
    frozen = pd.read_csv(FROZEN_REPEATS_CSV)
    # Self-check uses the H1 nested source_only rows (method=source_only, k=0):
    # whole-lake query, no calibration -- this isolates base-model reconstruction
    # from the RNG-loop-structure difference between H2 and H1 (see process_lake
    # docstring). One source_only row per (lake, learner) at k=0/rep=0.
    frozen_src = frozen[frozen.method == "source_only"].set_index(
        ["lake", "learner"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_self_check = []
    hp_provenance = []
    t0 = time.time()

    for li, target_lake in enumerate(lake_list):
        hp_row = hp.loc[target_lake]
        hp_provenance.append({
            "outer_lake": target_lake,
            "plsr_n_components": int(hp_row["plsr_n_components"]),
            "xgb_params": hp_row["xgb_params"],
            "mlp_params": hp_row["mlp_params"],
        })
        # Index frozen source_only rows for THIS lake (one per learner).
        lake_frozen_src = {}
        try:
            lake_src_df = frozen_src.loc[target_lake]
            if isinstance(lake_src_df, pd.Series):  # single-learner edge case
                lake_frozen_src = {lake_src_df.name: lake_src_df}
            else:
                for learner, fr in lake_src_df.iterrows():
                    lake_frozen_src[learner] = fr
        except KeyError:
            pass

        rows, sc = process_lake(
            target_lake, groups, band_cols, hp, lake_frozen_src)
        all_rows.extend(rows)
        all_self_check.extend(sc)
        print(f"  [{li+1}/{len(lake_list)}] {target_lake} done -- "
              f"{time.time()-t0:.0f}s elapsed, {len(all_rows)} rows")

    per_repeat = pd.DataFrame(all_rows)
    per_repeat.to_csv(OUTPUT_DIR / "h2_nested_per_repeat.csv", index=False)

    sc_df = pd.DataFrame(all_self_check)
    sc_df.to_csv( OUTPUT_DIR / "h2_nested_self_check_detail.csv", index=False)
    max_mae_diff = sc_df.mae_log_diff.max() if len(sc_df) else 0.0
    max_rer_diff = sc_df.rer_log_diff.max() if len(sc_df) else 0.0
    n_over = int(((sc_df.mae_log_diff > SELF_CHECK_TOL) |
                  (sc_df.rer_log_diff > SELF_CHECK_TOL)).sum())
    print(f"\n=== Self-check (source-only base model vs frozen nested H1 k=0, tol {SELF_CHECK_TOL:.0e}) ===")
    print(f"  max mae_log_diff = {max_mae_diff:.3e}; max rer_log_diff = {max_rer_diff:.3e}")
    print(f"  {n_over}/{len(sc_df)} rows over tolerance")

    pd.DataFrame(hp_provenance).to_csv(
        OUTPUT_DIR / "h2_nested_hyperparameter_provenance.csv", index=False)
    print(f"\nSaved per_repeat ({len(per_repeat)} rows) + self_check + hp_provenance to {OUTPUT_DIR}")
    print(f"Total time: {time.time()-t0:.0f}s")


def run_smoke():
    """Smoke test: XGBoost, k=5, two lakes with different HP. Must verify the
    8 self-checks in the protocol before full run."""
    meta = gloria.load_meta()
    rrs = gloria.load_rrs()
    band_cols = gloria.rrs_band_columns(rrs, lo=400, hi=750)
    groups = gloria.build_tss_groups(meta, rrs, band_cols)
    groups["log10_TSS"] = np.log10(groups["TSS"].clip(lower=1e-3))
    eligible = gloria.eligible_tss_lakes(groups, min_group=MIN_GROUP, lakes_reservoirs_only=True)
    lake_list = [l for l in eligible.index if l not in STRICT_24_EXCLUDE]

    hp = pd.read_csv(HP_CSV).set_index("outer_lake")
    frozen = pd.read_csv(FROZEN_REPEATS_CSV)
    frozen_src = frozen[frozen.method == "source_only"].set_index(["lake", "learner"])

    # Pick two lakes with DIFFERENT XGBoost HP.
    import ast as _ast
    xgb_hp_by_lake = {l: _ast.literal_eval(hp.loc[l, "xgb_params"])["n_estimators"]
                       for l in lake_list}
    distinct_pair = None
    lakes_sorted = sorted(lake_list, key=lambda l: xgb_hp_by_lake[l])
    for i, l1 in enumerate(lakes_sorted):
        for l2 in lakes_sorted[i+1:]:
            if xgb_hp_by_lake[l1] != xgb_hp_by_lake[l2]:
                distinct_pair = (l1, l2)
                break
        if distinct_pair:
            break
    l1, l2 = distinct_pair
    print(f"=== SMOKE TEST: XGBoost, k=5, lakes={l1} (xgb_n_est={xgb_hp_by_lake[l1]}) "
          f"& {l2} (xgb_n_est={xgb_hp_by_lake[l2]}) ===")

    all_sc = []
    for target_lake in [l1, l2]:
        lake_frozen_src = {}
        try:
            lake_src_df = frozen_src.loc[target_lake]
            if isinstance(lake_src_df, pd.Series):
                lake_frozen_src = {lake_src_df.name: lake_src_df}
            else:
                for learner, fr in lake_src_df.iterrows():
                    lake_frozen_src[learner] = fr
        except KeyError:
            pass
        _, sc = process_lake(
            target_lake, groups, band_cols, hp, lake_frozen_src,
            learner_filter=["XGBoost"], k_filter=[5], max_reps=5, verbose=False)
        all_sc.extend(sc)

    sc_df = pd.DataFrame(all_sc)
    print(f"\n--- Smoke self-check (source-only base model vs frozen nested H1 k=0) ---")
    print(f"  rows: {len(sc_df)}; max mae_log_diff = {sc_df.mae_log_diff.max():.3e}")
    n_over = int((sc_df.mae_log_diff > SELF_CHECK_TOL).sum())
    print(f"  {n_over}/{len(sc_df)} rows over tolerance {SELF_CHECK_TOL:.0e}")

    # Verdict
    if n_over == 0:
        print("\n  [SMOKE PASS] source-only base model reproduces frozen nested H1 k=0 within tol.")
        print("  This verifies outer-specific HP loading + 23-lake rebuild + repr-PLS.")
        print("  (Random-support RNG path differs from H1 by design -- see process_lake docstring.)")
        print("  Proceed to full run.")
    else:
        print("\n  [SMOKE FAIL] source-only over tolerance -- base-model rebuild is wrong; investigate.")
        print(sc_df[sc_df.mae_log_diff > SELF_CHECK_TOL].head().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="run smoke test (XGBoost k=5, 2 lakes, 5 reps) only")
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
    else:
        run_full()
