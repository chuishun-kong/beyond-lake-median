"""Gate B.1: targeted Jasień physical-identity boundary sensitivity.

Only the Lake Jasień Południowy outer fold is eligible for recomputation.
Primary frozen outputs are read-only and guarded by SHA-256 checks.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lake_insitu import baselines, gloria, metrics, sampling, selection

from experiments.active_selection import build_h2_nested_summaries as h2_summaries
from experiments.active_selection import run_h2_true_nested as h2_pipeline
from experiments.capacity_boundary import run_h3_paired_source_audit_nested as h3_pipeline
from experiments.capacity_boundary import run_nested_capacity_boundary as h1_pipeline


RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoint"
H2_HYBRID_SUMMARY_DIR = RESULTS_DIR / "h2_hybrid_summaries"
TARGET_LAKE = "Lake Jasień Południowy"
RELATED_SOURCE_SITE = "Lake Jasień Północny"
SOURCE_EXCLUSIONS = {TARGET_LAKE, RELATED_SOURCE_SITE}
RNG_SEED = 20260712
K_LEVELS = [1, 3, 5, 10]
N_REPEATS = 100

CAPACITY_RESULTS = REPO_ROOT / "experiments" / "capacity_boundary" / "results"
H2_PRIMARY_DIR = (
    REPO_ROOT / "experiments" / "active_selection" / "results" / "true_nested_h2"
)
PRIMARY_FILES = [
    CAPACITY_RESULTS / "capacity_boundary_repeats_nested.csv",
    CAPACITY_RESULTS / "capacity_boundary_summary_nested.csv",
    CAPACITY_RESULTS / "nested_hyperparameters.csv",
    CAPACITY_RESULTS / "h1_global_vs_nested_comparison.csv",
    CAPACITY_RESULTS / "h3_paired_source_audit_nested_per_rep.csv",
    CAPACITY_RESULTS / "h3_paired_source_audit_nested_per_lake.csv",
    CAPACITY_RESULTS / "h3_paired_source_audit_nested_summary_fullprecision.csv",
    H2_PRIMARY_DIR / "h2_nested_per_repeat.csv",
    H2_PRIMARY_DIR / "h2_nested_per_lake.csv",
    H2_PRIMARY_DIR / "h2_nested_own_pool_summary.csv",
    H2_PRIMARY_DIR / "h2_nested_pairwise_common_query.csv",
    H2_PRIMARY_DIR / "h2_nested_fourway_common_query.csv",
]


def h1_random_split_signatures(group_ids):
    """Return the frozen H1/H3 rep-major random support/query trajectory."""
    group_ids = list(group_ids)
    rng = np.random.default_rng(
        RNG_SEED + zlib.crc32(TARGET_LAKE.encode()) % 10_000
    )
    rows = []
    for rep in range(N_REPEATS):
        for k in K_LEVELS:
            support_idx, query_idx = sampling.random_support_query_split(
                np.arange(len(group_ids)), k=k, rng=rng
            )
            rows.append(
                {
                    "rep": rep,
                    "k": k,
                    "support_count": len(support_idx),
                    "query_count": len(query_idx),
                    "support_group_ids": json.dumps(
                        [group_ids[i] for i in support_idx],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "query_group_ids": json.dumps(
                        [group_ids[i] for i in query_idx],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
    return pd.DataFrame(rows)


def h2_random_split_signatures(group_ids):
    """Return the frozen H2 learner-major/k-major random trajectory."""
    group_ids = list(group_ids)
    rng = np.random.default_rng(
        RNG_SEED + zlib.crc32(TARGET_LAKE.encode()) % 10_000
    )
    rows = []
    for learner in ["PLSR", "XGBoost", "MLP"]:
        for k in K_LEVELS:
            for rep in range(N_REPEATS):
                support_idx, query_idx = sampling.random_support_query_split(
                    np.arange(len(group_ids)), k=k, rng=rng
                )
                rows.append(
                    {
                        "learner": learner,
                        "k": k,
                        "rep": rep,
                        "support_count": len(support_idx),
                        "query_count": len(query_idx),
                        "support_group_ids": json.dumps(
                            [group_ids[i] for i in support_idx],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "query_group_ids": json.dumps(
                            [group_ids[i] for i in query_idx],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def replace_target_rows(primary, sensitivity, *, target_lake=TARGET_LAKE,
                        lake_column="lake", key_columns=None):
    """Replace exactly one outer lake in a copy of a primary table."""
    primary = primary.copy()
    sensitivity = sensitivity.copy()
    primary_target = primary[primary[lake_column] == target_lake]
    sensitivity_target = sensitivity[sensitivity[lake_column] == target_lake]
    if len(primary_target) != len(sensitivity_target):
        raise AssertionError(
            f"Target replacement row mismatch: {len(primary_target)} != "
            f"{len(sensitivity_target)}"
        )

    if key_columns:
        left_keys = primary_target[key_columns].astype(str).agg("||".join, axis=1)
        right_keys = sensitivity_target[key_columns].astype(str).agg("||".join, axis=1)
        if set(left_keys) != set(right_keys):
            raise AssertionError("Target replacement keys do not match primary")
        sensitivity_target = sensitivity_target.assign(_replace_key=right_keys.values)
        sensitivity_target = sensitivity_target.set_index("_replace_key").loc[left_keys].reset_index(drop=True)

    primary.loc[primary_target.index, primary.columns] = sensitivity_target[
        primary.columns
    ].to_numpy()
    return primary


def build_h1_aggregate(per_lake_summary):
    """Reproduce the frozen true-nested H1 CI (fresh seed 0 per cell)."""
    calibrated = per_lake_summary[per_lake_summary["method"] == "calibrated"]
    rows = []
    for learner in ["PLSR", "XGBoost", "MLP"]:
        for k in K_LEVELS:
            cell = calibrated[
                (calibrated["learner"] == learner) & (calibrated["k"] == k)
            ]
            rer = cell["median_rer_log"].to_numpy(dtype=float)
            ci_lo, ci_hi = metrics.bootstrap_lake_ci(
                rer, n_boot=10_000, rng=np.random.default_rng(0)
            )
            rows.append(
                {
                    "learner": learner,
                    "k": k,
                    "median_rer_pct": round(float(np.median(rer)) * 100, 2),
                    "mean_rer_pct": round(float(np.mean(rer)) * 100, 2),
                    "ci_lo_pct": round(ci_lo * 100, 2),
                    "ci_hi_pct": round(ci_hi * 100, 2),
                    "zero_status": (
                        "excludes_zero_positive"
                        if ci_lo > 0
                        else "excludes_zero_negative"
                        if ci_hi < 0
                        else "crosses_zero"
                    ),
                    "n_positive_lakes": int((rer > 0).sum()),
                    "n_lakes": len(rer),
                }
            )
    return pd.DataFrame(rows)


def _mean_bootstrap_ci(values, n_boot=10_000, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array(
        [np.mean(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)]
    )
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def build_h3_aggregate(per_lake):
    """Reproduce the frozen H3 full-precision mean-bootstrap summary."""
    rows = []
    for learner in ["PLSR", "XGBoost", "MLP"]:
        for k in K_LEVELS:
            sub = per_lake[
                (per_lake["learner"] == learner) & (per_lake["k"] == k)
            ]
            rho = sub["delta_spearman"].to_numpy(dtype=float)
            r2_log = sub["delta_anomaly_r2_logspace"].to_numpy(dtype=float)
            r2_orig = sub["delta_anomaly_r2_origspace"].to_numpy(dtype=float)
            rho_ci = _mean_bootstrap_ci(rho)
            r2_log_ci = _mean_bootstrap_ci(r2_log)
            r2_orig_ci = _mean_bootstrap_ci(r2_orig)
            if k == 1:
                classification = "identity"
            elif rho_ci[1] < -1e-12:
                classification = "negative"
            elif rho_ci[0] > 1e-12:
                classification = "improvement"
            else:
                classification = "unresolved"
            rows.append(
                {
                    "learner": learner,
                    "k": k,
                    "mean_delta_rho": float(np.mean(rho)),
                    "mean_delta_rho_ci_lo": rho_ci[0],
                    "mean_delta_rho_ci_hi": rho_ci[1],
                    "mean_delta_R2_logspace": float(np.mean(r2_log)),
                    "mean_delta_R2_logspace_ci_lo": r2_log_ci[0],
                    "mean_delta_R2_logspace_ci_hi": r2_log_ci[1],
                    "mean_delta_R2_origspace": float(np.mean(r2_orig)),
                    "mean_delta_R2_origspace_ci_lo": r2_orig_ci[0],
                    "mean_delta_R2_origspace_ci_hi": r2_orig_ci[1],
                    "classification": classification,
                    "n_lakes": len(sub),
                }
            )
    return pd.DataFrame(rows)


def _write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.15g",
    )


def _load_groups():
    meta = gloria.load_meta(REPO_ROOT / "data" / "raw" / "gloria" / "GLORIA_2022")
    rrs = gloria.load_rrs(REPO_ROOT / "data" / "raw" / "gloria" / "GLORIA_2022")
    band_cols = gloria.rrs_band_columns(rrs, lo=400, hi=750)
    groups = gloria.build_tss_groups(meta, rrs, band_cols)
    groups["log10_TSS"] = np.log10(groups["TSS"].clip(lower=1e-3))
    eligible = gloria.eligible_tss_lakes(
        groups, min_group=20, lakes_reservoirs_only=True
    )
    lake_list = [
        lake
        for lake in eligible.index
        if lake not in {"Ba Be Lake", "Lake Constance"}
    ]
    if len(lake_list) != 24 or TARGET_LAKE not in lake_list:
        raise AssertionError("Strict-24 outer population does not match the frozen protocol")
    return groups, band_cols, lake_list


def _h1_detail(repeats, target_df):
    group_ids = target_df["group_id"].tolist()
    signatures = h1_random_split_signatures(group_ids).set_index(["rep", "k"])
    y = target_df["TSS"].to_numpy(dtype=float)
    y_log = target_df["log10_TSS"].to_numpy(dtype=float)
    rows = []
    calibrated = repeats[repeats["method"] == "calibrated"]
    for (rep, k), signature in signatures.iterrows():
        support_ids = json.loads(signature["support_group_ids"])
        query_ids = json.loads(signature["query_group_ids"])
        id_to_index = {group_id: index for index, group_id in enumerate(group_ids)}
        support_idx = np.array([id_to_index[group_id] for group_id in support_ids])
        query_idx = np.array([id_to_index[group_id] for group_id in query_ids])
        label = baselines.label_only_baselines(y[support_idx])["label_median"]
        label_mae = float(
            np.mean(np.abs(y_log[query_idx] - np.log10(label)))
        )
        for _, result in calibrated[
            (calibrated["rep"] == rep) & (calibrated["k"] == k)
        ].iterrows():
            rows.append(
                {
                    "lake": TARGET_LAKE,
                    "learner": result["learner"],
                    "k": int(k),
                    "rep": int(rep),
                    "support_count": int(signature["support_count"]),
                    "query_count": int(signature["query_count"]),
                    "label_only_mae_log": label_mae,
                    "calibrated_mae_log": float(result["mae_log"]),
                    "paired_delta_mae_log": label_mae - float(result["mae_log"]),
                    "rer_log": float(result["rer_log"]),
                }
            )
    detail = pd.DataFrame(rows)
    return (
        detail.groupby(["lake", "learner", "k"], sort=True)
        .agg(
            support_count=("support_count", "first"),
            query_count=("query_count", "first"),
            label_only_mae_log=("label_only_mae_log", "median"),
            calibrated_mae_log=("calibrated_mae_log", "median"),
            paired_delta_mae_log=("paired_delta_mae_log", "median"),
            rer_log=("rer_log", "median"),
        )
        .reset_index()
    )


def _h1_summary(repeats):
    return (
        repeats.groupby(["lake", "learner", "k", "method"], sort=True)
        .agg(
            median_rer_log=("rer_log", "median"),
            median_mae_log=("mae_log", "median"),
            median_demeaned_spearman=("demeaned_spearman", "median"),
        )
        .reset_index()
    )


def _wide_comparison(primary, sensitivity, keys, numeric_columns):
    left = primary[keys + numeric_columns].copy()
    right = sensitivity[keys + numeric_columns].copy()
    merged = left.merge(right, on=keys, suffixes=("_primary", "_sensitivity"), validate="one_to_one")
    for column in numeric_columns:
        merged[f"{column}_change"] = (
            merged[f"{column}_sensitivity"] - merged[f"{column}_primary"]
        )
    return merged


def _deterministic_h2_supports(groups, band_cols, additional_exclusions):
    target, source = gloria.split_outer_target_source(
        groups, TARGET_LAKE, additional_exclusions
    )
    x_source_raw = source[band_cols].to_numpy(dtype=float)
    y_source_log = source["log10_TSS"].to_numpy(dtype=float)
    x_target_raw = target[band_cols].to_numpy(dtype=float)
    mu = x_source_raw.mean(axis=0)
    sigma = x_source_raw.std(axis=0)
    sigma[sigma == 0] = 1.0
    x_source = (x_source_raw - mu) / sigma
    x_target = (x_target_raw - mu) / sigma
    representation = baselines.fit_plsr_source(x_source, y_source_log, n_components=2)
    target_scores = representation.transform(x_target)
    source_scores = representation.transform(x_source)
    support_score = selection.support_score(target_scores, source_scores, k=10)
    density_score = selection.target_density_score(target_scores, k=10)
    weight = (1.0 / (support_score + 1e-6)) * (1.0 / (density_score + 1e-6))
    orders = {
        "farthest_point_kennard_stone": selection.farthest_point_ks_order(target_scores),
        "unweighted_d_optimal": selection.greedy_d_optimal_order(
            target_scores, weights=None, max_k=max(K_LEVELS)
        ),
        "proposed_weighted_d_optimal": selection.greedy_d_optimal_order(
            target_scores, weights=weight, max_k=max(K_LEVELS)
        ),
    }
    group_ids = target["group_id"].tolist()
    rows = []
    for method, order in orders.items():
        for k in K_LEVELS:
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "support_group_ids": json.dumps(
                        [group_ids[i] for i in order[:k]],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
    return pd.DataFrame(rows)


def _selected_hp_frame(hp_record):
    row = {
        "outer_lake": TARGET_LAKE,
        "plsr_n_components": int(hp_record["plsr_n_components"]),
        "xgb_params": repr(hp_record["xgb_params"]),
        "mlp_params": repr(hp_record["mlp_params"]),
    }
    return pd.DataFrame([row]).set_index("outer_lake")


def _compute_h3_target(groups, band_cols, hp_frame, sensitivity_h1_repeats):
    target, source = gloria.split_outer_target_source(
        groups, TARGET_LAKE, {RELATED_SOURCE_SITE}
    )
    x_source_raw = source[band_cols].to_numpy(dtype=float)
    y_source_log = source["log10_TSS"].to_numpy(dtype=float)
    x_target_raw = target[band_cols].to_numpy(dtype=float)
    y_target = target["TSS"].to_numpy(dtype=float)
    y_target_log = target["log10_TSS"].to_numpy(dtype=float)
    mu = x_source_raw.mean(axis=0)
    sigma = x_source_raw.std(axis=0)
    sigma[sigma == 0] = 1.0
    x_source = (x_source_raw - mu) / sigma
    x_target = (x_target_raw - mu) / sigma
    representation = baselines.fit_plsr_source(x_source, y_source_log, n_components=2)
    target_representation = representation.transform(x_target)
    learners = h2_pipeline.build_base_learners(
        x_source, y_source_log, hp_frame.loc[TARGET_LAKE]
    )
    base_predictions = {
        learner: np.asarray(model.predict(x_target)).ravel()
        for learner, model in learners.items()
    }
    frozen_calibrated = sensitivity_h1_repeats[
        sensitivity_h1_repeats["method"] == "calibrated"
    ].set_index(["learner", "k", "rep"])
    rng = np.random.default_rng(
        RNG_SEED + zlib.crc32(TARGET_LAKE.encode()) % 10_000
    )
    rows = []
    checks = []
    for rep in range(N_REPEATS):
        for k in K_LEVELS:
            support_idx, query_idx = sampling.random_support_query_split(
                np.arange(len(target)), k=k, rng=rng
            )
            y_query = y_target[query_idx]
            y_query_log = y_target_log[query_idx]
            y_support_log = y_target_log[support_idx]
            label = baselines.label_only_baselines(y_target[support_idx])["label_median"]
            label_mae = float(np.mean(np.abs(y_query_log - np.log10(label))))

            for learner, base_prediction in base_predictions.items():
                base_support = base_prediction[support_idx]
                base_query = base_prediction[query_idx]
                if k == 1:
                    intercept = baselines.intercept_only_calibration(
                        base_support, y_support_log
                    )
                    calibrated_log = base_query + intercept
                else:
                    z_support = h1_pipeline.z_features_for_k(
                        k, base_support, target_representation[support_idx]
                    )
                    z_query = h1_pipeline.z_features_for_k(
                        k, base_query, target_representation[query_idx]
                    )
                    residual = y_support_log - base_support
                    ridge = baselines.residual_ridge_calibration(
                        z_support, residual, alpha=1.0
                    )
                    calibrated_log = base_query + ridge.predict(z_query).ravel()

                calibrated_original = baselines.safe_pow10(calibrated_log)
                source_original = baselines.safe_pow10(base_query)
                calibrated_skill = metrics.lake_demeaned_skill(
                    y_query, calibrated_original
                )
                source_skill = metrics.lake_demeaned_skill(y_query, source_original)
                calibrated_mae = float(np.mean(np.abs(y_query_log - calibrated_log)))
                calibrated_rer = metrics.relative_error_reduction(
                    label_mae, calibrated_mae
                )
                expected = frozen_calibrated.loc[(learner, k, rep)]
                checks.append(
                    {
                        "learner": learner,
                        "k": k,
                        "rep": rep,
                        "mae_log_diff": abs(calibrated_mae - expected["mae_log"]),
                        "rer_log_diff": abs(calibrated_rer - expected["rer_log"]),
                        "spearman_diff": abs(
                            calibrated_skill["spearman"]
                            - expected["demeaned_spearman"]
                        ),
                        "anomaly_r2_diff": abs(
                            calibrated_skill["anomaly_r2"] - expected["anomaly_r2"]
                        ),
                    }
                )
                calibrated_log_r2 = h3_pipeline.log_space_skill(
                    y_query_log, calibrated_log
                )
                source_log_r2 = h3_pipeline.log_space_skill(
                    y_query_log, base_query
                )
                rows.append(
                    {
                        "lake": TARGET_LAKE,
                        "learner": learner,
                        "k": k,
                        "rep": rep,
                        "calibrated_spearman": calibrated_skill["spearman"],
                        "source_spearman": source_skill["spearman"],
                        "delta_spearman": calibrated_skill["spearman"]
                        - source_skill["spearman"],
                        "calibrated_anomaly_r2_logspace": calibrated_log_r2,
                        "source_anomaly_r2_logspace": source_log_r2,
                        "delta_anomaly_r2_logspace": calibrated_log_r2
                        - source_log_r2,
                        "delta_anomaly_r2_origspace": calibrated_skill["anomaly_r2"]
                        - source_skill["anomaly_r2"],
                    }
                )
    checks = pd.DataFrame(checks)
    if checks[["mae_log_diff", "rer_log_diff", "spearman_diff", "anomaly_r2_diff"]].max().max() > 2e-9:
        raise AssertionError("H3 regeneration does not reproduce sensitivity H1")
    return pd.DataFrame(rows), checks


def _h3_per_lake(per_repeat):
    return (
        per_repeat.groupby(["lake", "learner", "k"], sort=True)
        .agg(
            calibrated_spearman=("calibrated_spearman", "median"),
            source_spearman=("source_spearman", "median"),
            delta_spearman=("delta_spearman", "median"),
            calibrated_anomaly_r2_logspace=(
                "calibrated_anomaly_r2_logspace",
                "median",
            ),
            source_anomaly_r2_logspace=("source_anomaly_r2_logspace", "median"),
            delta_anomaly_r2_logspace=("delta_anomaly_r2_logspace", "median"),
            delta_anomaly_r2_origspace=("delta_anomaly_r2_origspace", "median"),
        )
        .reset_index()
    )


def _run_h2_summary_builder(per_repeat_path, output_dir):
    original_per_rep = h2_summaries.PER_REP
    original_results_dir = h2_summaries.RESULTS_DIR
    try:
        h2_summaries.PER_REP = per_repeat_path
        h2_summaries.RESULTS_DIR = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        h2_summaries.run()
    finally:
        h2_summaries.PER_REP = original_per_rep
        h2_summaries.RESULTS_DIR = original_results_dir


def _comparison_with_status(primary, sensitivity, keys, value_columns, status_column):
    merged = _wide_comparison(primary, sensitivity, keys, value_columns)
    statuses = primary[keys + [status_column]].merge(
        sensitivity[keys + [status_column]],
        on=keys,
        suffixes=("_primary", "_sensitivity"),
        validate="one_to_one",
    )
    merged = merged.merge(statuses, on=keys, validate="one_to_one")
    merged[f"{status_column}_changed"] = (
        merged[f"{status_column}_primary"]
        != merged[f"{status_column}_sensitivity"]
    )
    return merged


def _classify_decision(
    h1_comparison,
    h2_comparison,
    h3_comparison,
    *,
    target_material_numeric=False,
):
    conclusion_change = bool(
        h1_comparison["zero_status_changed"].any()
        or h1_comparison.loc[h1_comparison["k"] == 5, "threshold_10pct_changed"].any()
        or h2_comparison["classification_changed"].any()
        or h3_comparison["classification_changed"].any()
    )
    if conclusion_change:
        return "CONCLUSION_CHANGING"
    material_numeric = bool(
        target_material_numeric
        or
        (h1_comparison["mean_rer_pct_change"].abs() > 1.0).any()
        or (h1_comparison["ci_lo_pct_change"].abs() > 1.0).any()
        or (h1_comparison["ci_hi_pct_change"].abs() > 1.0).any()
        or (h2_comparison["mean_delta_mae_change"].abs() > 0.005).any()
        or (h3_comparison["mean_delta_rho_change"].abs() > 0.01).any()
    )
    return (
        "NUMERICALLY_SENSITIVE_BUT_INTERPRETIVELY_STABLE"
        if material_numeric
        else "STABLE"
    )


def file_hashes(paths):
    return {
        str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in paths
    }


def verify_hashes_unchanged(expected):
    for path_text, digest in expected.items():
        path = Path(path_text)
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        if current != digest:
            raise AssertionError(f"Primary frozen output changed: {path}")


def compare_support_sequences(primary_json, sensitivity_json):
    primary = json.loads(primary_json)
    sensitivity = json.loads(sensitivity_json)
    return {
        "support_order_changed": primary != sensitivity,
        "support_set_changed": set(primary) != set(sensitivity),
    }


def run(aggregate_only=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    primary_hashes = file_hashes(PRIMARY_FILES)
    groups, band_cols, lake_list = _load_groups()
    target, primary_source = gloria.split_outer_target_source(groups, TARGET_LAKE)
    sensitivity_target, sensitivity_source = gloria.split_outer_target_source(
        groups, TARGET_LAKE, {RELATED_SOURCE_SITE}
    )
    if len(target) != 22 or len(primary_source) - len(sensitivity_source) != 4:
        raise AssertionError("Jasień target/source counts do not match Gate B")
    if sensitivity_source["Site_name"].isin(SOURCE_EXCLUSIONS).any():
        raise AssertionError("Identity cluster remains in sensitivity source")

    target_ids = target["group_id"].tolist()
    sensitivity_target_ids = sensitivity_target["group_id"].tolist()
    if target_ids != sensitivity_target_ids:
        raise AssertionError("Sensitivity changed target group IDs or order")
    h1_signatures = h1_random_split_signatures(target_ids)
    h1_sensitivity_signatures = h1_random_split_signatures(sensitivity_target_ids)
    h2_signatures = h2_random_split_signatures(target_ids)
    h2_sensitivity_signatures = h2_random_split_signatures(sensitivity_target_ids)
    h1_signature_bytes_equal = (
        h1_signatures.to_csv(index=False, lineterminator="\n").encode("utf-8")
        == h1_sensitivity_signatures.to_csv(
            index=False, lineterminator="\n"
        ).encode("utf-8")
    )
    h2_signature_bytes_equal = (
        h2_signatures.to_csv(index=False, lineterminator="\n").encode("utf-8")
        == h2_sensitivity_signatures.to_csv(
            index=False, lineterminator="\n"
        ).encode("utf-8")
    )
    if not (h1_signature_bytes_equal and h2_signature_bytes_equal):
        raise AssertionError("Sensitivity changed a random support/query trajectory")
    _write_csv(h1_signatures, RESULTS_DIR / "h1_random_split_signatures.csv")
    _write_csv(h2_signatures, RESULTS_DIR / "h2_random_split_signatures.csv")

    h1_checkpoint = CHECKPOINT_DIR / "h1_jasien_sensitivity_repeats.csv"
    hp_checkpoint = CHECKPOINT_DIR / "selected_hyperparameters.json"
    if not aggregate_only and not (h1_checkpoint.exists() and hp_checkpoint.exists()):
        print("Running targeted inner-LOLO tuning + H1 for Lake Jasień Południowy...")
        inner_lakes = [lake for lake in lake_list if lake != TARGET_LAKE]
        h1_repeats, hp_record = h1_pipeline.process_outer_lake(
            TARGET_LAKE,
            groups,
            band_cols,
            inner_lakes,
            additional_source_exclusions={RELATED_SOURCE_SITE},
        )
        _write_csv(h1_repeats, h1_checkpoint)
        hp_checkpoint.write_text(
            json.dumps(hp_record, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if not h1_checkpoint.exists() or not hp_checkpoint.exists():
        raise FileNotFoundError("H1 sensitivity checkpoint is missing")
    h1_sensitivity = pd.read_csv(h1_checkpoint)
    hp_record = json.loads(hp_checkpoint.read_text(encoding="utf-8"))
    hp_frame = _selected_hp_frame(hp_record)

    primary_hp = pd.read_csv(CAPACITY_RESULTS / "nested_hyperparameters.csv")
    old_hp = primary_hp[primary_hp["outer_lake"] == TARGET_LAKE].iloc[0]
    selected_hp = pd.DataFrame(
        [
            {
                "outer_lake": TARGET_LAKE,
                "primary_source_group_count": len(primary_source),
                "sensitivity_source_group_count": len(sensitivity_source),
                "excluded_related_groups": len(primary_source) - len(sensitivity_source),
                "primary_plsr_n_components": int(old_hp["plsr_n_components"]),
                "sensitivity_plsr_n_components": int(hp_record["plsr_n_components"]),
                "primary_xgb_params": old_hp["xgb_params"],
                "sensitivity_xgb_params": repr(hp_record["xgb_params"]),
                "primary_mlp_params": old_hp["mlp_params"],
                "sensitivity_mlp_params": repr(hp_record["mlp_params"]),
                "inner_validation_lakes": 23,
                "additional_source_exclusion": RELATED_SOURCE_SITE,
            }
        ]
    )
    _write_csv(selected_hp, RESULTS_DIR / "selected_hyperparameters.csv")

    primary_h1_repeats = pd.read_csv(
        CAPACITY_RESULTS / "nested" / "Lake_Jasień_Południowy_repeats.csv"
    )
    primary_h1_detail = _h1_detail(primary_h1_repeats, target)
    sensitivity_h1_detail = _h1_detail(h1_sensitivity, target)
    h1_target_comparison = _wide_comparison(
        primary_h1_detail,
        sensitivity_h1_detail,
        ["lake", "learner", "k"],
        [
            "support_count",
            "query_count",
            "label_only_mae_log",
            "calibrated_mae_log",
            "paired_delta_mae_log",
            "rer_log",
        ],
    )
    _write_csv(
        h1_target_comparison,
        RESULTS_DIR / "h1_jasien_original_vs_sensitivity.csv",
    )

    sensitivity_h1_summary = _h1_summary(h1_sensitivity)
    _write_csv(sensitivity_h1_summary, RESULTS_DIR / "h1_sensitivity_per_lake.csv")
    primary_h1_summary = pd.read_csv(
        CAPACITY_RESULTS / "capacity_boundary_summary_nested.csv"
    )
    hybrid_h1_summary = replace_target_rows(
        primary_h1_summary,
        sensitivity_h1_summary,
        key_columns=["lake", "learner", "k", "method"],
    )
    pd.testing.assert_frame_equal(
        hybrid_h1_summary[hybrid_h1_summary["lake"] != TARGET_LAKE].reset_index(drop=True),
        primary_h1_summary[primary_h1_summary["lake"] != TARGET_LAKE].reset_index(drop=True),
    )
    _write_csv(hybrid_h1_summary, RESULTS_DIR / "h1_hybrid_per_lake.csv")
    primary_h1_aggregate = build_h1_aggregate(primary_h1_summary)
    sensitivity_h1_aggregate = build_h1_aggregate(hybrid_h1_summary)
    h1_aggregate_comparison = _comparison_with_status(
        primary_h1_aggregate,
        sensitivity_h1_aggregate,
        ["learner", "k"],
        ["median_rer_pct", "mean_rer_pct", "ci_lo_pct", "ci_hi_pct"],
        "zero_status",
    )
    h1_aggregate_comparison["threshold_10pct_primary"] = (
        h1_aggregate_comparison["median_rer_pct_primary"] >= 10
    )
    h1_aggregate_comparison["threshold_10pct_sensitivity"] = (
        h1_aggregate_comparison["median_rer_pct_sensitivity"] >= 10
    )
    h1_aggregate_comparison["threshold_10pct_changed"] = (
        h1_aggregate_comparison["threshold_10pct_primary"]
        != h1_aggregate_comparison["threshold_10pct_sensitivity"]
    )
    _write_csv(
        h1_aggregate_comparison,
        RESULTS_DIR / "h1_aggregate_primary_vs_sensitivity.csv",
    )

    primary_supports = _deterministic_h2_supports(groups, band_cols, set())
    sensitivity_supports = _deterministic_h2_supports(
        groups, band_cols, {RELATED_SOURCE_SITE}
    )
    support_comparison = primary_supports.merge(
        sensitivity_supports,
        on=["method", "k"],
        suffixes=("_primary", "_sensitivity"),
        validate="one_to_one",
    )
    support_change_flags = support_comparison.apply(
        lambda row: compare_support_sequences(
            row["support_group_ids_primary"],
            row["support_group_ids_sensitivity"],
        ),
        axis=1,
        result_type="expand",
    )
    support_comparison = pd.concat(
        [support_comparison, support_change_flags], axis=1
    )
    _write_csv(
        support_comparison,
        RESULTS_DIR / "h2_deterministic_supports_primary_vs_sensitivity.csv",
    )

    h2_sensitivity_path = CHECKPOINT_DIR / "h2_jasien_sensitivity_per_repeat.csv"
    if not aggregate_only and not h2_sensitivity_path.exists():
        print("Running targeted H2 own-pool and common-query layers...")
        rows, self_checks = h2_pipeline.process_lake(
            TARGET_LAKE,
            groups,
            band_cols,
            hp_frame,
            {},
            additional_source_exclusions={RELATED_SOURCE_SITE},
        )
        if self_checks:
            raise AssertionError("Sensitivity H2 unexpectedly used primary self-check rows")
        _write_csv(pd.DataFrame(rows), h2_sensitivity_path)
    if not h2_sensitivity_path.exists():
        raise FileNotFoundError("H2 sensitivity checkpoint is missing")
    h2_sensitivity = pd.read_csv(h2_sensitivity_path)
    primary_h2_repeat = pd.read_csv(H2_PRIMARY_DIR / "h2_nested_per_repeat.csv")
    hybrid_h2_repeat = replace_target_rows(
        primary_h2_repeat,
        h2_sensitivity,
        key_columns=["lake", "learner", "k", "estimand", "rep", "method"],
    )
    pd.testing.assert_frame_equal(
        hybrid_h2_repeat[hybrid_h2_repeat["lake"] != TARGET_LAKE].reset_index(drop=True),
        primary_h2_repeat[primary_h2_repeat["lake"] != TARGET_LAKE].reset_index(drop=True),
    )
    hybrid_h2_path = RESULTS_DIR / "h2_hybrid_per_repeat.csv"
    _write_csv(hybrid_h2_repeat, hybrid_h2_path)
    _run_h2_summary_builder(hybrid_h2_path, H2_HYBRID_SUMMARY_DIR)

    primary_h2_per_lake = pd.read_csv(H2_PRIMARY_DIR / "h2_nested_per_lake.csv")
    hybrid_h2_per_lake = pd.read_csv(
        H2_HYBRID_SUMMARY_DIR / "h2_nested_per_lake.csv"
    )
    primary_target_h2 = primary_h2_per_lake[
        primary_h2_per_lake["lake"] == TARGET_LAKE
    ]
    sensitivity_target_h2 = hybrid_h2_per_lake[
        hybrid_h2_per_lake["lake"] == TARGET_LAKE
    ]
    h2_target_comparison = _wide_comparison(
        primary_target_h2,
        sensitivity_target_h2,
        ["lake", "learner", "k", "estimand", "method"],
        [
            "mae_log_median",
            "rer_log_median",
            "label_mae_log_median",
            "delta_mae_median_manuscript",
            "mae_method_median",
            "mae_random_median",
            "query_size_median",
            "n_usable_repeats",
        ],
    )
    _write_csv(
        h2_target_comparison,
        RESULTS_DIR / "h2_jasien_original_vs_sensitivity.csv",
    )

    primary_h2_own = pd.read_csv(H2_PRIMARY_DIR / "h2_nested_own_pool_summary.csv")
    sensitivity_h2_own = pd.read_csv(
        H2_HYBRID_SUMMARY_DIR / "h2_nested_own_pool_summary.csv"
    )
    h2_aggregate_comparison = _comparison_with_status(
        primary_h2_own,
        sensitivity_h2_own,
        ["learner", "k"],
        ["mean_delta_mae", "median_delta_mae", "ci_lo", "ci_hi"],
        "classification",
    )
    _write_csv(
        h2_aggregate_comparison,
        RESULTS_DIR / "h2_aggregate_primary_vs_sensitivity.csv",
    )

    common_frames = []
    for estimand, filename in [
        ("pairwise_common_query", "h2_nested_pairwise_common_query.csv"),
        ("fourway_common_query", "h2_nested_fourway_common_query.csv"),
    ]:
        primary_common = pd.read_csv(H2_PRIMARY_DIR / filename)
        sensitivity_common = pd.read_csv(H2_HYBRID_SUMMARY_DIR / filename)
        comparison = _comparison_with_status(
            primary_common,
            sensitivity_common,
            ["learner", "k", "method"],
            ["mean_delta_mae", "median_delta_mae", "ci_lo", "ci_hi"],
            "classification",
        )
        comparison.insert(0, "estimand", estimand)
        common_frames.append(comparison)
    h2_common_comparison = pd.concat(common_frames, ignore_index=True)
    _write_csv(
        h2_common_comparison,
        RESULTS_DIR / "h2_common_query_primary_vs_sensitivity.csv",
    )

    h3_sensitivity_path = CHECKPOINT_DIR / "h3_jasien_sensitivity_per_repeat.csv"
    h3_check_path = CHECKPOINT_DIR / "h3_self_check.csv"
    if not aggregate_only and not h3_sensitivity_path.exists():
        print("Running targeted paired H3 regeneration...")
        h3_sensitivity, h3_checks = _compute_h3_target(
            groups, band_cols, hp_frame, h1_sensitivity
        )
        _write_csv(h3_sensitivity, h3_sensitivity_path)
        _write_csv(h3_checks, h3_check_path)
    if not h3_sensitivity_path.exists():
        raise FileNotFoundError("H3 sensitivity checkpoint is missing")
    h3_sensitivity = pd.read_csv(h3_sensitivity_path)
    sensitivity_h3_per_lake = _h3_per_lake(h3_sensitivity)
    _write_csv(sensitivity_h3_per_lake, RESULTS_DIR / "h3_sensitivity_per_lake.csv")
    primary_h3_per_lake = pd.read_csv(
        CAPACITY_RESULTS / "h3_paired_source_audit_nested_per_lake.csv"
    )
    hybrid_h3_per_lake = replace_target_rows(
        primary_h3_per_lake,
        sensitivity_h3_per_lake,
        key_columns=["lake", "learner", "k"],
    )
    pd.testing.assert_frame_equal(
        hybrid_h3_per_lake[hybrid_h3_per_lake["lake"] != TARGET_LAKE].reset_index(drop=True),
        primary_h3_per_lake[primary_h3_per_lake["lake"] != TARGET_LAKE].reset_index(drop=True),
    )
    _write_csv(hybrid_h3_per_lake, RESULTS_DIR / "h3_hybrid_per_lake.csv")
    primary_h3_aggregate = build_h3_aggregate(primary_h3_per_lake)
    sensitivity_h3_aggregate = build_h3_aggregate(hybrid_h3_per_lake)
    _write_csv(
        sensitivity_h3_aggregate,
        RESULTS_DIR / "h3_hybrid_summary_fullprecision.csv",
    )
    h3_aggregate_comparison = _comparison_with_status(
        primary_h3_aggregate,
        sensitivity_h3_aggregate,
        ["learner", "k"],
        [
            "mean_delta_rho",
            "mean_delta_rho_ci_lo",
            "mean_delta_rho_ci_hi",
            "mean_delta_R2_logspace",
            "mean_delta_R2_logspace_ci_lo",
            "mean_delta_R2_logspace_ci_hi",
        ],
        "classification",
    )
    _write_csv(
        h3_aggregate_comparison,
        RESULTS_DIR / "h3_aggregate_primary_vs_sensitivity.csv",
    )
    h3_target_comparison = _wide_comparison(
        primary_h3_per_lake[primary_h3_per_lake["lake"] == TARGET_LAKE],
        sensitivity_h3_per_lake,
        ["lake", "learner", "k"],
        [
            "calibrated_spearman",
            "source_spearman",
            "delta_spearman",
            "delta_anomaly_r2_logspace",
            "delta_anomaly_r2_origspace",
        ],
    )
    _write_csv(
        h3_target_comparison,
        RESULTS_DIR / "h3_jasien_original_vs_sensitivity.csv",
    )

    target_material_numeric = bool(
        (h1_target_comparison["rer_log_change"].abs() >= 0.05).any()
        or (h3_target_comparison["delta_spearman_change"].abs() >= 0.05).any()
    )
    overall = _classify_decision(
        h1_aggregate_comparison,
        h2_aggregate_comparison,
        h3_aggregate_comparison,
        target_material_numeric=target_material_numeric,
    )
    impact_rows = []
    for _, row in h1_aggregate_comparison.iterrows():
        impact_rows.append(
            {
                "domain": "H1",
                "learner": row["learner"],
                "k": row["k"],
                "headline": row["k"] == 5,
                "primary_value": row["mean_rer_pct_primary"],
                "sensitivity_value": row["mean_rer_pct_sensitivity"],
                "absolute_change": row["mean_rer_pct_change"],
                "primary_status": row["zero_status_primary"],
                "sensitivity_status": row["zero_status_sensitivity"],
                "status_changed": row["zero_status_changed"]
                or row["threshold_10pct_changed"],
            }
        )
    for _, row in h2_aggregate_comparison.iterrows():
        impact_rows.append(
            {
                "domain": "H2_own_pool",
                "learner": row["learner"],
                "k": row["k"],
                "headline": True,
                "primary_value": row["mean_delta_mae_primary"],
                "sensitivity_value": row["mean_delta_mae_sensitivity"],
                "absolute_change": row["mean_delta_mae_change"],
                "primary_status": row["classification_primary"],
                "sensitivity_status": row["classification_sensitivity"],
                "status_changed": row["classification_changed"],
            }
        )
    for _, row in h3_aggregate_comparison.iterrows():
        impact_rows.append(
            {
                "domain": "H3",
                "learner": row["learner"],
                "k": row["k"],
                "headline": row["k"] == 5,
                "primary_value": row["mean_delta_rho_primary"],
                "sensitivity_value": row["mean_delta_rho_sensitivity"],
                "absolute_change": row["mean_delta_rho_change"],
                "primary_status": row["classification_primary"],
                "sensitivity_status": row["classification_sensitivity"],
                "status_changed": row["classification_changed"],
            }
        )
    decision_impact = pd.DataFrame(impact_rows)
    decision_impact.insert(0, "overall_classification", overall)
    _write_csv(decision_impact, RESULTS_DIR / "decision_impact.csv")

    verify_hashes_unchanged(primary_hashes)
    manifest = {
        "target_lake": TARGET_LAKE,
        "related_source_site": RELATED_SOURCE_SITE,
        "target_group_count": len(target),
        "primary_source_group_count": len(primary_source),
        "sensitivity_source_group_count": len(sensitivity_source),
        "excluded_related_group_count": len(primary_source) - len(sensitivity_source),
        "other_outer_folds_recomputed": 0,
        "target_group_ids_and_order_identical": target_ids == sensitivity_target_ids,
        "h1_h3_random_split_bytes_identical": h1_signature_bytes_equal,
        "h2_random_split_bytes_identical": h2_signature_bytes_equal,
        "other_outer_rows_identical": True,
        "primary_hashes_verified_unchanged": True,
        "material_single_fold_change": target_material_numeric,
        "overall_classification": overall,
        "primary_hashes": primary_hashes,
    }
    (RESULTS_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"Gate B.1 aggregation complete: classification={overall}; "
        f"source {len(primary_source)} -> {len(sensitivity_source)} groups."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="reuse isolated sensitivity checkpoints and rebuild comparisons",
    )
    args = parser.parse_args()
    run(aggregate_only=args.aggregate_only)


if __name__ == "__main__":
    main()
