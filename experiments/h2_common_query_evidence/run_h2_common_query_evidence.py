"""Audit frozen H2 common-query sensitivity evidence without rerunning models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


WEIGHTED_SELECTOR = "proposed_weighted_d_optimal"
PAIRWISE_ESTIMAND = "pairwise_common_query"
FOURWAY_ESTIMAND = "fourway_common_query"
SOURCE_RELATIVE = Path("experiments") / "active_selection" / "results" / "true_nested_h2"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
FROZEN_INPUT_NAMES = (
    "h2_nested_own_pool_summary.csv",
    "h2_nested_pairwise_common_query.csv",
    "h2_nested_fourway_common_query.csv",
    "h2_nested_query_attrition.csv",
    "h2_nested_per_repeat.csv",
    "h2_nested_per_lake.csv",
)
FOURWAY_MIN_USABLE_REPEATS = 50
N_BOOT = 10_000
BOOTSTRAP_SEED = 0

def classify_nominal_ci(ci_lo: float, ci_hi: float) -> str:
    """Apply the frozen manuscript sign: positive delta-MAE favors selector."""
    if ci_lo > 0:
        return "selector_better"
    if ci_hi < 0:
        return "random_better"
    return "unresolved"


def build_decision_matrix(
    own_pool: pd.DataFrame, pairwise: pd.DataFrame, fourway: pd.DataFrame
) -> pd.DataFrame:
    """Compare frozen primary and common-query classifications by selector cell."""
    pair = pairwise.rename(columns={
        "method": "selector",
        "classification": "pairwise_classification",
        "n_lakes": "pairwise_n_lakes",
    })
    four = fourway.rename(columns={
        "method": "selector",
        "classification": "fourway_classification",
        "n_lakes": "fourway_n_lakes",
    })
    matrix = pair.merge(
        four[["learner", "k", "selector", "fourway_classification", "fourway_n_lakes"]],
        on=["learner", "k", "selector"],
        validate="one_to_one",
    )
    weighted = own_pool[["learner", "k", "classification"]].rename(
        columns={"classification": "own_pool_classification"}
    )
    matrix = matrix.merge(weighted, on=["learner", "k"], how="left", validate="many_to_one")
    matrix["own_pool_classification"] = matrix["own_pool_classification"].where(
        matrix["selector"] == WEIGHTED_SELECTOR, "not_applicable"
    )
    matrix["classification_change_pairwise_vs_fourway"] = (
        matrix["pairwise_classification"] != matrix["fourway_classification"]
    )
    matrix["stable_selector_better_all_available_estimands"] = (
        (matrix["pairwise_classification"] == "selector_better")
        & (matrix["fourway_classification"] == "selector_better")
        & (
            (matrix["own_pool_classification"] == "selector_better")
            | (matrix["own_pool_classification"] == "not_applicable")
        )
    )
    return matrix


def _audited_classification(frame: pd.DataFrame) -> pd.DataFrame:
    audited = frame.copy()
    audited["frozen_classification"] = audited["classification"]
    audited["classification"] = audited["classification"].replace(
        {"method_better": "selector_better"}
    )
    calculated = audited.apply(
        lambda row: classify_nominal_ci(row["ci_lo"], row["ci_hi"]), axis=1
    )
    if not (audited["classification"] == calculated).all():
        raise ValueError("A frozen H2 summary classification disagrees with its CI.")
    return audited


def _repeat_coverage(per_repeat: pd.DataFrame, estimand: str, usable_only: bool) -> pd.DataFrame:
    rows = per_repeat.loc[per_repeat["estimand"] == estimand].copy()
    if usable_only:
        rows = rows.loc[rows["usable"] == True].copy()
    rows = rows.dropna(subset=["delta_mae"])
    lake_keys = ["lake", "learner", "k", "method"]
    by_lake = rows.groupby(lake_keys, sort=True).agg(
        n_repeat_rows=("rep", "count"),
        min_repeat=("rep", "min"),
        max_repeat=("rep", "max"),
        min_query_size=("query_size", "min"),
        max_query_size=("query_size", "max"),
    ).reset_index()
    duplicate_counts = (
        rows.groupby(lake_keys + ["rep"], sort=True).size().sub(1).clip(lower=0)
        .groupby(level=[0, 1, 2, 3]).sum().rename("duplicate_repeat_keys")
        .reset_index()
    )
    by_lake = by_lake.merge(duplicate_counts, on=lake_keys, how="left", validate="one_to_one")
    keys = ["learner", "k", "method"]
    return by_lake.groupby(keys, sort=True).agg(
        n_lakes_with_repeat_provenance=("lake", "nunique"),
        n_repeat_rows=("n_repeat_rows", "sum"),
        min_repeats_per_lake=("n_repeat_rows", "min"),
        max_repeats_per_lake=("n_repeat_rows", "max"),
        min_repeat=("min_repeat", "min"),
        max_repeat=("max_repeat", "max"),
        min_query_size=("min_query_size", "min"),
        max_query_size=("max_query_size", "max"),
        duplicate_repeat_keys=("duplicate_repeat_keys", "sum"),
    ).reset_index()


def audit_available_repeat_coverage(
    per_repeat: pd.DataFrame, frozen_fourway: pd.DataFrame
) -> pd.DataFrame:
    """Describe the frozen all-lake four-way summaries without a lake-entry filter."""
    rows = per_repeat.loc[per_repeat["estimand"] == FOURWAY_ESTIMAND].copy()
    keys = ["lake", "learner", "k", "method"]
    per_lake = rows.groupby(keys, sort=True).agg(
        candidate_repeats=("rep", "count"),
        usable_repeats=("usable", lambda values: int(values.eq(True).sum())),
    ).reset_index()
    per_lake["usable_repeat_fraction"] = (
        per_lake["usable_repeats"] / per_lake["candidate_repeats"]
    )
    coverage = per_lake.groupby(["learner", "k", "method"], sort=True).agg(
        candidate_repeats=("candidate_repeats", "sum"),
        usable_repeats=("usable_repeats", "sum"),
        usable_repeat_fraction=("usable_repeat_fraction", "mean"),
        n_lakes_available_repeat=("lake", "nunique"),
        n_lakes_usable_repeats_50plus=(
            "usable_repeats", lambda values: int((values >= FOURWAY_MIN_USABLE_REPEATS).sum())
        ),
        n_lakes_usable_repeats_lt50=(
            "usable_repeats", lambda values: int((values < FOURWAY_MIN_USABLE_REPEATS).sum())
        ),
        min_usable_repeats_per_lake=("usable_repeats", "min"),
        median_usable_repeats_per_lake=("usable_repeats", "median"),
        max_usable_repeats_per_lake=("usable_repeats", "max"),
    ).reset_index()
    expected = frozen_fourway[["learner", "k", "method", "n_lakes"]].rename(
        columns={"n_lakes": "frozen_summary_n_lakes"}
    )
    coverage = coverage.merge(expected, on=["learner", "k", "method"], validate="one_to_one")
    coverage["all_lakes_entered_frozen_summary"] = (
        coverage["n_lakes_available_repeat"] == coverage["frozen_summary_n_lakes"]
    )
    if not coverage["all_lakes_entered_frozen_summary"].all():
        raise ValueError("Frozen four-way summary lake counts disagree with repeat provenance.")
    return coverage


def _mean_bootstrap_ci(values: pd.Series) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.array([
        array[rng.integers(0, len(array), size=len(array))].mean()
        for _ in range(N_BOOT)
    ])
    return tuple(float(value) for value in np.percentile(samples, [2.5, 97.5]))


def derive_fourway_threshold50(per_repeat: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Derive a post hoc >=50-usable-repeat four-way sensitivity from frozen rows."""
    all_rows = per_repeat.loc[per_repeat["estimand"] == FOURWAY_ESTIMAND].copy()
    keys = ["lake", "learner", "k", "method"]
    candidate = all_rows.groupby(keys, sort=True).agg(
        candidate_repeats=("rep", "count"),
        usable_repeats=("usable", lambda values: int(values.eq(True).sum())),
    ).reset_index()
    usable = all_rows.loc[
        (all_rows["usable"] == True) & all_rows["delta_mae"].notna()
    ].copy()
    effects = usable.groupby(keys, sort=True).agg(
        lake_delta_mae=("delta_mae", lambda values: float(-values.median())),
        median_query_size=("query_size", "median"),
    ).reset_index()
    per_lake = candidate.merge(effects, on=keys, how="left", validate="one_to_one")
    per_lake["eligible_threshold50"] = (
        per_lake["usable_repeats"] >= FOURWAY_MIN_USABLE_REPEATS
    )
    rows = []
    for (learner, k, method), cell in per_lake.groupby(["learner", "k", "method"], sort=True):
        eligible = cell.loc[cell["eligible_threshold50"]].copy()
        ci_lo, ci_hi = _mean_bootstrap_ci(eligible["lake_delta_mae"])
        rows.append({
            "learner": learner,
            "k": int(k),
            "method": method,
            "estimand": "post_hoc_threshold50_fourway_common_query",
            "query_scope": "Q_all with >=50 usable repeats per lake-learner unit",
            "delta_definition": "MAE_random - MAE_method",
            "positive_means": "method_better_than_random",
            "threshold_usable_repeats": FOURWAY_MIN_USABLE_REPEATS,
            "n_lakes": int(len(eligible)),
            "n_lakes_excluded": int(len(cell) - len(eligible)),
            "mean_delta_mae": float(eligible["lake_delta_mae"].mean()),
            "median_delta_mae": float(eligible["lake_delta_mae"].median()),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "positive_lake_fraction": float((eligible["lake_delta_mae"] > 0).mean()),
            "classification": classify_nominal_ci(ci_lo, ci_hi),
            "bootstrap_n": N_BOOT,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "multiplicity_adjusted": False,
        })
    return {"per_lake": per_lake, "summary": pd.DataFrame(rows)}


def compare_available_vs_threshold50(
    frozen_fourway: pd.DataFrame, threshold50: pd.DataFrame
) -> pd.DataFrame:
    """Keep frozen available-repeat and post hoc thresholded results separate."""
    available = frozen_fourway[[
        "learner", "k", "method", "n_lakes", "mean_delta_mae", "median_delta_mae",
        "ci_lo", "ci_hi", "classification",
    ]].rename(columns={
        "n_lakes": "available_repeat_n_lakes",
        "mean_delta_mae": "available_repeat_mean_delta_mae",
        "median_delta_mae": "available_repeat_median_delta_mae",
        "ci_lo": "available_repeat_ci_lo",
        "ci_hi": "available_repeat_ci_hi",
        "classification": "available_repeat_classification",
    })
    restricted = threshold50[[
        "learner", "k", "method", "n_lakes", "n_lakes_excluded", "mean_delta_mae",
        "median_delta_mae", "ci_lo", "ci_hi", "classification", "positive_lake_fraction",
    ]].rename(columns={
        "n_lakes": "threshold50_n_lakes",
        "n_lakes_excluded": "threshold50_n_lakes_excluded",
        "mean_delta_mae": "threshold50_mean_delta_mae",
        "median_delta_mae": "threshold50_median_delta_mae",
        "ci_lo": "threshold50_ci_lo",
        "ci_hi": "threshold50_ci_hi",
        "classification": "threshold50_classification",
        "positive_lake_fraction": "threshold50_positive_lake_fraction",
    })
    comparison = available.merge(restricted, on=["learner", "k", "method"], validate="one_to_one")
    comparison["mean_delta_mae_change_threshold50_minus_available"] = (
        comparison["threshold50_mean_delta_mae"] - comparison["available_repeat_mean_delta_mae"]
    )
    comparison["classification_changed"] = (
        comparison["available_repeat_classification"]
        != comparison["threshold50_classification"]
    )
    return comparison


def audit_frozen_outputs(root: Path) -> dict[str, object]:
    """Read and validate frozen H2 summaries and their repeat-level provenance."""
    source = root / SOURCE_RELATIVE
    own = _audited_classification(pd.read_csv(source / "h2_nested_own_pool_summary.csv"))
    pairwise = _audited_classification(pd.read_csv(source / "h2_nested_pairwise_common_query.csv"))
    fourway = _audited_classification(pd.read_csv(source / "h2_nested_fourway_common_query.csv"))
    attrition = pd.read_csv(source / "h2_nested_query_attrition.csv").copy()
    per_repeat = pd.read_csv(source / "h2_nested_per_repeat.csv")
    per_lake = pd.read_csv(source / "h2_nested_per_lake.csv")
    available_repeat_coverage = audit_available_repeat_coverage(per_repeat, fourway)

    if not (len(own) == 12 and len(pairwise) == 36 and len(fourway) == 36 and len(attrition) == 4):
        raise ValueError("Unexpected frozen H2 summary cell counts.")
    raw_common = per_repeat.loc[per_repeat["estimand"].isin([PAIRWISE_ESTIMAND, FOURWAY_ESTIMAND])]
    raw_sign_error = np.abs(
        raw_common["delta_mae"] - (raw_common["mae_log_method"] - raw_common["mae_log_random"])
    ).max()
    if raw_sign_error > 1e-12:
        raise ValueError("Raw common-query delta-MAE no longer matches its run-time sign.")

    pair_coverage = _repeat_coverage(per_repeat, PAIRWISE_ESTIMAND, usable_only=False)
    four_coverage = _repeat_coverage(per_repeat, FOURWAY_ESTIMAND, usable_only=True)
    pairwise = pairwise.merge(
        pair_coverage,
        on=["learner", "k", "method"],
        how="left",
        validate="one_to_one",
    )
    fourway = fourway.merge(
        four_coverage,
        on=["learner", "k", "method"],
        how="left",
        validate="one_to_one",
    )
    attrition_for_merge = attrition.rename(columns={
        "frac_usable": "usable_repeat_fraction", "note": "restricted_pool_note"
    })[
        ["k", "usable_repeat_fraction", "n_lake_learner_units_50plus_repeats", "restricted_pool_note"]
    ]
    fourway = fourway.merge(attrition_for_merge, on="k", validate="many_to_one")
    fourway["row_level_usable_repeat_restriction"] = (
        fourway["usable_repeat_fraction"] < 1.0
    )

    for frame in (pairwise, fourway):
        if frame[["n_repeat_rows", "duplicate_repeat_keys"]].isna().any().any():
            raise ValueError("A frozen common-query summary lacks repeat-level provenance.")
        if (frame["duplicate_repeat_keys"] != 0).any():
            raise ValueError("Duplicate common-query repeat keys found.")
    if not (pairwise["n_repeat_rows"] == 2400).all():
        raise ValueError("Pairwise common-query coverage is not 24 lakes × 100 repeats.")
    if not fourway.loc[fourway["k"] < 10, "n_repeat_rows"].eq(2400).all():
        raise ValueError("Full-pool four-way coverage is incomplete below k=10.")

    matrix = build_decision_matrix(own, pairwise, fourway)
    matrix = matrix.merge(
        fourway[["learner", "k", "method", "usable_repeat_fraction", "row_level_usable_repeat_restriction"]].rename(
            columns={"method": "selector"}
        ),
        on=["learner", "k", "selector"],
        validate="one_to_one",
    )
    matrix["direction_agrees_pairwise_fourway"] = (
        matrix["pairwise_classification"] == matrix["fourway_classification"]
    )
    matrix["any_selector_better"] = (
        (matrix["pairwise_classification"] == "selector_better")
        | (matrix["fourway_classification"] == "selector_better")
        | (matrix["own_pool_classification"] == "selector_better")
    )
    k10_units = available_repeat_coverage.loc[
        (available_repeat_coverage["k"] == 10)
        & (available_repeat_coverage["method"] == WEIGHTED_SELECTOR)
    ]
    k10_low_coverage = per_lake.loc[
        (per_lake["estimand"] == FOURWAY_ESTIMAND)
        & (per_lake["k"] == 10)
        & (per_lake["method"] == WEIGHTED_SELECTOR)
        & (per_lake["n_usable_repeats"] < FOURWAY_MIN_USABLE_REPEATS)
    ]
    checks = {
        "raw_common_runtime_sign_max_abs_error": float(raw_sign_error),
        "pairwise_summary_rows": int(len(pairwise)),
        "fourway_summary_rows": int(len(fourway)),
        "decision_matrix_rows": int(len(matrix)),
        "pairwise_all_lakes_all_repeats": bool((pairwise["n_repeat_rows"] == 2400).all()),
        "fourway_k10_usable_fraction": float(
            attrition.loc[attrition["k"] == 10, "frac_usable"].item()
        ),
        "fourway_k10_lake_learner_units_total": int(k10_units["n_lakes_available_repeat"].sum()),
        "fourway_k10_lake_learner_units_50plus": int(
            k10_units["n_lakes_usable_repeats_50plus"].sum()
        ),
        "fourway_k10_lake_learner_units_lt50": int(
            k10_units["n_lakes_usable_repeats_lt50"].sum()
        ),
        "fourway_k10_low_coverage_min_usable_repeats": int(
            k10_low_coverage["n_usable_repeats"].min()
        ),
        "fourway_k10_low_coverage_max_usable_repeats": int(
            k10_low_coverage["n_usable_repeats"].max()
        ),
        "fourway_k10_all_lake_learner_units_entered": bool(
            available_repeat_coverage.loc[
                available_repeat_coverage["k"] == 10,
                "all_lakes_entered_frozen_summary",
            ].all()
        ),
        "frozen_per_lake_rows": int(len(per_lake)),
    }
    return {
        "own_pool": own,
        "pairwise": pairwise,
        "fourway": fourway,
        "attrition": attrition,
        "available_repeat_coverage": available_repeat_coverage,
        "decision_matrix": matrix,
        "checks": checks,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_audit(path: Path, audit: dict[str, object], input_hashes: dict[str, str]) -> None:
    own = audit["own_pool"]
    pairwise = audit["pairwise"]
    fourway = audit["fourway"]
    attrition = audit["attrition"]
    matrix = audit["decision_matrix"]
    checks = audit["checks"]
    k10 = audit["attrition"].loc[audit["attrition"]["k"] == 10].iloc[0]
    own_counts = own["classification"].value_counts().to_dict()
    pair_counts = pairwise["classification"].value_counts().to_dict()
    four_counts = fourway["classification"].value_counts().to_dict()
    changed = int(matrix["classification_change_pairwise_vs_fourway"].sum())
    any_selector_better = int(matrix["any_selector_better"].sum())
    k10 = attrition.loc[attrition["k"] == 10].iloc[0]
    lines = [
        "# H2 common-query supporting analyses",
        "",
        "The available-repeat four-way analysis and the post hoc threshold-50 sensitivity use different target inclusion rules, described below.",
        "",
        "## Scope and hierarchy",
        "",
        "The primary H2 estimand is the weighted D-optimal versus repeated-random comparison on each method's own remaining pool. Pairwise common-query and frozen available-repeat four-way common-query results are supporting sensitivity estimands only. RER remains a separate support-conditioned diagnostic and is not merged into this decision table. All intervals are nominal, cell-wise 10,000-resample percentile bootstrap intervals over lakes; they are not multiplicity-adjusted.",
        "",
        "## Frozen inputs and repeat-level checks",
        "",
        "The analysis reads the stored true-nested H2 summaries plus their per-repeat and per-lake provenance. It does not call the experiment runner, tune a model, or rewrite an input artifact. Pairwise rows use the frozen repeat-specific `Q_sR = pool \\ (S_method ∪ S_random)`; four-way rows use `Q_all = pool \\ (S_random ∪ S_KS ∪ S_unwD ∪ S_wD)`. The frozen aggregator removed unusable repeat rows but did not apply the configured >=50-usable-repeat threshold before forming lake-level summaries.",
        "",
        f"- Summary cells: own-pool {len(own)}, pairwise {len(pairwise)}, four-way {len(fourway)}; cross-estimand rows {len(matrix)}.",
        f"- Raw common-query run-time sign check: max |delta_mae - (mae_method - mae_random)| = {checks['raw_common_runtime_sign_max_abs_error']:.3e}.",
        "- The audited tables explicitly flip the persisted common-query run-time sign to the frozen manuscript sign, `MAE_random - MAE_selector`; positive values favor the selector.",
        f"- Pairwise repeat provenance is complete: {checks['pairwise_all_lakes_all_repeats']} (24 lakes x 100 repeats for every learner--budget--selector cell).",
        f"- At k=10, all {checks['fourway_k10_lake_learner_units_total']} lake--learner units entered the frozen all-lake available-repeat summary; {checks['fourway_k10_lake_learner_units_50plus']} had >=50 usable repeats and {checks['fourway_k10_lake_learner_units_lt50']} had fewer. The usable-repeat fraction was {checks['fourway_k10_usable_fraction']:.2%}.",
        "",
        "## Classification results",
        "",
        f"- Primary own-pool (weighted selector only): {own_counts}.",
        f"- Pairwise common-query: {pair_counts}.",
        f"- Four-way common-query: {four_counts}.",
        f"- Pairwise versus four-way class changes: {changed} of {len(matrix)} selector cells; selector-better cells across any audited estimand: {any_selector_better}.",
        "",
        "The frozen Jasień identity sensitivity is retained as a separate numerical-stability note: only the pairwise MLP/k=3 Kennard--Stone cell changes from `random_better` to `unresolved` under the reported perturbation; no four-way class changes. This note neither changes the primary own-pool result nor replaces any frozen H2 output.",
        "",
        "## Attrition boundary",
        "",
        f"At k=10, the frozen four-way analysis is an all-24-lake available-repeat sensitivity: {k10.n_usable_units}/{k10.n_lake_learner_rep_units_total} candidate repeat rows were usable ({k10.frac_usable:.2%}), and lake-level medians therefore had unequal within-lake repeat coverage. The >=50 count is a coverage diagnostic, not a frozen lake-entry rule. The distinct post hoc threshold-50 restricted-pool sensitivity is reported separately in Table S10.",
        "",
        "## Conclusion boundary",
        "",
        "The reported median-over-repetitions own-pool and common-query comparisons provide no consistent selector advantage over repeated random selection. The mean-over-repetitions sensitivity is a separate estimand. These results concern transductive, pool-based label acquisition.",
        "",
        "## Frozen input SHA-256",
        "",
    ]
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in input_hashes.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_fourway_correction(
    path: Path, audit: dict[str, object], threshold50: pd.DataFrame, comparison: pd.DataFrame
) -> None:
    """Describe available-repeat and threshold-50 aggregation separately."""
    checks = audit["checks"]
    k10 = audit["attrition"].loc[audit["attrition"]["k"] == 10].iloc[0]
    changed = comparison.loc[comparison["classification_changed"]]
    threshold_counts = threshold50["classification"].value_counts().to_dict()
    lines = [
        "# Four-way aggregation and repeat coverage",
        "",
        "## Available-repeat aggregation",
        "",
        "The script `build_h2_nested_summaries.py` groups every lake--learner--budget--method unit with any usable four-way repeat and takes its median before the across-lake summary. It does not filter `n_usable_repeats >= 50`; that threshold belongs to the separate post hoc sensitivity.",
        "",
        "## Available-repeat estimand",
        "",
        "The frozen four-way result is an **all-24-lake available-repeat four-way common-query sensitivity with unequal within-lake usable-repeat coverage**. At k=10, all "
        f"{checks['fourway_k10_lake_learner_units_total']} lake--learner units entered; "
        f"{checks['fourway_k10_lake_learner_units_50plus']} had >=50 usable repeats and "
        f"{checks['fourway_k10_lake_learner_units_lt50']} had only "
        f"{checks['fourway_k10_low_coverage_min_usable_repeats']}--{checks['fourway_k10_low_coverage_max_usable_repeats']}. "
        "The row-level usable fraction is "
        f"{checks['fourway_k10_usable_fraction']:.2%} ({k10.n_usable_units}/{k10.n_lake_learner_rep_units_total} candidate repeat rows).",
        "",
        "## Post hoc threshold-50 sensitivity",
        "",
        "A separate post hoc >=50-usable-repeat restricted-pool sensitivity was derived only from frozen repeat-level four-way metrics. It includes a lake--learner unit only when it has at least 50 usable repeats, then applies the frozen lake median and a fresh `default_rng(0)` 10,000-resample percentile bootstrap over included lakes. It is a different usability-defined estimand and does not replace the frozen available-repeat result.",
        "",
        f"- Threshold-50 classifications: {threshold_counts}.",
        f"- Threshold-50 selector-better cells: {int((threshold50['classification'] == 'selector_better').sum())}.",
        f"- Frozen versus threshold-50 classification changes: {len(changed)}.",
        "",
        "## Interpretation",
        "",
        "Available-repeat and threshold-50 results answer different questions. The >=50 coverage count is descriptive for the available-repeat analysis and is an inclusion rule only for the threshold-50 sensitivity.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_analysis(
    root: Path = ROOT, output_dir: Path = DEFAULT_OUTPUT_DIR, *, write_audit: bool = True
) -> dict[str, object]:
    """Write supporting H2 summaries from stored repeat-level results."""
    source = root / SOURCE_RELATIVE
    before_hashes = {
        str((SOURCE_RELATIVE / name).as_posix()): _sha256(source / name)
        for name in FROZEN_INPUT_NAMES
    }
    audit = audit_frozen_outputs(root)
    pairwise = audit["pairwise"].copy()
    fourway = audit["fourway"].copy()
    attrition = audit["attrition"].copy()
    matrix = audit["decision_matrix"].copy()
    available_coverage = audit["available_repeat_coverage"].copy()
    per_repeat = pd.read_csv(source / "h2_nested_per_repeat.csv")
    frozen_fourway = pd.read_csv(source / "h2_nested_fourway_common_query.csv")
    threshold50 = derive_fourway_threshold50(per_repeat)["summary"]
    comparison = compare_available_vs_threshold50(frozen_fourway, threshold50)
    available_fourway = frozen_fourway.merge(
        available_coverage,
        on=["learner", "k", "method"],
        validate="one_to_one",
    )
    for frame in (pairwise, fourway):
        frame["audited_delta_definition"] = "MAE_random - MAE_selector"
        frame["classification_rule"] = "ci_lo > 0: selector_better; ci_hi < 0: random_better; otherwise unresolved"
        frame["bootstrap_note"] = "nominal cell-wise 10000-resample percentile lake bootstrap; not multiplicity-adjusted"
        frame["repeat_level_pairing_status"] = "verified from frozen lake-learner-k-method-repeat rows"
    attrition["fourway_eligibility"] = "query_size >= 5; query_size/pool >= 0.20"
    attrition["frozen_summary_entry_rule"] = "all lake-learner units with available usable repeats"
    entered_by_k = available_coverage.loc[
        available_coverage["method"] == WEIGHTED_SELECTOR
    ].groupby("k")["n_lakes_available_repeat"].sum()
    attrition["lake_learner_units_entered_frozen_summary"] = attrition["k"].map(entered_by_k)
    attrition["n_lake_learner_units_lt50_repeats"] = (
        attrition["lake_learner_units_entered_frozen_summary"]
        - attrition["n_lake_learner_units_50plus_repeats"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "h2_pairwise_common_query_audited.csv": pairwise,
        "h2_fourway_common_query_audited.csv": fourway,
        "h2_fourway_available_repeat_audited.csv": available_fourway,
        "h2_fourway_repeat_coverage.csv": available_coverage,
        "h2_fourway_threshold50_sensitivity.csv": threshold50,
        "h2_fourway_available_vs_threshold50.csv": comparison,
        "h2_query_attrition_audited.csv": attrition,
        "h2_cross_estimand_decision_matrix.csv": matrix,
    }
    for filename, frame in outputs.items():
        _write_csv(frame, output_dir / filename)

    try:
        from experiments.h2_common_query_evidence.render_supplementary_tables import render_all
    except ModuleNotFoundError:  # Direct script execution from this directory.
        from render_supplementary_tables import render_all

    tables = render_all(root)
    for filename, text in tables.items():
        (output_dir / filename).write_text(text, encoding="utf-8", newline="\n")

    after_hashes = {
        str((SOURCE_RELATIVE / name).as_posix()): _sha256(source / name)
        for name in FROZEN_INPUT_NAMES
    }
    if before_hashes != after_hashes:
        raise RuntimeError("An H2 input changed during aggregation.")
    if write_audit:
        _write_audit(output_dir / "analysis_summary.md", audit, after_hashes)
        _write_fourway_correction(
            output_dir / "fourway_aggregation.md",
            audit,
            threshold50,
            comparison,
        )

    output_names = tuple(outputs) + tuple(tables)
    manifest = {
        "analysis": "h2_available_repeat_and_threshold50_sensitivity",
        "scope": "frozen_available_repeat_fourway_audit_plus_post_hoc_threshold50_supporting_sensitivity",
        "frozen_input_sha256_before_and_after": {
            name: {"before": before_hashes[name], "after": after_hashes[name]}
            for name in before_hashes
        },
        "primary_h2_replaced": False,
        "manuscripts_edited": False,
        "model_training_or_refit": False,
        "support_or_query_regeneration": False,
        "frozen_h2_outputs_rewritten": False,
        "aggregation_provenance_source_sha256": {
            "experiments/active_selection/build_h2_nested_summaries.py": _sha256(
                root / "experiments" / "active_selection" / "build_h2_nested_summaries.py"
            ),
            "experiments/active_selection/true_nested_h2_protocol.yaml": _sha256(
                root / "experiments" / "active_selection" / "true_nested_h2_protocol.yaml"
            ),
        },
        "frozen_fourway_entry_rule": "all lake-learner units with available usable repeats",
        "post_hoc_threshold50_replaces_frozen_fourway": False,
        "classification_boundary": "nominal cell-wise CI only; no multiplicity adjustment",
        "jasien_identity_sensitivity": {
            "pairwise_only_change": "MLP/k=3/Kennard--Stone random_better_to_unresolved",
            "fourway_class_changes": 0,
        },
        "integrity_checks": audit["checks"],
        "threshold50": {
            "threshold_usable_repeats": FOURWAY_MIN_USABLE_REPEATS,
            "classification_counts": threshold50["classification"].value_counts().to_dict(),
            "selector_better_cells": int((threshold50["classification"] == "selector_better").sum()),
            "classification_changes_vs_available": int(comparison["classification_changed"].sum()),
        },
        "outputs_sha256": {name: _sha256(output_dir / name) for name in output_names},
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    result = run_analysis(ROOT, parse_args().output_dir)
    print(pd.DataFrame([result["checks"]]).to_string(index=False))


if __name__ == "__main__":
    main()
