"""Derive absolute H1 losses from frozen true-nested repeat metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CAPACITY_RESULTS = ROOT / "experiments" / "capacity_boundary" / "results"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
FROZEN_REPEATS = CAPACITY_RESULTS / "capacity_boundary_repeats_nested.csv"
FROZEN_SUMMARY = CAPACITY_RESULTS / "capacity_boundary_summary_nested.csv"
FROZEN_H1_TABLE = CAPACITY_RESULTS / "h1_global_vs_nested_comparison.csv"
LEARNERS = ("PLSR", "XGBoost", "MLP")
BUDGETS = (1, 3, 5, 10)
EXPECTED_LAKES = 24
EXPECTED_REPEATS = 100
N_BOOT = 10_000
BOOTSTRAP_SEED = 0
RER_TOLERANCE = 1e-12
RER_EPSILON = 1e-8


def derive_repeat_level(frozen_repeats: pd.DataFrame) -> pd.DataFrame:
    """Recover same-support label-only loss and paired gain for each H1 row."""
    calibrated = frozen_repeats.loc[
        (frozen_repeats["method"] == "calibrated") & (frozen_repeats["k"] > 0)
    ].copy()
    calibrated = calibrated.rename(columns={"mae_log": "calibrated_log_mae"})
    calibrated["rer_reconstruction_denominator"] = 1.0 - calibrated["rer_log"]
    calibrated["label_only_log_mae"] = (
        (calibrated["calibrated_log_mae"] + calibrated["rer_log"] * RER_EPSILON)
        / calibrated["rer_reconstruction_denominator"]
    )
    calibrated["paired_delta_mae"] = (
        calibrated["label_only_log_mae"] - calibrated["calibrated_log_mae"]
    )
    calibrated["reconstructed_rer"] = (
        calibrated["paired_delta_mae"]
        / (calibrated["label_only_log_mae"] + RER_EPSILON)
    )
    calibrated["source_only_log_mae"] = np.nan
    return calibrated[
        [
            "lake",
            "learner",
            "k",
            "rep",
            "label_only_log_mae",
            "calibrated_log_mae",
            "source_only_log_mae",
            "paired_delta_mae",
            "rer_log",
            "rer_reconstruction_denominator",
            "reconstructed_rer",
        ]
    ]


def summarize_per_lake(repeat_level: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each lake after preserving its repeat-level pairing."""
    grouped = repeat_level.groupby(["lake", "learner", "k"], sort=True)
    per_lake = grouped.agg(
        lake_label_only_log_mae=("label_only_log_mae", "median"),
        lake_calibrated_log_mae=("calibrated_log_mae", "median"),
        lake_source_only_log_mae=("source_only_log_mae", "median"),
        lake_paired_delta_mae=("paired_delta_mae", "median"),
        repeat_count=("rep", "count"),
        positive_delta_repeat_fraction=(
            "paired_delta_mae", lambda values: float((values > 0).mean())
        ),
    ).reset_index()
    per_lake["median_component_difference"] = (
        per_lake["lake_label_only_log_mae"] - per_lake["lake_calibrated_log_mae"]
    )
    per_lake["median_pairing_gap"] = (
        per_lake["lake_paired_delta_mae"]
        - per_lake["median_component_difference"]
    )
    return per_lake


def _classify_interval(ci_lo: float, ci_hi: float) -> str:
    if ci_lo > 0:
        return "positive"
    if ci_hi < 0:
        return "negative"
    return "crosses_zero"


def _bootstrap_mean_ci(values: pd.Series, n_boot: int, seed: int) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.array([
        array[rng.integers(0, len(array), size=len(array))].mean()
        for _ in range(n_boot)
    ])
    return tuple(float(value) for value in np.percentile(samples, [2.5, 97.5]))


def summarize_across_lakes(
    per_lake: pd.DataFrame, n_boot: int = 10_000, seed: int = 0
) -> pd.DataFrame:
    """Summarize paired lake effects with lake-resampled percentile intervals."""
    rows = []
    for (learner, k), cell in per_lake.groupby(["learner", "k"], sort=True):
        delta = cell["lake_paired_delta_mae"]
        ci_lo, ci_hi = _bootstrap_mean_ci(delta, n_boot=n_boot, seed=seed)
        rows.append({
            "learner": learner,
            "k": int(k),
            "n_lakes": len(cell),
            "mean_label_only_log_mae": float(cell["lake_label_only_log_mae"].mean()),
            "median_label_only_log_mae": float(cell["lake_label_only_log_mae"].median()),
            "mean_calibrated_log_mae": float(cell["lake_calibrated_log_mae"].mean()),
            "median_calibrated_log_mae": float(cell["lake_calibrated_log_mae"].median()),
            "mean_source_only_log_mae": float(cell["lake_source_only_log_mae"].mean()),
            "median_source_only_log_mae": float(cell["lake_source_only_log_mae"].median()),
            "mean_paired_delta_mae": float(delta.mean()),
            "median_paired_delta_mae": float(delta.median()),
            "delta_ci_lo": ci_lo,
            "delta_ci_hi": ci_hi,
            "frac_lakes_delta_positive": float((delta > 0).mean()),
            "classification": _classify_interval(ci_lo, ci_hi),
            "bootstrap_n": n_boot,
            "bootstrap_seed": seed,
        })
    return pd.DataFrame(rows)


def validate_repeat_level(
    repeat_level: pd.DataFrame,
    *,
    learners: tuple[str, ...],
    budgets: tuple[int, ...],
    expected_lakes: int,
    expected_repeats: int,
    tolerance: float = 1e-12,
) -> dict[str, float | bool | int]:
    """Validate exact metric-level pairing and expected frozen-row coverage."""
    expected_rows = expected_lakes * len(learners) * len(budgets) * expected_repeats
    duplicate_rows = int(
        repeat_level.duplicated(["lake", "learner", "k", "rep"]).sum()
    )
    coverage = repeat_level.groupby(["lake", "learner", "k"], sort=False).size()
    rer_difference = np.abs(
        repeat_level["rer_log"] - repeat_level["reconstructed_rer"]
    )
    checks: dict[str, float | bool | int] = {
        "n_rows": len(repeat_level),
        "expected_rows": expected_rows,
        "n_lakes": int(repeat_level["lake"].nunique()),
        "duplicate_pairing_rows": duplicate_rows,
        "min_repeats_per_lake_cell": int(coverage.min()),
        "max_repeats_per_lake_cell": int(coverage.max()),
        "max_abs_rer_difference": float(rer_difference.max()),
        "all_rer_reconstruction_denominators_positive": bool(
            (repeat_level["rer_reconstruction_denominator"] > 0).all()
        ),
        "all_label_only_mae_positive": bool((repeat_level["label_only_log_mae"] > 0).all()),
    }
    checks["complete_coverage"] = bool(
        checks["n_rows"] == expected_rows
        and checks["n_lakes"] == expected_lakes
        and duplicate_rows == 0
        and checks["min_repeats_per_lake_cell"] == expected_repeats
        and checks["max_repeats_per_lake_cell"] == expected_repeats
        and set(repeat_level["learner"].unique()) == set(learners)
        and set(repeat_level["k"].unique()) == set(budgets)
    )
    if not checks["all_rer_reconstruction_denominators_positive"]:
        raise ValueError("Encountered a nonpositive RER reconstruction denominator.")
    if not checks["all_label_only_mae_positive"]:
        raise ValueError("Encountered nonpositive reconstructed label-only log-MAE.")
    if not checks["complete_coverage"]:
        raise ValueError("Frozen H1 repeat-level coverage is incomplete or duplicated.")
    if checks["max_abs_rer_difference"] > tolerance:
        raise ValueError("Reconstructed RER does not match frozen RER within tolerance.")
    return checks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _format(value: object) -> str:
    if pd.isna(value):
        return "not available"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6f}"
    return str(value)


def build_h1_comparison(
    absolute_summary: pd.DataFrame, frozen_h1_table: pd.DataFrame
) -> pd.DataFrame:
    """Join the new supporting absolute estimand to frozen H1 RER results."""
    frozen = frozen_h1_table.loc[
        frozen_h1_table["protocol"] == "true-nested",
        ["learner", "k", "median_RER", "mean_RER", "mean_CI_lo", "mean_CI_hi", "frac_pos"],
    ].copy()
    frozen = frozen.rename(columns={
        "median_RER": "frozen_median_rer_pct",
        "mean_RER": "frozen_mean_rer_pct",
        "mean_CI_lo": "frozen_rer_ci_lo_pct",
        "mean_CI_hi": "frozen_rer_ci_hi_pct",
        "frac_pos": "frozen_rer_frac_lakes_positive",
    })
    frozen["rer_classification"] = frozen.apply(
        lambda row: _classify_interval(
            row["frozen_rer_ci_lo_pct"], row["frozen_rer_ci_hi_pct"]
        ), axis=1
    )
    comparison = frozen.merge(absolute_summary, on=["learner", "k"], validate="one_to_one")
    comparison["delta_mae_classification"] = comparison["classification"]
    comparison["zero_classification_differs"] = (
        comparison["rer_classification"] != comparison["delta_mae_classification"]
    )
    comparison["k5_qualitative_interpretation"] = np.where(
        comparison["k"] != 5,
        "not_primary_budget",
        np.where(
            comparison["zero_classification_differs"],
            "different_zero_crossing_status",
            "same_zero_crossing_status",
        ),
    )
    return comparison


def _write_audit(
    path: Path,
    *,
    checks: dict[str, float | bool | int],
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    input_hashes: dict[str, str],
) -> None:
    k5 = summary.loc[summary["k"] == 5]
    lines = [
        "# H1 absolute log-MAE and paired delta-MAE",
        "",
        "This post hoc absolute-loss analysis complements the primary H1 relative-error-reduction endpoint.",
        "",
        "## Scope and provenance",
        "",
        "The authoritative input is the stored metric-level artifact `experiments/capacity_boundary/results/capacity_boundary_repeats_nested.csv`. It contains one calibrated row for every lake × learner × positive budget × repeat, with the stored same-support RER. The label-only log-MAE is recovered algebraically per row as `(calibrated_log_mae + frozen_rer × 1e-8) / (1 - frozen_rer)`; therefore paired delta-MAE is computed before any lake aggregation. No model fit, support/query generation, prediction generation, or stored-result rewrite occurred.",
        "",
        "The artifact does not store support/query IDs or source-only predictions on the positive-budget query subsets. Source-only output exists only at k=0 on full-lake query sets, so it is deliberately reported as unavailable here rather than mislabelled as same-query context. Row identity is verified at the frozen metric level by lake, learner, budget, and repeat keys; raw support/query identities cannot be independently rechecked from this artifact.",
        "",
        "## Pairing and reconstruction checks",
        "",
        f"- Repeat-level rows: {checks['n_rows']} / expected {checks['expected_rows']}; complete coverage: {checks['complete_coverage']}.",
        f"- Lakes: {checks['n_lakes']}; repeats per lake cell: {checks['min_repeats_per_lake_cell']}–{checks['max_repeats_per_lake_cell']}; duplicate keys: {checks['duplicate_pairing_rows']}.",
        f"- Reconstructed RER maximum absolute difference: {checks['max_abs_rer_difference']:.3e} (tolerance {RER_TOLERANCE:.0e}).",
        f"- All reconstructed label-only log-MAE values positive: {checks['all_label_only_mae_positive']}.",
        "- No denominator-zero row was removed: all 28,800 retained calibrated rows have positive recovered label-only log-MAE.",
        "",
        "## k=5 supporting absolute results",
        "",
        "Positive paired delta-MAE means lower calibrated log-MAE than the same-support label-only baseline. Values are dimensionless log10-ratio units. Intervals are 10,000-resample percentile bootstrap intervals over the 24 lake-level median paired effects, using a fresh `default_rng(0)` per learner-by-budget cell.",
        "",
        "| Learner | Mean label-only | Mean calibrated | Mean paired delta | 95% CI | Positive-lake fraction | Classification |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in k5.itertuples(index=False):
        lines.append(
            f"| {row.learner} | {_format(row.mean_label_only_log_mae)} | "
            f"{_format(row.mean_calibrated_log_mae)} | {_format(row.mean_paired_delta_mae)} | "
            f"[{_format(row.delta_ci_lo)}, {_format(row.delta_ci_hi)}] | "
            f"{_format(row.frac_lakes_delta_positive)} | {row.classification} |"
        )
    lines.extend([
        "",
        "## Relation to frozen RER",
        "",
        "The absolute and ratio effects may legitimately have different zero-crossing classifications because they weight lake-level error scales differently. Any disagreement is reported rather than used to replace the frozen primary endpoint.",
        "",
        "| Learner | k | Frozen RER class | Delta-MAE class | Different? | k=5 interpretation |",
        "|---|---:|---|---|---|---|",
    ])
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.learner} | {row.k} | {row.rer_classification} | "
            f"{row.delta_mae_classification} | {row.zero_classification_differs} | "
            f"{row.k5_qualitative_interpretation} |"
        )
    lines.extend(["", "## Frozen input hashes", ""])
    for name, digest in input_hashes.items():
        lines.append(f"- `{name}`: `{digest}`")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "This is a post hoc supporting analysis of the absolute same-support model–spectrum calibration increment beyond the label-only baseline. It is not a satellite-attributable effect, a causal decomposition, independent validation, or a new confirmatory endpoint.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_analysis(output_dir: Path) -> dict[str, object]:
    """Write absolute-loss summaries and a report from stored H1 metrics."""
    inputs = (FROZEN_REPEATS, FROZEN_SUMMARY, FROZEN_H1_TABLE)
    before_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in inputs}
    frozen_repeats = pd.read_csv(FROZEN_REPEATS)
    repeat_level = derive_repeat_level(frozen_repeats)
    checks = validate_repeat_level(
        repeat_level,
        learners=LEARNERS,
        budgets=BUDGETS,
        expected_lakes=EXPECTED_LAKES,
        expected_repeats=EXPECTED_REPEATS,
        tolerance=RER_TOLERANCE,
    )
    per_lake = summarize_per_lake(repeat_level)
    summary = summarize_across_lakes(
        per_lake, n_boot=N_BOOT, seed=BOOTSTRAP_SEED
    )
    comparison = build_h1_comparison(summary, pd.read_csv(FROZEN_H1_TABLE))

    output_dir.mkdir(parents=True, exist_ok=True)
    identity = repeat_level.assign(
        abs_rer_difference=lambda frame: np.abs(
            frame["rer_log"] - frame["reconstructed_rer"]
        ),
        label_only_mae_positive=lambda frame: frame["label_only_log_mae"] > 0,
        source_only_same_query_recoverable=False,
        source_only_unavailable_reason="source_only was stored only for k=0 full-lake query sets",
    )
    outputs = {
        "h1_absolute_repeat_level.csv": repeat_level,
        "h1_absolute_per_lake.csv": per_lake,
        "h1_absolute_summary.csv": summary,
        "h1_k5_primary_vs_absolute_comparison.csv": comparison,
        "h1_rer_identity_check.csv": identity,
    }
    for filename, frame in outputs.items():
        _write_csv(frame, output_dir / filename)

    after_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in inputs}
    if before_hashes != after_hashes:
        raise RuntimeError("An H1 input changed during aggregation.")

    audit_path = output_dir / "analysis_summary.md"
    _write_audit(
        audit_path,
        checks=checks,
        summary=summary,
        comparison=comparison,
        input_hashes=after_hashes,
    )

    manifest = {
        "analysis": "h1_absolute_loss",
        "scope": "post_hoc_supporting_absolute_same_support_increment",
        "frozen_input_sha256_before_and_after": {
            path: {"before": before_hashes[path], "after": after_hashes[path]}
            for path in before_hashes
        },
        "authoritative_artifact": str(FROZEN_REPEATS.relative_to(ROOT)),
        "source_only_same_query_recoverable": False,
        "source_only_unavailable_reason": "Frozen source-only metrics are k=0 full-lake evaluations, not positive-budget query subsets.",
        "pairing": "repeat-level; label-only loss recovered from the same row's calibrated loss and frozen RER",
        "lake_aggregation": "median of repeat-level paired delta-MAE",
        "bootstrap": {
            "resampling_unit": "lake",
            "statistic": "mean of 24 lake-level paired delta-MAE values",
            "n_boot": N_BOOT,
            "seed": BOOTSTRAP_SEED,
            "rng_scope": "fresh default_rng(0) per learner-by-budget cell",
            "interval": "percentile 2.5/97.5",
        },
        "integrity_checks": checks,
        "support_query_identity_status": "not stored in the frozen metric artifact; no split was regenerated",
        "primary_h1_replaced": False,
        "manuscripts_edited": False,
        "outputs_sha256": {
            filename: _sha256(output_dir / filename)
            for filename in outputs
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return {"checks": checks, "summary": summary, "comparison": comparison}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    result = run_analysis(parse_args().output_dir)
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
