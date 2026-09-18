"""Close H1 bootstrap provenance without changing frozen primary results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CAPACITY_RESULTS = ROOT / "experiments" / "capacity_boundary" / "results"
DEFAULT_GLOBAL = CAPACITY_RESULTS / "capacity_boundary_summary_final.csv"
DEFAULT_NESTED = CAPACITY_RESULTS / "capacity_boundary_summary_nested.csv"
DEFAULT_FROZEN = CAPACITY_RESULTS / "h1_global_vs_nested_comparison.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"

LEARNERS = ("PLSR", "XGBoost", "MLP")
BUDGETS = (1, 3, 5, 10)
SCHEMES = (
    "single_stream_seed1",
    "single_stream_seed0",
    "fresh_seed1_per_cell",
    "fresh_seed0_per_cell",
)
DIAGNOSTIC_SEEDS = tuple(range(20))
HISTORICAL_N_BOOT = 10_000
STABILITY_N_BOOT = 10_000
HIGH_B_N_BOOT = 100_000
HIGH_B_SEED = 0
ROBUST_K = 5
WINSOR_FRACTION = 0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean_ci(
    values: Iterable[float], n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Match the historical list-comprehension implementation exactly."""
    array = np.asarray(tuple(values), dtype=float)
    n_lakes = len(array)
    boot = np.array(
        [np.mean(array[rng.integers(0, n_lakes, size=n_lakes)]) for _ in range(n_boot)]
    )
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _scheme_rng(scheme: str) -> tuple[np.random.Generator | None, int, bool]:
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown RNG scheme: {scheme}")
    seed = 1 if "seed1" in scheme else 0
    fresh_per_cell = scheme.startswith("fresh_")
    return (None if fresh_per_cell else np.random.default_rng(seed), seed, fresh_per_cell)


def build_h1_table(
    global_summary_path: Path,
    nested_summary_path: Path,
    scheme: str,
    n_boot: int = HISTORICAL_N_BOOT,
) -> pd.DataFrame:
    """Rebuild the complete 24-row H1 comparison table under one RNG scheme."""
    global_summary = pd.read_csv(global_summary_path)
    nested_summary = pd.read_csv(nested_summary_path)
    shared_rng, seed, fresh_per_cell = _scheme_rng(scheme)
    rows: list[dict[str, object]] = []
    unrounded_endpoints: list[tuple[float, float]] = []

    for learner in LEARNERS:
        for k in BUDGETS:
            for protocol, frame in (
                ("global-tuned", global_summary),
                ("true-nested", nested_summary),
            ):
                subset = frame[
                    (frame["learner"] == learner)
                    & (frame["k"] == k)
                    & (frame["method"] == "calibrated")
                ]
                rer_fraction = subset["median_rer_log"].to_numpy(dtype=float)
                if len(rer_fraction) != 24:
                    raise ValueError(
                        f"Expected 24 lakes for {learner}, k={k}, {protocol}; "
                        f"found {len(rer_fraction)}"
                    )
                rng = np.random.default_rng(seed) if fresh_per_cell else shared_rng
                assert rng is not None
                ci_lo, ci_hi = bootstrap_mean_ci(rer_fraction, n_boot=n_boot, rng=rng)
                unrounded_endpoints.append((ci_lo * 100, ci_hi * 100))
                rows.append(
                    {
                        "learner": learner,
                        "k": k,
                        "protocol": protocol,
                        "median_RER": round(float(np.median(rer_fraction) * 100), 2),
                        "mean_RER": round(float(np.mean(rer_fraction) * 100), 2),
                        "mean_CI_lo": round(ci_lo * 100, 2),
                        "mean_CI_hi": round(ci_hi * 100, 2),
                        "frac_pos": round(float(np.mean(rer_fraction > 0)), 2),
                    }
                )
    table = pd.DataFrame(rows)
    table.attrs["unrounded_ci_endpoints_pct"] = unrounded_endpoints
    return table


def compare_with_frozen(
    candidate: pd.DataFrame, frozen: pd.DataFrame, scheme: str
) -> dict[str, object]:
    """Summarize equality at the precision actually stored in the frozen CSV."""
    candidate = candidate.reset_index(drop=True)
    frozen = frozen.reset_index(drop=True)
    endpoint_diff = np.abs(
        candidate[["mean_CI_lo", "mean_CI_hi"]].to_numpy(dtype=float)
        - frozen[["mean_CI_lo", "mean_CI_hi"]].to_numpy(dtype=float)
    )
    unrounded = np.asarray(
        candidate.attrs["unrounded_ci_endpoints_pct"], dtype=float
    )
    unrounded_to_stored_diff = np.abs(
        unrounded
        - frozen[["mean_CI_lo", "mean_CI_hi"]].to_numpy(dtype=float)
    )
    exact = candidate.equals(frozen) or all(
        candidate[column].equals(frozen[column])
        for column in frozen.columns
    )
    return {
        "scheme": scheme,
        "rng_scope": "fresh_per_cell" if scheme.startswith("fresh_") else "single_stream",
        "seed": 1 if "seed1" in scheme else 0,
        "n_boot": HISTORICAL_N_BOOT,
        "exact_full_table_match": bool(exact),
        "matching_endpoint_values": int(np.sum(endpoint_diff == 0)),
        "total_endpoint_values": int(endpoint_diff.size),
        "max_abs_endpoint_diff_pct": float(np.max(endpoint_diff)),
        "max_abs_unrounded_endpoint_to_stored_pct": float(
            np.max(unrounded_to_stored_diff)
        ),
    }


def ci_classification(lo: float, hi: float) -> str:
    if lo > 0:
        return "above_zero"
    if hi < 0:
        return "below_zero"
    return "crosses_zero"


def build_seed_stability(nested_summary_path: Path) -> pd.DataFrame:
    nested = pd.read_csv(nested_summary_path)
    rows: list[dict[str, object]] = []
    for learner in LEARNERS:
        for k in BUDGETS:
            subset = nested[
                (nested["learner"] == learner)
                & (nested["k"] == k)
                & (nested["method"] == "calibrated")
            ]
            values = subset["median_rer_log"].to_numpy(dtype=float)
            median_pct = float(np.median(values) * 100)
            for seed in DIAGNOSTIC_SEEDS:
                lo, hi = bootstrap_mean_ci(
                    values, STABILITY_N_BOOT, np.random.default_rng(seed)
                )
                rows.append(
                    {
                        "learner": learner,
                        "k": k,
                        "seed": seed,
                        "n_boot": STABILITY_N_BOOT,
                        "median_RER_pct": median_pct,
                        "median_ge_10pct": bool(median_pct >= 10),
                        "mean_CI_lo_pct": lo * 100,
                        "mean_CI_hi_pct": hi * 100,
                        "zero_classification": ci_classification(lo, hi),
                    }
                )
    return pd.DataFrame(rows)


def summarize_seed_stability(stability: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (learner, k), group in stability.groupby(["learner", "k"], sort=False):
        classifications = sorted(group["zero_classification"].unique())
        rows.append(
            {
                "learner": learner,
                "k": int(k),
                "n_seeds": int(group["seed"].nunique()),
                "min_CI_lo_pct": float(group["mean_CI_lo_pct"].min()),
                "max_CI_lo_pct": float(group["mean_CI_lo_pct"].max()),
                "min_CI_hi_pct": float(group["mean_CI_hi_pct"].min()),
                "max_CI_hi_pct": float(group["mean_CI_hi_pct"].max()),
                "classifications": ";".join(classifications),
                "classification_changes_across_seeds": len(classifications) > 1,
            }
        )
    return pd.DataFrame(rows)


def build_k5_high_b(nested_summary_path: Path) -> pd.DataFrame:
    nested = pd.read_csv(nested_summary_path)
    rows: list[dict[str, object]] = []
    for learner in LEARNERS:
        subset = nested[
            (nested["learner"] == learner)
            & (nested["k"] == 5)
            & (nested["method"] == "calibrated")
        ]
        values = subset["median_rer_log"].to_numpy(dtype=float)
        lo, hi = bootstrap_mean_ci(
            values, HIGH_B_N_BOOT, np.random.default_rng(HIGH_B_SEED)
        )
        rows.append(
            {
                "learner": learner,
                "k": 5,
                "seed": HIGH_B_SEED,
                "n_boot": HIGH_B_N_BOOT,
                "median_RER_pct": float(np.median(values) * 100),
                "mean_CI_lo_pct": lo * 100,
                "mean_CI_hi_pct": hi * 100,
                "zero_classification": ci_classification(lo, hi),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_samples(
    values: np.ndarray, n_boot: int, seed: int
) -> np.ndarray:
    """Draw target-unit bootstrap samples with a fresh deterministic stream."""
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_boot, len(array)))
    return array[indices]


def _bca_mean_ci(
    values: np.ndarray, bootstrap_means: np.ndarray, alpha: float = 0.05
) -> tuple[float, float]:
    """Bias-corrected and accelerated interval for the arithmetic mean."""
    array = np.asarray(values, dtype=float)
    observed = float(np.mean(array))
    less_fraction = float(np.mean(bootstrap_means < observed))
    eps = 0.5 / len(bootstrap_means)
    less_fraction = min(max(less_fraction, eps), 1.0 - eps)

    normal = NormalDist()
    z0 = normal.inv_cdf(less_fraction)
    jackknife = np.array(
        [np.mean(np.delete(array, i)) for i in range(len(array))], dtype=float
    )
    jackknife_center = float(np.mean(jackknife))
    deviations = jackknife_center - jackknife
    denominator = 6.0 * float(np.sum(deviations**2)) ** 1.5
    acceleration = (
        float(np.sum(deviations**3)) / denominator if denominator > 0 else 0.0
    )

    adjusted = []
    for probability in (alpha / 2.0, 1.0 - alpha / 2.0):
        z_alpha = normal.inv_cdf(probability)
        numerator = z0 + z_alpha
        adjusted_probability = normal.cdf(
            z0 + numerator / (1.0 - acceleration * numerator)
        )
        adjusted.append(min(max(adjusted_probability, 0.0), 1.0))
    lo, hi = np.quantile(bootstrap_means, adjusted)
    return float(lo), float(hi)


def _winsorized_rows(
    samples: np.ndarray, fraction: float = WINSOR_FRACTION
) -> np.ndarray:
    """Winsorize each bootstrap sample independently at both tails."""
    ordered = np.sort(np.asarray(samples, dtype=float), axis=1)
    tail_count = int(np.floor(fraction * ordered.shape[1]))
    if tail_count == 0:
        return ordered
    ordered[:, :tail_count] = ordered[:, [tail_count]]
    ordered[:, -tail_count:] = ordered[:, [-tail_count - 1]]
    return ordered


def build_k5_robust_inference(
    nested_summary_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post hoc H1 sensitivity using frozen k=5 target-unit summaries only."""
    nested = pd.read_csv(nested_summary_path)
    summary_rows: list[dict[str, object]] = []
    delete_one_rows: list[dict[str, object]] = []

    for learner in LEARNERS:
        subset = nested[
            (nested["learner"] == learner)
            & (nested["k"] == ROBUST_K)
            & (nested["method"] == "calibrated")
        ][["lake", "median_rer_log"]].dropna().sort_values("lake")
        if len(subset) != 24 or subset["lake"].nunique() != 24:
            raise ValueError(
                f"Expected 24 unique target units for {learner}, "
                f"k={ROBUST_K}; found {len(subset)} rows and "
                f"{subset['lake'].nunique()} unique names"
            )

        values = subset["median_rer_log"].to_numpy(dtype=float)
        samples = _bootstrap_samples(values, HIGH_B_N_BOOT, HIGH_B_SEED)
        bootstrap_means = np.mean(samples, axis=1)
        percentile_lo, percentile_hi = np.percentile(
            bootstrap_means, [2.5, 97.5]
        )
        bca_lo, bca_hi = _bca_mean_ci(values, bootstrap_means)

        winsorized = _winsorized_rows(samples)
        winsorized_means = np.mean(winsorized, axis=1)
        winsor_lo, winsor_hi = np.percentile(winsorized_means, [2.5, 97.5])
        observed_winsorized_mean = float(
            np.mean(_winsorized_rows(values.reshape(1, -1))[0])
        )

        for omitted_lake in subset["lake"]:
            retained = subset.loc[
                subset["lake"] != omitted_lake, "median_rer_log"
            ].to_numpy(dtype=float)
            retained_samples = _bootstrap_samples(
                retained, HIGH_B_N_BOOT, HIGH_B_SEED
            )
            retained_means = np.mean(retained_samples, axis=1)
            loo_lo, loo_hi = np.percentile(retained_means, [2.5, 97.5])
            delete_one_rows.append(
                {
                    "learner": learner,
                    "k": ROBUST_K,
                    "omitted_target_unit": omitted_lake,
                    "n_target_units": len(retained),
                    "mean_RER_pct": float(np.mean(retained) * 100),
                    "percentile_CI_lo_pct": float(loo_lo * 100),
                    "percentile_CI_hi_pct": float(loo_hi * 100),
                    "zero_classification": ci_classification(loo_lo, loo_hi),
                }
            )

        learner_loo = [
            row for row in delete_one_rows if row["learner"] == learner
        ]
        dianchi = next(
            row
            for row in learner_loo
            if row["omitted_target_unit"] == "Dianchi"
        )
        summary_rows.append(
            {
                "learner": learner,
                "k": ROBUST_K,
                "n_target_units": len(values),
                "mean_RER_pct": float(np.mean(values) * 100),
                "Dianchi_RER_pct": float(
                    subset.loc[
                        subset["lake"] == "Dianchi", "median_rer_log"
                    ].iloc[0]
                    * 100
                ),
                "percentile_highB_CI_lo_pct": float(percentile_lo * 100),
                "percentile_highB_CI_hi_pct": float(percentile_hi * 100),
                "percentile_highB_class": ci_classification(
                    percentile_lo, percentile_hi
                ),
                "BCa_CI_lo_pct": float(bca_lo * 100),
                "BCa_CI_hi_pct": float(bca_hi * 100),
                "BCa_class": ci_classification(bca_lo, bca_hi),
                "winsor_fraction_each_tail": WINSOR_FRACTION,
                "winsorized_mean_RER_pct": observed_winsorized_mean * 100,
                "winsorized_percentile_CI_lo_pct": float(winsor_lo * 100),
                "winsorized_percentile_CI_hi_pct": float(winsor_hi * 100),
                "winsorized_class": ci_classification(winsor_lo, winsor_hi),
                "delete_one_above_zero_count": sum(
                    row["zero_classification"] == "above_zero"
                    for row in learner_loo
                ),
                "delete_one_crosses_zero_count": sum(
                    row["zero_classification"] == "crosses_zero"
                    for row in learner_loo
                ),
                "delete_one_min_CI_lo_pct": min(
                    float(row["percentile_CI_lo_pct"]) for row in learner_loo
                ),
                "delete_one_max_CI_lo_pct": max(
                    float(row["percentile_CI_lo_pct"]) for row in learner_loo
                ),
                "Dianchi_omitted_mean_RER_pct": dianchi["mean_RER_pct"],
                "Dianchi_omitted_CI_lo_pct": dianchi[
                    "percentile_CI_lo_pct"
                ],
                "Dianchi_omitted_CI_hi_pct": dianchi[
                    "percentile_CI_hi_pct"
                ],
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(delete_one_rows)


def _historical_evidence(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {"available": False}
    lines = path.read_bytes().splitlines(keepends=True)
    evidence: dict[str, object] = {
        "available": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "one_based_line_numbers": [832, 833],
    }
    line_records = []
    for line_number in (832, 833):
        raw = lines[line_number - 1]
        record = json.loads(raw)
        line_records.append(
            {
                "line": line_number,
                "timestamp_utc": record.get("timestamp"),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    evidence["line_records"] = line_records
    return evidence


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\r\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-summary", type=Path, default=DEFAULT_GLOBAL)
    parser.add_argument("--nested-summary", type=Path, default=DEFAULT_NESTED)
    parser.add_argument("--frozen-table", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--historical-session-log", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = (args.global_summary, args.nested_summary, args.frozen_table)
    before_hashes = {str(path.resolve()): sha256_file(path) for path in inputs}
    frozen = pd.read_csv(args.frozen_table)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    canonical = None
    for scheme in SCHEMES:
        candidate = build_h1_table(
            args.global_summary, args.nested_summary, scheme, HISTORICAL_N_BOOT
        )
        comparison_rows.append(compare_with_frozen(candidate, frozen, scheme))
        if scheme == "fresh_seed0_per_cell":
            canonical = candidate
    assert canonical is not None

    comparison = pd.DataFrame(comparison_rows)
    stability = build_seed_stability(args.nested_summary)
    stability_summary = summarize_seed_stability(stability)
    high_b = build_k5_high_b(args.nested_summary)
    robust_summary, delete_one = build_k5_robust_inference(args.nested_summary)

    output_frames = {
        "rng_scheme_comparison.csv": comparison,
        "canonical_h1_reconstruction.csv": canonical,
        "bootstrap_seed_stability.csv": stability,
        "bootstrap_seed_stability_summary.csv": stability_summary,
        "k5_highB_diagnostic.csv": high_b,
        "k5_robust_inference_summary.csv": robust_summary,
        "k5_delete_one_diagnostic.csv": delete_one,
    }
    for filename, frame in output_frames.items():
        write_csv(frame, args.output_dir / filename)

    after_hashes = {str(path.resolve()): sha256_file(path) for path in inputs}
    if before_hashes != after_hashes:
        raise RuntimeError("A frozen input changed during the audit")

    manifest = {
        "analysis": "h1_bootstrap_provenance_closure",
        "evidence_class": "VERIFIED",
        "historical_algorithm": {
            "resampling_unit": "lake",
            "statistic": "mean of 24 lake-level median RER values",
            "interval": "percentile 2.5/97.5",
            "n_boot": HISTORICAL_N_BOOT,
            "rng_api": "numpy.random.default_rng",
            "rng_scope": "fresh per learner-budget-protocol cell",
            "seed": 0,
            "historical_iteration_order": {
                "learner": list(LEARNERS),
                "k": list(BUDGETS),
                "protocol": ["global-tuned", "true-nested"],
            },
        },
        "historical_execution_evidence": _historical_evidence(
            args.historical_session_log
        ),
        "git_provenance": {
            "commit": "215b44c4fa0b5e3004b7d13c62544258276440c7",
            "commit_time": "2026-07-13T12:34:55+08:00",
            "frozen_csv_blob": "1845f148d0ea59e3a2018c7b3aa550dfbb7d3d79",
            "contemporary_script_blob": "84cf894cefe8b37da3587d1104e2d09547470eee",
            "note": "The contemporary repository script did not generate the frozen comparison CSV; the historical session command did.",
        },
        "diagnostics_predeclared": {
            "stability_seeds": list(DIAGNOSTIC_SEEDS),
            "stability_n_boot": STABILITY_N_BOOT,
            "k5_high_b_seed": HIGH_B_SEED,
            "k5_high_b_n_boot": HIGH_B_N_BOOT,
        },
        "post_hoc_robust_inference_sensitivity": {
            "budget": ROBUST_K,
            "n_boot": HIGH_B_N_BOOT,
            "seed": HIGH_B_SEED,
            "methods": [
                "percentile mean interval",
                "bias-corrected and accelerated mean interval",
                "10 percent per-tail winsorized-mean percentile interval",
                "delete-one-target-unit percentile mean intervals",
            ],
            "winsorization_scope": "each bootstrap sample independently",
            "primary_results_replaced": False,
        },
        "frozen_input_sha256_before_and_after": {
            path: {"before": before_hashes[path], "after": after_hashes[path]}
            for path in before_hashes
        },
        "outputs_sha256": {
            filename: sha256_file(args.output_dir / filename)
            for filename in output_frames
        },
        "canonical_exact_full_table_match": bool(
            comparison.loc[
                comparison["scheme"] == "fresh_seed0_per_cell",
                "exact_full_table_match",
            ].iloc[0]
        ),
        "primary_results_replaced": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(comparison.to_string(index=False))
    print("\nSeed-stability classification changes:")
    print(
        stability_summary[
            ["learner", "k", "classification_changes_across_seeds"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
