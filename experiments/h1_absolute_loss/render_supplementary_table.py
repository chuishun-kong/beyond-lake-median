"""Render deterministic Supplementary Table S6 from absolute-loss results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "h1_absolute_loss" / "results"
INPUT = RESULTS / "h1_k5_primary_vs_absolute_comparison.csv"
OUTPUT = RESULTS / "manuscript_table_s6.md"
LEARNER_ORDER = {"PLSR": 0, "XGBoost": 1, "MLP": 2}


def render(root: Path = ROOT) -> str:
    """Return the complete, manuscript-ready post hoc absolute-loss table."""
    comparison = pd.read_csv(root / INPUT.relative_to(ROOT))
    comparison["learner_order"] = comparison["learner"].map(LEARNER_ORDER)
    comparison = comparison.sort_values(["learner_order", "k"])
    lines = [
        "## Supplementary Table S6 — Post hoc supporting absolute-loss analysis",
        "",
        "**Table S6. Post hoc supporting absolute-loss analysis for H1.** Values are dimensionless log10-ratio units. Label-only and calibrated values are across-lake summaries of lake-level medians; paired delta-MAE is the across-lake mean of lake-level medians of repeat-paired differences. Positive delta-MAE favors calibrated prediction. Intervals are 10,000-resample percentile-bootstrap intervals over lakes. This table does not replace the primary RER inference.",
        "",
        "| Learner | k | Mean label-only | Median label-only | Mean calibrated | Median calibrated | Mean paired delta-MAE | Median paired delta-MAE | 95% CI on mean paired delta-MAE | Positive-lake fraction | Frozen RER class | Absolute delta-MAE class | Classification differs? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in comparison.itertuples(index=False):
        differs = "yes" if row.zero_classification_differs else "no"
        lines.append(
            f"| {row.learner} | {row.k} | {row.mean_label_only_log_mae:.6f} | "
            f"{row.median_label_only_log_mae:.6f} | {row.mean_calibrated_log_mae:.6f} | "
            f"{row.median_calibrated_log_mae:.6f} | {row.mean_paired_delta_mae:.6f} | "
            f"{row.median_paired_delta_mae:.6f} | [{row.delta_ci_lo:.6f}, {row.delta_ci_hi:.6f}] | "
            f"{row.frac_lakes_delta_positive:.0%} | {row.rer_classification} | "
            f"{row.delta_mae_classification} | {differs} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"Rendered {OUTPUT}")


if __name__ == "__main__":
    main()
