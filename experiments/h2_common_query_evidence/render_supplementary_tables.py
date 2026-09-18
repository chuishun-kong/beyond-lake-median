"""Render deterministic H2 common-query supporting tables from frozen outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from experiments.h2_common_query_evidence.run_h2_common_query_evidence import (
        audit_available_repeat_coverage,
        audit_frozen_outputs,
        compare_available_vs_threshold50,
        derive_fourway_threshold50,
    )
except ModuleNotFoundError:  # Direct execution alongside the audit script.
    from run_h2_common_query_evidence import (
        audit_available_repeat_coverage,
        audit_frozen_outputs,
        compare_available_vs_threshold50,
        derive_fourway_threshold50,
    )


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "h2_common_query_evidence" / "results"
LEARNER_ORDER = {"PLSR": 0, "XGBoost": 1, "MLP": 2}
METHOD_LABELS = {
    "farthest_point_kennard_stone": "Kennard--Stone",
    "unweighted_d_optimal": "Unweighted D-optimal",
    "proposed_weighted_d_optimal": "Weighted D-optimal",
}


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(
        _learner_order=frame["learner"].map(LEARNER_ORDER),
        _method_order=frame["method"].map(METHOD_LABELS),
    ).sort_values(["_learner_order", "k", "_method_order"])


def _effect_rows(frame: pd.DataFrame) -> list[str]:
    rows = []
    for row in _ordered(frame).itertuples(index=False):
        rows.append(
            f"| {row.learner} | {row.k} | {METHOD_LABELS[row.method]} | "
            f"{row.mean_delta_mae:.6f} | {row.median_delta_mae:.6f} | "
            f"[{row.ci_lo:.6f}, {row.ci_hi:.6f}] | {row.classification} | "
            f"{row.n_lakes} | {row.n_repeat_rows} | "
            f"{row.min_repeats_per_lake}--{row.max_repeats_per_lake} | "
            f"{row.primary_own_pool_classification} |"
        )
    return rows


def _available_fourway_rows(frame: pd.DataFrame) -> list[str]:
    rows = []
    for row in _ordered(frame).itertuples(index=False):
        rows.append(
            f"| {row.learner} | {row.k} | {METHOD_LABELS[row.method]} | "
            f"{row.mean_delta_mae:.6f} | {row.median_delta_mae:.6f} | "
            f"[{row.ci_lo:.6f}, {row.ci_hi:.6f}] | {row.classification} | "
            f"{row.n_lakes_available_repeat} | {row.usable_repeats} | "
            f"{row.usable_repeat_fraction:.2%} | {row.min_usable_repeats_per_lake:g} | "
            f"{row.median_usable_repeats_per_lake:g} | {row.max_usable_repeats_per_lake:g} | "
            f"{row.n_lakes_usable_repeats_50plus} | {row.n_lakes_usable_repeats_lt50} |"
        )
    return rows


def _threshold50_rows(frame: pd.DataFrame) -> list[str]:
    rows = []
    for row in _ordered(frame.rename(columns={"method": "method"})).itertuples(index=False):
        changed = "yes" if row.classification_changed else "no"
        rows.append(
            f"| {row.learner} | {row.k} | {METHOD_LABELS[row.method]} | "
            f"{row.threshold50_n_lakes} | {row.threshold50_n_lakes_excluded} | "
            f"{row.threshold50_mean_delta_mae:.6f} | {row.threshold50_median_delta_mae:.6f} | "
            f"[{row.threshold50_ci_lo:.6f}, {row.threshold50_ci_hi:.6f}] | "
            f"{row.threshold50_positive_lake_fraction:.0%} | "
            f"{row.available_repeat_classification} | {row.threshold50_classification} | {changed} |"
        )
    return rows


def render_all(root: Path = ROOT) -> dict[str, str]:
    """Return the complete Tables S7--S10; no manuscript files are edited."""
    audit = audit_frozen_outputs(root)
    pairwise = audit["pairwise"].merge(
        audit["decision_matrix"]["learner k selector own_pool_classification".split()].rename(
            columns={
                "selector": "method",
                "own_pool_classification": "primary_own_pool_classification",
            }
        ),
        on=["learner", "k", "method"],
        validate="one_to_one",
    )
    pairwise["primary_own_pool_classification"] = pairwise[
        "primary_own_pool_classification"
    ].replace({"not_applicable": "not_primary_comparison"})
    fourway = audit["fourway"]
    attrition = audit["attrition"].sort_values("k")
    source = root / "experiments" / "active_selection" / "results" / "true_nested_h2"
    per_repeat = pd.read_csv(source / "h2_nested_per_repeat.csv")
    frozen_fourway = pd.read_csv(source / "h2_nested_fourway_common_query.csv")
    available = frozen_fourway.merge(
        audit_available_repeat_coverage(per_repeat, frozen_fourway),
        on=["learner", "k", "method"],
        validate="one_to_one",
    )
    threshold50 = derive_fourway_threshold50(per_repeat)["summary"]
    comparison = compare_available_vs_threshold50(frozen_fourway, threshold50)
    frozen_units_by_k = available.loc[
        available["method"] == "proposed_weighted_d_optimal"
    ].groupby("k")["n_lakes_available_repeat"].sum()
    s7 = [
        "## Supplementary Table S7 — H2 pairwise common-query sensitivity",
        "",
        "**Table S7. Pairwise common-query H2 sensitivity.** Each selector is "
        "compared with the same random support on repeat-specific "
        "$Q_{sR}=\\mathrm{pool}\\setminus(S_s\\cup S_R)$. Delta-MAE is "
        "`MAE_random - MAE_selector`; positive values favor the selector. "
        "Each cell is the across-lake mean of lake-level median paired effects; "
        "intervals are nominal, cell-wise 10,000-resample percentile bootstrap "
        "intervals over lakes and are not multiplicity-adjusted. This is a secondary "
        "sensitivity; unresolved does not establish equivalence. The primary own-pool "
        "classification applies only to weighted D-optimal.",
        "",
        "| Learner | k | Selector | Mean delta-MAE | Median delta-MAE | 95% CI | Classification | Lakes | Repeat rows | Repeats/lake | Primary own-pool class |",
        "|---|---:|---|---:|---:|---:|---|---:|---:|---:|---|",
        *_effect_rows(pairwise),
        "",
    ]
    s8 = [
        "## Supplementary Table S8 — Frozen all-24-lake available-repeat four-way common-query sensitivity",
        "",
        "**Table S8. Available-repeat four-way common-query H2 sensitivity.** Every selector is "
        "compared with the same random support on repeat-specific "
        "$Q_{all}=\\mathrm{pool}\\setminus(S_R\\cup S_{KS}\\cup S_{unweighted}\\cup S_{weighted})$. "
        "Delta-MAE is `MAE_random - MAE_selector`; positive values favor the "
        "selector. The frozen analysis retained every lake with available usable "
        "repetitions, so all 24 lakes entered each cell while within-lake usable-repeat "
        "coverage varied. Intervals are nominal cell-wise 10,000-resample lake-bootstrap "
        "intervals and are not multiplicity-adjusted. This is a secondary sensitivity; "
        "it is not a >=50-repeat lake-entry analysis, and unresolved does not establish "
        "equivalence. Table S9 gives coverage details.",
        "",
        "| Learner | k | Selector | Mean delta-MAE | Median delta-MAE | 95% CI | Classification | Lakes entered | Usable repeats | Usable fraction | Min usable repeats/lake | Median | Max | Lakes >=50 | Lakes <50 |",
        "|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *_available_fourway_rows(available),
        "",
    ]
    s9 = [
        "## Supplementary Table S9 — Four-way common-query usable-repeat coverage",
        "",
        "**Table S9. Four-way common-query eligibility and attrition.** A repeat "
        "is usable when its common query has at least 5 observations and at least "
        "20% of the target pool. The frozen all-24-lake summary omitted unusable repeat "
        "rows, then used every lake--learner unit with available usable repeats; >=50 is "
        "only a coverage diagnostic. This standalone table is included because the k-level "
        "query-size and coverage pattern cannot be represented transparently within the "
        "36-cell Table S8. At k=10, all 72 lake--learner units entered the frozen summary; "
        "the 18 low-coverage units had 1--29 usable repeats. This is a secondary sensitivity; "
        "unresolved does not establish equivalence.",
        "",
        "| k | Candidate repeats | Usable repeats | Usable fraction | Lake--learner units >=50 (of 72) | Units <50 | Units entering frozen summary | Median query size | Minimum | Maximum | Scope |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in attrition.itertuples(index=False):
        units_entered = int(frozen_units_by_k.loc[row.k])
        scope = (
            "all-lake available-repeat sensitivity (attrition > 0)"
            if row.k == 10
            else "full pool"
        )
        s9.append(
            f"| {row.k} | {row.n_lake_learner_rep_units_total} | {row.n_usable_units} | "
            f"{row.frac_usable:.2%} | {row.n_lake_learner_units_50plus_repeats} | "
            f"{units_entered - row.n_lake_learner_units_50plus_repeats} | {units_entered} | "
            f"{row.query_size_median:g} | {row.query_size_min:g} | "
            f"{row.query_size_max:g} | {scope} |"
        )
    s10 = [
        "## Supplementary Table S10 — Post hoc >=50-usable-repeat restricted-pool sensitivity",
        "",
        "**Table S10. Post hoc >=50-usable-repeat restricted-pool four-way sensitivity.** "
        "Using the frozen repeat-level four-way metrics only, a lake--learner unit is "
        "included when it has at least 50 usable repetitions. Delta-MAE is "
        "`MAE_random - MAE_selector`; positive values favor the selector. Each cell "
        "is the across-lake mean of lake-level median paired effects with a nominal, "
        "cell-wise 10,000-resample percentile lake-bootstrap interval and no multiplicity "
        "adjustment. This post hoc usability-defined estimand does not replace the frozen "
        "all-24-lake available-repeat sensitivity; unresolved does not establish equivalence.",
        "",
        "| Learner | k | Selector | Included lakes | Excluded lakes | Mean delta-MAE | Median delta-MAE | 95% CI | Positive-lake fraction | Frozen available-repeat class | Threshold-50 class | Classification changed? |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        *_threshold50_rows(comparison),
        "",
    ]
    return {
        "manuscript_table_s7.md": "\n".join(s7),
        "manuscript_table_s8.md": "\n".join(s8),
        "manuscript_table_s9.md": "\n".join(s9) + "\n",
        "manuscript_table_s10.md": "\n".join(s10),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for filename, text in render_all().items():
        (RESULTS / filename).write_text(text, encoding="utf-8", newline="\n")
        print(f"Rendered {RESULTS / filename}")


if __name__ == "__main__":
    main()
