from experiments.h2_common_query_evidence.run_h2_common_query_evidence import (
    audit_frozen_outputs,
    build_decision_matrix,
    classify_nominal_ci,
)
import pandas as pd
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_classify_nominal_ci_uses_frozen_h2_direction():
    assert classify_nominal_ci(0.001, 0.02) == "selector_better"
    assert classify_nominal_ci(-0.02, -0.001) == "random_better"
    assert classify_nominal_ci(-0.01, 0.01) == "unresolved"


def test_decision_matrix_only_assigns_primary_to_weighted_selector():
    own = pd.DataFrame([
        {"learner": "PLSR", "k": 5, "classification": "unresolved"},
    ])
    pairwise = pd.DataFrame([
        {"learner": "PLSR", "k": 5, "method": "farthest_point_kennard_stone", "classification": "random_better", "n_lakes": 24},
        {"learner": "PLSR", "k": 5, "method": "proposed_weighted_d_optimal", "classification": "unresolved", "n_lakes": 24},
    ])
    fourway = pairwise.copy()

    matrix = build_decision_matrix(own, pairwise, fourway)

    ks = matrix.loc[matrix["selector"] == "farthest_point_kennard_stone"].iloc[0]
    weighted = matrix.loc[matrix["selector"] == "proposed_weighted_d_optimal"].iloc[0]
    assert ks["own_pool_classification"] == "not_applicable"
    assert weighted["own_pool_classification"] == "unresolved"
    assert weighted["classification_change_pairwise_vs_fourway"] == False


def test_frozen_h2_audit_retains_all_cells_and_k10_restriction():
    audited = audit_frozen_outputs(REPO_ROOT)

    assert len(audited["pairwise"]) == 36
    assert len(audited["fourway"]) == 36
    assert len(audited["decision_matrix"]) == 36
    assert audited["attrition"].loc[audited["attrition"]["k"] == 10, "frac_usable"].item() == 0.7547
    assert audited["checks"]["fourway_k10_low_coverage_min_usable_repeats"] == 1
    assert audited["checks"]["fourway_k10_low_coverage_max_usable_repeats"] == 29
    assert audited["fourway"].loc[
        audited["fourway"]["k"] == 10, "row_level_usable_repeat_restriction"
    ].all()


def test_renderer_makes_attrition_a_standalone_table():
    from experiments.h2_common_query_evidence.render_supplementary_tables import render_all

    tables = render_all(REPO_ROOT)

    assert set(tables) == {
        "manuscript_table_s7.md",
        "manuscript_table_s8.md",
        "manuscript_table_s9.md",
        "manuscript_table_s10.md",
    }
    assert "## Supplementary Table S7" in tables["manuscript_table_s7.md"]
    assert "Pairwise common-query" in tables["manuscript_table_s7.md"]
    assert "Primary own-pool class" in tables["manuscript_table_s7.md"]
    assert "not_primary_comparison" in tables["manuscript_table_s7.md"]
    assert "Available-repeat" in tables["manuscript_table_s8.md"]
    assert "## Supplementary Table S9" in tables["manuscript_table_s9.md"]
    assert "75.47%" in tables["manuscript_table_s9.md"]
    assert "At k=10, all 72 lake--learner units entered" in tables["manuscript_table_s9.md"]
    assert "1--29" in tables["manuscript_table_s9.md"]
    assert "restricted-pool sensitivity" not in tables["manuscript_table_s9.md"]
    for name in [
        "manuscript_table_s7.md",
        "manuscript_table_s8.md",
        "manuscript_table_s9.md",
        "manuscript_table_s10.md",
    ]:
        assert "unresolved does not establish equivalence" in tables[name]
    assert "Post hoc >=50-usable-repeat" in tables["manuscript_table_s10.md"]


def test_run_analysis_writes_a_complete_frozen_provenance_manifest(tmp_path):
    from experiments.h2_common_query_evidence.run_h2_common_query_evidence import run_analysis

    result = run_analysis(REPO_ROOT, tmp_path)

    assert result["checks"]["raw_common_runtime_sign_max_abs_error"] <= 1e-12
    assert (tmp_path / "h2_pairwise_common_query_audited.csv").is_file()
    assert (tmp_path / "h2_fourway_common_query_audited.csv").is_file()
    assert (tmp_path / "h2_fourway_available_repeat_audited.csv").is_file()
    assert (tmp_path / "h2_fourway_repeat_coverage.csv").is_file()
    assert (tmp_path / "h2_fourway_threshold50_sensitivity.csv").is_file()
    assert (tmp_path / "h2_fourway_available_vs_threshold50.csv").is_file()
    assert (tmp_path / "h2_query_attrition_audited.csv").is_file()
    assert (tmp_path / "h2_cross_estimand_decision_matrix.csv").is_file()
    assert (tmp_path / "manuscript_table_s10.md").is_file()
    assert (tmp_path / "run_manifest.json").is_file()
    assert (tmp_path / "analysis_summary.md").is_file()
    assert (tmp_path / "fourway_aggregation.md").is_file()


def test_fourway_threshold50_derivation_preserves_frozen_and_restricted_estimands():
    from experiments.h2_common_query_evidence.run_h2_common_query_evidence import (
        derive_fourway_threshold50,
    )

    per_repeat = pd.read_csv(
        REPO_ROOT / "experiments" / "active_selection" / "results" / "true_nested_h2" /
        "h2_nested_per_repeat.csv"
    )
    result = derive_fourway_threshold50(per_repeat)

    assert len(result["per_lake"]) == 864
    assert len(result["summary"]) == 36
    assert result["summary"].loc[result["summary"]["k"] == 10, "n_lakes"].eq(18).all()
    assert result["summary"].loc[result["summary"]["k"] == 10, "n_lakes_excluded"].eq(6).all()
    assert not result["summary"]["classification"].eq("selector_better").any()


def test_available_repeat_coverage_reports_all_lakes_and_low_coverage_units():
    from experiments.h2_common_query_evidence.run_h2_common_query_evidence import (
        audit_available_repeat_coverage,
    )

    source = REPO_ROOT / "experiments" / "active_selection" / "results" / "true_nested_h2"
    coverage = audit_available_repeat_coverage(
        pd.read_csv(source / "h2_nested_per_repeat.csv"),
        pd.read_csv(source / "h2_nested_fourway_common_query.csv"),
    )

    k10 = coverage.loc[coverage["k"] == 10]
    assert len(coverage) == 36
    assert k10["n_lakes_available_repeat"].eq(24).all()
    assert k10["n_lakes_usable_repeats_50plus"].eq(18).all()
    assert k10["n_lakes_usable_repeats_lt50"].eq(6).all()
    assert k10["min_usable_repeats_per_lake"].isin([1, 3]).all()


def test_available_vs_threshold50_comparison_preserves_no_selector_advantage():
    from experiments.h2_common_query_evidence.run_h2_common_query_evidence import (
        compare_available_vs_threshold50,
        derive_fourway_threshold50,
    )

    source = REPO_ROOT / "experiments" / "active_selection" / "results" / "true_nested_h2"
    comparison = compare_available_vs_threshold50(
        pd.read_csv(source / "h2_nested_fourway_common_query.csv"),
        derive_fourway_threshold50(pd.read_csv(source / "h2_nested_per_repeat.csv"))["summary"],
    )

    assert len(comparison) == 36
    assert not comparison["threshold50_classification"].eq("selector_better").any()
    assert not comparison["classification_changed"].any()
