import pandas as pd
import pytest
from pathlib import Path

from experiments.h1_absolute_loss.run_h1_absolute_loss import (
    derive_repeat_level,
    summarize_across_lakes,
    summarize_per_lake,
    validate_repeat_level,
)
from experiments.h1_absolute_loss.render_supplementary_table import render


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_derive_repeat_level_recovers_same_support_label_loss_and_pairing():
    frozen = pd.DataFrame(
        [
            {
                "lake": "Example Lake",
                "learner": "PLSR",
                "k": 5,
                "rep": 0,
                "method": "calibrated",
                "mae_log": 0.3,
                "rer_log": 0.25,
            }
        ]
    )

    derived = derive_repeat_level(frozen)

    assert derived.loc[0, "label_only_log_mae"] == pytest.approx(0.4)
    assert derived.loc[0, "calibrated_log_mae"] == pytest.approx(0.3)
    assert derived.loc[0, "paired_delta_mae"] == pytest.approx(0.1)
    assert derived.loc[0, "reconstructed_rer"] == pytest.approx(0.25)
    assert pd.isna(derived.loc[0, "source_only_log_mae"])


def test_summarize_per_lake_uses_median_of_paired_repeat_differences():
    repeat_level = pd.DataFrame(
        [
            {
                "lake": "Example Lake", "learner": "PLSR", "k": 5, "rep": 0,
                "label_only_log_mae": 0.8, "calibrated_log_mae": 0.7,
                "source_only_log_mae": float("nan"), "paired_delta_mae": 0.1,
            },
            {
                "lake": "Example Lake", "learner": "PLSR", "k": 5, "rep": 1,
                "label_only_log_mae": 0.3, "calibrated_log_mae": 0.1,
                "source_only_log_mae": float("nan"), "paired_delta_mae": 0.2,
            },
            {
                "lake": "Example Lake", "learner": "PLSR", "k": 5, "rep": 2,
                "label_only_log_mae": 0.1, "calibrated_log_mae": 0.0,
                "source_only_log_mae": float("nan"), "paired_delta_mae": 0.1,
            },
        ]
    )

    per_lake = summarize_per_lake(repeat_level)

    assert per_lake.loc[0, "lake_paired_delta_mae"] == pytest.approx(0.1)
    assert per_lake.loc[0, "median_component_difference"] == pytest.approx(0.2)
    assert per_lake.loc[0, "median_pairing_gap"] == pytest.approx(-0.1)
    assert per_lake.loc[0, "repeat_count"] == 3
    assert per_lake.loc[0, "positive_delta_repeat_fraction"] == pytest.approx(1.0)


def test_summarize_across_lakes_classifies_percentile_ci_on_lake_means():
    per_lake = pd.DataFrame(
        [
            {
                "lake": "A", "learner": "PLSR", "k": 5,
                "lake_label_only_log_mae": 0.3, "lake_calibrated_log_mae": 0.2,
                "lake_source_only_log_mae": float("nan"), "lake_paired_delta_mae": 0.1,
            },
            {
                "lake": "B", "learner": "PLSR", "k": 5,
                "lake_label_only_log_mae": 0.5, "lake_calibrated_log_mae": 0.3,
                "lake_source_only_log_mae": float("nan"), "lake_paired_delta_mae": 0.2,
            },
        ]
    )

    summary = summarize_across_lakes(per_lake, n_boot=100, seed=0)

    assert summary.loc[0, "n_lakes"] == 2
    assert summary.loc[0, "mean_paired_delta_mae"] == pytest.approx(0.15)
    assert summary.loc[0, "frac_lakes_delta_positive"] == pytest.approx(1.0)
    assert summary.loc[0, "classification"] == "positive"


def test_validate_repeat_level_checks_pairing_identity_and_complete_cell_coverage():
    frozen = pd.DataFrame(
        [
            {
                "lake": "Example Lake", "learner": "PLSR", "k": 5, "rep": 0,
                "method": "calibrated", "mae_log": 0.3, "rer_log": 0.25,
            },
            {
                "lake": "Example Lake", "learner": "PLSR", "k": 5, "rep": 1,
                "method": "calibrated", "mae_log": 0.2, "rer_log": 0.2,
            },
        ]
    )

    checks = validate_repeat_level(
        derive_repeat_level(frozen),
        learners=("PLSR",),
        budgets=(5,),
        expected_lakes=1,
        expected_repeats=2,
    )

    assert checks["max_abs_rer_difference"] < 1e-12
    assert checks["all_label_only_mae_positive"] is True
    assert checks["complete_coverage"] is True


def test_validate_repeat_level_rejects_a_zero_rer_reconstruction_denominator():
    frozen = pd.DataFrame(
        [
            {
                "lake": "Example Lake", "learner": "PLSR", "k": 5, "rep": 0,
                "method": "calibrated", "mae_log": 0.0, "rer_log": 1.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="denominator"):
        validate_repeat_level(
            derive_repeat_level(frozen),
            learners=("PLSR",),
            budgets=(5,),
            expected_lakes=1,
            expected_repeats=1,
        )


def test_s6_renderer_includes_all_cells_and_plsr_scale_disagreement():
    table = render(REPO_ROOT)

    assert "## Supplementary Table S6 — Post hoc supporting absolute-loss analysis" in table
    assert "**Table S6. Post hoc supporting absolute-loss analysis for H1.**" in table
    assert table.count("| PLSR |") == 4
    assert table.count("| XGBoost |") == 4
    assert table.count("| MLP |") == 4
    assert "| PLSR | 5 | 0.223181 | 0.231050 | 0.200694 | 0.196407 | 0.026851 | 0.030491 | [0.008842, 0.044315] | 71% | crosses_zero | positive | yes |" in table
    assert "does not replace the primary RER inference" in table


def test_run_analysis_keeps_report_in_output_directory_and_preserves_readme(tmp_path):
    from experiments.h1_absolute_loss.run_h1_absolute_loss import run_analysis

    readme = tmp_path / "README.md"
    readme.write_text("Existing analysis documentation.\n", encoding="utf-8")
    output_dir = tmp_path / "results"

    result = run_analysis(output_dir)

    assert result["checks"]["complete_coverage"] is True
    assert (output_dir / "analysis_summary.md").is_file()
    assert (output_dir / "run_manifest.json").is_file()
    assert readme.read_text(encoding="utf-8") == "Existing analysis documentation.\n"
