"""H3 validation must reject invalid pairs without replacing accepted results."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "capacity_boundary"))
import run_h3_paired_source_audit_nested as h3


@pytest.mark.parametrize("actual,archived,status", [
    (np.nan, np.nan, "both_nan"), (np.nan, 0.0, "one_nan"),
    (0.0, np.nan, "one_nan"), (np.inf, np.inf, "infinite"),
    (-np.inf, 0.0, "infinite"),
])
def test_nonfinite_comparison_never_passes(actual, archived, status):
    _, passed, actual_status = h3.metric_self_check(actual, archived, "spearman")
    assert not passed
    assert actual_status == status


def test_ulp_allowance_only_for_extreme_original_space_r2():
    archived = -1.1e7
    next_float = np.nextafter(archived, 0)
    assert h3.metric_self_check(next_float, archived, "anomaly_r2")[1]
    assert not h3.metric_self_check(next_float, archived, "spearman")[1]
    assert not h3.metric_self_check(archived + 1e-5, archived, "anomaly_r2")[1]


@pytest.mark.parametrize("bootstrap", [h3.mean_bootstrap_ci, h3.median_bootstrap_ci])
@pytest.mark.parametrize("values", [[1, np.nan], [1, np.inf], []])
def test_bootstrap_does_not_silently_drop_missing_values(bootstrap, values):
    with pytest.raises(ValueError):
        bootstrap(values, n_boot=5)


def complete_pairs():
    keys = pd.MultiIndex.from_product(
        [[f"target{i}" for i in range(24)], ["PLSR", "XGBoost", "MLP"],
         [1, 3, 5, 10], range(100)], names=["lake", "learner", "k", "rep"])
    frame = keys.to_frame(index=False)
    frame["delta_spearman"] = 0.0
    return frame


def test_exact_episode_coverage_and_finiteness():
    frame = complete_pairs()
    h3.validate_complete_pairs(frame)
    with pytest.raises(ValueError):
        h3.validate_complete_pairs(frame.iloc[:-1])
    duplicated = pd.concat([frame.iloc[:-1], frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        h3.validate_complete_pairs(duplicated)
    frame.loc[0, "delta_spearman"] = np.nan
    with pytest.raises(ValueError):
        h3.validate_complete_pairs(frame)


def test_failed_replay_preserves_accepted_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(h3, "RESULTS_DIR", tmp_path)
    final = tmp_path / "h3_paired_source_audit_nested_per_rep.csv"
    final.write_bytes(b"accepted result\n")
    detail = pd.DataFrame({"n_failed_metrics": [1]})
    with pytest.raises(ValueError, match="provisional"):
        h3.save_checked_per_rep(pd.DataFrame({"delta_spearman": [np.nan]}), detail)
    assert final.read_bytes() == b"accepted result\n"
    assert (tmp_path / "h3_paired_source_audit_nested_per_rep_provisional.csv").exists()


def test_missing_target_metric_not_counted_as_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(h3, "RESULTS_DIR", tmp_path)
    frame = pd.DataFrame({"lake": [f"target{i}" for i in range(24)],
                          "learner": "PLSR", "k": 1,
                          "source_spearman": 0.5, "calibrated_spearman": 0.5,
                          "delta_spearman": 0.0,
                          "delta_anomaly_r2_logspace": 0.0,
                          "delta_anomaly_r2_origspace": 0.0})
    frame.loc[0, "delta_spearman"] = np.nan
    with pytest.raises(ValueError, match="1 NaN"):
        h3.compute_summary_from_per_lake(frame)
    assert not list(tmp_path.glob("*summary*.csv"))
