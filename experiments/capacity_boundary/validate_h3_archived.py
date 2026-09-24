"""Validate the stored H3 paired records without fitting source models.

Run from the repository root: python experiments/capacity_boundary/validate_h3_archived.py
This checks stored metrics, episode coverage, and both aggregation layers; it
does not reconstruct query predictions or independently refit source models.
"""
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import run_h3_paired_source_audit_nested as h3


def main():
    root = Path(__file__).resolve().parent / "results"
    read = lambda name: pd.read_csv(root / name, float_precision="round_trip")
    keys = ["lake", "learner", "k", "rep"]
    per_rep = read("h3_paired_source_audit_nested_per_rep.csv")
    h3.validate_complete_pairs(per_rep)
    archived = read("capacity_boundary_repeats_nested.csv")
    archived = archived[archived.method == "calibrated"]
    merged = per_rep.merge(archived, on=keys, how="outer", validate="one_to_one",
                           indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("H3 and mainline episode keys do not match")
    comparisons = {}
    for current, old, metric in [
        ("calibrated_spearman", "demeaned_spearman", "spearman"),
        ("calibrated_anomaly_r2_origspace", "anomaly_r2", "anomaly_r2"),
    ]:
        checks = [h3.metric_self_check(a, b, metric)
                  for a, b in zip(merged[current], merged[old])]
        if not all(passed for _, passed, _ in checks):
            raise ValueError(f"Stored H3/mainline mismatch: {metric}")
        comparisons[metric] = {
            "max_abs_difference": max(difference for difference, _, _ in checks),
            "finite_pairs": sum(status == "finite" for _, _, status in checks),
        }
    np.testing.assert_array_equal(
        per_rep.calibrated_spearman - per_rep.source_spearman,
        per_rep.delta_spearman)
    target = read("h3_paired_source_audit_nested_per_lake.csv")
    target_keys = keys[:-1]
    columns = target.columns.difference(target_keys).tolist()
    recomputed = per_rep.groupby(target_keys)[columns].median().sort_index()
    stored = target.set_index(target_keys).sort_index()
    pd.testing.assert_index_equal(recomputed.index, stored.index)
    np.testing.assert_allclose(recomputed[columns], stored[columns], rtol=0, atol=1e-12)
    target_error = float(np.max(abs(recomputed[columns].to_numpy() - stored[columns].to_numpy())))
    detail = read("h3_nested_self_check_detail.csv")
    detail_keys = pd.MultiIndex.from_frame(detail[keys])
    if detail_keys.has_duplicates or set(detail_keys) != set(pd.MultiIndex.from_frame(per_rep[keys])):
        raise ValueError("Historical self-check keys do not match H3")
    h3.require_finite(detail.diff_val, "historical self-check differences")
    over = detail[detail.diff_val > h3.SELF_CHECK_TOL]
    # Resolve the historical absolute-tolerance exceptions using the actual
    # stored values and representable precision, never just a zero exit code.
    for row in over.itertuples():
        if row.worst_metric != "anomaly_r2":
            raise ValueError("Unexplained historical self-check exception")
        match = merged.set_index(keys).loc[(row.lake, row.learner, row.k, row.rep)]
        diff, passed, _ = h3.metric_self_check(
            match.calibrated_anomaly_r2_origspace, match.anomaly_r2, "anomaly_r2")
        if not passed or diff != row.diff_val:
            raise ValueError("Historical exception does not match archived values")
    original_results = h3.RESULTS_DIR
    with tempfile.TemporaryDirectory(prefix="h3-summary-validation-") as tmp:
        try:
            h3.RESULTS_DIR = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                h3.compute_summary_from_per_lake(target)
            summary_errors = {}
            for name in ["h3_paired_source_audit_nested_summary.csv",
                         "h3_paired_source_audit_nested_summary_fullprecision.csv"]:
                actual = pd.read_csv(Path(tmp) / name, float_precision="round_trip")
                expected = read(name)
                # The full-precision original-space R2 tails reach ~2e4;
                # preserve a few-ULP comparison as well as an absolute floor.
                pd.testing.assert_frame_equal(actual, expected, check_exact=False,
                                               rtol=8*np.finfo(float).eps, atol=1e-12)
                numeric = actual.select_dtypes("number").columns
                summary_errors[name] = float(np.max(abs(actual[numeric] - expected[numeric])))
        finally:
            h3.RESULTS_DIR = original_results
    print(json.dumps({"status": "PASS", "episodes": len(per_rep),
        "target_learner_budget_cells": len(target), "draws_per_cell": 100,
        "targets_per_learner_budget": 24, "nonfinite_paired_metrics": 0,
        "comparisons": comparisons, "max_target_aggregation_error": target_error,
        "summary_max_abs_errors": summary_errors,
        "historical_original_space_R2_roundoff_exceptions": len(over),
        "scope": "stored-metric validation and reaggregation; no query-prediction replay"},
        indent=2))


if __name__ == "__main__":
    main()
