"""Table S1: exploratory direct lake-paired ΔMAE bootstrap statistics.

The table covers 12 learner-budget configurations and uses the B=1,000,000
high-resolution interval for PLSR k=3 in place of its B=10,000 interval.
The historical bootstrap inputs listed below are not bundled in this release;
the derived table CSV is included for inspection.

No hardcoded numbers -- reads directly from the two frozen result CSVs
(h2_direct_mae_paired_bootstrap.csv, h2_plsr_k3_bootstrap_highB.csv) so the
table stays in sync with them. Emits a markdown table to stdout and saves
the merged data as a CSV for the record.

Usage: python experiments/active_selection/make_table_s1_h2_direct_mae.py
"""
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
K_ORDER = [1, 3, 5, 10]
LEARNER_ORDER = ["PLSR", "XGBoost", "MLP"]


def classify(learner, k):
    if k == 1:
        return "adverse"
    if learner == "PLSR" and k == 3:
        return "unresolved boundary case"
    return "unresolved"


def relation_to_zero(ci_lower, ci_upper):
    if ci_lower > 0:
        return "above zero"
    if ci_upper < 0:
        return "below zero"
    return "spans zero"


def run():
    mae = pd.read_csv(RESULTS_DIR / "h2_direct_mae_paired_bootstrap.csv")
    high_b = pd.read_csv(RESULTS_DIR / "h2_plsr_k3_bootstrap_highB.csv")
    assert len(high_b) == 1, f"expected 1 high-B row, got {len(high_b)}"
    hb = high_b.iloc[0]

    is_plsr_k3 = (mae["learner"] == hb["learner"]) & (mae["k"] == hb["k"])
    assert is_plsr_k3.sum() == 1, "expected exactly one PLSR k=3 row to override"
    frozen_row = mae.loc[is_plsr_k3].iloc[0]
    # Sanity gate: the high-B script's own record of the original B=10,000
    # endpoints must match the frozen direct-MAE CSV bit-for-bit (up to
    # float noise) before we trust splicing its high-B CI into this table.
    assert abs(frozen_row["ci_lower"] - hb["frozen_B10000_ci_lower"]) < 1e-9
    assert abs(frozen_row["ci_upper"] - hb["frozen_B10000_ci_upper"]) < 1e-9

    mae["bootstrap_resolution"] = "B=10,000"
    mae.loc[is_plsr_k3, "ci_lower"] = hb["ci_lower"]
    mae.loc[is_plsr_k3, "ci_upper"] = hb["ci_upper"]
    mae.loc[is_plsr_k3, "bootstrap_resolution"] = "B=1,000,000"

    mae["interval_relation_to_zero"] = mae.apply(
        lambda r: relation_to_zero(r["ci_lower"], r["ci_upper"]), axis=1)
    mae["classification"] = mae.apply(
        lambda r: classify(r["learner"], r["k"]), axis=1)

    mae["learner"] = pd.Categorical(mae["learner"], categories=LEARNER_ORDER, ordered=True)
    mae["k"] = pd.Categorical(mae["k"], categories=K_ORDER, ordered=True)
    merged = mae.sort_values(["learner", "k"]).reset_index(drop=True)

    out_cols = ["learner", "k", "median_delta_mae", "mean_delta_mae",
                "ci_lower", "ci_upper", "bootstrap_resolution",
                "interval_relation_to_zero", "classification"]
    merged[out_cols].to_csv(RESULTS_DIR / "table_s1_h2_direct_mae_vs_rer.csv", index=False)

    header = ("| Learner | k | median ΔMAE | mean ΔMAE | "
               "95% CI used for inference | Bootstrap resolution | "
               "Interval relation to zero | Classification |")
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for _, r in merged.iterrows():
        lines.append(
            f"| {r['learner']} | {int(r['k'])} | {r['median_delta_mae']:+.4f} | "
            f"{r['mean_delta_mae']:+.4f} | "
            f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}] | "
            f"{r['bootstrap_resolution']} | {r['interval_relation_to_zero']} | "
            f"{r['classification']} |"
        )
    table_md = "\n".join(lines)
    print(table_md)
    print(f"\nSaved merged data to {RESULTS_DIR / 'table_s1_h2_direct_mae_vs_rer.csv'}")
    return table_md


if __name__ == "__main__":
    run()
