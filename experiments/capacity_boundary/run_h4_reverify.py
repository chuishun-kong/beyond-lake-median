"""H4 re-verification on the final strict-24 lake pool (a gap flagged
during internal review): does source-domain support distance
predict calibration risk?

An earlier H4 check used the pre-strict-24 26-lake pool and a separate
16-component PLSR embedding. This re-verification instead reuses
the SAME 2-component representation-PLS embedding (REPR_PLS_COMPONENTS=2)
that the main capacity_boundary/confirmatory_extension pipeline already
uses for calibration features, and correlates support_score against the
already-computed true-nested k=5 PLSR "calibrated" RER
(capacity_boundary_summary_nested.csv), rather than re-deriving a separate
calibration-form comparison.

The output includes Pearson and Spearman correlations with p-values and a
systematic leave-one-target-unit-out sweep over all 24 target units. This
keeps the outlier-sensitivity statement tied to the same true-nested result
family used by the current manuscript.

Usage: python experiments/capacity_boundary/run_h4_reverify.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lake_insitu import gloria, baselines, selection  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MIN_GROUP = 20
REPR_PLS_COMPONENTS = 2
STRICT_24_EXCLUDE = {"Ba Be Lake", "Lake Constance"}


def run():
    meta = gloria.load_meta()
    rrs = gloria.load_rrs()
    band_cols = gloria.rrs_band_columns(rrs, lo=400, hi=750)
    groups = gloria.build_tss_groups(meta, rrs, band_cols)
    groups["log10_TSS"] = np.log10(groups["TSS"].clip(lower=1e-3))

    eligible = gloria.eligible_tss_lakes(groups, min_group=MIN_GROUP,
                                          lakes_reservoirs_only=True)
    lake_list = [l for l in eligible.index if l not in STRICT_24_EXCLUDE]
    print(f"Eligible lakes (strict-24): {len(lake_list)}")

    rows = []
    for target_lake in lake_list:
        is_target = groups["Site_name"] == target_lake
        target_df = groups[is_target].reset_index(drop=True)
        source_df = groups[~is_target].reset_index(drop=True)

        X_source_raw = source_df[band_cols].to_numpy(dtype=float)
        y_source_log = source_df["log10_TSS"].to_numpy(dtype=float)
        X_target_raw = target_df[band_cols].to_numpy(dtype=float)

        mu, sigma = X_source_raw.mean(axis=0), X_source_raw.std(axis=0)
        sigma[sigma == 0] = 1.0
        X_source = (X_source_raw - mu) / sigma
        X_target = (X_target_raw - mu) / sigma

        repr_pls = baselines.fit_plsr_source(X_source, y_source_log,
                                              n_components=REPR_PLS_COMPONENTS)
        repr_scores_target = repr_pls.transform(X_target)
        repr_scores_source = repr_pls.transform(X_source)

        s_score = selection.support_score(repr_scores_target, repr_scores_source, k=10)
        rows.append({"lake": target_lake, "mean_support_distance": float(s_score.mean())})
        print(f"  done: {target_lake}")

    support_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    support_df.to_csv(RESULTS_DIR / "h4_support_distance_strict24.csv", index=False)

    rer_df = pd.read_csv(RESULTS_DIR / "capacity_boundary_summary_nested.csv")
    rer_k5 = rer_df[(rer_df.learner == "PLSR") & (rer_df.k == 5) & (rer_df.method == "calibrated")]
    merged = support_df.merge(rer_k5[["lake", "median_rer_log"]], on="lake", how="inner")
    merged.to_csv(RESULTS_DIR / "h4_reverify_merged.csv", index=False)

    x, y = merged["mean_support_distance"], merged["median_rer_log"]
    r_pearson, p_pearson = pearsonr(x, y)
    rho_spearman, p_spearman = spearmanr(x, y)
    print(f"\n=== H4 re-verification (strict-24 pool, true-nested PLSR k=5 "
          f"'calibrated' RER) ===")
    print(f"Pearson  r   = {r_pearson:.3f}  p = {p_pearson:.3f}  (n_lakes={len(merged)})")
    print(f"Spearman rho = {rho_spearman:.3f}  p = {p_spearman:.3f}")

    # Systematically recompute Pearson r after excluding each target unit once.
    loo_rows = []
    for lake in merged["lake"]:
        sub = merged[merged.lake != lake]
        r_loo, p_loo = pearsonr(sub["mean_support_distance"], sub["median_rer_log"])
        loo_rows.append({"excluded_lake": lake, "r_without_lake": round(r_loo, 3),
                          "p_without_lake": round(p_loo, 3),
                          "delta_from_full_r": round(r_loo - r_pearson, 3),
                          "sign_flips": bool(np.sign(r_loo) != np.sign(r_pearson))})
    loo_df = pd.DataFrame(loo_rows).sort_values("delta_from_full_r", key=abs, ascending=False)
    loo_df.to_csv(RESULTS_DIR / "h4_leave_one_out_strict24.csv", index=False)

    print(f"\n=== Leave-one-out sweep (largest |delta| from full-sample r first) ===")
    print(loo_df.head(5).to_string(index=False))
    n_flip = int(loo_df["sign_flips"].sum())
    print(f"\nExcluding a single lake flips the sign of r in {n_flip}/{len(loo_df)} cases")
    if n_flip:
        flippers = loo_df[loo_df.sign_flips]["excluded_lake"].tolist()
        print(f"Lake(s) whose removal flips the sign: {flippers}")


if __name__ == "__main__":
    run()
