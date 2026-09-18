"""China IWC-only lakes as replication/stress-test
(NOT independent validation -- this dataset is within FSResTL-Chla's
IWC target domain). Two things are being stress-tested simultaneously and
must be reported separately:
  1. Geographic/optical domain shift (GLORIA source -> unseen China lakes,
     zero overlap with GLORIA per overlap_audit_china.csv).
  2. Label DEFINITION shift (GLORIA TSS -> China TSM). Hard constraint: TSS and China TSM are not assumed interchangeable without
     verified metadata compatibility, which does not exist here. This run
     applies a TSS-trained protocol to TSM labels as a deliberate adverse
     stress test, never as a second copy of the main GLORIA-internal H1/H3
     result.

Usage: python experiments/china_stress_test/run_china_stress_test.py
"""
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lake_insitu import gloria, china, baselines, metrics, sampling, models  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RNG_SEED = 20260712
K = 5
N_REPEATS = 50
PLSR_N_COMPONENTS = 16
REPR_PLS_COMPONENTS = 2


def log_mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def run():
    gmeta = gloria.load_meta()
    grrs = gloria.load_rrs()
    band_cols = gloria.rrs_band_columns(grrs, lo=400, hi=750)
    ggroups = gloria.build_tss_groups(gmeta, grrs, band_cols)
    ggroups["log10_TSS"] = np.log10(ggroups["TSS"].clip(lower=1e-3))

    X_source_raw = ggroups[band_cols].to_numpy(dtype=float)
    y_source_log = ggroups["log10_TSS"].to_numpy(dtype=float)
    mu, sigma = X_source_raw.mean(axis=0), X_source_raw.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_source = (X_source_raw - mu) / sigma

    print(f"GLORIA source pool (ALL {len(ggroups)} TSS groups worldwide, "
          f"no leave-one-out needed -- these 13 China lakes have zero "
          f"overlap with GLORIA per overlap_audit_china.csv).")

    learners = {
        "PLSR": baselines.fit_plsr_source(X_source, y_source_log,
                                           n_components=PLSR_N_COMPONENTS),
        "XGBoost": models.fit_xgboost_source(X_source, y_source_log),
        "MLP": models.fit_mlp_source(X_source, y_source_log),
    }
    repr_pls = baselines.fit_plsr_source(X_source, y_source_log,
                                          n_components=REPR_PLS_COMPONENTS)

    merged = china.load_merged()
    cn_band_cols = china.band_columns(merged, lo=400, hi=750)
    assert cn_band_cols == [str(w) for w in range(400, 751)], \
        "China band columns must exactly match the GLORIA 400-750nm selection"
    cgroups = china.build_tsm_groups(merged, cn_band_cols)
    cgroups["log10_TSM"] = np.log10(cgroups["TSM"].clip(lower=1e-3))

    rows = []
    for lake in china.IWC_ONLY_LAKES:
        sub = cgroups[cgroups["Lake_name"] == lake].reset_index(drop=True)
        n = len(sub)
        if n <= K + 2:
            print(f"  SKIP {lake}: only {n} groups, too few for k={K} + a "
                  f"meaningful query set")
            continue

        X_raw = sub[cn_band_cols].to_numpy(dtype=float)
        X = (X_raw - mu) / sigma  # standardize with GLORIA source stats -- no refit
        y = sub["TSM"].to_numpy(dtype=float)
        y_log = sub["log10_TSM"].to_numpy(dtype=float)
        repr_scores = repr_pls.transform(X)

        rng = np.random.default_rng(RNG_SEED + zlib.crc32(lake.encode()) % 10_000)

        for learner_name, model in learners.items():
            base_pred_log = np.asarray(model.predict(X)).ravel()
            affine_z = base_pred_log

            for rep in range(N_REPEATS):
                support_idx, query_idx = sampling.random_support_query_split(
                    np.arange(n), k=K, rng=rng)
                y_sup, y_qry = y[support_idx], y[query_idx]
                y_sup_log, y_qry_log = y_log[support_idx], y_log[query_idx]
                base_pred_sup_log = base_pred_log[support_idx]
                base_pred_qry_log = base_pred_log[query_idx]

                # Fixed to a single pre-committed statistic
                # (support_median) for the frozen baseline decision.
                lbl = baselines.label_only_baselines(y_sup)
                best_label_mae_log = log_mae(
                    y_qry_log, np.full_like(y_qry_log, np.log10(lbl["label_median"])))

                z_sup = np.column_stack([affine_z[support_idx], repr_scores[support_idx, 0]])
                z_qry = np.column_stack([affine_z[query_idx], repr_scores[query_idx, 0]])
                resid_sup_log = y_sup_log - base_pred_sup_log
                ridge = baselines.residual_ridge_calibration(z_sup, resid_sup_log, alpha=1.0)
                pred_log = base_pred_qry_log + ridge.predict(z_qry).ravel()
                pred_orig = baselines.safe_pow10(pred_log)

                skill = metrics.lake_demeaned_skill(y_qry, pred_orig)
                rows.append({
                    "lake": lake, "learner": learner_name, "rep": rep,
                    "n_group": n,
                    "rer_log": metrics.relative_error_reduction(
                        best_label_mae_log, log_mae(y_qry_log, pred_log)),
                    "demeaned_spearman": skill["spearman"],
                })

        print(f"  done: {lake} (n_group={n})")

    repeats = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    repeats.to_csv(RESULTS_DIR / "china_stress_test_repeats.csv", index=False)

    summary = repeats.groupby(["lake", "learner"]).agg(
        median_rer_log=("rer_log", "median"),
        median_demeaned_spearman=("demeaned_spearman", "median"),
        n_group=("n_group", "first"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "china_stress_test_summary.csv", index=False)

    print("\n=== China TSM stress-test summary (k=5, GLORIA-TSS-trained models) ===")
    for learner in ["PLSR", "XGBoost", "MLP"]:
        m = summary[summary.learner == learner]
        print(f"\n-- {learner} --")
        print(f"  n lakes tested: {len(m)}")
        print(f"  median RER(log): {m['median_rer_log'].median()*100:.2f}%")
        print(f"  fraction lakes positive: {(m['median_rer_log']>0).mean():.2f}")
        print(f"  median demeaned Spearman: {m['median_demeaned_spearman'].median():.3f}")
        print(m[["lake", "n_group", "median_rer_log", "median_demeaned_spearman"]]
              .sort_values("median_rer_log", ascending=False).to_string(index=False))


if __name__ == "__main__":
    run()
