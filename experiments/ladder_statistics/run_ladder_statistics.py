"""Matched-control ladder statistics aggregation (protocol v0.2).

Pipeline:
  1. run unit tests (caller responsibility; this script asserts the S6 anchor itself)
  2. S6 Panel B anchor tripwire — any unit/protocol slip halts the run
  3. full-key validation of the ablation asset at k=5 (24 targets x 3 learners x 100 reps)
  4. episode-level paired deltas D14/D24/D34 (+ L0-L2, L2-L3 diagnostics)
  5. per-target median over 100 paired repeats -> equal-weight across-target mean
  6. nominal 95% percentile bootstrap (10k, default_rng(0) fresh per cell) for all 15 cells
  7. Bonferroni family (D24/D34 x 3 learners = 6 intervals; 100k resamples; quantiles
     0.05/12 and 1-0.05/12; fresh default_rng(0) per cell)
  8. non-finite counts per cell; manifest with input hashes

No refit, no retrain, no exclusion of episodes (protocol v0.2 KEEP_ALL_EPISODES).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ladder_stats import (  # noqa: E402
    aggregate_median_then_mean,
    bootstrap_mean_ci,
    bonferroni_quantiles,
    m1_label_only_loss,
)

ABLATION = REPO_ROOT / "experiments/capacity_boundary/results/calibration_direction_ablation_nested_per_rep.csv"
M1CSV = REPO_ROOT / "experiments/capacity_boundary/results/m1_label_only_per_rep_algebraic.csv"
OUTDIR = HERE / "results"
EPS = 1e-8
K = 5
LEARNERS = ("PLSR", "XGBoost", "MLP")
NOMINAL_BOOT = 10_000
FAMILY_BOOT = 100_000
ANCHOR = {"PLSR": 0.0269, "XGBoost": 0.0451, "MLP": 0.0489}  # published S6 Panel B, 4dp


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def anchor_tripwire(m1: pd.DataFrame) -> None:
    """Recompute published S6 Panel B (per-lake median of B-C, then across-lake mean)."""
    k5 = m1[m1["k"].eq(K)]
    per_lake = k5.groupby(["learner", "lake"])["delta_mae_label_minus_calibrated"].median()
    got = per_lake.groupby("learner").mean().round(4)
    for ln, target in ANCHOR.items():
        if abs(got[ln] - target) > 5e-5:
            raise SystemExit(f"ANCHOR FAIL {ln}: got {got[ln]} expected {target} — halting")
    print(f"anchor OK: {got.to_dict()}")


def main() -> None:
    abl = pd.read_csv(ABLATION)
    m1 = pd.read_csv(M1CSV)
    anchor_tripwire(m1)

    # ---- full-key validation at k=5 ----
    k5 = abl[abl["k"].eq(K)]
    counts = k5.groupby(["lake", "learner", "variant"]).size()
    assert counts.groupby("variant").apply(lambda s: set(s.index.get_level_values(0))).map(len).eq(24).all()
    assert counts.eq(100).all(), "every (lake,learner,variant) cell must have 100 reps"
    assert set(k5["learner"]) == set(LEARNERS)
    n_lakes = k5["lake"].nunique()
    assert n_lakes == 24, f"expected 24 target units, got {n_lakes}"
    m1_k5 = m1[m1["k"].eq(K)]
    assert m1_k5.groupby(["lake", "learner"]).size().eq(100).all()

    # M1 loss is learner-invariant up to ulp; one value per (lake, rep), broadcast to learners
    m1_ep = (m1_k5.groupby(["lake", "rep"])["label_only_mae_algebraic"].mean()
             .rename("L1").reset_index())

    losses = k5.pivot(index=["lake", "learner", "rep"], columns="variant",
                      values="mae_log").reset_index()
    losses = losses.merge(m1_ep, on=["lake", "rep"], how="inner", validate="many_to_one")
    assert len(losses) == 24 * 3 * 100, f"merge produced {len(losses)} rows (expect 7200)"

    contrasts = {
        "D14": ("L1", "full"),
        "D24": ("intercept_only", "full"),
        "D34": ("affine_only", "full"),
        "L0L2_diag": ("source_only", "intercept_only"),
        "L2L3_diag": ("intercept_only", "affine_only"),
    }

    target_rows, interval_rows = [], []
    lo_q, hi_q = bonferroni_quantiles(6, 0.05)

    for ln in LEARNERS:
        sub = losses[losses["learner"].eq(ln)]
        for cname, (a, b) in contrasts.items():
            delta = sub[a] - sub[b]
            agg = aggregate_median_then_mean(delta, sub["lake"])
            n_nonfinite = int((~np.isfinite(delta)).sum())
            row = {
                "learner": ln, "k": K, "contrast": cname,
                "estimand": "mean_target_median_repeat_paired_delta_logMAE",
                "effect": round(agg["estimand"], 6),
                "n_targets": agg["n_targets"], "valid_target_count": agg["valid_target_count"],
                "n_repeats_per_target": 100, "n_nonfinite_episodes": n_nonfinite,
                "positive_means": "full_variant_lower_logMAE",
            }
            target_rows.append(row)

            vals = agg["per_target_median"].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            nom_lo, nom_hi = bootstrap_mean_ci(finite, NOMINAL_BOOT, 0.025, 0.975, seed=0)
            row_nom = dict(row, interval="nominal_95", lo=round(nom_lo, 6), hi=round(nom_hi, 6),
                           n_boot=NOMINAL_BOOT, rng="default_rng(0)")
            interval_rows.append(row_nom)

            if cname in ("D24", "D34"):
                bf_lo, bf_hi = bootstrap_mean_ci(finite, FAMILY_BOOT, lo_q, hi_q, seed=0)
                interval_rows.append(dict(row, interval="bonferroni_95_family6",
                                          lo=round(bf_lo, 6), hi=round(bf_hi, 6),
                                          n_boot=FAMILY_BOOT, rng="default_rng(0)",
                                          quantiles=f"[{lo_q:.7f},{hi_q:.7f}]"))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    te = pd.DataFrame(target_rows)
    iv = pd.DataFrame(interval_rows)
    te.to_csv(OUTDIR / "target_effects.csv", index=False)
    iv.to_csv(OUTDIR / "interval_table.csv", index=False)

    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "ladder_statistics_v0.2",
        "inputs": {
            "ablation": {"path": str(ABLATION.relative_to(REPO_ROOT)), "sha256_16": sha16(ABLATION)},
            "m1": {"path": str(M1CSV.relative_to(REPO_ROOT)), "sha256_16": sha16(M1CSV)},
        },
        "code_version": subprocess.check_output(
            ["git", "log", "-1", "--format=%h"], cwd=REPO_ROOT, text=True).strip(),
        "rules": {"episodes": "KEEP_ALL (v0.2; no exclusion)", "eps": EPS, "rer_unit": "fraction",
                  "aggregation": "per-target median over 100 paired repeats, then across-target mean",
                  "nominal": f"15 cells, {NOMINAL_BOOT} resamples, percentile, rng(0) per cell",
                  "bonferroni": f"6 cells (D24/D34 x 3), {FAMILY_BOOT} resamples, quantiles [{lo_q:.7f}, {hi_q:.7f}], rng(0) per cell"},
        "anchor_check": "S6 Panel B 0.0269/0.0451/0.0489 reproduced before aggregation",
    }
    with open(OUTDIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== target effects (k=5) ===")
    print(te[["learner", "contrast", "effect", "n_nonfinite_episodes"]].to_string(index=False))
    print("\n=== intervals ===")
    print(iv[["learner", "contrast", "interval", "lo", "hi"]].to_string(index=False))
    print(f"\nwrote {OUTDIR/'target_effects.csv'}, interval_table.csv, run_manifest.json")


if __name__ == "__main__":
    main()
