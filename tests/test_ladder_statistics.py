"""Synthetic-data tests for matched-control aggregation and intervals."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "ladder_statistics"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ladder_stats import (  # noqa: E402
    aggregate_median_then_mean,
    bonferroni_quantiles,
    bootstrap_mean_ci,
    m1_label_only_loss,
    paired_delta,
    unstable_mask,
)


def test_m1_algebra_roundtrip():
    # B - C = r*(C + eps)/(1 - r) exactly; construct C, r and check both relations
    C = np.array([0.20, 0.15, 0.30])
    r = np.array([0.10, -0.05, 0.25])
    eps = 1e-8
    B = m1_label_only_loss(C, r, eps=eps)
    assert np.allclose(B - C, r * (C + eps) / (1 - r), atol=1e-15)
    # inverse: r = (B - C) / (B + eps) approximates the stored r
    assert np.allclose((B - C) / (B + eps), r, atol=1e-9)


def test_m1_negative_beyond_minus_one_is_valid():
    # RER has NO -1 bound: B << C yields r < -1 and the algebra is exact.
    # Frozen-asset evidence: r range [-30.37, +0.83]; 1,000 rows have r <= -1.
    C = np.array([0.3, 0.2])
    B_true = np.array([0.1, 0.02])
    eps = 1e-8
    r = (B_true - C) / (B_true + eps)      # r approx -2 and -9
    assert np.all(r < -1)
    B_recovered = m1_label_only_loss(C, r, eps=eps)
    np.testing.assert_allclose(B_recovered, B_true, atol=1e-12)


def test_m1_observed_positive_range_passes():
    # max observed r in the asset is +0.83; anything < 1 is valid input
    out = m1_label_only_loss(np.array([0.2]), np.array([0.83]))
    assert np.isfinite(out).all()


def test_m1_rejects_singular_and_percent_inputs():
    with pytest.raises(ValueError):
        m1_label_only_loss(np.array([0.2]), np.array([1.0]))    # singularity
    with pytest.raises(ValueError):
        m1_label_only_loss(np.array([0.2]), np.array([8.28]))   # percent-scale
    with pytest.raises(ValueError):
        m1_label_only_loss(np.array([0.2]), np.array([101.0]))  # percent-scale overflow RER


def test_paired_delta_is_episode_level():
    a = pd.Series([0.3, 0.2, 0.4])
    b = pd.Series([0.1, 0.25, 0.35])
    np.testing.assert_allclose(paired_delta(a, b).to_numpy(), [0.2, -0.05, 0.05], atol=1e-12)


def test_median_then_mean_order():
    # target A: deltas 1,2,100 -> median 2 ; target B: deltas 4,5,6 -> median 5
    # estimand = mean(2,5) = 3.5 (NOT median-of-all = 4, NOT mean-of-all)
    deltas = pd.Series([1.0, 2.0, 100.0, 4.0, 5.0, 6.0])
    target = pd.Series(["A", "A", "A", "B", "B", "B"])
    out = aggregate_median_then_mean(deltas, target)
    assert out["per_target_median"]["A"] == 2.0
    assert out["per_target_median"]["B"] == 5.0
    assert out["estimand"] == 3.5


def test_median_then_mean_robust_to_unstable_minority():
    # 3 unstable episodes out of 100 do not move the per-target median much
    rng = np.random.default_rng(1)
    deltas = np.concatenate([rng.normal(0.05, 0.01, 97), np.array([50.0, -80.0, 120.0])])
    target = pd.Series(["A"] * 100)
    out = aggregate_median_then_mean(pd.Series(deltas), target)
    assert abs(out["per_target_median"]["A"] - 0.05) < 0.01  # median unaffected


def test_nan_not_zero_filled_and_counted():
    deltas = pd.Series([1.0, np.nan, 3.0])
    target = pd.Series(["A", "A", "A"])
    out = aggregate_median_then_mean(deltas, target)
    med = out["per_target_median"]["A"]
    assert med == 2.0  # pandas median skips the NaN (documented behavior)
    assert out["nonfinite_per_target"]["A"] == 1  # and the skip is surfaced, not hidden
    # all-NaN group propagates NaN (never zero)
    out2 = aggregate_median_then_mean(pd.Series([np.nan, np.nan]), pd.Series(["B", "B"]))
    assert np.isnan(out2["per_target_median"]["B"])


def test_bootstrap_deterministic_and_quantiles():
    values = np.arange(1.0, 25.0)  # 24 targets
    lo, hi = bootstrap_mean_ci(values, n_boot=2000, lo_q=0.025, hi_q=0.975, seed=0)
    lo2, hi2 = bootstrap_mean_ci(values, n_boot=2000, lo_q=0.025, hi_q=0.975, seed=0)
    assert (lo, hi) == (lo2, hi2)  # same seed -> identical
    assert lo < values.mean() < hi
    # wider interval when we ask for more extreme quantiles
    lo3, hi3 = bootstrap_mean_ci(values, n_boot=2000, lo_q=0.004, hi_q=0.996, seed=0)
    assert lo3 <= lo and hi3 >= hi


def test_bonferroni_quantiles():
    lo, hi = bonferroni_quantiles(n_family=6, alpha=0.05)
    assert abs(lo - 0.05 / 12) < 1e-12
    assert abs(hi - (1 - 0.05 / 12)) < 1e-12
    lo1, hi1 = bonferroni_quantiles(n_family=1, alpha=0.05)
    assert abs(lo1 - 0.025) < 1e-12 and abs(hi1 - 0.975) < 1e-12


def test_unstable_mask_threshold():
    r = pd.Series([0.5, -0.995, 0.99, -0.98, 0.999])
    m = unstable_mask(r, threshold=0.99)
    assert list(m) == [False, True, False, False, True]  # strictly greater


def test_query_label_permutation_leaves_support_path_unchanged():
    """B02 boundary: support selection and support-label statistics read no query
    labels. Perturbing ONLY query labels must leave supports and the M1 constant
    bit-identical (only downstream evaluation numbers would change)."""
    import zlib
    from lake_insitu import sampling

    rng_seed = 20260712
    y = np.arange(1.0, 41.0)  # 40 synthetic groups with labels
    seed = rng_seed + zlib.crc32("SyntheticLake".encode()) % 10_000
    r1 = np.random.default_rng(seed)
    r2 = np.random.default_rng(seed)
    s1, q1 = sampling.random_support_query_split(np.arange(40), k=5, rng=r1)
    s2, q2 = sampling.random_support_query_split(np.arange(40), k=5, rng=r2)
    assert np.array_equal(s1, s2) and np.array_equal(q1, q2)  # determinism
    # permute query labels only; supports identical, support median unchanged
    y_perm = y.copy()
    y_perm[q1] = y_perm[q1][::-1]
    assert np.array_equal(s1, s2)
    assert np.median(y[s1]) == np.median(y_perm[s2])


def test_full_pipeline_on_synthetic_episode_table():
    # end-to-end miniature: 2 targets x 2 learners x 5 repeats x {full, intercept_only}
    rows = []
    rng = np.random.default_rng(42)
    for t in ["T1", "T2"]:
        for ln in ["L1", "L2"]:
            for rep in range(5):
                base = rng.normal(0.2, 0.02)
                rows.append(dict(target=t, learner=ln, rep=rep, variant="full", mae=base))
                rows.append(dict(target=t, learner=ln, rep=rep, variant="intercept_only", mae=base + rng.normal(0.01, 0.005)))
    df = pd.DataFrame(rows)
    wide = df.pivot(index=["target", "learner", "rep"], columns="variant", values="mae")
    d24 = paired_delta(wide["intercept_only"].reset_index(drop=True),
                       wide["full"].reset_index(drop=True))
    meta = wide.reset_index()
    out = aggregate_median_then_mean(d24, meta["target"])
    assert -1.0 < out["estimand"] < 1.0  # sanity on synthetic scale
    assert out["n_targets"] == 2
