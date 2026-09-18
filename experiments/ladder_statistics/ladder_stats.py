"""Paired-loss aggregation for the matched-control ladder.

Conventions:
  - mainline rer_log is stored as a FRACTION (not percent); eps = 1e-8.
  - D-series deltas are episode-level paired differences of mae_log columns.
  - aggregation: per-target median over 100 paired repeats, then equal-weight
    mean across targets ("mean_target_median_repeat_paired_delta_logMAE").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS_DEFAULT = 1e-8


def m1_label_only_loss(calibrated_mae: np.ndarray, rer_fraction: np.ndarray,
                       eps: float = EPS_DEFAULT) -> np.ndarray:
    """Algebraic label-only loss B=(C + r*eps)/(1-r).

    The ONLY singularity is r = +1. RER has no -1 lower bound: B << C gives
    r = (B-C)/(B+eps) arbitrarily negative (observed range in the frozen asset:
    -30.37 to +0.83), and those rows reconstruct exactly. Percent-scale inputs
    (values like 8.28 for 8.28%) are caught by requiring r < 1 and by the
    runner's S6 anchor check, not by a symmetric |r| threshold.
    """
    r = np.asarray(rer_fraction, dtype=float)
    if np.any(r >= 1.0):
        raise ValueError("r >= 1 encountered: either percent-scale RER input or the "
                         "true singularity; fractional RER is required")
    return (np.asarray(calibrated_mae, dtype=float) + r * eps) / (1.0 - r)


def paired_delta(loss_a: pd.Series, loss_b: pd.Series) -> pd.Series:
    """Episode-level paired difference L_a - L_b on aligned episode keys."""
    return loss_a - loss_b


def count_nonfinite(deltas: pd.Series, target: pd.Series) -> pd.Series:
    """Per-target count of non-finite deltas — callers must report these before
    aggregating (pandas groupby().median() silently skips NaN otherwise)."""
    df = pd.DataFrame({"delta": deltas, "target": target})
    return df.groupby("target")["delta"].apply(lambda s: int((~np.isfinite(s)).sum()))


def aggregate_median_then_mean(deltas: pd.Series, target: pd.Series) -> dict:
    """Per-target median over repeats, then equal-weight mean across targets.

    Returns dict with per-target medians, the final estimand value, and per-target
    non-finite counts. NOTE: pandas median skips NaN by default; a group is NaN only
    when ALL its values are NaN. Use count_nonfinite() alongside to surface how many
    repeats were silently skipped. No zero-fill anywhere.
    """
    df = pd.DataFrame({"delta": deltas, "target": target})
    per_target = df.groupby("target")["delta"].median()
    valid = per_target[np.isfinite(per_target)]
    return {
        "per_target_median": per_target,
        "estimand": float(valid.mean()),          # mean over FINITE per-target medians only
        "n_targets": int(per_target.shape[0]),    # groups present in the input
        "valid_target_count": int(valid.shape[0]),  # groups actually contributing
        "nonfinite_per_target": count_nonfinite(deltas, target),
    }


def unstable_mask(rer_fraction: pd.Series, threshold: float = 0.99) -> pd.Series:
    """Boolean mask of episodes where M1 algebra is numerically unstable."""
    return rer_fraction.abs() > threshold


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, lo_q: float, hi_q: float,
                      seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the across-target mean; resamples target units."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return tuple(np.percentile(means, [100 * lo_q, 100 * hi_q]))


def bonferroni_quantiles(n_family: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Bonferroni quantile levels for a family of n intervals."""
    per_alpha = alpha / n_family
    return per_alpha / 2.0, 1.0 - per_alpha / 2.0
