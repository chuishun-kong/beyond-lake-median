"""Evaluation metrics: lake-macro relative error reduction, within-lake skill,
and lake-level bootstrap confidence intervals.

Query labels enter these functions only; they must never have been used to
pick k, representation dimension, ridge strength, or which baseline "wins"
(query-isolation hard constraint).
"""
import numpy as np
from scipy.stats import spearmanr, pearsonr


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def relative_error_reduction(mae_label_only: float, mae_method: float,
                              eps: float = 1e-8) -> float:
    """RER_l = (MAE_label_only - MAE_method) / (MAE_label_only + eps).
    Positive means the method beats the label-only baseline on this lake."""
    return (mae_label_only - mae_method) / (mae_label_only + eps)


def lake_demeaned_skill(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-lake skill on the query set. Demeaning by the lake's own query mean
    is a no-op for Spearman/Pearson (both shift-invariant) but is exactly what
    makes anomaly R2 measure within-lake variation instead of absolute level."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 3 or np.std(y_true) == 0:
        return {"spearman": np.nan, "pearson": np.nan, "anomaly_r2": np.nan}

    yt_dm = y_true - y_true.mean()
    yp_dm = y_pred - y_pred.mean()

    sp = spearmanr(yt_dm, yp_dm).statistic
    pe = pearsonr(yt_dm, yp_dm)[0] if np.std(yp_dm) > 0 else np.nan

    ss_res = np.sum((yt_dm - yp_dm) ** 2)
    ss_tot = np.sum(yt_dm ** 2)
    anomaly_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"spearman": sp, "pearson": pe, "anomaly_r2": anomaly_r2}


def bootstrap_lake_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                       rng: np.random.Generator = None) -> tuple:
    """Percentile bootstrap CI treating each element of `values` as one i.i.d.
    lake-level observation (lake is the resampling unit)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = rng or np.random.default_rng(0)
    boot_means = np.empty(n_boot)
    n = len(values)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_means[b] = np.mean(sample)
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))
