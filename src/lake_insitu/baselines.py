"""Label-only baselines and budget-bound low-freedom calibration.

All modeling happens in log10(TSS) space;
callers convert back to original space only for reporting MAE.

Calibration capacity is bound to the label budget k:
k=1 -> intercept-only; k=3/5/10 -> intercept plus 1/2/3 ridge-regularized
residual directions (source prediction, then representation-PLS scores
s1, s2), so a handful of support points never destabilizes the fit.
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge


def label_only_baselines(y_support: np.ndarray) -> dict:
    """Arithmetic mean, geometric/log-mean, median of support labels (original
    space). These are constant predictors applied to every query point."""
    y_support = np.asarray(y_support, dtype=float)
    return {
        "label_mean": float(np.mean(y_support)),
        # clip(lower=1e-3) matches the project-wide convention for TSS=0
        # (e.g. gloria groups' log10_TSS) -- GLORIA has 2 zero-TSS records,
        # which would otherwise make log10(0)=-inf -> label_log_mean=0.0.
        # Frozen baseline is label_median (a pre-committed decision),
        # which never touches this
        # value, but the dict is still returned for diagnostics/callers.
        "label_log_mean": float(10 ** np.mean(np.log10(y_support.clip(min=1e-3)))),
        "label_median": float(np.median(y_support)),
    }


def fit_plsr_source(X_source: np.ndarray, y_log_source: np.ndarray,
                     n_components: int) -> PLSRegression:
    """n_components has no default: every call site must state its choice
    explicitly (source-held-out-selected value is 16 for the TSS base
    learner, 2 for the representation-PLS embedding -- see
    PLSR_N_COMPONENTS / REPR_PLS_COMPONENTS in the experiment scripts)."""
    n_components = min(n_components, X_source.shape[0] - 1, X_source.shape[1])
    # scale=True is the sklearn default and what every frozen run used; it is
    # stated explicitly here and in the Supplement so reproductions do not
    # silently diverge by passing scale=False after manual standardization.
    model = PLSRegression(n_components=max(1, n_components), scale=True)
    model.fit(X_source, y_log_source)
    return model


def intercept_only_calibration(base_pred_support_log: np.ndarray,
                                y_support_log: np.ndarray) -> float:
    """b_l: mean residual between labels and source-only prediction, log space."""
    return float(np.mean(y_support_log - base_pred_support_log))


def safe_pow10(log_pred: np.ndarray, lo: float = -3.0, hi: float = 5.0) -> np.ndarray:
    """Clip log10-space predictions before exponentiating. A 1D residual ridge
    fit on k=5 support points can extrapolate its slope far outside the
    support's z-range for query points near the source domain's edge; without
    this clip a single such point can blow up to non-physical TSS values
    (observed: >1e7 mg/L) and dominate any mean-based aggregate. Bounds cover
    0.001-100,000 mg/L, far beyond any physically plausible TSS."""
    return 10 ** np.clip(log_pred, lo, hi)


def residual_ridge_calibration(z_support: np.ndarray, resid_support_log: np.ndarray,
                                alpha: float = 1.0) -> Ridge:
    """1D (or few-D) residual calibration: resid ~ intercept + beta^T z, ridge-
    regularized. z is a low-dim source-trained representation (e.g. first PLS
    score), NOT re-fit on target data (feature selection
    rules frozen on source-held-out tasks only)."""
    z_support = np.asarray(z_support).reshape(len(resid_support_log), -1)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(z_support, resid_support_log)
    return model
