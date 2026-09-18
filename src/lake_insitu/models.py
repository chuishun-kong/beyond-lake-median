"""Additional source base learners (PLSR main, XGBoost
robust contrast, small MLP high-capacity contrast). All operate in
log10(TSS) space and share the PLSR wrapper's fit(X, y_log) / predict(X)
convention so calibration code in baselines.py is base-learner-agnostic.
"""
import os

import numpy as np
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# 2026-07-12: default (untuned) values kept nameable via env override so the
# "does the H2 conclusion depend on hyperparameter choice" sensitivity check
# (appendix) can be reproduced on demand without duplicating run scripts.
_XGB_DEFAULT_PRE_TUNING = {"n_estimators": 300, "max_depth": 4}
_MLP_DEFAULT_PRE_TUNING = {"hidden_layer_sizes": (64, 32)}


def fit_xgboost_source(X_source: np.ndarray, y_log_source: np.ndarray,
                        seed: int = 0) -> XGBRegressor:
    # n_estimators=500, max_depth=6 source-held-out-selected 2026-07-12
    # (leave-one-lake-out sweep over strict-24 pool, see
    # experiments/capacity_boundary/run_hyperparam_sweep.py +
    # results/hyperparam_sweep.csv; objective = median held-out log-MAE,
    # same methodology as PLSR's n_components=16). Set
    # USE_PRE_TUNING_HP=1 to reproduce the pre-tuning sensitivity check.
    hp = _XGB_DEFAULT_PRE_TUNING if os.environ.get("USE_PRE_TUNING_HP") else \
        {"n_estimators": 500, "max_depth": 6}
    model = XGBRegressor(
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1, **hp,
    )
    model.fit(X_source, y_log_source)
    return model


def fit_mlp_source(X_source: np.ndarray, y_log_source: np.ndarray,
                    seed: int = 0) -> MLPRegressor:
    # hidden_layer_sizes=(128,64) source-held-out-selected 2026-07-12,
    # same sweep/objective as fit_xgboost_source above. Set
    # USE_PRE_TUNING_HP=1 to reproduce the pre-tuning sensitivity check.
    hp = _MLP_DEFAULT_PRE_TUNING if os.environ.get("USE_PRE_TUNING_HP") else \
        {"hidden_layer_sizes": (128, 64)}
    model = MLPRegressor(
        activation="relu", alpha=1e-3, learning_rate_init=1e-3,
        max_iter=2000, early_stopping=True, n_iter_no_change=20,
        random_state=seed, **hp,
    )
    model.fit(X_source, y_log_source)
    return model


def refit_mlp_support_only(X_support: np.ndarray, y_log_support: np.ndarray,
                            seed: int = 0) -> MLPRegressor:
    """High-capacity 'full fine-tuning' contrast: train a fresh network of
    the SAME architecture using only the k support points. This is expected
    to overfit badly at small k -- that expected failure is the point (plan
    section 6.2: full fine-tuning on 3-5 labels vs low-capacity methods is an
    intentionally unfair comparison to make visible, not a serious contender).
    A small hidden layer and strong L2 (alpha) keep it from being an
    absurd strawman; it still has far more free parameters than the
    calibration capacity table allows at the same k.
    """
    n = len(y_log_support)
    model = MLPRegressor(
        hidden_layer_sizes=(8,), activation="relu", alpha=1e-1,
        learning_rate_init=1e-2, max_iter=3000, random_state=seed,
    )
    model.fit(X_support, y_log_support)
    return model
