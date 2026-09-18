"""Smoke tests for lake_insitu.baselines (added after a parameter-name typo
in fit_plsr_source passed review unseen because no test imported the module)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lake_insitu import baselines  # noqa: E402


def test_fit_plsr_source_runs_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 20))
    y = X[:, 0] * 0.5 + rng.normal(scale=0.1, size=60)
    model = baselines.fit_plsr_source(X, y, n_components=2)
    assert getattr(model, "scale", None) is True
    pred = model.predict(X[:5])
    assert np.all(np.isfinite(pred))


def test_intercept_only_calibration_is_mean_residual():
    base = np.array([1.0, 2.0, 3.0])
    y = np.array([1.5, 2.5, 3.0])
    assert baselines.intercept_only_calibration(base, y) == np.float64(0.3333333333333333)


def test_label_only_median_in_original_units():
    out = baselines.label_only_baselines(np.array([10.0, 20.0, 160.0]))
    assert out["label_median"] == 20.0
