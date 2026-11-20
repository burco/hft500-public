"""Stage 10 models on SYNTHETIC data."""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hft500 import evaluation as E, models as MD  # noqa: E402

FE = ["a", "b", "c"]


def _data(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)); X[::11, 2] = np.nan
    y_lin = 0.5 * X[:, 0] - 0.3 * X[:, 1]
    y_nl = np.sign(X[:, 0]) * X[:, 1]                                  # interaction: invisible to a linear model
    Y = np.column_stack([y_lin + rng.normal(size=n), y_nl + 0.5 * rng.normal(size=n), y_lin + y_nl + rng.normal(size=n)])
    return X, Y


def _frame(X):
    return pl.DataFrame({f: X[:, i] for i, f in enumerate(FE)})


def test_linear_fit_save_load_and_predict(tmp_path):
    X, Y = _data()
    lo, hi, ind = np.full(3, -np.inf), np.full(3, np.inf), np.array([False, False, True])
    M = E.moments(E.design(X, lo, hi, ind), Y)
    m = MD.LinearModel.fit(M, FE, lo, hi, ind, used=["a", "b"], lam=0.0)
    p = m.predict(_frame(X))
    assert np.corrcoef(p["y_500ms"], 0.5 * X[:, 0] - 0.3 * X[:, 1])[0, 1] == pytest.approx(1.0, abs=1e-3)
    assert m.W[2, 0] == 0 and m.n_params() == 6                         # 'c' unused
    m.save(tmp_path / "lin.json")
    m2 = MD.LinearModel.load(tmp_path / "lin.json")
    assert np.array_equal(m2.predict(_frame(X))["y_2s"], p["y_2s"])


def test_lgb_beats_linear_on_interaction_and_is_deterministic(tmp_path):
    X, Y = _data()
    Xt, Yt = _data(seed=1)
    params = MD.LGB_BASE | MD.LGB_CONFIGS["A"] | {"min_data_in_leaf": 200}
    g = MD.LGBModel.fit(X.astype(np.float32), Y, FE, params, 100)
    g2 = MD.LGBModel.fit(X.astype(np.float32), Y, FE, params, 100)
    pt, pt2 = g.predict(_frame(Xt)), g2.predict(_frame(Xt))
    assert np.array_equal(pt["y_2s"], pt2["y_2s"])                      # same seed -> identical
    lo, hi, ind = np.full(3, -np.inf), np.full(3, np.inf), np.array([False, False, True])
    lin = MD.LinearModel.fit(E.moments(E.design(X, lo, hi, ind), Y), FE, lo, hi, ind, FE, 0.0)
    pl_ = lin.predict(_frame(Xt))
    c = lambda p, k: np.corrcoef(p, Yt[:, k])[0, 1]
    assert c(pt["y_2s"], 1) > 0.5 and abs(c(pl_["y_2s"], 1)) < 0.05      # tree finds the interaction
    g.save(tmp_path / "g")
    assert np.allclose(MD.LGBModel.load(tmp_path / "g").predict(_frame(Xt))["y_8s"], pt["y_8s"])
    assert np.isfinite(pt["y_500ms"]).all()                              # NaN inputs handled


def test_zero_model():
    z = MD.ZeroModel().predict(_frame(np.ones((5, 3))))
    assert all((v == 0).all() for v in z.values())
