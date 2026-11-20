"""Stage 9 evaluation helpers on SYNTHETIC data with known answers."""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hft500 import evaluation as E  # noqa: E402


def test_spearman_with_ties_matches_reference():
    x = np.array([1, 2, 2, 3, 10, 10, 10, 4.0]); y = np.array([3, 1, 2, 5, 9, 8, 7, 4.0])
    rx, ry = E.ranks(x), E.ranks(y)
    assert rx.tolist() == [0, 1.5, 1.5, 3, 6, 6, 6, 4]
    # Spearman = Pearson of average ranks
    assert E.corr(rx, ry) == pytest.approx(np.corrcoef(rx, ry)[0, 1])
    assert E.corr(np.ones(5), np.arange(5.0)) == 0.0                # constant -> 0 (baseline convention)


def test_day_stats_and_deciles():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50_000); y = 0.3 * x + rng.normal(size=50_000)
    x[:500] = np.nan
    r = {(d["feature"], d["horizon"]): d for d in E.day_stats({"x": x, "neg": -x}, {"y": y})}
    assert r[("x", "y")]["pearson"] == pytest.approx(0.3 / np.sqrt(1.09), abs=0.01)
    assert r[("neg", "y")]["pearson"] == pytest.approx(-r[("x", "y")]["pearson"])
    assert r[("x", "y")]["coverage"] == pytest.approx(0.99)
    dec = np.array([r[("x", "y")][f"d{i}"] for i in range(10)])
    assert np.all(np.diff(dec) > 0) and E.monotonicity(dec) == pytest.approx(1.0)
    u = np.array([3, 2, 1, 0, 0, 0, 0, 1, 2, 3.0])                       # U-shape
    assert abs(E.monotonicity(u)) < 0.3


def test_summarise_t_stat_and_flips():
    d = pl.DataFrame({"instrument": "A", "horizon": "y", "feature": "f",
                      "date": ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"],
                      "pearson": [0.2, 0.4, -0.1, -0.1], "spearman": 0.0, "coverage": 1.0})
    s = E.summarise(d).row(0, named=True)
    v = np.array([0.2, 0.4, -0.1, -0.1])
    assert s["t"] == pytest.approx(v.mean() / (v.std(ddof=1) / 2))
    assert s["pos_share"] == 0.5 and s["months"] == 2 and s["months_flipped"] == 1


def test_clusters_single_linkage():
    c = np.array([[1, .95, .1], [.95, 1, -.92], [.1, -.92, 1]])
    assert E.clusters(c, ["a", "b", "c"], 0.9) == [["a", "b", "c"]]
    assert E.clusters(c, ["a", "b", "c"], 0.94) == [["a", "b"], ["c"]]


def test_month_folds_embargo_and_order():
    ds = [f"2026-0{m}-{d:02d}" for m in (1, 2, 3) for d in range(1, 29)]
    f = E.month_folds(ds, "2026-02", embargo_days=1)
    assert len(f) == 2
    tr, te = f[0]
    assert te[0] == "2026-02-01" and tr[-1] == "2026-01-28"             # fixture has days 1-28 only
    ds2 = ds + ["2026-01-31"]
    tr2, _ = E.month_folds(ds2, "2026-02")[0]
    assert "2026-01-31" not in tr2 and max(tr2) < "2026-02-01"
    assert all(max(t) < min(s) for t, s in f)


def test_ridge_from_moments_matches_direct_fit():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20_000, 3)); X[::7, 2] = np.nan
    Y = np.column_stack([X[:, 0] - 0.5 * X[:, 1] + rng.normal(size=20_000) * 0.1, rng.normal(size=20_000)])
    lo, hi = np.full(3, -10.0), np.full(3, 10.0)
    Z = E.design(X, lo, hi, np.array([False, False, True]))
    assert Z.shape == (20_000, 4) and Z[0, 3] == 1.0 and Z[0, 2] == 0.0
    M = None
    for part in np.array_split(np.arange(20_000), 5):                    # moments add up across "days"
        M = E.add_moments(M, E.moments(Z[part], Y[part]))
    cols = np.arange(4)
    b, mu, sd = E.ridge_from_moments(M, 0, cols, lam=0.0)
    pred = ((Z - mu) / sd) @ b
    ref = np.linalg.lstsq(np.column_stack([np.ones(len(Z)), Z]), Y[:, 0], rcond=None)[0]
    assert np.corrcoef(pred, np.column_stack([np.ones(len(Z)), Z]) @ ref)[0, 1] == pytest.approx(1.0, abs=1e-9)
    b2, _, _ = E.ridge_from_moments(M, 0, np.array([1]), lam=0.0)        # subset = ablation
    assert b2.shape == (1,) and b2[0] < 0
