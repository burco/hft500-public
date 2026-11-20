"""Stage 6: metric and OLS baseline on hand-checkable inputs."""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hft500 import align as A, baseline_ols as B, metric as M  # noqa: E402


def test_pearson_conventions():
    assert M.pearson(np.ones(5), np.arange(5.0)) == 0.0                     # constant prediction -> 0
    assert M.pearson(np.arange(5.0), np.arange(5.0) * 3 + 1) == pytest.approx(1.0)
    assert M.pearson(np.arange(5.0), -np.arange(5.0)) == pytest.approx(-1.0)


def test_score_day_clips_targets_and_zero_fills_predictions():
    y = np.array([0.0, 1.0, 2.0, 100.0, np.nan])
    p = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    # clipping 100 -> 3 makes the relation perfectly linear; the NaN-target row is dropped
    assert M.score_day(p, y, -10, 3) == pytest.approx(1.0)
    assert M.score_day(p, y, -10, 1000) < 0.99
    p_missing = np.array([0.0, 1.0, np.nan, 3.0, 5.0])                     # missing -> 0
    assert M.score_day(p_missing, y, -10, 3) == pytest.approx(np.corrcoef([0, 1, 0, 3], [0, 1, 2, 3])[0, 1])
    assert M.score_day(p, y, -10, 3, min_points=10) is None


def test_score_predictions_and_aggregate():
    ts = np.arange(4) * 500
    tg = pl.DataFrame({"ts": ts, "y_500ms": [0.0, 1.0, 2.0, 3.0], "y_2s": [3.0, 2.0, 1.0, 0.0], "y_8s": [1.0, 1.0, 1.0, 1.0]})
    preds = pl.DataFrame({"ts": ts[:3], "instrument": "X", "pred_500ms": [0.0, 1.0, 2.0], "pred_2s": [0.0, 1.0, 2.0], "pred_8s": [1.0, 2.0, 3.0]})
    clips = {("X", h): (-9, 9) for h in M.HORIZONS}
    rows = M.score_predictions(preds, {"X": tg}, clips, "2026-07-15")
    r = {x["horizon"]: x["corr"] for x in rows}
    assert r["y_500ms"] == pytest.approx(np.corrcoef([0, 1, 2, 0], [0, 1, 2, 3])[0, 1])   # last row missing -> 0
    assert r["y_8s"] == 0.0                                                     # constant target -> 0
    cells, score = M.aggregate(pl.DataFrame(rows))
    assert score == pytest.approx(np.mean(list(r.values())))


def test_ols_fit_and_predict_day(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "d"
    days = ["2025-09-15", "2025-09-16"]
    for d in days:
        t = A.grid(d)[:2000]
        bq, aq = rng.random(len(t)) * 5, rng.random(len(t)) * 5
        imb = (bq - aq) / (bq + aq)
        y = 1e-4 + 2e-4 * imb + rng.normal(0, 1e-5, len(t))
        (root / "book" / "X").mkdir(parents=True, exist_ok=True); (root / "targets" / "X").mkdir(parents=True, exist_ok=True)
        keep = np.ones(len(t), bool); keep[5] = False                       # one missing book row
        pl.DataFrame({"ts": t[keep], "bid_qty": bq[keep], "ask_qty": aq[keep]}).write_parquet(root / "book" / "X" / f"{d}.parquet")
        pl.DataFrame({"ts": t, "y_500ms": y, "y_2s": 2 * y, "y_8s": 4 * y}).write_parquet(root / "targets" / "X" / f"{d}.parquet")
    R = A.DirRoot(root)
    coef = B.fit(R, "X", *days, {h: (-1, 1) for h in B.HORIZONS})
    a, b = coef["y_500ms"]
    assert a == pytest.approx(1e-4, rel=0.02) and b == pytest.approx(2e-4, rel=0.02)
    assert coef["y_8s"][1] == pytest.approx(4 * b)
    g, p = B.predict_day(R, "X", days[0], coef)
    assert len(g) == 172_800 and p["y_500ms"][5] == 0.0 and p["y_500ms"][2000] == 0.0   # no book row -> 0
    assert p["y_500ms"][0] != 0.0
