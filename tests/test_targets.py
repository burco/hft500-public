"""Stage 5 target helpers on hand-built examples."""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hft500 import targets as T  # noqa: E402


def test_rebuild_uses_asof_and_offset():
    ev_ts = np.array([0, 100, 600, 2_000, 8_000])
    ev_mid = np.array([100.0, 101.0, 102.0, 104.0, 108.0])
    t = np.array([0, 500])
    y = T.rebuild(ev_ts, ev_mid, t, offset_ms=10)
    # t=0: M(0)=mid@10=100, M(500)=mid@510=101  -> ln(101/100)
    assert y["y_500ms"][0] == pytest.approx(np.log(101 / 100))
    # t=500: M=mid@510=101, mid@1010=102
    assert y["y_500ms"][1] == pytest.approx(np.log(102 / 101))
    assert y["y_8s"][0] == pytest.approx(np.log(108 / 100))
    # with 100 ms quotes, the 10 ms offset changes nothing unless a record sits in (t, t+10]
    assert T.rebuild(ev_ts, ev_mid, t, 0)["y_500ms"][0] == y["y_500ms"][0]


def test_additivity_and_implied_mid():
    rng = np.random.default_rng(0)
    lm = np.cumsum(rng.normal(0, 1e-4, 200)) + np.log(100)
    ts = np.arange(200) * 500
    y = {h: np.r_[lm[k:] - lm[:-k], np.full(k, np.nan)] for h, k in T.STEPS.items()}
    tg = pl.DataFrame({"ts": ts, **y})
    err = T.additivity_error(tg)
    assert err["y_2s"] < 1e-12 and err["y_8s"] < 1e-12
    assert np.allclose(T.implied_log_mid(y["y_500ms"], lm), lm)
    y5 = y["y_500ms"].copy(); y5[50] = np.nan
    assert np.allclose(T.implied_log_mid(y5, lm), lm)          # re-anchors after a gap


def test_acf_of_overlapping_sums_is_high():
    rng = np.random.default_rng(1)
    e = rng.normal(size=20_000)
    y8 = np.lib.stride_tricks.sliding_window_view(e, 16).sum(1)
    a = T.acf(y8, [1, 8, 16])
    assert a[0] == pytest.approx(15 / 16, abs=0.02) and a[1] == pytest.approx(0.5, abs=0.03) and abs(a[2]) < 0.03


def test_discreteness_and_tails():
    mid = np.full(6, 100.0)
    tick = 0.01
    moves = np.array([0, 0, 0.005, -0.005, 0.01, 0.05])          # 0, 0, ±1/2 tick, 1 tick, 5 ticks
    y = np.log((mid + moves) / mid)
    d = T.discreteness(y, mid, tick)
    assert d["pct_0_half_ticks"] == pytest.approx(200 / 6) and d["pct_1_half_ticks"] == pytest.approx(200 / 6)
    assert d["pct_gt_4_half_ticks"] == pytest.approx(100 / 6)
    ts = T.tail_share(np.r_[np.zeros(998), 1.0, -1.0], -0.5, 0.5)
    assert ts["clipped_pct"] == pytest.approx(0.2) and ts["ss_share_in_tails_pct"] == pytest.approx(100.0)


def test_clip_targets():
    tg = pl.DataFrame({"ts": [0, 500], "y_500ms": [0.01, -0.01], "y_2s": [0.0, 0.0], "y_8s": [0.0, 0.0]})
    c = T.clip_targets(tg, "X", {("X", "y_500ms"): (-0.001, 0.002)})
    assert c["y_500ms"].to_list() == [0.002, -0.001]
