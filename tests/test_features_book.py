"""Stage 7 features: hand-checkable values, the registry, and the truncation test."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hft500 import align as A, causality as C, features_book as F  # noqa: E402

D1, D2 = "2025-09-15", "2025-09-16"
INST = "BTCUSDT_PERP"


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> A.DirRoot:
    r = tmp_path_factory.mktemp("fb")
    rng = np.random.default_rng(3)
    for day in (D1, D2):
        t0 = A.day_start(day)
        ts = np.arange(t0, t0 + A.DAY_MS, 500, dtype=np.int64)
        if day == D1:
            ts = ts[3:]                                       # first day of data misses its first snapshots
        if day == D2:
            ts = np.delete(ts, np.arange(5000, 5400))          # a 200 s book gap
        n = len(ts)
        mid = 50_000 + np.cumsum(rng.integers(-1, 2, n)) * 0.1
        cols = {"ts": ts, "bid_px": mid - 0.05, "ask_px": mid + 0.05,
                "bid_qty": (rng.random(n) * 5 + 0.1).astype(np.float32), "ask_qty": (rng.random(n) * 5 + 0.1).astype(np.float32)}
        for c in A.BUCKETS:
            cols[c] = (rng.random(n) * 1e5).astype(np.float32)
        p = r / "book" / INST / f"{day}.parquet"; p.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(cols).write_parquet(p)
        bts = np.sort(rng.choice(np.arange(t0 + 2000, t0 + A.DAY_MS, 100), 40_000, replace=False))
        bts = bts[bts % 500 != 0]
        m = len(bts)
        bm = 50_000 + rng.integers(-30, 30, m) * 0.1
        p = r / "bbo" / INST / f"{day}.parquet"; p.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"ts": bts, "bid_px": bm - 0.05, "ask_px": bm + 0.05,
                      "bid_qty": np.ones(m, np.float32), "ask_qty": np.full(m, 3.0, np.float32)}).write_parquet(p)
    return A.DirRoot(r)


def test_shape_registry_and_ranges(root):
    df = F.build(root, INST, D2, meta_tick=0.1)
    assert df.height == 172_800 and set(F.FEATURES) <= set(df.columns)
    assert len(F.registry_frame()) == len(F.FEATURES)
    assert df["imb_l1"].drop_nans().is_between(-1, 1).all()
    assert df["mp_dev_ticks"].drop_nans().is_between(-2, 2).all()
    assert (df["spread_ticks"].drop_nans() == 1).all()                     # tick learnt from day 1 = 0.1
    assert df["frozen_60s"].drop_nans().is_in([0.0, 1.0]).all()
    # warm-up: windows are already full at 00:00:00.000
    assert np.isfinite(df["ret_60s"][0]) and np.isfinite(df["rv_5min"][0]) and np.isfinite(df["imb_l1_ewm_30s"][0])


def test_values_by_hand(root):
    df = F.build(root, INST, D2, meta_tick=0.1)
    q = A.quote_grid(root, INST, D2, warmup=True, step_ms=500)
    k = 1234
    b, a = q["bid_qty"][k], q["ask_qty"][k]
    assert df["imb_l1"][k] == pytest.approx((b - a) / (b + a))
    m = (q["bid_px"] + q["ask_px"]) / 2
    assert df["ret_2s"][k] == pytest.approx(np.log(m[k] / m[k - 4]) * 1e4)
    assert df["imb_l1_chg_1s"][k] == pytest.approx(df["imb_l1"][k] - df["imb_l1"][k - 2])


def test_helpers():
    x = np.array([1.0, np.nan, 3.0, 5.0])
    assert F.rolling_sum(x, 2).tolist() == [1.0, 1.0, 3.0, 8.0]
    e = F.ewm(np.array([0.0, np.nan, 1.0]), 1.0)                            # alpha = 0.5
    assert e.tolist() == [0.0, 0.0, 0.5]
    assert F.lag_diff(np.array([1.0, 2.0, 4.0]), 1)[1:].tolist() == [1.0, 2.0]


def test_tick_comes_from_the_previous_day_only(root):
    ev = A.quote_events(root, INST, D2, warmup=True)
    assert F.day_tick(ev, A.day_start(D2), meta_tick=0.5) == pytest.approx(0.1)
    cold = A.quote_events(root, INST, D2, warmup=False)                     # no previous day -> meta fallback
    assert F.day_tick(cold, A.day_start(D2), meta_tick=0.09999999999) == pytest.approx(0.1)


def test_features_pass_the_truncation_check(root):
    d0 = A.day_start(D2)
    res = C.truncation_check(lambda r, d: F.build(r, INST, d, meta_tick=0.1), root, D2, n=3, seed=11,
                             Ts=[d0, d0 + 1], inst_for_gaps=INST)
    assert all(r["ok"] for r in res), [r for r in res if not r["ok"]]


def test_a_whole_day_tick_estimate_would_have_leaked(root):
    """The trap we avoid: estimating the tick from the CURRENT day's prices."""
    from hft500.quality import empirical_tick
    def leaky(r, d):
        ev = A.quote_events(r, INST, d, warmup=False)
        tk = empirical_tick(ev["bid_px"].to_numpy()) if ev is not None else np.nan
        return pl.DataFrame({"ts": A.grid(d), "tick": np.full(172_800, tk)})
    d0 = A.day_start(D2)
    res = C.truncation_check(leaky, root, D2, n=0, Ts=[d0 + 50])            # 50 ms in: one snapshot, no step known yet
    assert not all(r["ok"] for r in res)


def test_first_day_with_nothing_known_yet(root):
    """Real failure from Stage 7: first day of data, no previous day, cut at midnight."""
    d0 = A.day_start(D1)
    full = F.build(root, INST, D1, meta_tick=0.1)
    assert all(np.isnan(full[f][0]) for f in F.FEATURES)                  # nothing known at 00:00:00.000
    res = C.truncation_check(lambda r, d: F.build(r, INST, d, meta_tick=0.1), root, D1, n=2, seed=2,
                             Ts=[d0, d0 + 600, d0 + 1500], inst_for_gaps=INST)
    assert all(r["ok"] for r in res), [r for r in res if not r["ok"]]
