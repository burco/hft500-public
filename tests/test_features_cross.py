"""Stage 8 cross/trade features on a SYNTHETIC multi-instrument market with known answers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hft500 import align as A, causality as C, features_cross as X  # noqa: E402

D1, D2 = "2025-09-15", "2025-09-16"
INST = "BTCUSDT_PERP"
SPOT_PREMIUM = 2e-4          # spot = perp * (1 + 2 bps)
USDC_RATE = 0.9998           # USDCUSDT mid


def _book(path: Path, ts, mid, rng, tick=0.1):
    n = len(ts)
    cols = {"ts": ts, "bid_px": mid - tick / 2, "ask_px": mid + tick / 2,
            "bid_qty": (rng.random(n) * 5 + 0.1).astype(np.float32), "ask_qty": (rng.random(n) * 5 + 0.1).astype(np.float32)}
    for c in A.BUCKETS:
        cols[c] = np.full(n, 1e5, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(cols).write_parquet(path)


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> A.DirRoot:
    return make_root(tmp_path_factory.mktemp("x"))


def make_root(r: Path) -> A.DirRoot:
    """SYNTHETIC two-day market: BTC perp, ETH perp, BTC spot (+2 bps), BTC USDC spot, stable, BTC trades."""
    rng = np.random.default_rng(8)
    for day in (D1, D2):
        t0 = A.day_start(day)
        ts = np.arange(t0, t0 + A.DAY_MS, 500, dtype=np.int64)
        perp = 100_000 + np.cumsum(rng.integers(-1, 2, len(ts))) * 0.1
        _book(r / "book" / INST / f"{day}.parquet", ts, perp, rng)
        eth = 3_000 + np.cumsum(rng.integers(-1, 2, len(ts))) * 0.01
        _book(r / "book" / "ETHUSDT_PERP" / f"{day}.parquet", ts, eth, rng, 0.01)
        _book(r / "book" / "BTCUSDT_SPOT" / f"{day}.parquet", ts, perp * (1 + SPOT_PREMIUM), rng, 0.0)
        usdc = perp / USDC_RATE
        if day == D2:
            usdc[20_000:20_200] = np.nan                       # PDF null rule during a USDC depeg
        _book(r / "book" / "BTCUSDC_SPOT" / f"{day}.parquet", ts, usdc, rng, 0.0)
        sts = np.arange(t0, t0 + A.DAY_MS, 1000, dtype=np.int64)
        p = r / "stable" / f"{day}.parquet"; p.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"ts": sts, "pair": "USDCUSDT", "bid_px": USDC_RATE, "ask_px": USDC_RATE,
                      "stale_ms": np.zeros(len(sts), np.int32)}).write_parquet(p)
        # trades: 3 buy bars and 1 sell bar inside (t0+10.0 s, t0+10.5 s]
        tts = np.array([t0 + 10_100, t0 + 10_200, t0 + 10_300, t0 + 10_500], np.int64)
        p = r / "trades" / INST / f"{day}.parquet"; p.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"ts": tts, "buy_qty": np.array([1, 1, 1, 0], np.float32), "buy_vwap": [1e5, 1e5, 1e5, np.nan],
                      "buy_n": np.array([1, 1, 1, 0], np.int32), "buy_max_qty": np.array([1, 1, 1, np.nan], np.float32),
                      "buy_last_ms": np.array([5, 5, 5, 0], np.uint8), "sell_qty": np.array([0, 0, 0, 1], np.float32),
                      "sell_vwap": [np.nan, np.nan, np.nan, 1e5], "sell_n": np.array([0, 0, 0, 1], np.int32),
                      "sell_max_qty": np.array([np.nan, np.nan, np.nan, 1], np.float32),
                      "sell_last_ms": np.array([0, 0, 0, 20], np.uint8)}).write_parquet(p)
    return A.DirRoot(r)


def test_ofi_increments_by_hand():
    q = {"bid_px": np.array([10.0, 10.0, 10.1, 10.1]), "bid_qty": np.array([5.0, 7.0, 2.0, 1.0]),
         "ask_px": np.array([10.2, 10.2, 10.2, 10.1]), "ask_qty": np.array([4.0, 4.0, 6.0, 3.0])}
    e = X.ofi_increments(q)
    # step1: bid same price +2, ask same 0 -> +2 | step2: bid up -> +2 (new queue), ask same +2 -> 2-2=0
    # step3: bid same -1, ask DOWN -> new ask queue 3 counts as selling pressure -> -1-3 = -4
    assert np.isnan(e[0]) and e[1:].tolist() == [2.0, 0.0, -4.0]


@pytest.fixture(scope="module")
def feats(root):
    return X.build(root, INST, D2)


def test_shape_and_uniform_columns(feats):
    assert feats.height == 172_800 and set(X.FEATURES) <= set(feats.columns)
    assert feats["xret_btc_0.5s"].is_nan().all()                    # own coin
    assert feats["xret_sol_0.5s"].is_nan().all()                    # instrument absent from this fixture
    assert np.isfinite(feats["xret_eth_2s"][0])                      # warm-up across midnight


def test_basis_and_stablecoin_gap(feats):
    expected = (1 / (1 + SPOT_PREMIUM) - 1) * 1e4
    assert feats["basis_bps"].drop_nans().mean() == pytest.approx(expected, abs=1e-6)
    assert abs(feats["basis_dev_60s"][50_000]) < 1e-6
    assert abs(feats["usdc_gap_bps"][1_000]) < 1e-6                 # converted with the stable table -> no gap
    assert feats["usdc_gap_bps"][20_100] != feats["usdc_gap_bps"][20_100]          # NaN while nulled
    assert np.isfinite(feats["usdc_gap_dev_60s"][20_300])           # EWMA carried over the hole


def test_lead_lag_matches_the_other_instruments_return(root, feats):
    q = A.quote_grid(root, "ETHUSDT_PERP", D2, warmup=True, step_ms=500)
    m = ((q["bid_px"] + q["ask_px"]) / 2).to_numpy()
    k = 777
    assert feats["xret_eth_0.5s"][k] == pytest.approx(np.log(m[k] / m[k - 1]) * 1e4)


def test_trade_windows(feats):
    at = lambda s: feats.filter(pl.col("ts") == A.day_start(D2) + int(s * 1000))
    r = at(10.5)                                                    # window (10.0, 10.5] holds 3 buys, 1 sell
    assert r["tim_0.5s"][0] == pytest.approx((3 - 1) / 4)
    assert at(10.0)["tim_0.5s"][0] == 0.0                           # nothing yet
    assert r["last_side"][0] == pytest.approx(-1.0, abs=0.25)        # the last trade (at 10.48 s) was a sell


def test_cross_features_pass_the_truncation_check(root):
    d0 = A.day_start(D2)
    res = C.truncation_check(lambda r, d: X.build(r, INST, d), root, D2, n=2, seed=4,
                             Ts=[d0, d0 + 10_400, d0 + 10_500, d0 + 10_000_050], inst_for_gaps=INST)
    assert all(r["ok"] for r in res), [r for r in res if not r["ok"]]
