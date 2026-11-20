"""Stage 4: alignment rules and the causality check, on a small SYNTHETIC two-day dataset."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hft500 import align as A, causality as C  # noqa: E402

D1, D2 = "2025-09-15", "2025-09-16"
T1, T2 = A.day_start(D1), A.day_start(D2)
INST = "BTCUSDT_PERP"


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> A.DirRoot:
    r = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(1)
    for day, t0 in ((D1, T1), (D2, T2)):
        ts = np.arange(t0, t0 + A.DAY_MS, 500, dtype=np.int64)
        if day == D2:
            ts = ts[3:]                                     # day 2 misses its first 3 snapshots (real pattern)
        n = len(ts)
        mid = 100_000 + np.cumsum(rng.integers(-1, 2, n)) * 0.1
        book = {"ts": ts, "bid_px": mid - 0.05, "ask_px": mid + 0.05,
                "bid_qty": rng.random(n).astype(np.float32) * 5, "ask_qty": rng.random(n).astype(np.float32) * 5}
        for c in A.BUCKETS:
            book[c] = (rng.random(n) * 1e5).astype(np.float32)
        p = r / "book" / INST / f"{day}.parquet"; p.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(book).write_parquet(p)
        bts = np.sort(rng.choice(np.arange(t0 + 100, t0 + A.DAY_MS, 100), 50_000, replace=False))
        bts = bts[bts % 500 != 0]                           # bbo never sits on 500 ms points
        m = len(bts)
        bmid = 100_000 + rng.integers(-50, 50, m) * 0.1
        p = r / "bbo" / INST / f"{day}.parquet"; p.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"ts": bts, "bid_px": bmid - 0.05, "ask_px": bmid + 0.05,
                      "bid_qty": np.ones(m, np.float32), "ask_qty": np.ones(m, np.float32)}).write_parquet(p)
        # trades: a bar at 12:00:00.300, plus (day 1 only) the final bar stamped at next-day midnight
        tts = [t0 + 43_200_300, t0 + 43_200_600] + ([t0 + A.DAY_MS] if day == D1 else [])
        k = len(tts)
        p = r / "trades" / INST / f"{day}.parquet"; p.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"ts": np.array(tts, np.int64), "buy_qty": np.array([1.0, 0.0, 7.0][:k], np.float32),
                      "buy_vwap": [100_000.0, np.nan, 100_001.0][:k], "buy_n": np.array([2, 0, 3][:k], np.int32),
                      "buy_max_qty": np.array([0.6, np.nan, 5.0][:k], np.float32), "buy_last_ms": np.array([20, 0, 1][:k], np.uint8),
                      "sell_qty": np.array([0.0, 2.0, 0.0][:k], np.float32), "sell_vwap": [np.nan, 99_999.0, np.nan][:k],
                      "sell_n": np.array([0, 1, 0][:k], np.int32), "sell_max_qty": np.array([np.nan, 2.0, np.nan][:k], np.float32),
                      "sell_last_ms": np.array([0, 50, 0][:k], np.uint8)}).write_parquet(p)
        sts = np.arange(t0, t0 + A.DAY_MS, 1000, dtype=np.int64)
        p = r / "stable" / f"{day}.parquet"; p.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"ts": np.repeat(sts, 2), "pair": ["USDCUSDT", "FDUSDUSDT"] * len(sts),
                      "bid_px": 0.9998, "ask_px": 1.0002, "stale_ms": np.zeros(2 * len(sts), np.int32)}).write_parquet(p)
    return A.DirRoot(r)


def official_quote_grid(root: A.DirRoot, date: str):
    """common.quote_grid from the assignment, re-typed (no warm-up)."""
    cols = ["ts", *A.QUOTE_COLS]
    b, q = root.read("book", date, INST, cols), root.read("bbo", date, INST, cols)
    ts = np.concatenate([b["ts"].to_numpy(), q["ts"].to_numpy()])
    arrs = [np.concatenate([b[c].to_numpy().astype("float64"), q[c].to_numpy().astype("float64")]) for c in cols[1:]]
    o = np.argsort(ts, kind="stable"); ts = ts[o]; arrs = [a[o] for a in arrs]
    g = A.grid(date, 100); i = np.searchsorted(ts, g, "right") - 1; ok = i >= 0; i = np.maximum(i, 0)
    return g, [np.where(ok, a[i], np.nan) for a in arrs]


def test_grid_is_built_from_the_date():
    g = A.grid(D2)
    assert len(g) == 172_800 and g[0] == T2 and g[-1] == T2 + A.DAY_MS - 500


def test_quote_grid_equals_official_recipe(root):
    g, ref = official_quote_grid(root, D2)
    ours = A.quote_grid(root, INST, D2, warmup=False)
    assert np.array_equal(ours["ts"].to_numpy(), g)
    for c, r in zip(A.QUOTE_COLS, ref):
        assert np.array_equal(ours[c].to_numpy(), r, equal_nan=True), c


def test_warmup_fills_the_start_of_day(root):
    cold = A.quote_grid(root, INST, D2, warmup=False)
    warm = A.quote_grid(root, INST, D2, warmup=True)
    assert np.isnan(cold["bid_px"][0])                    # book missing at 00:00:00.000, no bbo yet
    prev = root.read("book", D1, INST).sort("ts")
    prev_bbo = root.read("bbo", D1, INST).sort("ts")
    last = prev if prev["ts"][-1] > prev_bbo["ts"][-1] else prev_bbo
    assert warm["bid_px"][0] == pytest.approx(float(last["bid_px"][-1]))
    assert warm["quote_age_ms"][0] == T2 - int(last["ts"][-1])
    # once day-2 data exists, warm and cold agree
    k = 20  # 00:00:02.000
    assert warm["bid_px"][k] == cold["bid_px"][k]


def test_trade_bars_land_at_their_end_time(root):
    tr = A.trades_on_grid(root, INST, D2)
    at = lambda ms: tr.filter(pl.col("ts") == T2 + ms)
    assert at(43_200_000)["tr_buy_qty"][0] == 0           # bar ending 12:00:00.300 is NOT known at 12:00:00.000
    assert at(43_200_500)["tr_buy_qty"][0] == pytest.approx(1.0)
    assert at(43_201_000)["tr_sell_qty"][0] == pytest.approx(2.0)  # bar ending .600 -> window (.500, 1.000]
    assert at(0)["tr_buy_qty"][0] == pytest.approx(7.0)   # previous day's midnight bar lands on t = day start
    assert at(0)["tr_buy_age_ms"][0] == 1                 # its last trade was 1 ms before midnight
    assert at(43_200_500)["tr_buy_notional"][0] == pytest.approx(100_000.0)
    cold = A.trades_on_grid(root, INST, D2, warmup=False)
    assert cold.filter(pl.col("ts") == T2)["tr_buy_qty"][0] == 0


def test_load_day_shape_and_stable(root):
    df = A.load_day(root, INST, D2)
    assert df.height == 172_800 and df["ts"].is_sorted()
    assert {"q_bid_px", "q_px_age_ms", "bid_v01", "book_age_ms", "tr_buy_qty"} <= set(df.columns)
    assert df["q_bid_px"].is_nan().sum() == 0             # warm-up leaves no hole
    st = A.stable_on_grid(root, D2)
    assert st["st_USDCUSDT_mid"][1] == pytest.approx(1.0) and st["st_USDCUSDT_age_ms"][1] == 500


def test_truncated_root_hides_the_future(root):
    T = T2 + 3_600_000
    tr = A.TruncatedRoot(root, T)
    assert tr.read("bbo", D2, INST)["ts"].max() <= T
    assert tr.read("bbo", D1, INST).height == root.read("bbo", D1, INST).height   # past day untouched


def test_load_day_passes_the_causality_check(root):
    res = C.truncation_check(lambda r, d: A.load_day(r, INST, d), root, D2, n=4,
                             Ts=[T2, T2 + 43_200_300, T2 + 43_200_500], inst_for_gaps=INST)
    assert all(r["ok"] for r in res), res


@pytest.mark.parametrize("leak", ["shift_minus_1", "day_mean", "backward_fill", "grid_from_data"])
def test_leaky_features_fail(root, leak):
    def fn(r, d):
        df = A.load_day(r, INST, d).select("ts", "q_bid_px")
        if leak == "shift_minus_1":
            return df.with_columns(pl.col("q_bid_px").shift(-1).alias("f"))
        if leak == "day_mean":
            return df.with_columns((pl.col("q_bid_px") - pl.col("q_bid_px").mean()).alias("f"))
        if leak == "backward_fill":
            raw = r.read("book", d, INST, ["ts", "bid_px"])
            return df.join(raw, on="ts", how="left").with_columns(pl.col("bid_px").fill_null(strategy="backward").alias("f"))
        # grid_from_data: normalising time by the last timestamp present in the data
        last = r.read("bbo", d, INST, ["ts"])["ts"].max()
        return df.with_columns(((pl.col("ts") - last) / 1000).alias("f"))
    res = C.truncation_check(fn, root, D2, n=3, seed=7, inst_for_gaps=INST)
    assert not all(r["ok"] for r in res), f"{leak} slipped through"


def test_gap_cuts_target_the_missing_start_of_day(root):
    cuts = C.gap_cuts(root, INST, D2)
    assert any(T2 <= c < T2 + 1500 for c in cuts)          # inside the 3 missing snapshots


def test_no_data_yet_still_gives_the_full_grid(root):
    """First day of the dataset (no previous day), cut at midnight: nothing is known yet."""
    tr = A.TruncatedRoot(root, T1 - 1)                 # D1 has no previous day; nothing at all is known
    df = A.load_day(tr, INST, D1)
    assert df.height == 172_800
    assert df["q_bid_px"].is_nan().all() and df["bid_v01"].is_nan().all() and (df["tr_buy_qty"] == 0).all()
    at_cut = A.load_day(A.TruncatedRoot(root, T1), INST, D1)   # only the 00:00:00.000 snapshot is known
    assert not np.isnan(at_cut["q_bid_px"][0]) and at_cut["q_quote_age_ms"][1] == 500
    res = C.truncation_check(lambda r, d: A.load_day(r, INST, d), root, D1, n=2, Ts=[T1, T1 + 1])
    assert all(r["ok"] for r in res), res


def test_one_sided_trades_do_not_crash(root):
    """Cut right after the first bar of day 1 (buy-only): the sell side has never traded."""
    tr = A.trades_on_grid(A.TruncatedRoot(root, T1 + 43_200_300), INST, D1, warmup=False)
    assert tr["tr_sell_age_ms"].is_nan().all() and tr["tr_buy_qty"].sum() == pytest.approx(1.0)
