"""Stage 3 checks on SYNTHETIC days with deliberately planted faults."""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from hft500 import quality as q  # noqa: E402

DAY = "2025-09-15"
D0 = 1757894400000           # 2025-09-15 00:00:00 UTC
TICK = 0.1


def make_book(n_drop=0, crossed_at=None, nan_at=None, offtick_at=None):
    ts = np.arange(D0, D0 + 86_400_000, 500, dtype=np.int64)
    if n_drop:
        ts = np.delete(ts, np.arange(1000, 1000 + n_drop))            # one contiguous gap
    n = len(ts)
    bid = np.full(n, 100_000.0); ask = bid + TICK
    if crossed_at is not None: bid[crossed_at] = ask[crossed_at] + TICK
    if nan_at is not None: ask[nan_at] = np.nan
    if offtick_at is not None: bid[offtick_at] += 0.03
    cols = {"ts": ts, "bid_px": bid, "ask_px": ask, "bid_qty": np.full(n, 2.0, np.float32), "ask_qty": np.full(n, 1.0, np.float32)}
    for s in ("bid", "ask"):
        for i in range(1, 12):
            cols[f"{s}_v{i:02d}"] = np.full(n, 1e5 * i, np.float32)
    return pl.DataFrame(cols)


def make_bbo(on500=False):
    ts = np.array([D0 + 100, D0 + 200, D0 + 300] + ([D0 + 500] if on500 else []), dtype=np.int64)
    n = len(ts)
    return pl.DataFrame({"ts": ts, "bid_px": [100_000.0 - TICK * k for k in range(n)],
                         "ask_px": [100_000.0 + TICK * (1 - k) for k in range(n)],
                         "bid_qty": np.ones(n, np.float32), "ask_qty": np.ones(n, np.float32)})


def make_trades(bad=False):
    ts = np.array([D0 + 100, D0 + 200, D0 + 86_400_000], dtype=np.int64)  # last bar = next-day midnight (valid)
    df = pl.DataFrame({
        "ts": ts, "buy_qty": np.array([1, 0, 2], np.float32), "buy_vwap": [100_000.05, np.nan, 100_000.0],
        "buy_n": np.array([3, 0, 1], np.int32), "buy_max_qty": np.array([0.5, np.nan, 2], np.float32),
        "buy_last_ms": np.array([10, 0, 5], np.uint8),
        "sell_qty": np.array([0, 1, 0], np.float32), "sell_vwap": [np.nan, 99_999.95, np.nan],
        "sell_n": np.array([0, 2, 0], np.int32), "sell_max_qty": np.array([np.nan, 0.6, np.nan], np.float32),
        "sell_last_ms": np.array([0, 40, 0], np.uint8)})
    if bad:
        df = df.with_columns(pl.Series("buy_vwap", [np.nan, np.nan, 100_000.0]),       # qty>0 but no vwap
                             pl.Series("sell_last_ms", np.array([0, 150, 0], np.uint8)))  # >= bar length
    return df


def test_clean_day_has_no_faults():
    bm = q.book_metrics(make_book(), D0, TICK)
    assert bm["missing_pts"] == 0 and bm["crossed"] == 0 and bm["off_tick"] == 0 and bm["nonfinite_px_qty"] == 0
    assert bm["pct_1tick"] == pytest.approx(100.0) and bm["longest_missing_s"] == 0
    tm = q.trades_metrics(make_trades(), D0, 99_000, 101_000)
    assert tm["consistency_violations"] == 0 and tm["outside_day"] == 0
    assert tm["buy_notional"] == pytest.approx(1 * 100_000.05 + 2 * 100_000.0, rel=1e-6)
    assert q.bbo_metrics(make_bbo(), D0, TICK)["on_500ms"] == 0


def test_planted_faults_are_caught():
    bm = q.book_metrics(make_book(n_drop=200, crossed_at=5, nan_at=7, offtick_at=9), D0, TICK)
    assert bm["missing_pts"] == 200 and bm["longest_missing_s"] == pytest.approx(100.0)
    assert bm["crossed"] == 1 and bm["nonfinite_px_qty"] == 1 and bm["off_tick"] == 1
    assert q.bbo_metrics(make_bbo(on500=True), D0, TICK)["on_500ms"] == 1
    tm = q.trades_metrics(make_trades(bad=True), D0, 99_000, 101_000)
    assert tm["buy_vwap_missing"] == 1 and tm["sell_last_ms_ge_100"] == 1 and tm["consistency_violations"] == 2


def test_quote_grid_matches_common_py_rule():
    g = q.quote_grid(make_book(), make_bbo(), D0)
    assert len(g["ts"]) == 864_000
    # 0 from book, 100/200/300 from bbo, 400 forward-filled from 300, 500 from book again
    assert g["bid_px"][:6].tolist() == pytest.approx([100_000.0, 100_000.0, 99_999.9, 99_999.8, 99_999.8, 100_000.0])
    qm = q.quote_metrics(g)
    assert qm["grid_nan_pts"] == 0 and qm["px_changes"] == 3


def test_targets_and_stable():
    ts = np.arange(D0, D0 + 86_400_000, 500, dtype=np.int64)
    y = np.zeros(len(ts)); y[:10] = 0.01; y[10] = np.nan
    tg = pl.DataFrame({"ts": ts, "y_500ms": y, "y_2s": y, "y_8s": y})
    tm = q.targets_metrics(tg, D0, {"y_500ms": (-0.001, 0.001)})
    assert tm["y_500ms_nonfinite"] == 1 and tm["y_500ms_clipped_pct"] == pytest.approx(10 / (len(ts) - 1) * 100)
    sts = np.arange(D0, D0 + 86_400_000, 1000, dtype=np.int64)
    st = pl.DataFrame({"ts": np.repeat(sts, 2), "pair": ["USDCUSDT", "FDUSDUSDT"] * len(sts),
                       "bid_px": 0.9999, "ask_px": 1.0001, "stale_ms": 0}).with_columns(
        pl.when((pl.col("pair") == "FDUSDUSDT") & (pl.col("ts") < D0 + 5000)).then(0.99).otherwise(pl.col("bid_px")).alias("bid_px"),
        pl.when((pl.col("pair") == "FDUSDUSDT") & (pl.col("ts") < D0 + 5000)).then(0.991).otherwise(pl.col("ask_px")).alias("ask_px"))
    r = {x["pair"]: x for x in q.stable_metrics(st, D0)}
    assert r["FDUSDUSDT"]["depeg_s"] == 5 and r["USDCUSDT"]["depeg_s"] == 0 and r["USDCUSDT"]["missing_s"] == 0


def test_runner_and_report_end_to_end(tmp_path: Path):
    import stage3_quality, stage3_report
    data = tmp_path / "data"; (data / "meta").mkdir(parents=True); (data / "stable").mkdir()
    stage = tmp_path / "stage"
    for inst, kw in (("BTCUSDT_PERP", dict(n_drop=200, crossed_at=5)), ("ETHUSDT_PERP", {})):
        for src, df in (("book", make_book(**kw)), ("bbo", make_bbo()), ("trades", make_trades()),
                        ("targets", pl.DataFrame({"ts": np.arange(D0, D0 + 86_400_000, 500, dtype=np.int64),
                                                  "y_500ms": 0.0, "y_2s": 0.0, "y_8s": 0.0}))):
            p = stage / "data" / src / inst / f"{DAY}.parquet"; p.parent.mkdir(parents=True, exist_ok=True); df.write_parquet(p)
        with tarfile.open(tmp_path / f"data-{inst}-001.tar", "w") as tf:
            tf.add(stage / "data", arcname="data", filter=lambda m: m if inst in m.name or m.isdir() else None)
    pl.DataFrame({"instrument": ["BTCUSDT_PERP", "ETHUSDT_PERP"], "date": [DAY] * 2, "tick_size": [TICK, TICK],
                  "bucket_edges_bps": ["0.5"] * 2}).write_parquet(data / "meta" / "daily.parquet")
    pl.DataFrame({"instrument": ["BTCUSDT_PERP"], "horizon": ["y_500ms"], "lo": [-1e-3], "hi": [1e-3], "n": [1]}
                 ).write_parquet(data / "meta" / "target_clip.parquet")
    sts = np.arange(D0, D0 + 86_400_000, 1000, dtype=np.int64)
    pl.DataFrame({"ts": sts, "pair": "USDCUSDT", "bid_px": 0.9999, "ask_px": 1.0001, "bid_qty": 1.0, "ask_qty": 1.0,
                  "buy_qty": 0.0, "sell_qty": 0.0, "stale_ms": 0}).write_parquet(data / "stable" / f"{DAY}.parquet")
    out = tmp_path / "rep"
    args = ["--archive-dir", str(tmp_path), "--data-dir", str(data), "--out", str(out), "--workers", "1"]
    assert stage3_quality.main(args) == 0
    assert stage3_quality.done_units(out) == {("BTCUSDT_PERP", DAY), ("ETHUSDT_PERP", DAY), ("__stable__", DAY)}
    assert stage3_quality.main(args) == 0                 # resume: nothing left, nothing duplicated
    assert stage3_report.main(["--out", str(out)]) == 0
    flags = pl.read_csv(out / "flags.csv")
    got = set(zip(flags["flag"], flags["instrument"]))
    assert ("book_gap", "BTCUSDT_PERP") in got and ("book_crossed", "BTCUSDT_PERP") in got
    # synthetic prices never change, so prices_frozen is CORRECTLY raised for both; nothing else on ETH
    assert {f for f, i in got if i == "ETHUSDT_PERP"} == {"prices_frozen"}
    for f in ("summary_instruments.csv", "summary_targets.csv", "summary_stable.csv", "report.md", "fig_book_missing.png"):
        assert (out / f).exists(), f


def test_tick_float_noise_is_not_off_tick():
    px = np.array([69036.16, 69036.17, 100000.01])
    assert q._off_tick(px, 0.00999999999476131) == 0          # real meta value; was a false positive
    assert q._off_tick(np.array([130.921]), 0.01) == 1
    assert q.empirical_tick(np.array([100.00, 100.01, 100.02, 100.05])) == pytest.approx(0.01)
