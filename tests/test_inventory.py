"""Tests on a tiny SYNTHETIC fixture. These check the code, not the market data."""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from hft500 import inventory as inv  # noqa: E402
import stage1_inventory  # noqa: E402

DAY0 = int(datetime(2025, 9, 15, tzinfo=timezone.utc).timestamp() * 1000)


def _write(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    for k, day in enumerate(["2025-09-15", "2025-09-16", "2025-09-18"]):  # 09-17 missing
        t0 = DAY0 + k * 86_400_000
        ts = np.arange(t0, t0 + 3_600_000, 500, dtype=np.int64)            # 1h of 500 ms grid
        df = pl.DataFrame({"ts": ts, "bid_px": 1.0, "ask_px": 1.1})
        if day == "2025-09-18":
            df = df.with_columns(pl.lit(3.0).alias("new_col"))              # schema drift
        _write(d / "book" / "BTCUSDT_PERP" / f"{day}.parquet", df)
    bbo_ts = np.array([DAY0, DAY0 + 100, DAY0 + 100, DAY0 + 250], dtype=np.int64)  # dup + off-grid
    _write(d / "bbo" / "BTCUSDT_PERP" / "2025-09-15.parquet",
           pl.DataFrame({"ts": bbo_ts, "bid_px": [1.0] * 4}))
    stable_ts = np.repeat(np.arange(DAY0, DAY0 + 5000, 1000, dtype=np.int64), 3)
    _write(d / "stable" / "2025-09-15.parquet",
           pl.DataFrame({"ts": stable_ts, "pair": ["USDCUSDT", "FDUSDUSDT", "USD1USDT"] * 5}))
    trades = pl.DataFrame({"timestamp": pl.Series(np.arange(DAY0 + 100, DAY0 + 1100, 100) * 1_000_000)
                           .cast(pl.Datetime("ns", "UTC")), "qty": 1.0})
    _write(d / "trades" / "BTCUSDT_PERP" / "2025-09-15.parquet", trades)
    (d / "trades" / "BTCUSDT_PERP" / "2025-09-16.parquet").write_bytes(b"not parquet")
    return d


def test_parse_date():
    assert inv.parse_date(Path("a/2024-01-01/BTC/2025-09-15.parquet")) == date(2025, 9, 15)
    assert inv.parse_date(Path("x/20250916.parquet")) == date(2025, 9, 16)
    assert inv.parse_date(Path("x/nodate.parquet")) is None


@pytest.mark.parametrize("scale,unit", [(1, "ms"), (1000, "us"), (1_000_000, "ns")])
def test_unit_inference(scale, unit):
    import pyarrow as pa
    ms, desc = inv.to_epoch_ms(pa.chunked_array([np.array([DAY0, DAY0 + 500]) * scale]))
    assert unit in desc and list(ms) == [DAY0, DAY0 + 500]


def test_end_to_end(data_dir: Path, tmp_path: Path):
    files = inv.add_footers(inv.discover(data_dir))
    assert files.height == 7
    assert files.filter(pl.col("source") == "stable")["instrument"][0] == "(none)"
    summ = inv.summarise(files)
    book = summ.filter(pl.col("source") == "book").row(0, named=True)
    assert book["files"] == 3 and book["schema_variants"] == 2
    assert book["missing_train_days"] == (date(2026, 7, 14) - date(2025, 9, 15)).days + 1 - 3
    assert summ.filter(pl.col("source") == "trades")["unreadable"][0] == 1

    bbo_file = files.filter(pl.col("source") == "bbo")["path"][0]
    c = inv.check_timestamps("bbo", "BTCUSDT_PERP", date(2025, 9, 15), bbo_file)
    assert c.dup_ts == 1 and c.pct_on_grid == 75.0 and c.monotonic
    assert c.pct_on_500ms == 25.0          # DAY0 is a 500 ms point -> must be flagged for bbo

    tr_file = files.filter((pl.col("source") == "trades") & pl.col("readable"))["path"][0]
    c = inv.check_timestamps("trades", "BTCUSDT_PERP", date(2025, 9, 15), tr_file)
    assert c.ts_col == "timestamp" and c.pct_on_grid == 100.0 and c.pct_inside_day == 100.0

    out = tmp_path / "rep"
    rc = stage1_inventory.main(["--data-dir", str(data_dir), "--out", str(out)])
    assert rc == 1  # FAIL: unreadable file + missing ETH/SOL/PEPE
    for f in ("inventory_summary.csv", "schema_report.json", "readiness.md", "ts_checks.csv"):
        assert (out / f).exists()
    text = (out / "readiness.md").read_text()
    assert "| predict.py | MISSING |" in text and "| dir targets/ | MISSING |" in text
    assert "| PDF-required columns present | FAIL |" in text and "bid_v01" in text
    assert "| dir stable/ | PASS |" in text


def test_archive_matches_extracted(data_dir: Path, tmp_path: Path):
    """Reading Parquet straight out of a .tar must give the same inventory as the extracted files."""
    import tarfile
    good = [p for p in data_dir.rglob("*.parquet") if p.stat().st_size > 20]
    tar_path = tmp_path / "data-BTCUSDT_PERP-014.tar"
    with tarfile.open(tar_path, "w") as tf:                    # uncompressed, like the real archives
        for p in good:
            tf.add(p, arcname="data/" + str(p.relative_to(data_dir)).replace("\\", "/"))
    (tmp_path / "data-BTCUSDT_PERP-015 (1).tar").write_bytes(tar_path.read_bytes())  # a duplicate copy

    chosen, groups = inv.pick_archives(sorted(tmp_path.glob("*.tar")))
    assert chosen["BTCUSDT_PERP"].name == "data-BTCUSDT_PERP-014.tar" and len(groups["BTCUSDT_PERP"]) == 2

    a = inv.add_footers(inv.discover_archives([tar_path])).sort("source", "instrument", "date")
    e = inv.add_footers(inv.discover(data_dir)).filter(pl.col("readable")).sort("source", "instrument", "date")
    cols = ["source", "instrument", "date", "num_rows", "schema_id", "null_counts"]
    assert a.select(cols).equals(e.select(cols))

    vp = a.filter(pl.col("source") == "bbo")["path"][0]
    c = inv.check_timestamps("bbo", "BTCUSDT_PERP", date(2025, 9, 15), vp)
    assert c.dup_ts == 1 and c.pct_on_500ms == 25.0

    out = tmp_path / "rep"
    rc = stage1_inventory.main(["--archive-dir", str(tmp_path), "--out", str(out)])
    assert rc == 1 and (out / "inventory_summary.csv").exists()
