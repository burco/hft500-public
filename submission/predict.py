"""Submission entry point — Binance perpetuals, next 500 ms / 2 s / 8 s log-return prediction.

    python predict.py --data-dir <DIR> --date <YYYY-MM-DD> --out <PATH>.parquet

<DIR> has the official layout without targets/: the requested day plus the previous day.
Output: 691,200 rows (4 perps x 172,800 grid points), columns
ts (int64 ms UTC), instrument, pred_500ms, pred_2s, pred_8s (float64, finite).

Model: per (perp, horizon) LightGBM, 127 leaves x 100 trees, on 46 causal features built from the
perp's own book / quotes / trades, the other perps, the matching spot pair and the USDC-quoted twin
converted with the stablecoin table. Every feature at t uses only rows with ts <= t (verified by the
official truncation test, see README). No state is kept between calls.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hft500 import align as A, models as MD, pipeline as P  # noqa: E402

MODEL_DIR = HERE / "models"
GRID_POINTS = 172_800


def meta_ticks(data_dir: Path, date: str) -> dict:
    """Tick sizes for the day from meta/daily.parquet if it is supplied; otherwise the feature code
    estimates the tick from the previous day's prices (it does that first anyway)."""
    p = data_dir / "meta" / "daily.parquet"
    if not p.is_file():
        return {}
    m = pl.read_parquet(p, columns=["date", "instrument", "tick_size"]).filter(pl.col("date").cast(pl.Utf8) == date)
    return dict(zip(m["instrument"], m["tick_size"]))


def load_models(model_dir: Path) -> dict:
    return {inst: MD.LGBModel.load(model_dir / inst) for inst in P.INSTRUMENTS}


def check(out: pl.DataFrame, date: str) -> None:
    assert out.columns == ["ts", "instrument", "pred_500ms", "pred_2s", "pred_8s"], out.columns
    assert out.height == 4 * GRID_POINTS, out.height
    g = A.grid(date)
    for inst in P.INSTRUMENTS:
        s = out.filter(pl.col("instrument") == inst)
        assert np.array_equal(s["ts"].to_numpy(), g), f"grid mismatch for {inst}"
        for c in P.PRED.values():
            v = s[c].to_numpy()
            assert np.isfinite(v).all(), f"non-finite {inst} {c}"
            if v.std() == 0:   # only when nothing at all is known for the day; scores 0 either way
                print(f"warning: constant predictions for {inst} {c}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default=str(MODEL_DIR), help="weights directory (default: ./models)")
    a = ap.parse_args()
    t0 = time.time()
    data_dir = Path(a.data_dir)
    root = A.DirRoot(data_dir)
    out = P.predict_day(root, a.date, load_models(Path(a.model_dir)), meta_ticks(data_dir, a.date))
    out = out.with_columns(pl.col("ts").cast(pl.Int64), pl.col("instrument").cast(pl.Utf8),
                           *[pl.col(c).cast(pl.Float64) for c in P.PRED.values()])
    check(out, a.date)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out, compression="zstd")
    print(f"{out.height:,} rows -> {a.out} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
