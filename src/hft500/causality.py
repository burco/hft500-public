"""Stage 4 — the organisers' causality check, reproduced.

    1. run f on the full data for a day;
    2. hide every row with ts > T in every table (TruncatedRoot) and run f again;
    3. every output row with ts <= T must be IDENTICAL (bit-for-bit; NaN == NaN).

f(root, date) must return a Polars DataFrame with an int64 'ts' column.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl

from .align import DAY_MS, Root, TruncatedRoot, day_start


def _equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        a, b = a.astype(np.float64), b.astype(np.float64)
        return (a == b) | (np.isnan(a) & np.isnan(b))
    return a == b


def compare(full: pl.DataFrame, trunc: pl.DataFrame, T: int) -> dict:
    """Mismatches among rows with ts <= T."""
    if full is None or trunc is None:
        return dict(T=T, rows=0, ok=False, problem="function returned None (output must always cover the full grid)")
    f = full.filter(pl.col("ts") <= T).sort("ts")
    g = trunc.filter(pl.col("ts") <= T).sort("ts")
    if f.height != g.height or not np.array_equal(f["ts"].to_numpy(), g["ts"].to_numpy()):
        return dict(T=T, rows=f.height, ok=False, problem=f"row sets differ ({f.height} vs {g.height})")
    bad_cols, first_bad, n_bad = [], None, 0
    for c in f.columns:
        if c == "ts" or c not in g.columns or not f[c].dtype.is_numeric():
            continue
        eq = _equal(f[c].to_numpy(), g[c].to_numpy())
        if not eq.all():
            n_bad += int((~eq).sum())
            bad_cols.append(c)
            k = int(np.argmin(eq))
            first_bad = min(first_bad, int(f["ts"][k])) if first_bad is not None else int(f["ts"][k])
    return dict(T=T, rows=f.height, ok=not bad_cols, mismatches=n_bad, bad_columns=bad_cols, first_bad_ts=first_bad)


def gap_cuts(root: Root, inst: str, date: str, min_gap_ms: int = 1000, k: int = 3) -> list[int]:
    """Cut times INSIDE the k longest holes of the book (incl. the start of the day).
    Random cuts rarely land there, but leaks that 'fill' holes (bfill, interpolation)
    only show up there."""
    b = root.read("book", date, inst, ["ts"])
    d0 = day_start(date)
    ts = np.sort(np.concatenate([[d0 - 500], b["ts"].to_numpy() if b is not None else [], [d0 + DAY_MS]]))
    gaps = np.diff(ts)
    idx = np.argsort(gaps)[::-1][:k]
    return [int(ts[i] + gaps[i] // 2) for i in idx if gaps[i] >= min_gap_ms]


def truncation_check(fn: Callable[[Root, str], pl.DataFrame], root: Root, date: str,
                     n: int = 5, seed: int = 0, Ts: list[int] | None = None,
                     inst_for_gaps: str | None = None) -> list[dict]:
    """Run the check at n random cut times inside the day, plus explicit Ts, plus (if
    inst_for_gaps is given) cut times inside that instrument's largest book gaps."""
    rng = np.random.default_rng(seed)
    d0 = day_start(date)
    cuts = list(Ts or []) + [int(d0 + rng.integers(0, DAY_MS)) for _ in range(n)]
    if inst_for_gaps:
        cuts += gap_cuts(root, inst_for_gaps, date)
    full = fn(root, date)
    return [compare(full, fn(TruncatedRoot(root, T), date), T) for T in cuts]
