"""Stage 4 — causal reconstruction and alignment.

Everything here answers one question for each prediction time t (a multiple of
500 ms): "what was KNOWN at t?". The rules, from the PDF and common.py:

* quotes  : bbo records only changes on a 100 ms grid; book holds the 500 ms points.
            The state at t is the LAST record with ts <= t (backward as-of join).
* book    : one row per 500 ms grid point, but some are missing -> use the last row <= t.
* trades  : a bar labelled ts covers (ts-100, ts] and is known from ts on.
            Window ending at t = bars with ts in (t-500, t].
* stable  : 1 s grid, forward-filled by the provider; as-of backward join.
* warm-up : the previous day is supplied. Its last records are legitimately known at
            00:00:00.000, so state is carried across midnight (and the previous day's
            final trade bar, stamped exactly at midnight, lands on t = day start).

Never: shift(-k), centred windows, backward fill, whole-day statistics, or anything
keyed on the last timestamp in the data. The truncation test in causality.py checks it.

All timestamps are int64 epoch-milliseconds, UTC (the collector's receive time).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from .tarsource import index_tar, open_any

DAY_MS = 86_400_000
GRID_MS, QUOTE_MS, TRADE_MS, STABLE_MS = 500, 100, 100, 1000
QUOTE_COLS = ["bid_px", "ask_px", "bid_qty", "ask_qty"]
BUCKETS = [f"{s}_v{i:02d}" for s in ("bid", "ask") for i in range(1, 12)]


# --------------------------------------------------------------------------- dates / grids
def day_start(date: str) -> int:
    return int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1000


def prev_date(date: str) -> str:
    return (dt.date.fromisoformat(date) - dt.timedelta(days=1)).isoformat()


def grid(date: str, step_ms: int = GRID_MS) -> np.ndarray:
    """Grid built from the DATE, never from data timestamps (missing rows must not drop points)."""
    t0 = day_start(date)
    return np.arange(t0, t0 + DAY_MS, step_ms, dtype=np.int64)


# --------------------------------------------------------------------------- data roots
class Root(Protocol):
    def read(self, table: str, date: str, instrument: str | None = None,
             columns: list[str] | None = None) -> pl.DataFrame | None: ...


class DirRoot:
    """The official layout: <root>/<table>/<instrument>/<date>.parquet, stable/<date>.parquet."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, table: str, date: str, instrument: str | None) -> Path:
        return self.root / table / f"{date}.parquet" if instrument is None else self.root / table / instrument / f"{date}.parquet"

    def read(self, table, date, instrument=None, columns=None):
        p = self.path(table, date, instrument)
        return pl.read_parquet(p, columns=columns) if p.is_file() else None


class ArchiveRoot:
    """Same interface, reading members of the uncompressed data-<INST>.tar archives in place.
    Tables not found in any archive (stable, meta) fall back to `fallback` (a DirRoot)."""

    def __init__(self, archives: list[str | Path], fallback: DirRoot | None = None):
        self.fallback = fallback
        self.vpaths: dict[tuple[str, str | None, str], str] = {}
        for ar in archives:
            for r in index_tar(ar):
                name = Path(r["member"]).stem
                inst = None if r["instrument"] == "(none)" else r["instrument"]
                self.vpaths[(r["source"], inst, name)] = r["path"]

    def read(self, table, date, instrument=None, columns=None):
        vp = self.vpaths.get((table, instrument, date))
        if vp is None:
            return self.fallback.read(table, date, instrument, columns) if self.fallback else None
        src = open_any(vp)
        try:
            return pl.from_arrow(pq.read_table(src, columns=columns))
        finally:
            src.close()


class TruncatedRoot:
    """Wraps a root and hides every row with ts > T (all tables, all days) — exactly what the
    organisers' causality check does. Files that become empty are returned empty, not None."""

    def __init__(self, base: Root, T: int):
        self.base, self.T = base, int(T)

    def read(self, table, date, instrument=None, columns=None):
        cols = None if columns is None else list(dict.fromkeys(["ts", *columns]))
        df = self.base.read(table, date, instrument, cols)
        if df is None:
            return None
        df = df.filter(pl.col("ts") <= self.T)
        return df if columns is None else df.select(columns)


def _read_two_days(root: Root, table: str, date: str, inst: str | None, columns: list[str] | None,
                   warmup: bool) -> pl.DataFrame | None:
    parts = []
    if warmup:
        p = root.read(table, prev_date(date), inst, columns)
        if p is not None and p.height:
            parts.append(p)
    c = root.read(table, date, inst, columns)
    if c is not None and c.height:
        parts.append(c)
    if not parts:
        return None
    return pl.concat(parts, how="vertical_relaxed")


def _asof_index(event_ts: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Index of the last event with ts <= t (event_ts sorted), -1 if none."""
    return np.searchsorted(event_ts, t, side="right") - 1


def _take(values: np.ndarray, idx: np.ndarray, fill=np.nan) -> np.ndarray:
    out = values[np.maximum(idx, 0)].astype(np.float64, copy=True)
    out[idx < 0] = fill
    return out


# --------------------------------------------------------------------------- quotes (100 ms)
def quote_events(root: Root, inst: str, date: str, warmup: bool = True) -> pl.DataFrame | None:
    """Merged, time-sorted best-quote records from book (500 ms) and bbo (changes, 100 ms).
    Same rule as common.quote_grid: concatenate, stable sort by ts, forward-fill only."""
    book = _read_two_days(root, "book", date, inst, ["ts", *QUOTE_COLS], warmup)
    if book is None:
        return None
    bbo = _read_two_days(root, "bbo", date, inst, ["ts", *QUOTE_COLS], warmup)
    parts = [book] + ([bbo] if bbo is not None else [])
    ev = (pl.concat([p.with_columns(pl.col(QUOTE_COLS).cast(pl.Float64)) for p in parts], how="vertical_relaxed")
          .sort("ts", maintain_order=True))
    return ev


def quote_state(ev: pl.DataFrame, t: np.ndarray) -> dict[str, np.ndarray]:
    """Best bid/ask/sizes as of each t, plus two staleness clocks (ms):
    quote_age_ms = t - time of the last quote record; px_age_ms = t - time the bid or ask PRICE last changed."""
    ts = ev["ts"].to_numpy()
    i = _asof_index(ts, t)
    out = {c: _take(ev[c].to_numpy(), i) for c in QUOTE_COLS}
    out["quote_age_ms"] = np.where(i >= 0, t - ts[np.maximum(i, 0)], np.nan)
    bid, ask = ev["bid_px"].to_numpy(), ev["ask_px"].to_numpy()
    changed = np.ones(len(ts), dtype=bool)
    changed[1:] = (bid[1:] != bid[:-1]) | (ask[1:] != ask[:-1])
    last_change_ts = np.maximum.accumulate(np.where(changed, ts, np.iinfo(np.int64).min))
    out["px_age_ms"] = np.where(i >= 0, t - last_change_ts[np.maximum(i, 0)], np.nan)
    return out


def quote_grid(root: Root, inst: str, date: str, warmup: bool = True, step_ms: int = QUOTE_MS) -> pl.DataFrame | None:
    """Full quote grid for the day (default 100 ms, the official one)."""
    ev = quote_events(root, inst, date, warmup)
    if ev is None:
        return None
    t = grid(date, step_ms)
    return pl.DataFrame({"ts": t, **quote_state(ev, t)})


# --------------------------------------------------------------------------- book buckets (500 ms)
def book_on_grid(root: Root, inst: str, date: str, warmup: bool = True) -> pl.DataFrame | None:
    """Depth buckets as of each 500 ms point (last book row <= t) and their age."""
    bk = _read_two_days(root, "book", date, inst, ["ts", *BUCKETS], warmup)
    if bk is None:
        return None
    bk = bk.sort("ts", maintain_order=True)
    t = grid(date)
    ts = bk["ts"].to_numpy()
    i = _asof_index(ts, t)
    cols = {c: _take(bk[c].to_numpy(), i) for c in BUCKETS}
    return pl.DataFrame({"ts": t, **cols, "book_age_ms": np.where(i >= 0, t - ts[np.maximum(i, 0)], np.nan)})


# --------------------------------------------------------------------------- trades (500 ms windows)
def trades_on_grid(root: Root, inst: str, date: str, warmup: bool = True) -> pl.DataFrame:
    """Aggregate 100 ms bars into the window (t-500, t] for every 500 ms point t.
    Missing bars = no trades (qty 0, n 0). Also: ms since the last buy / sell trade."""
    t = grid(date)
    out = {"ts": t}
    tr = _read_two_days(root, "trades", date, inst, None, warmup)
    sides = ("buy", "sell")
    if tr is None:
        for s in sides:
            out.update({f"tr_{s}_qty": np.zeros(len(t)), f"tr_{s}_n": np.zeros(len(t)),
                        f"tr_{s}_notional": np.zeros(len(t)), f"tr_{s}_max_qty": np.zeros(len(t)),
                        f"tr_{s}_age_ms": np.full(len(t), np.nan)})
        return pl.DataFrame(out)
    tr = tr.sort("ts", maintain_order=True)
    bts = tr["ts"].to_numpy()
    # bar (ts-100, ts] belongs to the window ending at the first grid point >= ts
    gpt = -(-bts // GRID_MS) * GRID_MS
    pos = (gpt - t[0]) // GRID_MS
    keep = (pos >= 0) & (pos < len(t))
    for s in sides:
        q = tr[f"{s}_qty"].cast(pl.Float64).to_numpy()
        n = tr[f"{s}_n"].cast(pl.Float64).to_numpy()
        vw = tr[f"{s}_vwap"].cast(pl.Float64).fill_nan(None).fill_null(0.0).to_numpy()
        mx = tr[f"{s}_max_qty"].cast(pl.Float64).fill_nan(None).fill_null(0.0).to_numpy()
        for name, v, how in (("qty", q, "sum"), ("n", n, "sum"), ("notional", q * vw, "sum"), ("max_qty", mx, "max")):
            acc = np.zeros(len(t))
            (np.add if how == "sum" else np.maximum).at(acc, pos[keep], v[keep])
            out[f"tr_{s}_{name}"] = acc
        # time of the last trade on this side = bar end - last_ms (always <= bar end <= t)
        has = q > 0
        last_trade_ts = np.sort((bts - tr[f"{s}_last_ms"].cast(pl.Int64).to_numpy())[has])
        if len(last_trade_ts) == 0:                     # this side has not traded yet
            out[f"tr_{s}_age_ms"] = np.full(len(t), np.nan)
            continue
        j = _asof_index(last_trade_ts, t)
        out[f"tr_{s}_age_ms"] = np.where(j >= 0, t - last_trade_ts[np.maximum(j, 0)], np.nan)
    return pl.DataFrame(out)


# --------------------------------------------------------------------------- stablecoins (1 s)
def stable_on_grid(root: Root, date: str, warmup: bool = True) -> pl.DataFrame | None:
    """Mid and quote age of each stablecoin pair as of every 500 ms point."""
    st = _read_two_days(root, "stable", date, None, ["ts", "pair", "bid_px", "ask_px", "stale_ms"], warmup)
    if st is None:
        return None
    t = grid(date)
    out = {"ts": t}
    for pair in sorted(st["pair"].unique().to_list()):
        g = st.filter(pl.col("pair") == pair).sort("ts", maintain_order=True)
        ts = g["ts"].to_numpy()
        i = _asof_index(ts, t)
        mid = (g["bid_px"].cast(pl.Float64) + g["ask_px"].cast(pl.Float64)).to_numpy() / 2
        out[f"st_{pair}_mid"] = _take(mid, i)
        out[f"st_{pair}_age_ms"] = np.where(i >= 0, _take(g["stale_ms"].to_numpy(), i) + (t - ts[np.maximum(i, 0)]), np.nan)
    return pl.DataFrame(out)


# --------------------------------------------------------------------------- one instrument-day
def load_day(root: Root, inst: str, date: str, warmup: bool = True, with_targets: bool = False) -> pl.DataFrame:
    """One row per 500 ms prediction point: quotes (q_*), book buckets, trades (tr_*),
    staleness clocks, and optionally the supplied targets (training only — never a feature).

    ALWAYS returns all 172,800 rows. A source with no data known yet gives NaN columns —
    the shape of the output must never depend on how much data has arrived (truncation check)."""
    t = grid(date)
    ev = quote_events(root, inst, date, warmup)
    if ev is not None:
        qs = quote_state(ev, t)
    else:
        qs = {c: np.full(len(t), np.nan) for c in [*QUOTE_COLS, "quote_age_ms", "px_age_ms"]}
    df = pl.DataFrame({"ts": t, **{f"q_{k}": v for k, v in qs.items()}})
    bk = book_on_grid(root, inst, date, warmup)
    if bk is None:
        bk = pl.DataFrame({"ts": t, **{c: np.full(len(t), np.nan) for c in [*BUCKETS, "book_age_ms"]}})
    df = df.join(bk, on="ts", how="left")
    df = df.join(trades_on_grid(root, inst, date, warmup), on="ts", how="left")
    if with_targets:
        tg = root.read("targets", date, inst)
        if tg is not None:
            df = df.join(tg, on="ts", how="left")
    return df.with_columns(pl.lit(inst).alias("instrument"))
