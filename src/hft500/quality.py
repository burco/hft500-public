"""Stage 3 — per-day data-quality metrics.

One function per source. Each takes ONE day of ONE instrument (eager Polars
DataFrame) plus context, and returns a flat dict of numbers. Nothing here
changes data; it only measures it. Thresholds for "bad" live in the report.

Timestamps: int64 epoch-ms UTC (collector receive time). day0 = 00:00:00.000.
"""
from __future__ import annotations

import numpy as np
import polars as pl

DAY_MS = 86_400_000
BPS = 1e4


def _f(df: pl.DataFrame, c: str) -> np.ndarray:
    return df[c].cast(pl.Float64).to_numpy()


def _nonfinite(*arrs: np.ndarray) -> int:
    return int(sum((~np.isfinite(a)).sum() for a in arrs))


def clean_tick(tick: float) -> float:
    """meta/daily.parquet stores ticks with float noise (0.01 -> 0.00999999999476131).
    Multiplied by ~7 million ticks that noise exceeds any sane tolerance, so round to
    8 significant digits first (the noise sits around the 10th)."""
    return float(f"{tick:.8g}")


def empirical_tick(px: np.ndarray) -> float:
    """Most common smallest step between distinct observed prices (robust to a few odd prints)."""
    u = np.unique(px[np.isfinite(px)])
    if len(u) < 2:
        return float("nan")
    d = np.diff(u)
    d = d[d > np.abs(u).max() * 1e-10]           # drop float noise between "equal" prices
    if not len(d):
        return float("nan")
    d = np.array([float(f"{x:.6g}") for x in d])  # 0.09999999999 and 0.1000000001 are one step
    vals, cnt = np.unique(d, return_counts=True)
    return float(vals[np.argmax(cnt)])


def _off_tick(px: np.ndarray, tick: float | None) -> int:
    """Prices that are not a multiple of the tick (tolerance 0.1% of a tick)."""
    if not tick or not np.isfinite(tick) or tick <= 0:
        return -1
    tick = clean_tick(tick)
    ok = np.isfinite(px)
    k = np.round(px[ok] / tick)
    return int((np.abs(px[ok] - k * tick) > tick * 1e-3).sum())


def _gaps(ts: np.ndarray, start: int, end: int, step: int) -> tuple[int, float]:
    """Missing grid points in [start, end) and the longest missing stretch (seconds)."""
    ts = np.unique(ts[(ts >= start) & (ts < end) & (ts % step == 0)])
    missing = (end - start) // step - len(ts)
    edges = np.concatenate([[start - step], ts, [end]])
    longest = float((np.diff(edges) - step).max() / 1000) if len(edges) > 1 else (end - start) / 1000
    return int(missing), max(longest, 0.0)


# --------------------------------------------------------------------------- book
def book_metrics(book: pl.DataFrame, day0: int, tick: float | None) -> dict:
    ts = book["ts"].to_numpy()
    bid, ask = _f(book, "bid_px"), _f(book, "ask_px")
    bq, aq = _f(book, "bid_qty"), _f(book, "ask_qty")
    missing, longest = _gaps(ts, day0, day0 + DAY_MS, 500)
    mid = (bid + ask) / 2
    good = np.isfinite(mid) & (mid > 0) & (ask > bid)
    spr_bps = np.where(good, (ask - bid) / mid * BPS, np.nan)
    spr_ticks = np.where(good, np.round((ask - bid) / tick), np.nan) if tick else np.full(len(ts), np.nan)
    lr = np.diff(np.log(np.where(good, mid, np.nan))) * BPS          # consecutive 500 ms mid moves
    lr = lr[np.isfinite(lr)]
    bcols = [f"bid_v{i:02d}" for i in range(1, 12)]
    acols = [f"ask_v{i:02d}" for i in range(1, 12)]
    B = book.select(bcols).cast(pl.Float64).to_numpy()
    A = book.select(acols).cast(pl.Float64).to_numpy()
    return dict(
        rows=len(ts), missing_pts=missing, longest_missing_s=longest,
        off_grid=int((ts % 500 != 0).sum()), dup_ts=int(len(ts) - len(np.unique(ts))),
        nonfinite_px_qty=_nonfinite(bid, ask, bq, aq), nonpos_px=int(((bid <= 0) | (ask <= 0)).sum()),
        crossed=int((bid > ask).sum()), locked=int((bid == ask).sum()),
        zero_best_qty=int(((bq <= 0) | (aq <= 0)).sum()),
        off_tick=_off_tick(np.concatenate([bid, ask]), tick),
        pct_1tick=float(np.nanmean(spr_ticks == 1) * 100) if tick else np.nan,
        max_spread_ticks=float(np.nanmax(spr_ticks)) if np.isfinite(spr_ticks).any() else np.nan,
        spread_bps_med=float(np.nanmedian(spr_bps)), spread_bps_p99=float(np.nanpercentile(spr_bps, 99)),
        mid_first=float(mid[0]) if len(mid) else np.nan, mid_min=float(np.nanmin(mid)), mid_max=float(np.nanmax(mid)),
        max_abs_move_500ms_bps=float(np.abs(lr).max()) if len(lr) else np.nan,
        moves_gt_50bps=int((np.abs(lr) > 50).sum()),
        best_bid_qty_med=float(np.nanmedian(bq)), best_ask_qty_med=float(np.nanmedian(aq)),
        best_notional_med=float(np.nanmedian((bq * bid + aq * ask) / 2)),
        depth4_notional_med=float(np.nanmedian((B[:, :4].sum(1) + A[:, :4].sum(1)) / 2)),
        depth11_notional_med=float(np.nanmedian((B.sum(1) + A.sum(1)) / 2)),
        bucket_negative=int((B < 0).sum() + (A < 0).sum()), bucket_nonfinite=_nonfinite(B.ravel(), A.ravel()),
        bucket1_both_empty_pct=float(np.mean((B[:, 0] <= 0) & (A[:, 0] <= 0)) * 100),
    )


# ------------------------------------------------------------------ bbo + quotes
def bbo_metrics(bbo: pl.DataFrame, day0: int, tick: float | None) -> dict:
    ts = bbo["ts"].to_numpy()
    bid, ask = _f(bbo, "bid_px"), _f(bbo, "ask_px")
    bq, aq = _f(bbo, "bid_qty"), _f(bbo, "ask_qty")
    d = np.diff(ts)
    return dict(
        rows=len(ts), on_500ms=int((ts % 500 == 0).sum()), off_grid=int((ts % 100 != 0).sum()),
        dup_ts=int((d == 0).sum()), non_monotonic=int((d < 0).sum()),
        outside_day=int(((ts < day0) | (ts >= day0 + DAY_MS)).sum()),
        nonfinite_px_qty=_nonfinite(bid, ask, bq, aq), crossed=int((bid > ask).sum()),
        zero_best_qty=int(((bq <= 0) | (aq <= 0)).sum()), off_tick=_off_tick(np.concatenate([bid, ask]), tick),
    )


def quote_grid(book: pl.DataFrame, bbo: pl.DataFrame | None, day0: int) -> dict[str, np.ndarray]:
    """Full 100 ms grid for one day, same rule as common.quote_grid: forward-fill only,
    NaN before the day's first quote (no prior-day warm-up here)."""
    cols = ["bid_px", "ask_px", "bid_qty", "ask_qty"]
    parts = [book.select(["ts"] + cols)] + ([bbo.select(["ts"] + cols)] if bbo is not None and bbo.height else [])
    q = pl.concat([p.with_columns(pl.col(cols).cast(pl.Float64)) for p in parts]).sort("ts", maintain_order=True)
    ts = q["ts"].to_numpy()
    grid = np.arange(day0, day0 + DAY_MS, 100, dtype=np.int64)
    i = np.searchsorted(ts, grid, "right") - 1
    ok = i >= 0
    i = np.maximum(i, 0)
    out = {"ts": grid}
    for c in cols:
        v = q[c].to_numpy()
        out[c] = np.where(ok, v[i], np.nan)
    return out


def _longest_run(same: np.ndarray) -> int:
    """Longest run of consecutive True values."""
    if not same.any():
        return 0
    x = np.concatenate([[0], same.astype(np.int8), [0]])
    starts, ends = np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)
    return int((ends - starts).max())


def quote_metrics(g: dict[str, np.ndarray]) -> dict:
    """Activity / staleness on the reconstructed 100 ms grid."""
    px = np.stack([g["bid_px"], g["ask_px"]], 1)
    al = np.stack([g[c] for c in ("bid_px", "ask_px", "bid_qty", "ask_qty")], 1)
    same_px = np.all(px[1:] == px[:-1], 1)
    same_all = np.all(al[1:] == al[:-1], 1)
    mid = (g["bid_px"] + g["ask_px"]) / 2
    return dict(
        grid_nan_pts=int(np.isnan(mid).sum()),
        px_changes=int((~same_px).sum()), any_changes=int((~same_all).sum()),
        longest_static_px_s=_longest_run(same_px) / 10, longest_static_all_s=_longest_run(same_all) / 10,
    )


# ------------------------------------------------------------------------- trades
def trades_metrics(tr: pl.DataFrame, day0: int, mid_lo: float, mid_hi: float) -> dict:
    ts = tr["ts"].to_numpy()
    d = np.diff(ts)
    out = dict(bars=len(ts), off_grid=int((ts % 100 != 0).sum()), dup_ts=int((d == 0).sum()),
               non_monotonic=int((d < 0).sum()),
               outside_day=int(((ts <= day0) | (ts > day0 + DAY_MS)).sum()))   # bar (ts-100, ts]
    viol = 0
    for s in ("buy", "sell"):
        q, n = _f(tr, f"{s}_qty"), tr[f"{s}_n"].cast(pl.Int64).to_numpy()
        vw, mx = _f(tr, f"{s}_vwap"), _f(tr, f"{s}_max_qty")
        last = tr[f"{s}_last_ms"].cast(pl.Int64).to_numpy()
        has = q > 0
        out[f"{s}_qty"] = float(np.nansum(q))
        out[f"{s}_notional"] = float(np.nansum(np.where(has, q * vw, 0.0)))
        out[f"{s}_trades"] = int(n.sum())
        out[f"{s}_bars"] = int(has.sum())
        v = dict(
            neg_qty=int((q < 0).sum()),
            qty_n_mismatch=int(((q > 0) != (n > 0)).sum()),
            vwap_missing=int((has & ~np.isfinite(vw)).sum()),
            vwap_on_empty=int((~has & np.isfinite(vw)).sum()),
            maxqty_gt_qty=int((has & (mx > q * (1 + 1e-5))).sum()),
            last_ms_ge_100=int((has & (last >= 100)).sum()),
            vwap_outside_day_range=int((has & ((vw < mid_lo * 0.99) | (vw > mid_hi * 1.01))).sum()),
        )
        for k, x in v.items():
            out[f"{s}_{k}"] = x
            viol += x
    out["two_sided_bars_pct"] = float(np.mean((_f(tr, "buy_qty") > 0) & (_f(tr, "sell_qty") > 0)) * 100) if len(ts) else 0.0
    out["consistency_violations"] = viol
    return out


# ------------------------------------------------------------------------ targets
def targets_metrics(tg: pl.DataFrame, day0: int, clip: dict[str, tuple[float, float]] | None) -> dict:
    ts = tg["ts"].to_numpy()
    missing, _ = _gaps(ts, day0, day0 + DAY_MS, 500)
    out = dict(rows=len(ts), missing_pts=missing)
    for h in ("y_500ms", "y_2s", "y_8s"):
        y = _f(tg, h)
        fin = np.isfinite(y)
        out[f"{h}_nonfinite"] = int((~fin).sum())
        out[f"{h}_zero_pct"] = float(np.mean(y[fin] == 0) * 100) if fin.any() else np.nan
        out[f"{h}_std_bps"] = float(np.std(y[fin]) * BPS) if fin.any() else np.nan
        out[f"{h}_max_abs_bps"] = float(np.abs(y[fin]).max() * BPS) if fin.any() else np.nan
        if clip and h in clip:
            lo, hi = clip[h]
            out[f"{h}_clipped_pct"] = float(np.mean((y[fin] < lo) | (y[fin] > hi)) * 100)
    return out


# ------------------------------------------------------------------------- stable
def stable_metrics(st: pl.DataFrame, day0: int) -> list[dict]:
    """One dict per stablecoin pair for the day."""
    rows = []
    for pair, g in st.group_by("pair", maintain_order=True):
        pair = pair[0] if isinstance(pair, tuple) else pair
        ts = g["ts"].to_numpy()
        bid, ask = _f(g, "bid_px"), _f(g, "ask_px")
        mid = (bid + ask) / 2
        dev = np.abs(mid - 1) * BPS
        stale = g["stale_ms"].cast(pl.Float64).to_numpy()
        missing, longest = _gaps(ts, day0, day0 + DAY_MS, 1000)
        rows.append(dict(
            pair=pair, rows=len(ts), missing_s=missing, longest_missing_s=longest,
            dup_ts=int(len(ts) - len(np.unique(ts))), nonfinite_px=_nonfinite(bid, ask),
            crossed=int((bid > ask).sum()),
            dev_bps_max=float(np.nanmax(dev)) if np.isfinite(dev).any() else np.nan,
            dev_bps_med=float(np.nanmedian(dev)) if np.isfinite(dev).any() else np.nan,
            depeg_s=int((dev > 50).sum()),
            stale_ms_max=float(np.nanmax(stale)) if len(stale) else np.nan,
            stale_gt_10s=int((stale > 10_000).sum()),
        ))
    return rows
