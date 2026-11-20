"""Stage 5 — the prediction targets.

What the data showed (BTCUSDT_PERP, 2026-04-06; re-checked across the year in stage5_analyze.py):

* All three horizons come from ONE label-mid series M(t) := mid(t + 10 ms), sampled on the
  500 ms grid:   y_H(t) = ln M(t + H) - ln M(t).   Hence y_2s(t) = sum of 4 consecutive y_500ms,
  y_8s(t) = sum of 16 (verified to 1e-15).
* M(t) equals the visible 100 ms quote-grid mid at t on ~99.35% of points. On the rest the
  price had already moved in (t, t+10 ms] — information we cannot see. So supplied labels
  are ground truth; a rebuild from quotes is only a development proxy.

TRAINING-ONLY module: nothing here may be used to build a feature.
"""
from __future__ import annotations

import numpy as np
import polars as pl

HORIZONS = {"y_500ms": 500, "y_2s": 2000, "y_8s": 8000}
STEPS = {h: ms // 500 for h, ms in HORIZONS.items()}          # 1, 4, 16 grid steps
BPS = 1e4


# --------------------------------------------------------------------------- clipping
def load_clip(meta_dir) -> dict[tuple[str, str], tuple[float, float]]:
    t = pl.read_parquet(f"{meta_dir}/target_clip.parquet")
    return {(r["instrument"], r["horizon"]): (r["lo"], r["hi"]) for r in t.iter_rows(named=True)}


def clip_targets(tg: pl.DataFrame, inst: str, clips: dict) -> pl.DataFrame:
    """Clip each horizon to the official training-set 0.1% / 99.9% bounds (the metric does this)."""
    return tg.with_columns([pl.col(h).clip(*clips[(inst, h)]) for h in HORIZONS if (inst, h) in clips])


# --------------------------------------------------------------------------- rebuild (proxy)
def mid_asof(ev_ts: np.ndarray, ev_mid: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Mid of the last quote record with ts <= time (NaN before the first record)."""
    times = np.atleast_1d(times)
    i = np.searchsorted(ev_ts, times, side="right") - 1
    out = ev_mid[np.maximum(i, 0)].astype(np.float64)
    out[i < 0] = np.nan
    return out


def rebuild(ev_ts: np.ndarray, ev_mid: np.ndarray, t: np.ndarray, offset_ms: int = 10) -> dict[str, np.ndarray]:
    """Proxy labels from quote records. ev must extend >= 8.01 s past the last t (next day's data):
    labels look forward by design — which is exactly why they can never be features."""
    m0 = mid_asof(ev_ts, ev_mid, t + offset_ms)
    return {h: np.log(mid_asof(ev_ts, ev_mid, t + offset_ms + ms) / m0) for h, ms in HORIZONS.items()}


# --------------------------------------------------------------------------- structure
def implied_log_mid(y500: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """ln M(t) recovered by cumulating y_500ms. `anchor` is an observable log-mid per grid point
    (e.g. ln grid-mid(t)); it is used at the first point and again after every missing target."""
    out = np.full(len(y500), np.nan)
    prev = np.nan
    for j in range(len(y500)):
        if j == 0 or not np.isfinite(y500[j - 1]) or not np.isfinite(prev):
            out[j] = anchor[j]
        else:
            out[j] = prev + y500[j - 1]
        prev = out[j]
    return out


def additivity_error(tg: pl.DataFrame) -> dict[str, float]:
    """max |y_H(t) - sum of the k consecutive y_500ms| over rows where all are finite."""
    y = tg.sort("ts")["y_500ms"].to_numpy()
    out = {}
    for h in ("y_2s", "y_8s"):
        k = STEPS[h]
        n = len(y) - k + 1
        s = np.lib.stride_tricks.sliding_window_view(y, k).sum(axis=1)[:n]
        yh = tg.sort("ts")[h].to_numpy()[:n]
        ok = np.isfinite(s) & np.isfinite(yh)
        out[h] = float(np.max(np.abs(s[ok] - yh[ok]))) if ok.any() else float("nan")
    return out


def acf(x: np.ndarray, lags) -> np.ndarray:
    """Autocorrelation at the given lags (NaN-aware, pairwise)."""
    out = []
    for L in lags:
        a, b = x[:-L], x[L:]
        ok = np.isfinite(a) & np.isfinite(b)
        out.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 else np.nan)
    return np.array(out)


def discreteness(y: np.ndarray, mid: np.ndarray, tick: float, max_half_ticks: int = 4) -> dict[str, float]:
    """Share of 500 ms moves that are exactly 0, ±1/2 tick, ±1 tick ... of the mid (in mid units)."""
    ok = np.isfinite(y) & np.isfinite(mid)
    dm = np.round((np.exp(y[ok]) - 1) * mid[ok] / (tick / 2))      # move in half-ticks
    res = {f"pct_{k}_half_ticks": float(np.mean(np.abs(dm) == k) * 100) for k in range(max_half_ticks + 1)}
    res[f"pct_gt_{max_half_ticks}_half_ticks"] = float(np.mean(np.abs(dm) > max_half_ticks) * 100)
    return res


def tail_share(y: np.ndarray, lo: float, hi: float) -> dict[str, float]:
    """How much of the target's variance (sum of squares) sits in the clipped 0.2% tails."""
    ok = np.isfinite(y)
    y = y[ok]
    out_mask = (y < lo) | (y > hi)
    ss = np.sum(y ** 2)
    return dict(clipped_pct=float(out_mask.mean() * 100),
                ss_share_in_tails_pct=float(np.sum(y[out_mask] ** 2) / ss * 100) if ss > 0 else float("nan"),
                kurtosis=float(np.mean((y - y.mean()) ** 4) / np.var(y) ** 2) if np.var(y) > 0 else float("nan"),
                kurtosis_clipped=float(_kurt(np.clip(y, lo, hi))))


def _kurt(y):
    v = np.var(y)
    return np.mean((y - y.mean()) ** 4) / v ** 2 if v > 0 else np.nan
