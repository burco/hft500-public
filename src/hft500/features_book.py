"""Stage 7 — order-book features on the 500 ms prediction grid.

Every feature at t uses only data stamped <= t. To give rolling windows and EWMAs real
history at 00:00:00.000, everything is computed on an EXTENDED grid that starts at the
previous day's midnight (the previous day is always supplied, in training and in
predict.py alike), then the requested day is sliced out.

Causality traps handled here:
* the tick size comes from the PREVIOUS day's prices (a whole-day estimate on the current
  day would change when the day is truncated); meta's float-cleaned tick is the fallback;
* EWMAs and rolling sums only look backwards; nothing is normalised by a whole-day statistic.

Units: returns and volatility in basis points; times in seconds; depth in quote notional.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import align as A
from .quality import clean_tick, empirical_tick

STEP_MS = 500
RET_LAGS = {"0.5s": 1, "1s": 2, "2s": 4, "5s": 10, "10s": 20, "30s": 60, "60s": 120}
Z_LAGS = {"0.5s": 1, "2s": 4, "10s": 20}
DEPTH_K = {"0.5bps": 1, "1bps": 2, "2bps": 4, "10bps": 5}

# name -> (family, lookback seconds, description). The single source of truth for Stage 9.
REGISTRY: dict[str, tuple[str, float, str]] = {
    "imb_l1": ("imbalance", 0, "best-level queue imbalance (Qb-Qa)/(Qb+Qa) from the 100 ms quote grid"),
    "imb_l1_chg_1s": ("imbalance", 1, "imb_l1 now minus 1 s ago"),
    "imb_l1_ewm_5s": ("imbalance", 30, "EWMA of imb_l1, half-life 5 s"),
    "imb_l1_ewm_30s": ("imbalance", 180, "EWMA of imb_l1, half-life 30 s"),
    **{f"dimb_{k}": ("depth", 0, f"cumulative notional imbalance of book buckets within {k} of mid")
       for k in DEPTH_K},
    "depth2bps_rel": ("depth", 1800, "log(depth within 2 bps / its EWMA, half-life 5 min)"),
    "depth_near_share": ("depth", 0, "share of book notional within 2 bps out of all 11 buckets"),
    "spread_ticks": ("spread", 0, "spread in ticks (tick from the previous day's prices)"),
    "spread_bps": ("spread", 0, "spread / mid in bps"),
    "mp_dev_ticks": ("spread", 0, "microprice - mid in ticks = imb_l1 * spread_ticks / 2, clipped to +-2"),
    **{f"ret_{k}": ("momentum", s * 0.5, f"past log mid return over {k}, bps") for k, s in RET_LAGS.items()},
    **{f"retz_{k}": ("momentum", 300, f"ret_{k} divided by its expected size from 5 min realised vol")
       for k in Z_LAGS},
    "rv_30s": ("volatility", 30, "realised vol of 500 ms mid returns over 30 s, bps"),
    "rv_5min": ("volatility", 300, "realised vol of 500 ms mid returns over 5 min, bps"),
    "rv_ratio": ("volatility", 300, "rv_30s / rv_5min scaled to the same horizon (vol is rising if > 1)"),
    "moves_10s": ("activity", 10, "number of best-price changes in the last 10 s"),
    "moves_60s": ("activity", 60, "number of best-price changes in the last 60 s"),
    "px_age_log": ("staleness", 0, "log(1 + seconds since the best bid or ask price last changed)"),
    "frozen_60s": ("staleness", 0, "1 if the best prices have not changed for > 60 s (feed-freeze clock)"),
    "quote_age_s": ("staleness", 0, "seconds since the last quote record of any kind"),
    "book_age_s": ("staleness", 0, "seconds since the last 500 ms book snapshot"),
}
FEATURES = list(REGISTRY)


def registry_frame() -> pl.DataFrame:
    return pl.DataFrame([dict(feature=k, family=v[0], lookback_s=float(v[1]), description=v[2])
                         for k, v in REGISTRY.items()])


# --------------------------------------------------------------------------- helpers
def ewm(x: np.ndarray, halflife_steps: float) -> np.ndarray:
    """Backward-only EWMA; NaN inputs are skipped (the state carries over)."""
    a = 1.0 - 0.5 ** (1.0 / halflife_steps)
    out = np.empty(len(x))
    m = np.nan
    for i, v in enumerate(x):
        if v == v:  # not NaN
            m = v if m != m else m + a * (v - m)
        out[i] = m
    return out


def rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Sum over the last w points INCLUDING the current one (NaN counts as 0)."""
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=0.0))])
    i = np.arange(1, len(x) + 1)
    return c[i] - c[np.maximum(i - w, 0)]


def lag_diff(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[k:] = x[k:] - x[:-k]
    return out


def safe_ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def day_tick(ev: pl.DataFrame | None, d0: int, meta_tick: float | None) -> float:
    """Tick from the PREVIOUS day's quote prices only (causal); meta tick as fallback."""
    if ev is not None:
        prev = ev.filter(pl.col("ts") < d0)
        if prev.height > 100:
            tk = empirical_tick(np.concatenate([prev["bid_px"].to_numpy(), prev["ask_px"].to_numpy()]))
            if np.isfinite(tk) and tk > 0:
                return float(f"{tk:.8g}")
    return clean_tick(meta_tick) if meta_tick else float("nan")


# --------------------------------------------------------------------------- main
def build(root: A.Root, inst: str, date: str, meta_tick: float | None = None) -> pl.DataFrame:
    """All registered features for one instrument-day: 172,800 rows, always."""
    d0 = A.day_start(date)
    t = A.grid(date)
    t_ext = np.arange(d0 - A.DAY_MS, d0 + A.DAY_MS, STEP_MS, dtype=np.int64)
    today = slice(len(t_ext) - len(t), len(t_ext))
    out: dict[str, np.ndarray] = {"ts": t}

    ev = A.quote_events(root, inst, date, warmup=True)
    if ev is None:
        for f in FEATURES:
            out[f] = np.full(len(t), np.nan)
        return pl.DataFrame(out).with_columns(pl.lit(inst).alias("instrument"))

    tick = day_tick(ev, d0, meta_tick)
    q = A.quote_state(ev, t_ext)
    bid, ask, bq, aq = q["bid_px"], q["ask_px"], q["bid_qty"], q["ask_qty"]
    mid = (bid + ask) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        lm = np.log(mid)

    # imbalance
    imb = np.where(np.isfinite(bq + aq), safe_ratio(bq - aq, bq + aq), np.nan)
    out["imb_l1"] = imb[today]
    out["imb_l1_chg_1s"] = lag_diff(imb, 2)[today]
    out["imb_l1_ewm_5s"] = ewm(imb, 10)[today]
    out["imb_l1_ewm_30s"] = ewm(imb, 60)[today]

    # spread / microprice
    spr = ask - bid
    st = np.round(spr / tick) if np.isfinite(tick) else np.full(len(t_ext), np.nan)
    out["spread_ticks"] = st[today]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["spread_bps"] = (spr / mid * 1e4)[today]
    out["mp_dev_ticks"] = np.clip(imb * st / 2, -2, 2)[today]

    # momentum & volatility (500 ms mid returns on the extended grid)
    for name, k in RET_LAGS.items():
        out[f"ret_{name}"] = (lag_diff(lm, k) * 1e4)[today]
    r = lag_diff(lm, 1) * 1e4
    r2 = np.where(np.isfinite(r), r * r, np.nan)
    rv30 = np.sqrt(rolling_sum(r2, 60))
    rv300 = np.sqrt(rolling_sum(r2, 600))
    out["rv_30s"], out["rv_5min"] = rv30[today], rv300[today]
    out["rv_ratio"] = safe_ratio(rv30 * np.sqrt(10.0), rv300)[today]       # sqrt(600/60) puts both on one scale
    sig_step = rv300 / np.sqrt(600.0)
    for name, k in Z_LAGS.items():
        ret = lag_diff(lm, k) * 1e4
        out[f"retz_{name}"] = np.where(np.isfinite(ret), safe_ratio(ret, sig_step * np.sqrt(k)), np.nan)[today]

    # activity: best-PRICE changes among quote records (100 ms resolution)
    ets = ev["ts"].to_numpy()
    eb, ea = ev["bid_px"].to_numpy(), ev["ask_px"].to_numpy()
    chg = np.zeros(len(ets), bool)
    chg[1:] = (eb[1:] != eb[:-1]) | (ea[1:] != ea[:-1])
    cts = ets[chg]
    for name, w in (("moves_10s", 10_000), ("moves_60s", 60_000)):
        out[name] = (np.searchsorted(cts, t, "right") - np.searchsorted(cts, t - w, "right")).astype(np.float64)

    # staleness
    out["px_age_log"] = np.log1p(q["px_age_ms"][today] / 1000)
    out["frozen_60s"] = np.where(np.isfinite(q["px_age_ms"][today]), (q["px_age_ms"][today] > 60_000).astype(float), np.nan)
    out["quote_age_s"] = q["quote_age_ms"][today] / 1000

    # depth buckets (as of each extended grid point)
    bk = A._read_two_days(root, "book", date, inst, ["ts", *A.BUCKETS], True)
    if bk is not None:
        bk = bk.sort("ts", maintain_order=True)
        bts = bk["ts"].to_numpy()
        i = A._asof_index(bts, t_ext)
        B = np.stack([A._take(bk[f"bid_v{j:02d}"].to_numpy(), i) for j in range(1, 12)], 1)
        S = np.stack([A._take(bk[f"ask_v{j:02d}"].to_numpy(), i) for j in range(1, 12)], 1)
        cb, cs = np.cumsum(B, 1), np.cumsum(S, 1)
        for name, k in DEPTH_K.items():
            out[f"dimb_{name}"] = safe_ratio(cb[:, k - 1] - cs[:, k - 1], cb[:, k - 1] + cs[:, k - 1])[today]
            out[f"dimb_{name}"][~np.isfinite(cb[today, k - 1])] = np.nan
        d4 = cb[:, 3] + cs[:, 3]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["depth2bps_rel"] = np.log(d4 / ewm(d4, 600))[today]
        out["depth_near_share"] = safe_ratio(d4, cb[:, 10] + cs[:, 10])[today]
        out["book_age_s"] = np.where(i >= 0, (t_ext - bts[np.maximum(i, 0)]) / 1000, np.nan)[today]
    else:
        for f in [f"dimb_{k}" for k in DEPTH_K] + ["depth2bps_rel", "depth_near_share", "book_age_s"]:
            out[f] = np.full(len(t), np.nan)

    # "Nothing known yet" must look the same whichever code path produced it (truncation check):
    # where no quote record exists at or before t, every feature is NaN; where no book snapshot
    # exists yet, every depth feature is NaN.
    no_quote = ~np.isfinite(q["quote_age_ms"][today])
    for f in FEATURES:
        out[f] = np.asarray(out[f], dtype=np.float64)
        out[f][no_quote] = np.nan
    if bk is not None:
        no_book = (i < 0)[today]
        for f in [f"dimb_{k}" for k in DEPTH_K] + ["depth2bps_rel", "depth_near_share", "book_age_s"]:
            out[f][no_book] = np.nan
    df = pl.DataFrame({"ts": t, **{f: out[f] for f in FEATURES}})
    return df.with_columns(pl.lit(inst).alias("instrument"))
