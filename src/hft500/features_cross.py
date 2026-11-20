"""Stage 8 — trade-flow, order-flow-imbalance and cross-market features on the 500 ms grid.

Same causal discipline as features_book: everything at t uses only records stamped <= t,
computed on an extended grid that starts at the previous day's midnight, then sliced.
Data from OTHER instruments obeys the same rule (and is truncated with everything else by
the causality check).

Notes
* OFI here is SAMPLED: Cont-Kukanov-Stoikov increments between consecutive 100 ms quote states,
  not exact event-level OFI (the raw event stream is not supplied).
* ts is the collector's receive time, so an apparent lead of one venue over another can partly
  be data latency. That is a hypothesis to measure, not an assumption.
* Stablecoin conversion uses the `stable` table's mid, never price ratios between pairs. Spot
  pairs that the PDF nulls during a depeg stay NaN (EWMAs skip NaN).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import align as A
from .features_book import ewm, lag_diff, rolling_sum, safe_ratio

STEP_MS = 500
TARGETS = ("BTCUSDT_PERP", "ETHUSDT_PERP", "SOLUSDT_PERP", "1000PEPEUSDT_PERP")
COIN = {"BTCUSDT_PERP": "btc", "ETHUSDT_PERP": "eth", "SOLUSDT_PERP": "sol", "1000PEPEUSDT_PERP": "pepe"}
SPOT = {"BTCUSDT_PERP": ("BTCUSDT_SPOT", 1.0), "ETHUSDT_PERP": ("ETHUSDT_SPOT", 1.0),
        "SOLUSDT_PERP": ("SOLUSDT_SPOT", 1.0), "1000PEPEUSDT_PERP": ("PEPEUSDT_SPOT", 1000.0)}
# a USDC-quoted twin of each target, converted to USDT with stable's USDCUSDT mid
USDC = {"BTCUSDT_PERP": ("BTCUSDC_SPOT", 1.0), "ETHUSDT_PERP": ("ETHUSDC_SPOT", 1.0),
        "SOLUSDT_PERP": ("SOLUSDC_SPOT", 1.0), "1000PEPEUSDT_PERP": ("1000PEPEUSDC_PERP", 1.0)}
TRADE_W = {"0.5s": 1, "2s": 4, "10s": 20, "30s": 60}
SVN_W = {"0.5s": 1, "2s": 4, "10s": 20}
OFI_W = {"0.5s": 5, "1s": 10, "2s": 20, "5s": 50}          # in 100 ms steps
XRET = {"0.5s": 1, "2s": 4}

REGISTRY: dict[str, tuple[str, float, str]] = {
    **{f"tim_{w}": ("trades", n * 0.5, f"trade imbalance (buy-sell)/(buy+sell) qty over the last {w}") for w, n in TRADE_W.items()},
    **{f"svn_{w}": ("trades", 300, f"signed notional over {w} / its expected size from a 5 min EWMA of notional") for w in SVN_W},
    "intensity_10s": ("trades", 300, "trades in the last 10 s / expected count from a 5 min EWMA (activity, directionless)"),
    "last_side": ("trades", 0, "(sell age - buy age)/(sell age + buy age): +1 if the last trade was a buy"),
    **{f"ofi_{w}": ("ofi", 300, f"sampled order-flow imbalance over {w} / 5 min EWMA of best-level depth") for w in OFI_W},
    **{f"xret_{c}_{w}": ("leadlag", n * 0.5, f"{c.upper()} perp past return over {w}, bps (NaN for the target's own coin)")
       for c in COIN.values() for w, n in XRET.items()},
    **{f"sret_{w}": ("spot", n * 0.5, f"own USDT spot past return over {w}, bps") for w, n in XRET.items()},
    "spot_minus_perp_0.5s": ("spot", 0.5, "sret_0.5s minus the perp's own 0.5 s return (spot moved, perp not yet)"),
    "basis_bps": ("spot", 0, "perp mid / spot mid - 1, bps"),
    "basis_dev_10s": ("spot", 60, "basis_bps minus its EWMA (half-life 10 s)"),
    "basis_dev_60s": ("spot", 360, "basis_bps minus its EWMA (half-life 60 s)"),
    "usdc_gap_bps": ("stable", 0, "USDC-quoted twin converted with stable USDCUSDT mid, vs the perp mid, bps"),
    "usdc_gap_dev_60s": ("stable", 360, "usdc_gap_bps minus its EWMA (half-life 60 s)"),
}
FEATURES = list(REGISTRY)


def registry_frame() -> pl.DataFrame:
    return pl.DataFrame([dict(feature=k, family=v[0], lookback_s=float(v[1]), description=v[2]) for k, v in REGISTRY.items()])


# --------------------------------------------------------------------------- building blocks
def ext_grid(date: str, step_ms: int = STEP_MS, start_ms: int = A.DAY_MS) -> np.ndarray:
    d0 = A.day_start(date)
    return np.arange(d0 - start_ms, d0 + A.DAY_MS, step_ms, dtype=np.int64)


def quotes_on(root: A.Root, inst: str, date: str, times: np.ndarray) -> dict[str, np.ndarray] | None:
    ev = A.quote_events(root, inst, date, warmup=True)
    return None if ev is None else A.quote_state(ev, times)


def log_mid(q: dict | None, n: int, scale: float = 1.0) -> np.ndarray:
    if q is None:
        return np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log((q["bid_px"] + q["ask_px"]) / 2 * scale)


def ofi_increments(q: dict[str, np.ndarray]) -> np.ndarray:
    """Cont-Kukanov-Stoikov e_n between consecutive quote states (base units). e_0 = NaN."""
    pb, qb, pa, qa = q["bid_px"], q["bid_qty"], q["ask_px"], q["ask_qty"]
    e = np.full(len(pb), np.nan)
    pb0, qb0, pa0, qa0 = pb[:-1], qb[:-1], pa[:-1], qa[:-1]
    pb1, qb1, pa1, qa1 = pb[1:], qb[1:], pa[1:], qa[1:]
    eb = np.where(pb1 > pb0, qb1, np.where(pb1 == pb0, qb1 - qb0, -qb0))
    ea = np.where(pa1 < pa0, qa1, np.where(pa1 == pa0, qa1 - qa0, -qa0))
    e[1:] = eb - ea
    return e


def trades_ext(root: A.Root, inst: str, date: str, t_ext: np.ndarray) -> dict[str, np.ndarray]:
    """Per 500 ms point of the extended grid: buy/sell qty, counts and notional of bars in (t-500, t]."""
    n = len(t_ext)
    out = {k: np.zeros(n) for k in ("bq", "sq", "bn", "sn", "bnot", "snot")}
    tr = A._read_two_days(root, "trades", date, inst, None, True)
    if tr is None:
        return out
    bts = tr["ts"].to_numpy()
    pos = (-(-bts // STEP_MS) * STEP_MS - t_ext[0]) // STEP_MS
    keep = (pos >= 0) & (pos < n)
    for side, k in (("buy", "b"), ("sell", "s")):
        q = tr[f"{side}_qty"].cast(pl.Float64).to_numpy()
        c = tr[f"{side}_n"].cast(pl.Float64).to_numpy()
        vw = tr[f"{side}_vwap"].cast(pl.Float64).fill_nan(None).fill_null(0.0).to_numpy()
        np.add.at(out[f"{k}q"], pos[keep], q[keep])
        np.add.at(out[f"{k}n"], pos[keep], c[keep])
        np.add.at(out[f"{k}not"], pos[keep], (q * vw)[keep])
    return out


def stable_mid_ext(root: A.Root, date: str, pair: str, t_ext: np.ndarray) -> np.ndarray:
    st = A._read_two_days(root, "stable", date, None, ["ts", "pair", "bid_px", "ask_px"], True)
    if st is None:
        return np.full(len(t_ext), np.nan)
    g = st.filter(pl.col("pair") == pair).sort("ts", maintain_order=True)
    if g.height == 0:
        return np.full(len(t_ext), np.nan)
    i = A._asof_index(g["ts"].to_numpy(), t_ext)
    return A._take(((g["bid_px"].cast(pl.Float64) + g["ask_px"].cast(pl.Float64)) / 2).to_numpy(), i)


# --------------------------------------------------------------------------- main
def build(root: A.Root, inst: str, date: str) -> pl.DataFrame:
    """All registered cross features for one target instrument-day: always 172,800 rows."""
    t = A.grid(date)
    t_ext = ext_grid(date)
    n_ext = len(t_ext)
    today = slice(n_ext - len(t), n_ext)
    out: dict[str, np.ndarray] = {}

    q_own = quotes_on(root, inst, date, t_ext)
    lm_own = log_mid(q_own, n_ext)

    # ---- trade flow
    tr = trades_ext(root, inst, date, t_ext)
    for w, k in TRADE_W.items():
        b, s = rolling_sum(tr["bq"], k), rolling_sum(tr["sq"], k)
        out[f"tim_{w}"] = safe_ratio(b - s, b + s)[today]
    tot_not = tr["bnot"] + tr["snot"]
    exp_not = ewm(tot_not, 600)                                   # expected notional per 500 ms
    for w, k in SVN_W.items():
        sv = rolling_sum(tr["bnot"] - tr["snot"], k)
        out[f"svn_{w}"] = safe_ratio(sv, exp_not * k)[today]
    cnt = tr["bn"] + tr["sn"]
    out["intensity_10s"] = safe_ratio(rolling_sum(cnt, 20), ewm(cnt, 600) * 20)[today]
    ta = A.trades_on_grid(root, inst, date, warmup=True)
    ab, asl = ta["tr_buy_age_ms"].to_numpy(), ta["tr_sell_age_ms"].to_numpy()
    out["last_side"] = np.where(np.isfinite(ab) & np.isfinite(asl), safe_ratio(asl - ab, asl + ab), np.nan)

    # ---- sampled OFI on the 100 ms grid (10 s of history before midnight is enough for 5 s windows)
    d0 = A.day_start(date)
    t100 = np.arange(d0 - 10_000, d0 + A.DAY_MS, 100, dtype=np.int64)
    ev = A.quote_events(root, inst, date, warmup=True)
    if ev is not None:
        q100 = A.quote_state(ev, t100)
        e = ofi_increments(q100)
        c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(e, nan=0.0))])
        k_at = (t - t100[0]) // 100 + 1                           # cumulative index just after the step AT t
        depth = ewm((q_own["bid_qty"] + q_own["ask_qty"]) / 2, 600)[today]
        for w, steps in OFI_W.items():
            out[f"ofi_{w}"] = safe_ratio(c[k_at] - c[k_at - steps], depth)
    else:
        for w in OFI_W:
            out[f"ofi_{w}"] = np.full(len(t), np.nan)

    # ---- lead-lag: the other perps' past returns
    for other, coin in COIN.items():
        if other == inst:
            for w in XRET:
                out[f"xret_{coin}_{w}"] = np.full(len(t), np.nan)
            continue
        lm = log_mid(quotes_on(root, other, date, t_ext), n_ext)
        for w, k in XRET.items():
            out[f"xret_{coin}_{w}"] = (lag_diff(lm, k) * 1e4)[today]

    # ---- own spot: returns and basis
    sp, scale = SPOT[inst]
    lm_sp = log_mid(quotes_on(root, sp, date, t_ext), n_ext, scale)
    for w, k in XRET.items():
        out[f"sret_{w}"] = (lag_diff(lm_sp, k) * 1e4)[today]
    out["spot_minus_perp_0.5s"] = ((lag_diff(lm_sp, 1) - lag_diff(lm_own, 1)) * 1e4)[today]
    basis = (np.exp(lm_own - lm_sp) - 1) * 1e4
    out["basis_bps"] = basis[today]
    out["basis_dev_10s"] = (basis - ewm(basis, 20))[today]
    out["basis_dev_60s"] = (basis - ewm(basis, 120))[today]

    # ---- USDC twin converted with the stable table
    tw, tscale = USDC[inst]
    lm_tw = log_mid(quotes_on(root, tw, date, t_ext), n_ext, tscale)
    with np.errstate(invalid="ignore", divide="ignore"):
        usdc = np.log(stable_mid_ext(root, date, "USDCUSDT", t_ext))
    gap = (np.exp(lm_tw + usdc - lm_own) - 1) * 1e4
    out["usdc_gap_bps"] = gap[today]
    out["usdc_gap_dev_60s"] = (gap - ewm(gap, 120))[today]

    # nothing known yet about the target itself -> every feature NaN (same rule as features_book)
    if q_own is None:
        no_quote = np.ones(len(t), bool)
    else:
        no_quote = ~np.isfinite(q_own["quote_age_ms"][today])
    cols = {}
    for f in FEATURES:
        v = np.asarray(out[f], dtype=np.float64).copy()
        v[no_quote] = np.nan
        cols[f] = v
    return pl.DataFrame({"ts": t, **cols}).with_columns(pl.lit(inst).alias("instrument"))
