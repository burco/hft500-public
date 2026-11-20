"""Stage 2 — microstructure primitives as Polars expressions.

Every function returns a `pl.Expr`, so the same code runs on one row, one day,
or a lazy scan of a year. All are point-in-time: they use only the row itself.

Conventions (match the assignment's common.py / baseline.py):
- prices in quote currency (USDT), quantities in base units (BTC),
  book buckets bid_v01..v11 / ask_v01..v11 in quote-currency NOTIONAL;
- trade vwap / max_qty are NaN (not null) on a side with no trades;
- imbalance is defined as 0 when both sides are empty (baseline.py does this).
"""
from __future__ import annotations

import polars as pl

BID_BUCKETS = [f"bid_v{i:02d}" for i in range(1, 12)]
ASK_BUCKETS = [f"ask_v{i:02d}" for i in range(1, 12)]
BPS = 1e4


def _c(x: str | pl.Expr) -> pl.Expr:
    return pl.col(x) if isinstance(x, str) else x


def _safe_ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """num/den, but 0 where den == 0 (never inf/NaN)."""
    return pl.when(den > 0).then(num / den).otherwise(0.0)


# --- prices -------------------------------------------------------------------
def mid(bid="bid_px", ask="ask_px") -> pl.Expr:
    """mid = (bid + ask) / 2."""
    return (_c(bid) + _c(ask)) / 2


def spread(bid="bid_px", ask="ask_px") -> pl.Expr:
    """Absolute spread in price units (USDT)."""
    return _c(ask) - _c(bid)


def spread_ticks(tick: float | str | pl.Expr, bid="bid_px", ask="ask_px") -> pl.Expr:
    """Spread measured in ticks, rounded (prices are tick multiples)."""
    t = pl.lit(tick) if isinstance(tick, float) else _c(tick)
    return (spread(bid, ask) / t).round(0)


def spread_bps(bid="bid_px", ask="ask_px") -> pl.Expr:
    """Relative spread: (ask - bid) / mid, in basis points."""
    return spread(bid, ask) / mid(bid, ask) * BPS


# --- top-of-book imbalance ----------------------------------------------------------
def queue_imbalance(bid_qty="bid_qty", ask_qty="ask_qty") -> pl.Expr:
    """imb = (Qb - Qa) / (Qb + Qa), in [-1, 1]. +1 = all size on the bid."""
    b, a = _c(bid_qty).cast(pl.Float64), _c(ask_qty).cast(pl.Float64)
    return _safe_ratio(b - a, b + a)


def microprice(bid="bid_px", ask="ask_px", bid_qty="bid_qty", ask_qty="ask_qty") -> pl.Expr:
    """Size-weighted mid: (bid*Qa + ask*Qb) / (Qb + Qa).

    Leans toward the side that is likely to be hit next: a thin ask (small Qa)
    pulls the microprice up toward the ask. Identity:
        microprice - mid = imbalance * spread / 2
    """
    b, a = _c(bid_qty).cast(pl.Float64), _c(ask_qty).cast(pl.Float64)
    return pl.when(b + a > 0).then((_c(bid) * a + _c(ask) * b) / (b + a)).otherwise(mid(bid, ask))


def microprice_dev_ticks(tick: float, **cols) -> pl.Expr:
    """(microprice - mid) in ticks, in [-0.5, 0.5] when the spread is one tick."""
    return (microprice(**cols) - mid(cols.get("bid", "bid_px"), cols.get("ask", "ask_px"))) / tick


# --- depth (book buckets) -----------------------------------------------------------
def depth_imbalance(k: int) -> pl.Expr:
    """Imbalance of CUMULATIVE notional within the first k buckets (k=1..11).

    k=1 -> within 0.5 bps of mid; k=4 -> within 2 bps; k=11 -> within ~100 bps.
    """
    if not 1 <= k <= 11:
        raise ValueError("k must be in 1..11")
    b = pl.sum_horizontal(BID_BUCKETS[:k]).cast(pl.Float64)
    a = pl.sum_horizontal(ASK_BUCKETS[:k]).cast(pl.Float64)
    return _safe_ratio(b - a, b + a)


def depth_notional(side: str, k: int = 11) -> pl.Expr:
    """Total resting notional on one side within the first k buckets."""
    cols = BID_BUCKETS if side == "bid" else ASK_BUCKETS
    return pl.sum_horizontal(cols[:k]).cast(pl.Float64)


# --- trades (aggressor flow) --------------------------------------------------------
def signed_volume() -> pl.Expr:
    """Buyer-initiated minus seller-initiated quantity in the bar (base units).
    Positive = takers lifted offers (buying pressure)."""
    return (_c("buy_qty") - _c("sell_qty")).cast(pl.Float64)


def trade_imbalance() -> pl.Expr:
    """(buy_qty - sell_qty) / (buy_qty + sell_qty), 0 when no trades."""
    b, s = _c("buy_qty").cast(pl.Float64), _c("sell_qty").cast(pl.Float64)
    return _safe_ratio(b - s, b + s)


def log_return_bps(price_now: pl.Expr, price_later: pl.Expr) -> pl.Expr:
    """ln(later / now) in basis points (the assignment's target, scaled by 1e4)."""
    return (price_later / price_now).log() * BPS
