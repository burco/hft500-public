"""Stage 6 — the official baseline, re-implemented on our data layer.

imb(t) = (bid_qty - ask_qty) / (bid_qty + ask_qty) from the 500 ms BOOK row at exactly t
(0 if both queues are empty);  y_hat_H(t) = a_H + b_H * imb(t), fitted by OLS on CLIPPED
training targets, separately per (perpetual, horizon). Mirrors the organisers' baseline.py
and predict.py, including their conventions:
* fit only on grid points that have a book row AND finite targets for all three horizons;
* at prediction time a grid point without a book row predicts 0 (not the intercept).
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .align import Root, grid

HORIZONS = ("y_500ms", "y_2s", "y_8s")


def dates(start: str, end: str) -> list[str]:
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    return [(d0 + dt.timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


def imbalance(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    tot = bid + ask
    return np.where(tot > 0, (bid - ask) / np.where(tot > 0, tot, 1.0), 0.0)


def imbalance_and_targets(root: Root, inst: str, date: str, clip: dict | None):
    """Same alignment as baseline.imbalance_and_targets: target rows that have a book row at the same ts."""
    book = root.read("book", date, inst, ["ts", "bid_qty", "ask_qty"])
    tg = root.read("targets", date, inst, ["ts", *HORIZONS])
    if book is None or tg is None:
        return None
    bts, tts = book["ts"].to_numpy(), tg["ts"].to_numpy()
    i = np.searchsorted(bts, tts)
    ok = (i < len(bts)) & (bts[np.minimum(i, len(bts) - 1)] == tts)
    i = i[ok]
    imb = imbalance(book["bid_qty"].to_numpy()[i].astype(np.float64), book["ask_qty"].to_numpy()[i].astype(np.float64))
    ys = {}
    for h in HORIZONS:
        y = tg[h].to_numpy()[ok]
        ys[h] = np.clip(y, *clip[h]) if clip is not None else y
    return imb, ys


def fit(root: Root, inst: str, start: str, end: str, clip: dict) -> dict[str, tuple[float, float]] | None:
    """Streaming OLS (one pass, sums only) — the training set does not fit in memory."""
    n = sx = sxx = 0.0
    sy = {h: 0.0 for h in HORIZONS}
    sxy = {h: 0.0 for h in HORIZONS}
    for d in dates(start, end):
        r = imbalance_and_targets(root, inst, d, clip)
        if r is None:
            continue
        imb, ys = r
        m = np.isfinite(imb)
        for h in HORIZONS:
            m &= np.isfinite(ys[h])
        x = imb[m]
        n += len(x); sx += x.sum(); sxx += (x * x).sum()
        for h in HORIZONS:
            y = ys[h][m]
            sy[h] += y.sum(); sxy[h] += (x * y).sum()
    if n == 0:
        return None
    var = sxx - sx * sx / n
    out = {}
    for h in HORIZONS:
        b = (sxy[h] - sx * sy[h] / n) / var
        out[h] = (sy[h] / n - b * sx / n, b)
    return out


def predict_day(root: Root, inst: str, date: str, coef: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Exactly predict.py: one value per grid point; points without a book row stay 0."""
    g = grid(date)
    preds = {h: np.zeros(len(g)) for h in HORIZONS}
    book = root.read("book", date, inst, ["ts", "bid_qty", "ask_qty"])
    if book is None:
        return g, preds
    ts = book["ts"].to_numpy()
    imb = imbalance(book["bid_qty"].to_numpy().astype(np.float64), book["ask_qty"].to_numpy().astype(np.float64))
    i = np.searchsorted(g, ts)
    hit = (i < len(g)) & (g[np.minimum(i, len(g) - 1)] == ts)
    for h in HORIZONS:
        a, b = coef[h]
        preds[h][i[hit]] = a + b * imb[hit]
    return g, preds
