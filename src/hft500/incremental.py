"""Stage 13 — evaluate(): does a candidate feature add anything to the frozen 46-feature model?

One question, answered the same way every time, fast enough for an agent loop:

    "Out of sample, on TRAINING days only, does adding this feature to the Stage 9 base set
     raise the official metric (mean daily Pearson, clipped targets)?"

Protocol (fixed in advance; do not tune it on results):
* days: whatever cached training days you pass; sorted; the LAST `test_frac` of them are the
  test block, the `embargo_days` before it are dropped, everything earlier is train.
  Walk-forward, never shuffled. Validation days (2026-07-15 ...) are refused by default.
* preparation fitted on train only: clip each column at its 0.1% / 99.9% train quantiles,
  standardise with train mean / std, NaN -> 0 plus a missing-indicator column (the Stage 9
  linear preparation).
* model: ridge from the train Gram matrix (lam relative to n). A LINEAR probe on purpose:
  it is cheap, deterministic and comparable across candidates. A feature that only helps
  through interactions will look weak here — the regime table is the hint for that, and the
  final word belongs to a LightGBM refit.
* reported per (instrument, horizon):
    score_base / score_full / gain   — base vs base + all candidates
    addone_gain[c]                   — base + this one candidate
    resid_ic[c], resid_t[c], pos[c]  — daily Pearson of the candidate with the base model's
                                        out-of-sample residual (what is left to explain)
    regime gains                     — gain inside wide/tight spread and high/low volatility,
                                        thresholds from TRAIN days

TRAINING-SIDE module: it reads targets. Nothing here may be imported by feature code.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from .metric import pearson

HORIZONS = ("y_500ms", "y_2s", "y_8s")
VALID_START = "2026-07-15"
MIN_POINTS = 1000


# --------------------------------------------------------------------------- preparation
def _prep(Xtr: np.ndarray):
    with np.errstate(all="ignore"):
        lo = np.nanquantile(Xtr, 0.001, axis=0)
        hi = np.nanquantile(Xtr, 0.999, axis=0)
        C = np.clip(Xtr, lo, hi)
        mu = np.nanmean(C, axis=0)
        sd = np.nanstd(C, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    lo = np.where(np.isfinite(lo), lo, -np.inf)
    hi = np.where(np.isfinite(hi), hi, np.inf)
    has_nan = np.isnan(Xtr).any(axis=0)
    ind_mu = np.isnan(Xtr).mean(axis=0)
    return lo, hi, mu, sd, has_nan, ind_mu


def _design(X: np.ndarray, p) -> np.ndarray:
    lo, hi, mu, sd, has_nan, ind_mu = p
    with np.errstate(all="ignore"):
        Z = (np.clip(X, lo, hi) - mu) / sd
    nanm = np.isnan(Z)
    Z[nanm] = 0.0
    ind = (np.isnan(X[:, has_nan]).astype(np.float64) - ind_mu[has_nan])
    return np.hstack([Z, ind])


def _owners(names: list[str], has_nan: np.ndarray) -> list[str]:
    """Feature that owns each design column (value columns, then indicator columns)."""
    return list(names) + [n for n, h in zip(names, has_nan) if h]


def _daily_corr(pred, y, dates, mask=None) -> np.ndarray:
    out = []
    for d in np.unique(dates):
        m = dates == d
        if mask is not None:
            m &= mask
        if m.sum() < MIN_POINTS:
            continue
        out.append(pearson(pred[m], y[m]))
    return np.asarray(out, dtype=np.float64)


def _tstat(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if len(v) < 2 or v.std(ddof=1) == 0:
        return float("nan")
    return float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))


# --------------------------------------------------------------------------- core
def split_days(dates: list[str], test_frac: float, embargo_days: int) -> tuple[list[str], list[str]]:
    ds = sorted(set(dates))
    n_test = max(1, int(round(len(ds) * test_frac)))
    test = ds[-n_test:]
    train = ds[: max(0, len(ds) - n_test - embargo_days)]
    if len(train) < 2:
        raise ValueError(f"need at least 2 train days, got {len(train)} from {len(ds)} days")
    return train, test


def evaluate_frame(df: pl.DataFrame, base: list[str], candidates: list[str], horizons=HORIZONS,
                   test_frac: float = 0.35, embargo_days: int = 1, lam: float = 1e-4,
                   allow_validation: bool = False) -> dict[str, list[dict]]:
    """One instrument. df: date (str), ts, base + candidate feature columns, target columns."""
    missing = [c for c in [*base, *candidates, *horizons, "date"] if c not in df.columns]
    if missing:
        raise KeyError(f"columns missing from the frame: {missing}")
    if not allow_validation and (df["date"] >= VALID_START).any():
        raise ValueError("validation days in the frame: keep them for the final check (allow_validation=True to override)")
    dup = set(base) & set(candidates)
    if dup:
        raise ValueError(f"candidates already in the base set: {sorted(dup)}")

    train_days, test_days = split_days(df["date"].to_list(), test_frac, embargo_days)
    names = [*base, *candidates]
    tr = df.filter(pl.col("date").is_in(train_days))
    te = df.filter(pl.col("date").is_in(test_days))
    Xtr = tr.select(names).to_numpy().astype(np.float64)
    Xte = te.select(names).to_numpy().astype(np.float64)
    p = _prep(Xtr)
    Dtr, Dte = _design(Xtr, p), _design(Xte, p)
    own = np.asarray(_owners(names, p[4]))
    base_cols = np.flatnonzero(np.isin(own, base))
    d_te = te["date"].to_numpy()

    # regime masks on the test block; thresholds from train only
    regimes = {}
    if "spread_ticks" in df.columns:
        regimes["spread>1tick"] = te["spread_ticks"].to_numpy() > 1.5
        regimes["spread=1tick"] = te["spread_ticks"].to_numpy() <= 1.5
    if "rv_5min" in df.columns:
        thr = float(np.nanmedian(tr["rv_5min"].to_numpy()))
        rv = te["rv_5min"].to_numpy()
        regimes["vol_high"] = rv > thr
        regimes["vol_low"] = rv <= thr

    cells, feats, regs = [], [], []
    for h in horizons:
        ytr = tr[h].to_numpy().astype(np.float64)
        yte = te[h].to_numpy().astype(np.float64)
        ok_tr, ok_te = np.isfinite(ytr), np.isfinite(yte)
        ym = ytr[ok_tr].mean()
        A_ = Dtr[ok_tr]
        G = A_.T @ A_
        b = A_.T @ (ytr[ok_tr] - ym)
        n = ok_tr.sum()

        def fit_pred(cols):
            Gs = G[np.ix_(cols, cols)] + lam * n * np.eye(len(cols))
            w = np.linalg.solve(Gs, b[cols])
            return Dte[:, cols] @ w + ym

        Dm, ym_te, dm = Dte[ok_te], yte[ok_te], d_te[ok_te]
        pb = fit_pred(base_cols)[ok_te]
        pf = fit_pred(np.arange(Dtr.shape[1]))[ok_te]
        sb, sf = _daily_corr(pb, ym_te, dm), _daily_corr(pf, ym_te, dm)
        cells.append(dict(horizon=h, score_base=float(sb.mean()), score_full=float(sf.mean()),
                          gain=float((sf - sb).mean()), gain_t=_tstat(sf - sb),
                          win_days=float((sf > sb).mean()), n_train_days=len(train_days),
                          n_test_days=len(test_days), test_from=test_days[0], test_to=test_days[-1]))
        resid = ym_te - pb
        for c in candidates:
            cc = np.concatenate([base_cols, np.flatnonzero(own == c)])
            pc = fit_pred(cc)[ok_te]
            g = _daily_corr(pc, ym_te, dm) - sb
            xc = Dm[:, names.index(c)]
            ric = _daily_corr(xc, resid, dm)
            ic = _daily_corr(xc, ym_te, dm)
            feats.append(dict(horizon=h, feature=c, ic=float(ic.mean()), resid_ic=float(ric.mean()),
                              resid_t=_tstat(ric), resid_pos=float((ric > 0).mean()),
                              addone_gain=float(g.mean()), addone_win=float((g > 0).mean())))
        for r, m in regimes.items():
            mm = m[ok_te]
            gb = _daily_corr(pf, ym_te, dm, mm) - _daily_corr(pb, ym_te, dm, mm)
            regs.append(dict(horizon=h, regime=r, share=float(mm.mean()),
                             gain=float(gb.mean()) if len(gb) else float("nan"), days=len(gb)))
    return dict(cells=cells, features=feats, regimes=regs)


def evaluate(candidates: list[str], instruments: list[str], cache_dir: str | Path, base: list[str],
             days: list[str] | None = None, horizons=HORIZONS, out_dir: str | Path | None = None,
             **kw) -> dict[str, pl.DataFrame]:
    """Agent entry point. Reads <cache_dir>/<INST>/<date>.parquet (built by stage13_build.py),
    evaluates every instrument, returns three tables and (optionally) writes them as CSV + a
    one-line JSON verdict to out_dir."""
    tabs = {"cells": [], "features": [], "regimes": []}
    for inst in instruments:
        files = sorted(Path(cache_dir, inst).glob("*.parquet"))
        if days is not None:
            files = [f for f in files if f.stem in set(days)]
        if not files:
            raise FileNotFoundError(f"no cached days for {inst} in {cache_dir}")
        df = pl.concat([pl.read_parquet(f).with_columns(pl.lit(f.stem).alias("date")) for f in files],
                       how="vertical_relaxed")
        r = evaluate_frame(df, base, candidates, horizons, **kw)
        for k in tabs:
            tabs[k] += [dict(instrument=inst, **row) for row in r[k]]
    out = {k: pl.DataFrame(v) for k, v in tabs.items()}
    if out_dir is not None:
        o = Path(out_dir)
        o.mkdir(parents=True, exist_ok=True)
        for k, t in out.items():
            t.write_csv(o / f"{k}.csv")
        c = out["cells"]
        verdict = dict(candidates=candidates, mean_gain=float(c["gain"].mean()),
                       cells_improved=int((c["gain"] > 0).sum()), cells=c.height,
                       score_base=float(c["score_base"].mean()), score_full=float(c["score_full"].mean()))
        (o / "verdict.json").write_text(json.dumps(verdict, indent=1))
    return out
