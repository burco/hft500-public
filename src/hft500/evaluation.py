"""Stage 9 — feature evaluation helpers (training days only).

* daily statistics of one feature vs one target: Pearson IC, Spearman IC, decile means;
* across-day summaries: mean IC, t-stat, share of positive days, sign flips by month;
* correlation clusters (single linkage on |Spearman| >= threshold);
* chronological walk-forward folds by month with an embargo;
* a ridge model fitted from per-day SUFFICIENT STATISTICS (sums of Z'Z, Z'y), so the
  whole training period never has to sit in memory and family ablations are just
  sub-matrix selections. Linear prep: clip at fixed bounds, NaN -> 0 plus a missing indicator.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

import numpy as np
import polars as pl

N_DECILES = 10


# --------------------------------------------------------------------------- per-day stats
def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), 0-based."""
    return pl.Series(x).rank("average").to_numpy() - 1.0


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def decile_means(x_rank: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean of y within each decile of x (by rank). Heavy ties collapse into one decile."""
    n = len(x_rank)
    d = np.minimum((x_rank * N_DECILES / n).astype(int), N_DECILES - 1)
    s = np.bincount(d, y, N_DECILES)
    c = np.bincount(d, minlength=N_DECILES)
    with np.errstate(invalid="ignore"):
        return s / c


def day_stats(feats: dict[str, np.ndarray], ys: dict[str, np.ndarray]) -> list[dict]:
    """One row per (feature, horizon): n, coverage, pearson, spearman, decile means."""
    rows = []
    rank_cache: dict[tuple, np.ndarray] = {}
    for h, y in ys.items():
        fy = np.isfinite(y)
        for f, x in feats.items():
            m = fy & np.isfinite(x)
            cov = float(m.sum() / max(fy.sum(), 1))
            row = dict(feature=f, horizon=h, n=int(m.sum()), coverage=cov)
            if m.sum() < 1000:
                rows.append(row | dict(pearson=np.nan, spearman=np.nan, **{f"d{i}": np.nan for i in range(N_DECILES)}))
                continue
            xm, ym = x[m], y[m]
            key = (h, m.sum(), hash(m[:: max(1, len(m) // 997)].tobytes()))
            if key not in rank_cache:
                rank_cache[key] = ranks(ym)
            rx = ranks(xm)
            dm = decile_means(rx, ym)
            rows.append(row | dict(pearson=corr(xm, ym), spearman=corr(rx, rank_cache[key]),
                                   **{f"d{i}": float(dm[i]) for i in range(N_DECILES)}))
    return rows


# --------------------------------------------------------------------------- across days
def summarise(daily: pl.DataFrame, by=("instrument", "horizon", "feature")) -> pl.DataFrame:
    """Mean IC, t-stat (mean / (sd / sqrt(n_days))), share of positive days, monthly sign flips."""
    d = daily.filter(pl.col("pearson").is_not_nan()).with_columns(pl.col("date").str.slice(0, 7).alias("month"))
    base = d.group_by(*by).agg(
        pl.len().alias("days"),
        pl.col("pearson").mean().alias("ic"),
        pl.col("spearman").mean().alias("rank_ic"),
        (pl.col("pearson").mean() / (pl.col("pearson").std() / pl.len().sqrt())).alias("t"),
        (pl.col("pearson") > 0).mean().alias("pos_share"),
        pl.col("coverage").mean().alias("coverage"),
    )
    mon = (d.group_by(*by, "month").agg(pl.col("pearson").mean().alias("m_ic"))
           .join(base.select(*by, "ic"), on=list(by))
           .group_by(*by).agg(pl.len().alias("months"),
                              (pl.col("m_ic").sign() != pl.col("ic").sign()).sum().alias("months_flipped")))
    return base.join(mon, on=list(by))


def monotonicity(dec: np.ndarray) -> float:
    """Spearman of decile index vs decile mean target: +-1 = monotone, near 0 = U-shape or noise."""
    ok = np.isfinite(dec)
    if ok.sum() < 3:
        return float("nan")
    return corr(np.arange(len(dec))[ok].astype(float), ranks(dec[ok]))


# --------------------------------------------------------------------------- clusters
def clusters(c: np.ndarray, names: list[str], thr: float) -> list[list[str]]:
    """Single-linkage groups: features joined if |corr| >= thr (union-find)."""
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.isfinite(c[i, j]) and abs(c[i, j]) >= thr:
                parent[find(i)] = find(j)
    g = defaultdict(list)
    for i, n in enumerate(names):
        g[find(i)].append(n)
    return sorted(g.values(), key=lambda v: (-len(v), v[0]))


# --------------------------------------------------------------------------- folds
def month_folds(dates: list[str], first_test_month: str, embargo_days: int = 1) -> list[tuple[list[str], list[str]]]:
    """Expanding walk-forward: test = one calendar month; train = every date before that month,
    minus the last `embargo_days` before the test month starts (targets up to 8 s cross midnight)."""
    ds = sorted(dates)
    months = sorted({d[:7] for d in ds if d[:7] >= first_test_month})
    out = []
    for m in months:
        start = dt.date.fromisoformat(m + "-01")
        cut = (start - dt.timedelta(days=embargo_days)).isoformat()
        train = [d for d in ds if d < cut]
        test = [d for d in ds if d[:7] == m]
        if train and test:
            out.append((train, test))
    return out


# --------------------------------------------------------------------------- linear model from moments
def design(X: np.ndarray, lo: np.ndarray, hi: np.ndarray, with_ind: np.ndarray) -> np.ndarray:
    """Clip each column at fixed bounds, 0-fill NaN, append a missing indicator for flagged columns."""
    Xc = np.clip(X, lo, hi)
    miss = ~np.isfinite(Xc)
    Xc = np.where(miss, 0.0, Xc)
    return np.hstack([Xc, miss[:, with_ind].astype(np.float64)])


def moments(Z: np.ndarray, Y: np.ndarray) -> dict:
    """Sufficient statistics for every target column of Y (rows with any NaN target are dropped per column)."""
    out = {"n": [], "sz": [], "zz": [], "sy": [], "zy": []}
    for k in range(Y.shape[1]):
        m = np.isfinite(Y[:, k])
        Zm, y = Z[m], Y[m, k]
        out["n"].append(m.sum()); out["sz"].append(Zm.sum(0)); out["zz"].append(Zm.T @ Zm)
        out["sy"].append(y.sum()); out["zy"].append(Zm.T @ y)
    return {k: np.array(v) for k, v in out.items()}


def add_moments(a: dict | None, b: dict) -> dict:
    return {k: v.copy() for k, v in b.items()} if a is None else {k: a[k] + b[k] for k in a}


def ridge_from_moments(M: dict, k: int, cols: np.ndarray, lam: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardised ridge for target k on the column subset `cols`. Returns (beta_std, mean, sd)
    so that pred = ((z - mean) / sd) @ beta_std. Columns with zero variance get beta 0."""
    n = M["n"][k]
    mu = M["sz"][k][cols] / n
    cov = M["zz"][k][np.ix_(cols, cols)] / n - np.outer(mu, mu)
    cxy = M["zy"][k][cols] / n - mu * (M["sy"][k] / n)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    ok = sd > 1e-12
    beta = np.zeros(len(cols))
    S = cov[np.ix_(ok, ok)] / np.outer(sd[ok], sd[ok])
    beta[ok] = np.linalg.solve(S + lam * np.eye(ok.sum()), cxy[ok] / sd[ok])
    return beta, mu, np.where(ok, sd, 1.0)
