"""Stage 6 — the competition metric, implemented from the PDF.

1. clip each target at the training-set 0.1% / 99.9% quantiles (meta/target_clip.parquet);
2. Pearson correlation per (perpetual, horizon, UTC day); a missing prediction counts as 0;
3. average over days, then over the 12 (perpetual, horizon) combinations.

Conventions taken from the organisers' baseline.py where the PDF is silent:
* rows whose target is NaN are dropped (baseline masks with np.isfinite);
* a day where predictions or targets are constant scores 0.0 (not NaN);
* baseline.py also skips days with < 10,000 valid points (`min_points`; the PDF does not say this,
  default here is 0 = keep every day).
"""
from __future__ import annotations

import numpy as np
import polars as pl

HORIZONS = ("y_500ms", "y_2s", "y_8s")
PRED_COLS = {"y_500ms": "pred_500ms", "y_2s": "pred_2s", "y_8s": "pred_8s"}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; 0.0 when either side is constant (baseline.py convention)."""
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def score_day(pred: np.ndarray, y: np.ndarray, lo: float, hi: float, min_points: int = 0) -> float | None:
    """One (perpetual, horizon, day) cell. pred and y are aligned on the full grid; NaN pred = missing = 0.
    Returns None if the day is skipped (fewer than min_points finite targets)."""
    pred = np.where(np.isfinite(pred), pred, 0.0).astype(np.float64)
    m = np.isfinite(y)
    if m.sum() < max(min_points, 1):
        return None
    return pearson(pred[m], np.clip(y[m], lo, hi))


def score_predictions(preds: pl.DataFrame, targets: dict[str, pl.DataFrame], clips: dict,
                      date: str, min_points: int = 0) -> list[dict]:
    """preds: one day in the submission format (ts, instrument, pred_500ms, pred_2s, pred_8s).
    targets: {instrument: targets frame for that day}. Returns one row per (instrument, horizon)."""
    rows = []
    for inst, tg in targets.items():
        p = preds.filter(pl.col("instrument") == inst).select(["ts", *PRED_COLS.values()])
        g = tg.select(["ts", *HORIZONS]).join(p, on="ts", how="left")      # prediction missing -> null -> 0
        for h in HORIZONS:
            c = score_day(g[PRED_COLS[h]].fill_null(0.0).to_numpy(), g[h].to_numpy(), *clips[(inst, h)], min_points)
            if c is not None:
                rows.append(dict(date=date, instrument=inst, horizon=h, corr=c))
    return rows


def aggregate(per_day: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """Mean over days per (instrument, horizon), plus the final score = mean of the cell means."""
    cells = (per_day.group_by("instrument", "horizon")
             .agg(pl.col("corr").mean().alias("mean_corr"), pl.col("corr").std().alias("sd"),
                  pl.len().alias("days"))
             .with_columns((pl.col("sd") / pl.col("days").sqrt()).alias("se"))
             .sort("instrument", "horizon"))
    return cells, float(cells["mean_corr"].mean())
