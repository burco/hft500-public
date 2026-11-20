"""End-to-end prediction for one UTC day: the exact code path predict.py (Stage 12) will run.

predict_day(root, date, models) -> 691,200 rows: ts, instrument, pred_500ms, pred_2s, pred_8s.
* features: Stage 7 (book) + Stage 8 (cross) builders, previous-day warm-up, grid from the date;
* models: anything with predict(frame) -> {horizon: array} (LGBModel, LinearModel, ...);
* fallback: any non-finite prediction (e.g. nothing known yet at the first instant) becomes 0,
  which the metric treats the same as a missing prediction.

Also: RestrictedRoot, which enforces the inference contract (only the requested day and the day
before, never targets) and records every read for the audit.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import align as A, features_book as FB, features_cross as X

INSTRUMENTS = ("BTCUSDT_PERP", "ETHUSDT_PERP", "SOLUSDT_PERP", "1000PEPEUSDT_PERP")
PRED = {"y_500ms": "pred_500ms", "y_2s": "pred_2s", "y_8s": "pred_8s"}


class ContractViolation(RuntimeError):
    pass


class RestrictedRoot:
    """Only `date` and the day before may be read, and never the targets table."""

    def __init__(self, base: A.Root, date: str):
        self.base, self.allowed = base, {date, A.prev_date(date)}
        self.reads: list[tuple[str, str, str | None]] = []

    def read(self, table, date, instrument=None, columns=None):
        if table == "targets":
            raise ContractViolation(f"targets read for {instrument} {date}")
        if date not in self.allowed:
            raise ContractViolation(f"{table}/{instrument} read for {date}; allowed {sorted(self.allowed)}")
        self.reads.append((table, date, instrument))
        return self.base.read(table, date, instrument, columns)


def features(root: A.Root, inst: str, date: str, meta_tick: float | None = None) -> pl.DataFrame:
    return (FB.build(root, inst, date, meta_tick=meta_tick).drop("instrument")
            .join(X.build(root, inst, date).drop("instrument"), on="ts"))


def predict_day(root: A.Root, date: str, models: dict, meta_ticks: dict | None = None,
                feature_fn=features) -> pl.DataFrame:
    out = []
    for inst in INSTRUMENTS:
        fr = feature_fn(root, inst, date, (meta_ticks or {}).get(inst))
        p = models[inst].predict(fr)
        out.append(pl.DataFrame({"ts": fr["ts"], "instrument": inst,
                                 **{PRED[h]: np.nan_to_num(np.asarray(p[h], np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                                    for h in PRED}}))
    return pl.concat(out)


def wide(df: pl.DataFrame) -> pl.DataFrame:
    """One row per ts, one column per (instrument, horizon): what the truncation check compares."""
    return df.pivot("instrument", index="ts", values=list(PRED.values())).sort("ts")
