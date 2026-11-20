"""Stage 10 — the candidate models behind one small interface: predict(frame) -> {horizon: array}.

* ZeroModel        — constant 0 (scores 0 under the baseline.py convention).
* LinearModel      — OLS / ridge on the Stage 9 linear design (clip at fixed bounds, NaN -> 0 + indicator),
                     fitted from per-day sufficient statistics (evaluation.moments). Stored as JSON.
* LGBModel         — one LightGBM booster per horizon; NaN passed through (trees route missing values).

Every model is trained on CLIPPED targets; LightGBM sees targets in bps (x1e4) for numerics only:
Pearson is scale-invariant, so the scale never reaches the metric.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from . import evaluation as E

HORIZONS = ("y_500ms", "y_2s", "y_8s")
Y_SCALE = 1e4

LGB_BASE = dict(objective="l2", learning_rate=0.05, min_data_in_leaf=2000, feature_fraction=0.8,
                bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0, max_bin=63, num_threads=2,
                seed=7, deterministic=True, force_row_wise=True, verbose=-1)
LGB_CONFIGS = {"A": dict(num_leaves=31), "B": dict(num_leaves=127, min_data_in_leaf=1000)}


class ZeroModel:
    name = "zero"

    def predict(self, frame: pl.DataFrame) -> dict[str, np.ndarray]:
        return {h: np.zeros(frame.height) for h in HORIZONS}


@dataclass
class LinearModel:
    feats_all: list[str]                 # design columns are built from these (fixed order)
    lo: np.ndarray
    hi: np.ndarray
    ind: np.ndarray                      # which features get a missing indicator
    W: np.ndarray                        # (n_design, 3): pred = design @ W (intercept irrelevant for Pearson)
    used: list[str] = field(default_factory=list)
    lam: float = 0.0

    @classmethod
    def fit(cls, M: dict, feats_all, lo, hi, ind, used: list[str], lam: float) -> "LinearModel":
        cols = cols_for(feats_all, ind, used)
        W = np.zeros((len(feats_all) + int(ind.sum()), 3))
        for k in range(3):
            b, mu, sd = E.ridge_from_moments(M, k, cols, lam)
            W[cols, k] = b / sd
        return cls(list(feats_all), np.asarray(lo), np.asarray(hi), np.asarray(ind, bool), W, list(used), lam)

    def design(self, frame: pl.DataFrame) -> np.ndarray:
        return E.design(frame.select(self.feats_all).to_numpy().astype(np.float64), self.lo, self.hi, self.ind)

    def predict(self, frame: pl.DataFrame) -> dict[str, np.ndarray]:
        P = self.design(frame) @ self.W
        return {h: P[:, k] for k, h in enumerate(HORIZONS)}

    def n_params(self) -> int:
        return int((np.abs(self.W) > 0).sum())

    def save(self, path: Path) -> None:
        json.dump(dict(feats_all=self.feats_all, lo=_enc(self.lo), hi=_enc(self.hi), ind=self.ind.tolist(),
                       W=self.W.tolist(), used=self.used, lam=self.lam), open(path, "w"))

    @classmethod
    def load(cls, path: Path) -> "LinearModel":
        d = json.load(open(path))
        return cls(d["feats_all"], _dec(d["lo"]), _dec(d["hi"]), np.array(d["ind"], bool), np.array(d["W"]), d["used"], d["lam"])


def _enc(a):  # JSON has no inf
    return ["inf" if v == np.inf else "-inf" if v == -np.inf else float(v) for v in a]


def _dec(a):
    return np.array([float(v) for v in a])


def cols_for(feats_all, ind, used) -> np.ndarray:
    pos = {f: len(feats_all) + int(np.sum(ind[:i])) for i, f in enumerate(feats_all) if ind[i]}
    return np.array([feats_all.index(f) for f in used] + [pos[f] for f in used if f in pos], dtype=int)


class LGBModel:
    def __init__(self, feats: list[str], boosters: dict, n_iter: int | None = None):
        self.feats, self.boosters, self.n_iter = list(feats), boosters, n_iter

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, feats: list[str], params: dict, rounds: int) -> "LGBModel":
        import lightgbm as lgb
        bs = {}
        for k, h in enumerate(HORIZONS):
            m = np.isfinite(Y[:, k])
            ds = lgb.Dataset(X[m], Y[m, k] * Y_SCALE, params={"max_bin": params.get("max_bin", 63)}, free_raw_data=True)
            bs[h] = lgb.train(params, ds, rounds)
        return cls(feats, bs)

    def predict(self, frame: pl.DataFrame, n_iter: int | None = None) -> dict[str, np.ndarray]:
        X = frame.select(self.feats).to_numpy().astype(np.float32)
        it = n_iter or self.n_iter
        return {h: b.predict(X, num_iteration=it) for h, b in self.boosters.items()}

    def n_params(self) -> int:
        return int(sum(b.num_trees() * 2 * b.params.get("num_leaves", 31) for b in self.boosters.values()))

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        for h, b in self.boosters.items():
            b.save_model(str(d / f"{h}.txt"), num_iteration=self.n_iter)
        json.dump({"feats": self.feats, "n_iter": self.n_iter}, open(d / "meta.json", "w"))

    @classmethod
    def load(cls, d: Path) -> "LGBModel":
        import lightgbm as lgb
        m = json.load(open(d / "meta.json"))
        return cls(m["feats"], {h: lgb.Booster(model_file=str(d / f"{h}.txt")) for h in HORIZONS}, None)
