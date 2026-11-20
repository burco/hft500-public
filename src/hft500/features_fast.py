"""Stage 13 — FAST features: 100 ms lead-lag, cross-perp OFI and a vol-scaled lead-lag gap.

Why these (from Stages 8-10):
* Lead-lag is real but short: BTC <- ETH IC 0.164 at lag 0, 0.060 at lag 0.5 s, ~0 at 1 s.
  Stage 8 only measured it on the 500 ms grid. The quote grid is 100 ms, so the freshest
  100-200 ms of a leader's move is still unused information.
* LightGBM's gain over ridge sits almost entirely at 500 ms (+0.027 to +0.045), i.e. in the
  fastest signals. Finer time resolution is the most direct way to feed that.
* Cross-impact (Cont et al., "Cross-Impact of Order Flow Imbalance"): another perp's OFI can
  carry information its RETURN does not show yet (queues move before prices).
* A "gap" feature (others moved, we did not yet) states the lead-lag hypothesis directly,
  in units of each coin's own recent volatility, so BTC and PEPE are comparable.

Causal discipline is the same as features_book / features_cross:
* every value at t uses only records with ts <= t (quote_state is an as-of lookup);
* rolling quantities use backward EWMAs / cumulative sums with a lag;
* warm-up comes from the previous day's files, never from later data;
* nothing known yet about the target itself -> the whole row is NaN.

Caveat carried from Stage 8: ts is the collector's RECEIVE time. A 100 ms "lead" can be feed
latency rather than price discovery. It is usable only while the test period keeps the same
latency structure — check stability by month before trusting it.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import align as A
from .features_book import ewm, lag_diff, safe_ratio
from .features_cross import COIN, ext_grid, log_mid, ofi_increments

STEP_MS = 500
Q_MS = 100
WARM_Q_MS = 10_000                       # 100 ms history before midnight (enough for 0.5 s windows)
FAST = {"0.1s": 1, "0.2s": 2}            # windows in 100 ms steps
XOFI_STEPS = 5                           # cross OFI over 0.5 s
OWN_OFI_STEPS = 2                        # own OFI over 0.2 s (Stage 8's shortest was 0.5 s)
VOL_HL = 600                             # 5 min half-life on the 500 ms grid
DEPTH_HL = 600
GAP_W = {"0.5s": 1, "2s": 4}             # gap windows in 500 ms steps

REGISTRY: dict[str, tuple[str, float, str]] = {
    **{f"xr_{c}_{w}": ("fastlead", n * 0.1, f"{c.upper()} perp mid change over the last {w} (100 ms grid), bps; NaN for own coin")
       for c in COIN.values() for w, n in FAST.items()},
    **{f"r_{w}": ("fastown", n * 0.1, f"own mid change over the last {w} (100 ms grid), bps") for w, n in FAST.items()},
    "ofi_0.2s": ("ofi", 300, "own sampled OFI over the last 200 ms / 5 min EWMA of best-level depth"),
    **{f"xofi_{c}_0.5s": ("crossofi", 300, f"{c.upper()} perp sampled OFI over 0.5 s / its own depth EWMA; NaN for own coin")
       for c in COIN.values()},
    **{f"gap_z_{w}": ("gap", 300, f"mean vol-scaled {w} return of the other 3 perps minus own vol-scaled {w} return")
       for w in GAP_W},
}
FEATURES = list(REGISTRY)


def registry_frame() -> pl.DataFrame:
    return pl.DataFrame([dict(feature=k, family=v[0], lookback_s=float(v[1]), description=v[2])
                         for k, v in REGISTRY.items()])


def _coin_state(root: A.Root, inst: str, date: str, t100: np.ndarray, t_ext: np.ndarray, today: slice) -> dict | None:
    """Everything one perp contributes, on both grids. None when the perp has no quotes at all."""
    ev = A.quote_events(root, inst, date, warmup=True)
    if ev is None:
        return None
    q100 = A.quote_state(ev, t100)
    qe = A.quote_state(ev, t_ext)
    lme = log_mid(qe, len(t_ext))
    r05 = lag_diff(lme, 1) * 1e4
    e = ofi_increments(q100)
    return dict(
        lm100=log_mid(q100, len(t100)),
        cum_ofi=np.concatenate([[0.0], np.cumsum(np.nan_to_num(e, nan=0.0))]),
        depth=ewm((qe["bid_qty"] + qe["ask_qty"]) / 2, DEPTH_HL)[today],
        vol=ewm(np.abs(r05), VOL_HL)[today],                       # typical |0.5 s move|, bps
        ret={w: (lag_diff(lme, k) * 1e4)[today] for w, k in GAP_W.items()},
        age100=q100["quote_age_ms"],
    )


def build(root: A.Root, inst: str, date: str) -> pl.DataFrame:
    """All Stage 13 features for one target instrument-day: always 172,800 rows."""
    t = A.grid(date)
    d0 = A.day_start(date)
    t100 = np.arange(d0 - WARM_Q_MS, d0 + A.DAY_MS, Q_MS, dtype=np.int64)
    k = (t - t100[0]) // Q_MS                                       # index of t on the 100 ms grid
    t_ext = ext_grid(date)
    today = slice(len(t_ext) - len(t), len(t_ext))
    nan = np.full(len(t), np.nan)

    st = {other: _coin_state(root, other, date, t100, t_ext, today) for other in COIN}
    out: dict[str, np.ndarray] = {}

    def fast_ret(s, n):
        return nan if s is None else (s["lm100"][k] - s["lm100"][k - n]) * 1e4

    def ofi(s, n):
        return nan if s is None else safe_ratio(s["cum_ofi"][k + 1] - s["cum_ofi"][k + 1 - n], s["depth"])

    for other, coin in COIN.items():
        own = other == inst
        for w, n in FAST.items():
            out[f"xr_{coin}_{w}"] = nan if own else fast_ret(st[other], n)
        out[f"xofi_{coin}_0.5s"] = nan if own else ofi(st[other], XOFI_STEPS)

    s_own = st[inst]
    for w, n in FAST.items():
        out[f"r_{w}"] = fast_ret(s_own, n)
    out["ofi_0.2s"] = ofi(s_own, OWN_OFI_STEPS)

    for w, steps in GAP_W.items():
        scale = np.sqrt(steps)                                      # |0.5 s move| -> expected |w move|
        zs = []
        for other in COIN:
            s = st[other]
            if other == inst or s is None:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                zs.append(np.where(s["vol"] > 0, s["ret"][w] / (s["vol"] * scale), np.nan))
        if s_own is None or not zs:
            out[f"gap_z_{w}"] = nan
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            z_own = np.where(s_own["vol"] > 0, s_own["ret"][w] / (s_own["vol"] * scale), np.nan)
            stack = np.vstack(zs)
            cnt = np.isfinite(stack).sum(0)
            mean_others = np.where(cnt > 0, np.nansum(stack, 0) / np.maximum(cnt, 1), np.nan)
        out[f"gap_z_{w}"] = mean_others - z_own

    # nothing known yet about the target itself -> every feature NaN (same rule as Stages 7-8)
    no_quote = np.ones(len(t), bool) if s_own is None else ~np.isfinite(s_own["age100"][k])
    cols = {}
    for f in FEATURES:
        v = np.asarray(out[f], dtype=np.float64).copy()
        v[no_quote] = np.nan
        cols[f] = v
    return pl.DataFrame({"ts": t, **cols}).with_columns(pl.lit(inst).alias("instrument"))
