"""Stage 5 — target analysis across the year (every k-th day), one instrument per call.

usage: stage5_analyze.py <INST> [--every 7] [--out reports/stage5]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from hft500 import align as A, inventory as inv, quality as Q, spec, targets as T  # noqa: E402

LAGS_500 = [1, 2, 3, 4]
LAGS_8S = [1, 4, 8, 15, 16, 17, 20]


def analyze_day(R, inst: str, day: str, clips: dict) -> dict:
    tg = R.read("targets", day, inst).sort("ts")
    t = tg["ts"].to_numpy()
    nxt = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
    parts = [A.quote_events(R, inst, day, warmup=True)]
    e2 = A.quote_events(R, inst, nxt, warmup=False)          # labels look 8 s past midnight
    if e2 is not None:
        parts.append(e2)
    ev = pl.concat(parts).sort("ts", maintain_order=True)
    ets = ev["ts"].to_numpy()
    emid = ((ev["bid_px"] + ev["ask_px"]) / 2).to_numpy()
    tick = Q.empirical_tick(ev["bid_px"].to_numpy())
    rb = T.rebuild(ets, emid, t, 10)
    out = dict(instrument=inst, date=day, split="train" if day <= spec.TRAIN_END.isoformat() else "valid", tick=tick)
    for h in T.HORIZONS:
        s = tg[h].to_numpy()
        ok = np.isfinite(s) & np.isfinite(rb[h])
        out[f"{h}_rebuild_exact_pct"] = float(np.mean(np.abs(s[ok] - rb[h][ok]) < 1e-12) * 100)
        out[f"{h}_rebuild_corr"] = float(np.corrcoef(s[ok], rb[h][ok])[0, 1])
        lo, hi = clips[(inst, h)]
        for k, v in T.tail_share(s, lo, hi).items():
            out[f"{h}_{k}"] = v
        out[f"{h}_std_bps"] = float(np.nanstd(s) * T.BPS)
        out[f"{h}_zero_pct"] = float(np.mean(s[np.isfinite(s)] == 0) * 100)
    y5 = tg["y_500ms"].to_numpy()
    for k, v in T.discreteness(y5, T.mid_asof(ets, emid, t + 10), tick).items():
        out[f"y_500ms_{k}"] = v
    g = np.log(T.mid_asof(ets, emid, t))
    lm = T.implied_log_mid(y5, g)
    d = (lm - g) * T.BPS
    fin = np.isfinite(d)
    out["label_mid_eq_grid_mid_pct"] = float(np.mean(np.abs(d[fin]) < 1e-6) * 100)
    out["label_mid_minus_grid_bps_p99_abs"] = float(np.quantile(np.abs(d[fin]), 0.99))
    for h, e in T.additivity_error(tg).items():
        out[f"additivity_maxerr_{h}"] = e
    for L, a in zip(LAGS_500, T.acf(y5, LAGS_500)):
        out[f"acf500_lag{L}"] = float(a)
    for L, a in zip(LAGS_8S, T.acf(tg["y_8s"].to_numpy(), LAGS_8S)):
        out[f"acf8s_lag{L}"] = float(a)
    out["nan_first_10"] = int(np.isnan(y5[:10]).sum())
    out["nan_last_16_y8s"] = int(np.isnan(tg["y_8s"].to_numpy()[-16:]).sum())
    out["nan_total_y500"] = int(np.isnan(y5).sum())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inst")
    ap.add_argument("--every", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("reports/stage5"))
    a = ap.parse_args(argv)
    D = HERE.parent
    chosen, _ = inv.pick_archives(sorted(D.glob("*.tar")))
    R = A.ArchiveRoot(list(chosen.values()), fallback=A.DirRoot(D / "data"))
    clips = T.load_clip(D / "data" / "meta")
    days = [(spec.TRAIN_START + dt.timedelta(days=i)).isoformat()
            for i in range(0, (spec.VALID_END - spec.TRAIN_START).days + 1, a.every)]
    t0 = time.time()
    rows = [analyze_day(R, a.inst, d, clips) for d in days]
    a.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(a.out / f"days_{a.inst}.parquet")
    print(f"{a.inst}: {len(rows)} days in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
