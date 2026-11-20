"""Stage 8 — EXPLORATORY daily IC of trade-flow / cross-market features on TRAINING days only,
plus 'residual IC': correlation with what the target does AFTER removing what imb_l1 explains
that day (does the feature carry information the baseline lacks?). Not feature selection (Stage 9).

usage: stage8_explore.py <INST> [--every 14]
"""
import argparse, datetime as dt, sys, time
from pathlib import Path
import numpy as np, polars as pl
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from hft500 import align as A, features_book as FB, features_cross as X, inventory as inv, spec, targets as T

ap = argparse.ArgumentParser(); ap.add_argument("inst"); ap.add_argument("--every", type=int, default=14)
a = ap.parse_args()
D = HERE.parent
chosen, _ = inv.pick_archives(sorted(D.glob("*.tar")))
pepe = Path.home() / "scratch/data-PEPEUSDT_SPOT.tar"
R = A.ArchiveRoot(list(chosen.values()) + ([pepe] if pepe.exists() else []), fallback=A.DirRoot(D / "data"))
clips = T.load_clip(D / "data" / "meta")
meta = pl.read_parquet(D / "data/meta/daily.parquet").filter(pl.col("instrument") == a.inst)
mt = dict(zip(meta["date"], meta["tick_size"]))
days = [(spec.TRAIN_START + dt.timedelta(days=i)).isoformat() for i in range(3, (spec.TRAIN_END - spec.TRAIN_START).days + 1, a.every)]
rows, t0 = [], time.time()
for d in days:
    fb = FB.build(R, a.inst, d, meta_tick=mt.get(d)).select("ts", "imb_l1", "imb_l1_chg_1s", "ret_0.5s")
    fx = X.build(R, a.inst, d).drop("instrument")
    tg = T.clip_targets(R.read("targets", d, a.inst), a.inst, clips)
    g = fb.join(fx, on="ts").join(tg, on="ts", how="inner")
    imb = g["imb_l1"].to_numpy()
    for h in T.HORIZONS:
        y = g[h].to_numpy()
        m0 = np.isfinite(imb) & np.isfinite(y)
        b = np.polyfit(imb[m0], y[m0], 1)
        resid = np.where(m0, y - np.polyval(b, imb), np.nan)
        for col in X.FEATURES + ["imb_l1_chg_1s", "ret_0.5s"]:
            x = g[col].to_numpy().astype(float)
            m = np.isfinite(x) & np.isfinite(y)
            ok = m.sum() > 1000 and x[m].std() > 0
            ic = np.corrcoef(x[m], y[m])[0, 1] if ok else np.nan
            mr = m & np.isfinite(resid)
            ric = np.corrcoef(x[mr], resid[mr])[0, 1] if ok else np.nan
            rows.append(dict(instrument=a.inst, date=d, feature=col, horizon=h, ic=ic, resid_ic=ric, coverage=m.mean()))
out = HERE / "reports/stage8"; out.mkdir(parents=True, exist_ok=True)
pl.DataFrame(rows).write_parquet(out / f"ic_{a.inst}.parquet")
print(f"{a.inst}: {len(days)} training days in {time.time()-t0:.0f}s")
