"""Stage 7 — EXPLORATORY daily correlations of the book features, TRAINING days only.
Not feature selection (that is Stage 9): a sanity check that features carry signal and sign.

usage: stage7_explore.py <INST> [--every 14]
"""
import argparse, datetime as dt, sys, time
from pathlib import Path
import numpy as np, polars as pl
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from hft500 import align as A, baseline_ols as B, features_book as F, inventory as inv, spec, targets as T

ap = argparse.ArgumentParser(); ap.add_argument("inst"); ap.add_argument("--every", type=int, default=14)
a = ap.parse_args()
D = HERE.parent
chosen, _ = inv.pick_archives(sorted(D.glob("*.tar")))
R = A.ArchiveRoot(list(chosen.values()), fallback=A.DirRoot(D / "data"))
clips = T.load_clip(D / "data" / "meta")
meta = pl.read_parquet(D / "data/meta/daily.parquet").filter(pl.col("instrument") == a.inst)
mt = dict(zip(meta["date"], meta["tick_size"]))
days = [(spec.TRAIN_START + dt.timedelta(days=i)).isoformat() for i in range(3, (spec.TRAIN_END - spec.TRAIN_START).days + 1, a.every)]
rows, t0 = [], time.time()
for d in days:
    f = F.build(R, a.inst, d, meta_tick=mt.get(d))
    bk = R.read("book", d, a.inst, ["ts", "bid_qty", "ask_qty"])
    f = f.join(bk.with_columns(B_imb_book=pl.lit(None)).select("ts").with_columns(
        imb_book=pl.Series(B.imbalance(bk["bid_qty"].cast(pl.Float64).to_numpy(), bk["ask_qty"].cast(pl.Float64).to_numpy()))), on="ts", how="left")
    tg = T.clip_targets(R.read("targets", d, a.inst), a.inst, clips)
    g = f.join(tg, on="ts", how="inner")
    for col in F.FEATURES + ["imb_book"]:
        x = g[col].to_numpy().astype(float)
        for h in T.HORIZONS:
            y = g[h].to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            c = np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 1000 and x[m].std() > 0 else np.nan
            rows.append(dict(instrument=a.inst, date=d, feature=col, horizon=h, ic=c, coverage=m.mean()))
out = HERE / "reports/stage7"; out.mkdir(parents=True, exist_ok=True)
pl.DataFrame(rows).write_parquet(out / f"ic_{a.inst}.parquet")
print(f"{a.inst}: {len(days)} training days in {time.time()-t0:.0f}s")
