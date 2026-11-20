"""Stage 9 — for every TRAINING day of one instrument: build Stage 7 + Stage 8 features on the full
500 ms grid, compute per-day Pearson / Spearman / decile stats against the clipped targets, and save a
stride-subsampled float32 cache (features + targets) for clustering and fold ablations (also Stage 10).
Resumable: days already done are skipped.

usage: stage9_build.py <INST> [--stride 10] [--cache-dir DIR] [--limit N]
"""
import argparse, datetime as dt, sys, time
from pathlib import Path
import numpy as np, polars as pl
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from hft500 import align as A, evaluation as E, features_book as FB, features_cross as X, inventory as inv, spec, targets as T

ap = argparse.ArgumentParser(); ap.add_argument("inst"); ap.add_argument("--stride", type=int, default=10)
ap.add_argument("--cache-dir", default=str(HERE / "cache" / "features")); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--period", default="train"); ap.add_argument("--max-seconds", type=float, default=0)
a = ap.parse_args()
D = HERE.parent
chosen, _ = inv.pick_archives(sorted(D.glob("*.tar")))
pepe = Path.home() / "scratch/data-PEPEUSDT_SPOT.tar"
R = A.ArchiveRoot(list(chosen.values()) + ([pepe] if pepe.exists() else []), fallback=A.DirRoot(D / "data"))
clips = T.load_clip(D / "data" / "meta")
meta = pl.read_parquet(D / "data/meta/daily.parquet").filter(pl.col("instrument") == a.inst)
mt = dict(zip(meta["date"], meta["tick_size"]))
s, e = (spec.TRAIN_START, spec.TRAIN_END) if a.period == "train" else (spec.VALID_START, spec.VALID_END)
days = [(s + dt.timedelta(days=i)).isoformat() for i in range((e - s).days + 1)]
stats_dir = HERE / "reports/stage9/daily" / a.inst; stats_dir.mkdir(parents=True, exist_ok=True)
cache_dir = Path(a.cache_dir) / a.inst; cache_dir.mkdir(parents=True, exist_ok=True)
FEATS = FB.FEATURES + X.FEATURES
done = 0; t0 = time.time()
for d in days:
    if (cache_dir / f"{d}.parquet").exists():
        continue
    if a.max_seconds and time.time() - t0 > a.max_seconds:
        break
    tg = R.read("targets", d, a.inst)
    if tg is None:
        print("no targets", d, flush=True); continue
    tg = T.clip_targets(tg, a.inst, clips)
    g = (FB.build(R, a.inst, d, meta_tick=mt.get(d)).drop("instrument")
         .join(X.build(R, a.inst, d).drop("instrument"), on="ts")
         .join(tg.select("ts", *T.HORIZONS), on="ts", how="left"))
    if a.period == "train":
        rows = E.day_stats({f: g[f].to_numpy().astype(np.float64) for f in FEATS},
                           {h: g[h].to_numpy().astype(np.float64) for h in T.HORIZONS})
        pl.DataFrame(rows).with_columns(pl.lit(d).alias("date"), pl.lit(a.inst).alias("instrument")) \
          .write_parquet(stats_dir / f"{d}.parquet")
    g.gather_every(a.stride).with_columns(pl.exclude("ts").cast(pl.Float32)) \
     .write_parquet(cache_dir / f"{d}.tmp", compression="zstd")
    (cache_dir / f"{d}.tmp").rename(cache_dir / f"{d}.parquet")         # atomic: a killed run never leaves a half file
    done += 1
    if a.limit and done >= a.limit:
        break
    if done % 10 == 0:
        print(f"{a.inst} {d} done={done} {time.time()-t0:.0f}s", flush=True)
print(f"{a.inst}: built {done} days in {time.time()-t0:.0f}s", flush=True)
