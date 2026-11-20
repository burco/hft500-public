"""Stage 13 — cache for the incremental evaluation: base features (Stage 7 + 8) + Stage 13 fast
features + clipped targets, stride-subsampled, float32, one file per instrument-day.
Resumable and time-boxed (re-run until it prints 'all done').

usage: stage13_build.py <INST> [--every 14] [--stride 10] [--cache-dir DIR] [--max-seconds 150]
Training days only: every k-th day from 2025-09-15 to 2026-07-14.
"""
import argparse, datetime as dt, sys, time
from pathlib import Path
import numpy as np, polars as pl
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from hft500 import align as A, features_book as FB, features_cross as X, features_fast as F, inventory as inv, spec, targets as T

ap = argparse.ArgumentParser(); ap.add_argument("inst")
ap.add_argument("--every", type=int, default=14); ap.add_argument("--stride", type=int, default=10)
ap.add_argument("--cache-dir", default=str(HERE / "cache" / "stage13")); ap.add_argument("--max-seconds", type=float, default=150)
a = ap.parse_args()
D = HERE.parent
chosen, _ = inv.pick_archives(sorted(D.glob("*.tar")))
R = A.ArchiveRoot(list(chosen.values()), fallback=A.DirRoot(D / "data"))
clips = T.load_clip(D / "data" / "meta")
meta = pl.read_parquet(D / "data/meta/daily.parquet").filter(pl.col("instrument") == a.inst)
mt = dict(zip(meta["date"], meta["tick_size"]))
s, e = spec.TRAIN_START, spec.TRAIN_END
days = [(s + dt.timedelta(days=i)).isoformat() for i in range(0, (e - s).days + 1, a.every)]
out = Path(a.cache_dir) / a.inst; out.mkdir(parents=True, exist_ok=True)
t0, done = time.time(), 0
for d in days:
    if (out / f"{d}.parquet").exists():
        continue
    if time.time() - t0 > a.max_seconds:
        print(f"{a.inst}: time box hit after {done} days; re-run to continue", flush=True); break
    tg = R.read("targets", d, a.inst)
    if tg is None:
        print("no targets", d, flush=True); continue
    tg = T.clip_targets(tg, a.inst, clips)
    g = (FB.build(R, a.inst, d, meta_tick=mt.get(d)).drop("instrument")
         .join(X.build(R, a.inst, d).drop("instrument"), on="ts")
         .join(F.build(R, a.inst, d).drop("instrument"), on="ts")
         .join(tg.select("ts", *T.HORIZONS), on="ts", how="left"))
    g.gather_every(a.stride).with_columns(pl.exclude("ts").cast(pl.Float32)) \
     .write_parquet(out / f"{d}.tmp", compression="zstd")
    (out / f"{d}.tmp").rename(out / f"{d}.parquet")
    done += 1
    print(f"{a.inst} {d} {time.time()-t0:.0f}s", flush=True)
else:
    print(f"{a.inst}: all done ({len(list(out.glob('*.parquet')))} days)", flush=True)
