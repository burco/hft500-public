"""Stage 12 — verify the submission exactly as the organisers will run it.

  datadir --date D --dest DIR          materialise the official layout for D and D-1 (no targets/)
  truncate --src DIR --T ms --dest DIR copy DIR keeping only rows with ts <= T (every table)
  score --pred P --date D              official metric of a prediction file against the targets
  compare --full P --trunc Q --T ms    bit-identical check for t <= T
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, polars as pl
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from hft500 import align as A, causality as C, inventory as inv, metric as MT, pipeline as P, targets as T

ap = argparse.ArgumentParser(); ap.add_argument("step"); ap.add_argument("--date"); ap.add_argument("--dest"); ap.add_argument("--src")
ap.add_argument("--T", type=int); ap.add_argument("--pred"); ap.add_argument("--full"); ap.add_argument("--trunc")
a = ap.parse_args()
D = HERE.parent

if a.step == "datadir":
    chosen, _ = inv.pick_archives(sorted(D.glob("*.tar")))
    pepe = Path.home() / "scratch/data-PEPEUSDT_SPOT.tar"
    R = A.ArchiveRoot(list(chosen.values()) + ([pepe] if pepe.exists() else []), fallback=A.DirRoot(D / "data"))
    dest = Path(a.dest); n = 0
    days = [A.prev_date(a.date), a.date]
    for (src, inst, name), _ in R.vpaths.items():
        if name in days and src in ("book", "bbo", "trades") and inst:
            df = R.read(src, name, inst); p = dest / src / inst / f"{name}.parquet"; p.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(p, compression="zstd"); n += 1
    for d in days:
        s = D / "data/stable" / f"{d}.parquet"
        if s.exists():
            (dest / "stable").mkdir(parents=True, exist_ok=True); shutil.copy(s, dest / "stable" / s.name); n += 1
    shutil.copytree(D / "data/meta", dest / "meta", dirs_exist_ok=True)
    print("files", n, "->", dest)

if a.step == "truncate":
    src, dest = Path(a.src), Path(a.dest)
    for p in src.rglob("*.parquet"):
        q = dest / p.relative_to(src); q.parent.mkdir(parents=True, exist_ok=True)
        if p.relative_to(src).parts[0] == "meta":
            shutil.copy(p, q); continue
        pl.read_parquet(p).filter(pl.col("ts") <= a.T).write_parquet(q, compression="zstd")
    print("truncated at", a.T)

if a.step == "score":
    pred = pl.read_parquet(a.pred); clips = T.load_clip(D / "data/meta")
    tg = {i: A.DirRoot(D / "data").read("targets", a.date, i) for i in P.INSTRUMENTS}
    if any(v is None for v in tg.values()):
        chosen, _ = inv.pick_archives(sorted(D.glob("*.tar")))
        R = A.ArchiveRoot(list(chosen.values()), fallback=A.DirRoot(D / "data"))
        tg = {i: R.read("targets", a.date, i) for i in P.INSTRUMENTS}
    rows = MT.score_predictions(pred, tg, clips, a.date)
    print(json.dumps(rows)); print("day mean", round(float(np.mean([r["corr"] for r in rows])), 6))

if a.step == "compare":
    f, t = P.wide(pl.read_parquet(a.full)), P.wide(pl.read_parquet(a.trunc))
    r = C.compare(f, t, a.T); r = {k: (v.item() if isinstance(v, (np.integer, np.bool_)) else v) for k, v in r.items()}
    print(json.dumps(r))
