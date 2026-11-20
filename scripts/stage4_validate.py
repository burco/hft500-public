"""Stage 4 validation on REAL data:
  official  : our quote grid (no warm-up) vs the assignment's own common.quote_grid, bit-for-bit
  causal    : truncation check of load_day at random cuts + cuts inside the largest book gaps
usage: stage4_validate.py <official|causal> <INST> <DATE> [<DATE> ...]
"""
import sys, os, time, json
from pathlib import Path
import numpy as np, polars as pl, pyarrow.parquet as pq
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / "src"), str(HERE.parent / "assignment_code")]
from hft500 import align as A, causality as C, inventory as inv
import common   # the organisers' helpers

D = HERE.parent
chosen, _ = inv.pick_archives(sorted(D.glob("*.tar")))
pepe = Path.home() / "scratch/data-PEPEUSDT_SPOT.tar"
ROOT = A.ArchiveRoot(list(chosen.values()) + ([pepe] if pepe.exists() else []), fallback=A.DirRoot(D / "data"))

def official(inst, day):
    # route the official loader through our archive reader, then call THEIR quote_grid unchanged
    common.read_day = lambda root, table, date, instrument=None, columns=None: (
        None if (df := ROOT.read(table, date, instrument, columns)) is None else df.to_arrow())
    g, *ref = common.quote_grid("unused", inst, day)
    ours = A.quote_grid(ROOT, inst, day, warmup=False)
    same = {c: bool(np.array_equal(ours[c].to_numpy(), r, equal_nan=True)) for c, r in zip(A.QUOTE_COLS, ref)}
    warm = A.quote_grid(ROOT, inst, day, warmup=True)
    holes_cold = int(np.isnan(ref[0]).sum()); holes_warm = int(warm["bid_px"].is_nan().sum())
    return dict(check="official", inst=inst, day=day, ts_equal=bool(np.array_equal(ours["ts"].to_numpy(), g)),
                **{f"eq_{k}": v for k, v in same.items()}, nan_pts_official=holes_cold, nan_pts_with_warmup=holes_warm)

def causal(inst, day):
    t0 = time.time(); df = A.load_day(ROOT, inst, day); t_load = time.time() - t0
    res = C.truncation_check(lambda r, d: A.load_day(r, inst, d), ROOT, day, n=4, seed=hash((inst, day)) % 2**31,
                             Ts=[A.day_start(day)], inst_for_gaps=inst)
    return dict(check="causal", inst=inst, day=day, rows=df.height, cols=df.width, load_s=round(t_load, 2),
                cuts=len(res), all_ok=all(r["ok"] for r in res), bad=[r for r in res if not r["ok"]])

mode, inst, days = sys.argv[1], sys.argv[2], sys.argv[3:]
for day in days:
    print(json.dumps((official if mode == "official" else causal)(inst, day)), flush=True)
