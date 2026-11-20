"""Re-check prices against the tick with float noise removed, plus the tick seen in the data."""
import sys, os, time, glob, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, polars as pl, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hft500 import inventory as inv, quality as q
from hft500.tarsource import index_tar, open_any

def one(vpath, inst, day, tick):
    src = open_any(vpath); t = pq.read_table(src, columns=["bid_px", "ask_px"]); src.close()
    px = np.concatenate([t["bid_px"].to_numpy(), t["ask_px"].to_numpy()])
    emp = q.empirical_tick(px)
    return dict(instrument=inst, date=day, meta_tick=q.clean_tick(tick), empirical_tick=emp,
                off_tick_clean=q._off_tick(px, tick), off_tick_vs_empirical=q._off_tick(px, emp), values=int(np.isfinite(px).sum()))

if __name__ == "__main__":
    D, insts, out = Path(sys.argv[1]), sys.argv[2].split(","), Path(sys.argv[3])
    chosen, _ = inv.pick_archives(sorted(D.glob("*.tar")))
    meta = pl.read_parquet(D / "data/meta/daily.parquet")
    ticks = {(r["instrument"], r["date"]): r["tick_size"] for r in meta.iter_rows(named=True)}
    done = set(pl.read_csv(out)["instrument"].to_list()) if out.exists() else set()
    rows = pl.read_csv(out).to_dicts() if out.exists() else []
    for inst in insts:
        if inst in done: continue
        jobs = [(r["path"], inst, inv.parse_date(Path(r["member"])).isoformat()) for r in index_tar(chosen[inst]) if r["source"] == "book"]
        with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as ex:
            rows += list(ex.map(one, *zip(*[(p, i, d, ticks[(i, d)]) for p, i, d in jobs])))
        pl.DataFrame(rows).write_csv(out); print(inst, "done", flush=True)
