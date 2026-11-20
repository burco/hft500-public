"""Run the organisers' baseline.py fit() and score() UNCHANGED for one instrument.
Only common.read_day is redirected to read inside the .tar archives.
usage: stage6_official.py <INST>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / "src"), str(HERE.parent / "assignment_code")]
from hft500 import align as A, inventory as inv
import common
D = HERE.parent
chosen, _ = inv.pick_archives(sorted(D.glob("*.tar")))
ROOT = A.ArchiveRoot(list(chosen.values()), fallback=A.DirRoot(D / "data"))
common.read_day = lambda root, table, date, instrument=None, columns=None: (
    None if (df := ROOT.read(table, date, instrument, columns)) is None else df.to_arrow())
import baseline                                   # imports the patched read_day
inst = sys.argv[1]
clips = common.load_clip(str(D / "data"))
clip = {h: clips[(inst, h)] for h in common.HORIZONS}
t0 = time.time()
coef = baseline.fit("unused", inst, ("2025-09-15", "2026-07-14"), clip)
t1 = time.time()
rho = baseline.score("unused", inst, ("2026-07-15", "2026-09-14"), clip)
out = dict(instrument=inst, coef={h: list(v) for h, v in coef.items()}, rho=rho, fit_s=round(t1 - t0, 1), score_s=round(time.time() - t1, 1))
(HERE / "reports/stage6" / f"official_{inst}.json").write_text(json.dumps(out))
import numpy as np
print(inst, {h: (f"slope={coef[h][1]:.3e}", f"corr={np.mean(rho[h]):.4f}", f"days={len(rho[h])}") for h in common.HORIZONS}, out["fit_s"], out["score_s"])
