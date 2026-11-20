"""Stage 6 — reproduce the official baseline with OUR code and score it two ways.

  compat  : baseline.py's own scoring (x = imb on rows with a book snapshot, days with < 10,000 points skipped)
            -> must match reports/stage6/official_<INST>.json to 1e-6
  official: the PDF metric on predict.py-style output (full 172,800-point grid, missing = 0, no day skipped)

usage: stage6_ours.py <INST>
"""
import json, sys, time
from pathlib import Path
import numpy as np, polars as pl
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from hft500 import align as A, baseline_ols as B, inventory as inv, metric as M, targets as T

D = HERE.parent
chosen, _ = inv.pick_archives(sorted(D.glob("*.tar")))
R = A.ArchiveRoot(list(chosen.values()), fallback=A.DirRoot(D / "data"))
inst = sys.argv[1]
clips = T.load_clip(D / "data" / "meta")
clip = {h: clips[(inst, h)] for h in B.HORIZONS}
off = json.loads((HERE / "reports/stage6" / f"official_{inst}.json").read_text())

t0 = time.time()
coef = B.fit(R, inst, "2025-09-15", "2026-07-14", clip)
coef_diff = {h: [abs(coef[h][k] - off["coef"][h][k]) / max(abs(off["coef"][h][k]), 1e-300) for k in (0, 1)] for h in B.HORIZONS}

rows = []
for d in B.dates("2026-07-15", "2026-09-14"):
    # compat mode: correlate raw imbalance with clipped targets on book rows (what baseline.score does)
    r = B.imbalance_and_targets(R, inst, d, clip)
    if r is not None:
        imb, ys = r
        for h in B.HORIZONS:
            m = np.isfinite(imb) & np.isfinite(ys[h])
            if m.sum() >= 10_000:
                rows.append(dict(mode="compat", date=d, instrument=inst, horizon=h, corr=M.pearson(imb[m], ys[h][m])))
    # official mode: full-grid predictions exactly as predict.py would emit them
    g, p = B.predict_day(R, inst, d, coef)
    tg = R.read("targets", d, inst)
    pred = pl.DataFrame({"ts": g, "instrument": inst, **{M.PRED_COLS[h]: p[h] for h in B.HORIZONS}})
    rows += [dict(mode="official", **x) for x in M.score_predictions(pred, {inst: tg}, clips, d)]
df = pl.DataFrame(rows)
df.write_parquet(HERE / "reports/stage6" / f"ours_{inst}.parquet")

# compare compat per-day correlations with the official per-day list
maxdiff = 0.0
for h in B.HORIZONS:
    ours = df.filter((pl.col("mode") == "compat") & (pl.col("horizon") == h)).sort("date")["corr"].to_numpy()
    theirs = np.array(off["rho"][h])
    assert len(ours) == len(theirs), (h, len(ours), len(theirs))
    maxdiff = max(maxdiff, float(np.max(np.abs(ours - theirs))))
print(json.dumps(dict(instrument=inst, coef_max_rel_diff=max(max(v) for v in coef_diff.values()),
                      compat_per_day_max_abs_diff=maxdiff, seconds=round(time.time() - t0))))
