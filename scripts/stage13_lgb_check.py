"""Stage 13 — the same split as stage13_eval.py, but with the production model class:
LightGBM config B (Stage 10/12 settings) on base 46 vs base 46 + candidates.

usage: stage13_lgb_check.py <INST> [--candidates ...] [--out reports/stage13/lgb_<INST>.csv]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, polars as pl
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from hft500 import features_fast as F, incremental as I, models as M
from hft500.metric import pearson

ap = argparse.ArgumentParser(); ap.add_argument("inst"); ap.add_argument("--candidates", nargs="*")
ap.add_argument("--cache-dir", default=str(HERE / "cache" / "stage13")); ap.add_argument("--rounds", type=int, default=100)
a = ap.parse_args()
base = json.loads((HERE / "reports/stage9/pruned_features.json").read_text())["features"]
cands = a.candidates or [f for f in F.FEATURES if not f.startswith("gap_z")]
files = sorted(Path(a.cache_dir, a.inst).glob("*.parquet"))
df = pl.concat([pl.read_parquet(f).with_columns(pl.lit(f.stem).alias("date")) for f in files], how="vertical_relaxed")
train, test = I.split_days(df["date"].to_list(), 0.35, 1)
tr, te = df.filter(pl.col("date").is_in(train)), df.filter(pl.col("date").is_in(test))
params = {**M.LGB_BASE, **M.LGB_CONFIGS["B"]}
Y = tr.select(M.HORIZONS).to_numpy().astype(np.float64)
rows = []
for name, feats in (("base", base), ("base+fast", base + cands)):
    m = M.LGBModel.fit(tr.select(feats).to_numpy().astype(np.float32), Y, feats, params, a.rounds)
    p = m.predict(te)
    for h in M.HORIZONS:
        y = te[h].to_numpy().astype(np.float64)
        for d in test:
            k = (te["date"] == d).to_numpy() & np.isfinite(y)
            rows.append(dict(model=name, horizon=h, date=d, corr=pearson(p[h][k], y[k])))
r = pl.DataFrame(rows)
w = r.pivot("model", index=["horizon", "date"], values="corr").with_columns((pl.col("base+fast") - pl.col("base")).alias("gain"))
s = w.group_by("horizon").agg(pl.col("base").mean(), pl.col("base+fast").mean(), pl.col("gain").mean(),
                              (pl.col("gain").mean() / (pl.col("gain").std() / pl.len().sqrt())).alias("gain_t"),
                              (pl.col("gain") > 0).mean().alias("win_days")).sort("horizon").with_columns(pl.lit(a.inst).alias("instrument"))
out = HERE / "reports/stage13"; out.mkdir(parents=True, exist_ok=True)
s.write_csv(out / f"lgb_{a.inst}.csv")
pl.Config.set_float_precision(4); print(s)
