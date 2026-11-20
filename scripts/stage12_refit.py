"""Stage 12 — ONE refit of the frozen Stage 10 model on train + validation (the brief allows it).
Hyperparameters are read from reports/stage10/choice.json and not changed.

usage: stage12_refit.py --inst I --out DIR
"""
import argparse, json, sys, time
from pathlib import Path
import polars as pl
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from hft500 import models as MD, spec

ap = argparse.ArgumentParser(); ap.add_argument("--inst", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
import lightgbm as lgb
CH = json.load(open(HERE / "reports/stage10/choice.json"))
feats = json.load(open(HERE / "reports/stage9/pruned_features.json"))["features"]
H = list(MD.HORIZONS)
files = sorted((Path.home() / "scratch/cache" / a.inst).glob("*.parquet")) + sorted((Path.home() / "scratch/cache_valid" / a.inst).glob("*.parquet"))
dates = [p.stem for p in files]
assert dates[0] == spec.TRAIN_START.isoformat() and dates[-1] == spec.VALID_END.isoformat() and len(set(dates)) == len(dates) == 365, (dates[0], dates[-1], len(dates))
t = time.time()
tr = pl.concat([pl.read_parquet(p, columns=feats + H).gather_every(4) for p in files])
params = MD.LGB_BASE | MD.LGB_CONFIGS[CH["lgb_config"]]
bs = {}
for h in H:
    g = tr.filter(pl.col(h).is_not_nan())
    bs[h] = lgb.train(params, lgb.Dataset(g.select(feats).to_numpy(), g[h].to_numpy() * MD.Y_SCALE, params={"max_bin": 63}), CH["lgb_iters"])
m = MD.LGBModel(feats, bs, CH["lgb_iters"]); out = Path(a.out) / a.inst; m.save(out)
info = dict(instrument=a.inst, days=len(dates), first=dates[0], last=dates[-1], rows=tr.height, config=CH["lgb_config"],
            num_leaves=params["num_leaves"], iters=CH["lgb_iters"], params={k: v for k, v in params.items()}, train_s=round(time.time() - t, 1))
json.dump(info, open(out / "fit_info.json", "w"), indent=1)
print(info["instrument"], info["days"], info["rows"], info["train_s"], "s")
