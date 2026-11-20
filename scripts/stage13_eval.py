"""Stage 13 — run evaluate() on the cache and print a compact verdict.

usage: stage13_eval.py [--candidates a b c | --family fastlead gap ...] [--out reports/stage13]
Default candidates: every Stage 13 feature. Base: the 46 frozen Stage 9 features.
"""
import argparse, json, sys
from pathlib import Path
import polars as pl
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from hft500 import features_fast as F, incremental as I

ap = argparse.ArgumentParser()
ap.add_argument("--candidates", nargs="*"); ap.add_argument("--family", nargs="*")
ap.add_argument("--instruments", nargs="*", default=["BTCUSDT_PERP", "ETHUSDT_PERP", "SOLUSDT_PERP", "1000PEPEUSDT_PERP"])
ap.add_argument("--cache-dir", default=str(HERE / "cache" / "stage13")); ap.add_argument("--out", default=str(HERE / "reports" / "stage13"))
a = ap.parse_args()
base = json.loads((HERE / "reports/stage9/pruned_features.json").read_text())["features"]
cands = a.candidates or [f for f, v in F.REGISTRY.items() if not a.family or v[0] in a.family]
r = I.evaluate(cands, a.instruments, a.cache_dir, base, out_dir=a.out)
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(20); pl.Config.set_float_precision(4)
print(r["cells"].select("instrument", "horizon", "score_base", "score_full", "gain", "gain_t", "win_days"))
print(r["features"].group_by("feature").agg(pl.col("addone_gain").mean(), pl.col("resid_ic").mean(),
      pl.col("resid_pos").mean(), (pl.col("addone_gain") > 0).sum().alias("cells_up")).sort("addone_gain", descending=True))
print(r["regimes"].group_by("regime").agg(pl.col("gain").mean(), pl.col("share").mean()))
print(json.loads((Path(a.out) / "verdict.json").read_text()))
