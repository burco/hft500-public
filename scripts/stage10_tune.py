"""Stage 10 — model comparison on INTERNAL walk-forward folds (training period only; validation untouched).

  linear --inst I        zero / imb_only (~official OLS) / OLS / ridge(lambda grid) on 8 monthly folds,
                         from the Stage 9 per-day moments; scored on the stride-10 cache.
  lgb --stage S          resumable LightGBM job queue (one job = instrument x horizon x fold x config x
                         feature set), stops before the time budget. Stages: tune (fold 2026-05),
                         confirm (folds 2026-03, 2026-07 with the chosen config).
  choose                 picks LightGBM config + rounds on the tune fold (mean of 12 cells) and the
                         ridge lambda on the 8 linear folds; writes choice.json.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, polars as pl
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from hft500 import evaluation as E, features_book as FB, features_cross as X, metric as MT, models as MD

ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["linear", "lgb", "choose"])
ap.add_argument("--inst"); ap.add_argument("--stage", default="tune"); ap.add_argument("--budget", type=float, default=150)
ap.add_argument("--cache-dir", default=str(Path.home() / "scratch/cache"))
a = ap.parse_args()
CACHE = Path(a.cache_dir); STATE9 = CACHE.parent / "stage9_state"
OUT = HERE / "reports/stage10"; FO = OUT / "folds"; FO.mkdir(parents=True, exist_ok=True)
INSTS = ["BTCUSDT_PERP", "ETHUSDT_PERP", "SOLUSDT_PERP", "1000PEPEUSDT_PERP"]
H = list(MD.HORIZONS); FEATS = FB.FEATURES + X.FEATURES
PRUNED = json.load(open(HERE / "reports/stage9/pruned_features.json"))["features"]
LAMS = [1e-6, 0.01, 0.1, 1.0, 10.0]
ITERS = [100, 200, 400, 600, 800]
log = lambda *s: print(f"[{time.strftime('%H:%M:%S')}]", *s, flush=True)
days_of = lambda inst: sorted(p.stem for p in (CACHE / inst).glob("*.parquet"))


def test_days(inst, month):
    return [d for d in days_of(inst) if d[:7] == month]


def train_days(inst, month):
    return next(tr for tr, te in E.month_folds(days_of(inst), "2025-12") if te[0][:7] == month)


def score_rows(preds: dict, df: pl.DataFrame) -> dict:
    out = {}
    for h in H:
        y = df[h].to_numpy().astype(np.float64); m = np.isfinite(y)
        out[h] = MT.pearson(np.where(np.isfinite(preds[h][m]), preds[h][m], 0.0), y[m])
    return out


# ------------------------------------------------------------------ linear on 8 folds
if a.step == "linear":
    inst = a.inst
    z = np.load(STATE9 / f"mom_{inst}.npz")
    ds = list(z["dates"]); lo, hi, ind = z["lo"], z["hi"], z["ind"]
    cum = {k: np.concatenate([np.zeros_like(z[k][:1]), np.cumsum(z[k], 0)]) for k in ("n", "sz", "zz", "sy", "zy")}
    specs = {"imb_only": (["imb_l1"], 1e-3), "ols_full": (FEATS, 1e-6)} | {f"ridge_pruned_{l:g}": (PRUNED, l) for l in LAMS}
    rows = []
    for fi, (tr, te) in enumerate(E.month_folds(ds, "2025-12")):
        i = ds.index(tr[-1]) + 1; M = {k: cum[k][i] for k in cum}
        mods = {n: MD.LinearModel.fit(M, FEATS, lo, hi, ind, u, l) for n, (u, l) in specs.items()}
        for d in te:
            df = pl.read_parquet(CACHE / inst / f"{d}.parquet")
            for n, m in mods.items():
                for h, c in score_rows(m.predict(df), df).items():
                    rows.append((n, inst, h, te[0][:7], d, c))
    pl.DataFrame(rows, schema=["model", "instrument", "horizon", "month", "date", "corr"], orient="row").write_parquet(FO / f"linear_{inst}.parquet")
    log("linear", inst, "done")

# ------------------------------------------------------------------ LightGBM job queue
def jobs(stage):
    if stage == "tune":
        j = [(i, h, "2026-05", c, "pruned") for i in INSTS for h in H for c in ("A", "B")]
        return j + [(i, h, "2026-05", "A", "full") for i in ("BTCUSDT_PERP", "1000PEPEUSDT_PERP") for h in H]   # feature-set check on 6 cells
    ch = json.load(open(OUT / "choice.json"))
    return [(i, h, m, ch["lgb_config"], ch["lgb_feats"]) for m in ("2026-03", "2026-07") for i in INSTS for h in H]


if a.step == "lgb":
    import lightgbm as lgb
    t0 = time.time(); done = 0; last = 0.0
    if a.stage == "confirm":
        ITERS = [json.load(open(OUT / "choice.json"))["lgb_iters"]]                 # train only the chosen rounds
    for inst, h, month, cfg, fs in jobs(a.stage):
        f = FO / f"lgb_{a.stage}_{inst}_{h}_{month}_{cfg}_{fs}.parquet"
        if f.exists():
            continue
        if time.time() - t0 + max(last, 30) * 1.3 > a.budget:
            break
        t1 = time.time()
        feats = PRUNED if fs == "pruned" else FEATS
        tr = pl.concat([pl.read_parquet(CACHE / inst / f"{d}.parquet", columns=feats + [h]).gather_every(4) for d in train_days(inst, month)])
        tr = tr.filter(pl.col(h).is_not_nan())
        params = MD.LGB_BASE | MD.LGB_CONFIGS[cfg]
        dset = lgb.Dataset(tr.select(feats).to_numpy(), tr[h].to_numpy() * MD.Y_SCALE, params={"max_bin": 63}, free_raw_data=True)
        n_rows = tr.height; del tr
        b = lgb.train(params, dset, max(ITERS)); fit_s = time.time() - t1
        rows = []
        for d in test_days(inst, month):
            df = pl.read_parquet(CACHE / inst / f"{d}.parquet", columns=feats + [h])
            y = df[h].to_numpy().astype(np.float64); m = np.isfinite(y); Xd = df.select(feats).to_numpy()[m]
            acc, prev = np.zeros(m.sum()), 0
            for it in ITERS:                                              # staged: each tree evaluated once
                acc = acc + b.predict(Xd, start_iteration=prev, num_iteration=it - prev, raw_score=True); prev = it
                rows.append((inst, h, month, cfg, fs, it, d, MT.pearson(acc, y[m])))
        pl.DataFrame(rows, schema=["instrument", "horizon", "month", "config", "feats", "iters", "date", "corr"], orient="row") \
          .with_columns(pl.lit(fit_s).alias("fit_s"), pl.lit(n_rows).alias("train_rows")).write_parquet(f)
        last = time.time() - t1; done += 1
        log("job", inst, h, month, cfg, fs, f"rows={n_rows} fit={fit_s:.0f}s total={last:.0f}s")
    left = sum(not (FO / f"lgb_{a.stage}_{i}_{h}_{m}_{c}_{fs}.parquet").exists() for i, h, m, c, fs in jobs(a.stage))
    log(f"lgb {a.stage}: {done} jobs this call, {left} left")

# ------------------------------------------------------------------ choose
if a.step == "choose":
    L = pl.read_parquet(str(FO / "linear_*.parquet"))
    lin = L.group_by("model", "instrument", "horizon").agg(pl.col("corr").mean()).group_by("model").agg(pl.col("corr").mean()).sort("corr", descending=True)
    ridge = lin.filter(pl.col("model").str.starts_with("ridge"))
    lam = float(ridge["model"][0].split("_")[-1])
    T = pl.read_parquet(str(FO / "lgb_tune_*.parquet"))
    g = (T.group_by("config", "feats", "iters", "instrument", "horizon").agg(pl.col("corr").mean())
         .group_by("config", "feats", "iters").agg(pl.col("corr").mean()).sort("corr", descending=True))
    best = g.filter(pl.col("feats") == "pruned").row(0, named=True)
    six = T.filter(pl.col("feats") == "full").select("instrument", "horizon").unique()          # the 6 feature-set cells
    fs = (T.join(six, on=["instrument", "horizon"]).filter((pl.col("config") == "A") & (pl.col("iters") == best["iters"]))
          .group_by("feats", "instrument", "horizon").agg(pl.col("corr").mean()).group_by("feats").agg(pl.col("corr").mean()))
    fsd = dict(zip(fs["feats"], fs["corr"])); print("feature-set check (config A, 6 cells):", fsd)
    use_full = best["config"] == "A" and fsd.get("full", -1) > fsd.get("pruned", 0) + 0.001
    choice = dict(featureset_check=fsd, ridge_lambda=lam, lgb_config=best["config"], lgb_iters=int(best["iters"]), lgb_feats="full" if use_full else "pruned",
                  tune_scores=g.to_dicts(), linear_scores=lin.to_dicts())
    json.dump(choice, open(OUT / "choice.json", "w"), indent=1)
    print(lin); print(g)
    log("choice:", {k: v for k, v in choice.items() if not k.endswith("scores")})
