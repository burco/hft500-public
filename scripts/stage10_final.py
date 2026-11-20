"""Stage 10 — final fits on the whole TRAINING period, then ONE logged look at the validation period.

  fit --inst I          linear models from the Stage 9 per-day moments (all 303 training days) and the
                        chosen LightGBM config per horizon (cache rows every 4th = every 40th grid point)
  validate --budget S   resumable: for each validation day x instrument, build features on the FULL grid,
                        predict with every model, score with the official metric (clipped targets,
                        missing prediction = 0); records feature/prediction time and peak memory
  report                results table: 12 per-combo scores, aggregate, train/inference time, memory, size
"""
import argparse, datetime as dt, json, resource, sys, time
from pathlib import Path
import numpy as np, polars as pl
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from hft500 import align as A, features_book as FB, features_cross as X, inventory as inv, metric as MT, models as MD, spec, targets as T

ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["fit", "validate", "report"])
ap.add_argument("--inst"); ap.add_argument("--budget", type=float, default=150); ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--workers", type=int, default=1); ap.add_argument("--cache-dir", default=str(Path.home() / "scratch/cache"))
a = ap.parse_args()
CACHE = Path(a.cache_dir); STATE9 = CACHE.parent / "stage9_state"
OUT = HERE / "reports/stage10"; MOD = HERE / "models/stage10"; MOD.mkdir(parents=True, exist_ok=True)
INSTS = ["BTCUSDT_PERP", "ETHUSDT_PERP", "SOLUSDT_PERP", "1000PEPEUSDT_PERP"]
H = list(MD.HORIZONS); FEATS = FB.FEATURES + X.FEATURES
PRUNED = json.load(open(HERE / "reports/stage9/pruned_features.json"))["features"]
CH = json.load(open(OUT / "choice.json"))
LIN = {"imb_only": (["imb_l1"], 1e-3), "ols_full": (FEATS, 1e-6), "ridge_pruned": (PRUNED, CH["ridge_lambda"])}
log = lambda *s: print(f"[{time.strftime('%H:%M:%S')}]", *s, flush=True)

if a.step == "fit":
    import lightgbm as lgb
    inst = a.inst; d = MOD / inst; d.mkdir(exist_ok=True); info = {}
    z = np.load(STATE9 / f"mom_{inst}.npz")
    M = {k: z[k].sum(0) for k in ("n", "sz", "zz", "sy", "zy")}
    for n, (u, l) in LIN.items():
        t = time.time(); m = MD.LinearModel.fit(M, FEATS, z["lo"], z["hi"], z["ind"], u, l); m.save(d / f"{n}.json")
        info[n] = dict(train_s=time.time() - t, n_params=m.n_params(), note="from per-day moments (moment pass ~60 s/instrument, Stage 9)")
    feats = PRUNED if CH["lgb_feats"] == "pruned" else FEATS
    params = MD.LGB_BASE | MD.LGB_CONFIGS[CH["lgb_config"]]
    t = time.time()
    tr = pl.concat([pl.read_parquet(p, columns=feats + H).gather_every(4) for p in sorted((CACHE / inst).glob("*.parquet"))])
    bs = {}
    for h in H:
        g = tr.filter(pl.col(h).is_not_nan())
        bs[h] = lgb.train(params, lgb.Dataset(g.select(feats).to_numpy(), g[h].to_numpy() * MD.Y_SCALE, params={"max_bin": 63}), CH["lgb_iters"])
    g = MD.LGBModel(feats, bs, CH["lgb_iters"]); g.save(d / "lgb")
    info["lgb"] = dict(train_s=time.time() - t, n_params=g.n_params(), train_rows=tr.height, config=CH["lgb_config"], iters=CH["lgb_iters"])
    json.dump(info, open(d / "fit_info.json", "w"), indent=1)
    log("fit", inst, {k: round(v["train_s"], 1) for k, v in info.items()})

if a.step == "validate":
    D = HERE.parent
    chosen, _ = inv.pick_archives(sorted(D.glob("*.tar")))
    pepe = Path.home() / "scratch/data-PEPEUSDT_SPOT.tar"
    R = A.ArchiveRoot(list(chosen.values()) + ([pepe] if pepe.exists() else []), fallback=A.DirRoot(D / "data"))
    clips = T.load_clip(D / "data" / "meta")
    meta = pl.read_parquet(D / "data/meta/daily.parquet")
    vlog = OUT / "validation_log.csv"
    if not vlog.exists():
        pl.DataFrame({"opened_utc": [dt.datetime.utcnow().isoformat(timespec="seconds")], "purpose": ["Stage 10 final comparison (single look; models fixed beforehand)"],
                      "models": [",".join(["zero", *LIN, "lgb"])], "choice": [json.dumps({k: v for k, v in CH.items() if not k.endswith("scores")})]}).write_csv(vlog)
    days = [(spec.VALID_START + dt.timedelta(days=i)).isoformat() for i in range((spec.VALID_END - spec.VALID_START).days + 1)]
    jobs = [(i, d) for d in days for i in INSTS][a.worker::a.workers]
    VD = OUT / "valid_daily"; VD.mkdir(exist_ok=True)
    models = {}
    for inst in INSTS:
        models[inst] = {n: MD.LinearModel.load(MOD / inst / f"{n}.json") for n in LIN} | {"lgb": MD.LGBModel.load(MOD / inst / "lgb")}
    t0 = time.time(); n = 0
    for inst, d in jobs:
        f = VD / f"{inst}_{d}.parquet"
        if f.exists():
            continue
        if time.time() - t0 > a.budget:
            break
        tg = R.read("targets", d, inst)
        if tg is None:
            log("no targets", inst, d); continue
        mt = dict(zip(*meta.filter(pl.col("instrument") == inst).select("date", "tick_size").to_dict(as_series=False).values()))
        t1 = time.time()
        fr = FB.build(R, inst, d, meta_tick=mt.get(d)).drop("instrument").join(X.build(R, inst, d).drop("instrument"), on="ts")
        t_feat = time.time() - t1
        g = tg.select("ts", *H).join(fr, on="ts", how="right") if False else fr.join(tg.select("ts", *H), on="ts", how="left")
        rows = []
        for name, m in models[inst].items():
            t2 = time.time()
            if name == "lgb":
                Xg = g.select(m.feats).to_numpy().astype(np.float32)
                p = {h: m.boosters[h].predict(Xg, num_threads=1) for h in H}
            else:
                p = m.predict(g)
            t_pred = time.time() - t2
            for h in H:
                y = g[h].to_numpy().astype(np.float64)
                c = MT.score_day(p[h], y, *clips[(inst, h)])
                rows.append((name, inst, h, d, c, int(np.isfinite(y).sum()), bool(np.isfinite(p[h]).all()), float(np.std(p[h])), t_feat, t_pred))
        for h in H:
            rows.append(("zero", inst, h, d, 0.0, int(np.isfinite(g[h].to_numpy()).sum()), True, 0.0, 0.0, 0.0))
        pl.DataFrame(rows, schema=["model", "instrument", "horizon", "date", "corr", "n_targets", "all_finite", "pred_sd", "feat_s", "pred_s"], orient="row") \
          .with_columns(pl.lit(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024).alias("maxrss_mb")).write_parquet(f)
        n += 1
    left = sum(not (VD / f"{i}_{d}.parquet").exists() for d in days for i in INSTS)
    log(f"worker {a.worker}: {n} instrument-days this call; {left} left overall")

if a.step == "report":
    V = pl.read_parquet(str(OUT / "valid_daily/*.parquet"))
    cells = V.group_by("model", "instrument", "horizon").agg(pl.col("corr").mean(), pl.len().alias("days"))
    wide = cells.pivot("model", index=["instrument", "horizon"], values="corr").sort("instrument", "horizon")
    agg = cells.group_by("model").agg(pl.col("corr").mean().alias("score")).sort("score", descending=True)
    b10k = (V.filter(pl.col("n_targets") >= 10_000).group_by("model", "instrument", "horizon").agg(pl.col("corr").mean())
            .group_by("model").agg(pl.col("corr").mean().alias("score_baseline_style")))
    fi = {i: json.load(open(MOD / i / "fit_info.json")) for i in INSTS}
    cost = (V.group_by("model").agg(pl.col("pred_s").sum().alias("pred_s_total"), pl.col("maxrss_mb").max().alias("max_rss_mb"))
            .with_columns((pl.col("pred_s_total") / V.select(pl.struct("instrument", "date").n_unique()).item()).alias("pred_s_per_inst_day")))
    feat_s = V.filter(pl.col("model") == "lgb").group_by("instrument", "date").agg(pl.col("feat_s").first())["feat_s"]
    rowsz = []
    for m in agg["model"]:
        tr = sum(fi[i][m]["train_s"] for i in INSTS) if m in fi[INSTS[0]] else 0.0
        npar = sum(fi[i][m]["n_params"] for i in INSTS) if m in fi[INSTS[0]] else 0
        rowsz.append(dict(model=m, train_s_total=tr, n_params_total=npar))
    tab = agg.join(b10k, on="model", how="left").join(pl.DataFrame(rowsz), on="model").join(cost, on="model", how="left")
    wide.write_csv(OUT / "validation_cells.csv"); tab.write_csv(OUT / "validation_summary.csv")
    pl.Config.set_tbl_width_chars(220)
    print(wide.with_columns(pl.selectors.float().round(4))); print(tab.with_columns(pl.selectors.float().round(4)))
    print("feature build s per instrument-day: median", round(float(feat_s.median()), 2), "max", round(float(feat_s.max()), 2))
    print("all predictions finite:", bool(V["all_finite"].all()), "| min pred sd (non-zero models):", float(V.filter(pl.col("model") != "zero")["pred_sd"].min()))
