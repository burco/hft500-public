"""Stage 9 — feature evaluation report (TRAINING days only; the validation period is not touched).

Phase 1  univariate: per (instrument, horizon, feature) mean daily IC, rank IC, t-stat, % positive days,
         monthly sign flips, coverage, decile monotonicity          (from reports/stage9/daily, full grid)
Phase 2  clusters: |Spearman| between features, averaged over the 4 instruments  (cache sample)
Phase 3  out-of-sample evidence: standardised ridge on monthly walk-forward folds (test Dec..Jul,
         expanding train, 1-day embargo), fitted from per-day sufficient statistics. Linear prep:
         clip at 0.1/99.9 % bounds from Sep-Nov (inside every fold's train), NaN -> 0 + indicator.
         Scored with the official daily Pearson on the stride-10 cache rows.
Phase 4  pre-registered pruning (see PRUNE RULES below) and a final full-vs-pruned check.

usage (steps run in order, each well under 3 min):
  stage9_report.py univariate|select1|select2|finalize --cache-dir DIR
  stage9_report.py moments --inst I --cache-dir DIR ;  stage9_report.py eval --round N --inst I --cache-dir DIR
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, polars as pl
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from hft500 import evaluation as E, features_book as FB, features_cross as X, metric as MT, targets as T

ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["univariate", "moments", "eval", "select1", "select2", "select_amended", "finalize"])
ap.add_argument("--cache-dir", required=True); ap.add_argument("--first-test", default="2025-12")
ap.add_argument("--inst"); ap.add_argument("--round", type=int)
a = ap.parse_args()
CACHE = Path(a.cache_dir); OUT = HERE / "reports/stage9"; OUT.mkdir(parents=True, exist_ok=True)
INSTS = ["BTCUSDT_PERP", "ETHUSDT_PERP", "SOLUSDT_PERP", "1000PEPEUSDT_PERP"]
H = list(T.HORIZONS)
FEATS = FB.FEATURES + X.FEATURES
FAM = {**{k: v[0] for k, v in FB.REGISTRY.items()}, **{k: v[0] for k, v in X.REGISTRY.items()}}
CONDITIONERS = {"volatility", "activity", "staleness"}           # little linear IC by design; kept for trees
log = lambda *s: print(f"[{time.strftime('%H:%M:%S')}]", *s, flush=True)


STATE = CACHE.parent / "stage9_state"; STATE.mkdir(exist_ok=True)


def cache_days(inst):
    return sorted(p.stem for p in (CACHE / inst).glob("*.parquet"))


def load_xy(inst, d):
    df = pl.read_parquet(CACHE / inst / f"{d}.parquet", columns=FEATS + H)
    return df.select(FEATS).to_numpy().astype(np.float64), df.select(H).to_numpy().astype(np.float64)


# ------------------------------------------------------------------ step: univariate + clusters
if a.step == "univariate":
    daily = pl.read_parquet(str(HERE / "reports/stage9/daily/*/*.parquet"))
    cell = E.summarise(daily)
    dec = daily.group_by("instrument", "horizon", "feature").agg([pl.col(f"d{i}").mean() for i in range(10)])
    dc = [f"d{i}" for i in range(10)]
    dec = dec.with_columns(pl.Series("monotonic", [E.monotonicity(np.array(r, float)) for r in dec.select(dc).iter_rows()]),
                           (pl.max_horizontal(dc) - pl.min_horizontal(dc)).alias("decile_range"))
    cell = cell.join(dec.select("instrument", "horizon", "feature", "monotonic", "decile_range", *dc), on=["instrument", "horizon", "feature"])
    cell.write_parquet(OUT / "cell_univariate.parquet")
    mon = (daily.filter(pl.col("pearson").is_not_nan()).with_columns(pl.col("date").str.slice(0, 7).alias("month"))
           .group_by("instrument", "horizon", "feature", "month").agg(pl.col("pearson").mean().alias("ic")))
    mon.write_parquet(OUT / "ic_by_month.parquet")
    absr = []
    for inst in INSTS:
        df = pl.concat([pl.read_parquet(CACHE / inst / f"{d}.parquet", columns=FEATS) for d in cache_days(inst)[::7]])
        R = np.column_stack([df[f].fill_nan(None).rank("average").cast(pl.Float64).fill_null(np.nan).to_numpy() for f in FEATS])
        with np.errstate(all="ignore"):
            R = np.where(np.isnan(R), np.nanmean(R, 0, keepdims=True), R)
            absr.append(np.abs(np.corrcoef(R, rowvar=False)))
    with np.errstate(all="ignore"):
        Cm = np.nanmean(np.stack(absr), 0)
    pl.DataFrame(Cm, schema=FEATS).with_columns(pl.Series("feature", FEATS)).write_parquet(OUT / "abs_spearman.parquet")
    CL = E.clusters(Cm, FEATS, 0.9)
    json.dump(CL, open(STATE / "clusters.json", "w"))
    pl.DataFrame({"cluster": [i for i, c in enumerate(CL) for _ in c], "feature": [f for c in CL for f in c]}).write_csv(OUT / "clusters.csv")
    log("univariate:", daily["date"].n_unique(), "days,", cell.height, "cells;", sum(len(c) > 1 for c in CL), "multi-member clusters")

# ------------------------------------------------------------------ step: moments (one instrument)
if a.step == "moments":
    inst = a.inst; ds = cache_days(inst)
    ref = np.vstack([load_xy(inst, d)[0] for d in ds if d < "2025-11-15"])
    with np.errstate(all="ignore"):
        lo, hi = np.nanpercentile(ref, 0.1, axis=0), np.nanpercentile(ref, 99.9, axis=0)
    ind = np.isnan(ref).mean(0) > 0
    lo = np.where(np.isfinite(lo), lo, -np.inf); hi = np.where(np.isfinite(hi), hi, np.inf)
    del ref
    acc = {k: [] for k in ("n", "sz", "zz", "sy", "zy")}
    for d in ds:
        x, y = load_xy(inst, d)
        m = E.moments(E.design(x, lo, hi, ind), y)
        for k in acc:
            acc[k].append(m[k])
    np.savez(STATE / f"mom_{inst}.npz", dates=np.array(ds), lo=lo, hi=hi, ind=ind, **{k: np.stack(v) for k, v in acc.items()})
    log("moments", inst, len(ds), "days")


def load_mom(inst):
    z = np.load(STATE / f"mom_{inst}.npz")
    cum = {k: np.concatenate([np.zeros_like(z[k][:1]), np.cumsum(z[k], 0)]) for k in ("n", "sz", "zz", "sy", "zy")}
    return list(z["dates"]), z["lo"], z["hi"], z["ind"], cum


def cols_for(ind, feats):
    ind_pos = {f: len(FEATS) + int(ind[:i].sum()) for i, f in enumerate(FEATS) if ind[i]}
    return np.array([FEATS.index(f) for f in feats] + [ind_pos[f] for f in feats if f in ind_pos])


def models_for(r):
    fams = sorted(set(FAM.values()))
    if r == 1:
        m = {"imb_only": ["imb_l1"], "full": FEATS}
        m |= {f"dropfam:{g}": [f for f in FEATS if FAM[f] != g] for g in fams}
        m |= {f"imbfam:{g}": sorted({"imb_l1", *[f for f in FEATS if FAM[f] == g]}) for g in fams}
        m |= {f"drop:{f}": [x for x in FEATS if x != f] for f in FEATS}
        m |= {f"imb+:{f}": sorted({"imb_l1", f}) for f in FEATS}
        return m
    sel = json.load(open(STATE / "select.json"))
    if r == 2:
        S1 = sel["S1"]
        return {"S1": S1} | {f"drop:{f}": [x for x in S1 if x != f] for f in S1}
    if r == 3:
        return {"imb_only": ["imb_l1"], "full": FEATS, "S1": sel["S1"], "pruned": sel["S2"]}
    return {"imb_only": ["imb_l1"], "full": FEATS, "amended": sel["SA"]}


# ------------------------------------------------------------------ step: eval (one round, one instrument)
if a.step == "eval":
    inst, models = a.inst, models_for(a.round)
    ds, lo, hi, ind, cum = load_mom(inst)
    names = list(models); kz = len(FEATS) + int(ind.sum())
    rows = []
    for fi, (tr, te) in enumerate(E.month_folds(ds, a.first_test)):
        i = ds.index(tr[-1]) + 1                                          # train = ds[:i] (expanding, embargoed)
        assert ds[:i] == tr
        M = {k: cum[k][i] for k in cum}
        W = np.zeros((kz, len(names) * 3))
        for j, nm in enumerate(names):
            cols = cols_for(ind, models[nm])
            for k in range(3):
                b, mu, sd = E.ridge_from_moments(M, k, cols)
                W[cols, j * 3 + k] = b / sd
        for d in te:
            x, y = load_xy(inst, d)
            P = E.design(x, lo, hi, ind) @ W
            for k in range(3):
                m = np.isfinite(y[:, k]); yk = y[m, k]
                for j, nm in enumerate(names):
                    rows.append((nm, inst, H[k], fi, d, MT.pearson(P[m, j * 3 + k], yk)))
    pl.DataFrame(rows, schema=["model", "instrument", "horizon", "fold", "date", "corr"], orient="row") \
      .write_parquet(STATE / f"oos_r{a.round}_{inst}.parquet")
    log("eval round", a.round, inst, len(names), "models")


def scores(r):
    o = pl.read_parquet(str(STATE / f"oos_r{r}_*.parquet"))
    o.write_parquet(OUT / f"oos_round{r}.parquet")
    return o.group_by("model", "instrument", "horizon").agg(pl.col("corr").mean())


def delta(s, base, prefix):
    b = s.filter(pl.col("model") == base).select("instrument", "horizon", pl.col("corr").alias("base"))
    return (s.filter(pl.col("model").str.starts_with(prefix)).join(b, on=["instrument", "horizon"])
            .with_columns(pl.col("model").str.replace(prefix, "", literal=True).alias("name"),
                          (pl.col("base") - pl.col("corr")).alias("loss_if_dropped"), (pl.col("corr") - pl.col("base")).alias("gain")))


agg = lambda s, m: float(s.filter(pl.col("model") == m)["corr"].mean())

# ------------------------------------------------------------------ PRUNE RULES (fixed before looking at results)
# 1. clusters (|rho| >= 0.9): keep the member with the largest out-of-sample gain over imb_l1 alone.
# 2. on the reduced set, drop-one again; remove a non-conditioner if dropping it does not hurt on average
#    (mean loss <= 0) AND hurts in fewer than 6 of 12 cells. One pass only.
# 3. accept the pruned set only if its aggregate OOS score >= full - 0.001; otherwise keep the step-1 set.
if a.step == "select1":
    s1 = scores(1)
    add1 = delta(s1, "imb_only", "imb+:").group_by("name").agg(pl.col("gain").mean())
    g = dict(zip(add1["name"], add1["gain"]))
    rep = {}
    for c in json.load(open(STATE / "clusters.json")):
        keep = max(c, key=lambda f: g.get(f, -9))
        rep |= {f: keep for f in c}
    json.dump({"S1": [f for f in FEATS if rep[f] == f], "rep": rep}, open(STATE / "select.json", "w"))
    log("select1: imb_only", round(agg(s1, "imb_only"), 4), "full", round(agg(s1, "full"), 4), "S1 size", sum(rep[f] == f for f in FEATS))

if a.step == "select2":
    sel = json.load(open(STATE / "select.json"))
    d2 = delta(scores(2), "S1", "drop:").group_by("name").agg(pl.col("loss_if_dropped").mean().alias("loss2"),
                                                            (pl.col("loss_if_dropped") > 0).sum().alias("cells_hurt2"))
    weak = sorted(r["name"] for r in d2.iter_rows(named=True)
                  if FAM[r["name"]] not in CONDITIONERS and r["loss2"] <= 0 and r["cells_hurt2"] < 6)
    sel |= {"S2": [f for f in sel["S1"] if f not in weak], "weak": weak}
    json.dump(sel, open(STATE / "select.json", "w"))
    log("select2: removed", len(weak), "S2 size", len(sel["S2"]))

# AMENDED RULE (written after round 3 showed the cluster step costs ~0.007; disclosed in FINDINGS):
#   flaw 1: the representative was chosen by gain over imb_l1 alone, which is 0 by construction for imb_l1;
#   flaw 2: |rho| >= 0.9 imbalance variants are still complementary in a linear model.
#   new rule: one pass of drop-one from the FULL model (round 1); remove a non-conditioner if mean loss <= 0
#   AND it hurts in fewer than 6 of 12 cells; accept only if the OOS aggregate >= full - 0.001, else keep full.
if a.step == "select_amended":
    sel = json.load(open(STATE / "select.json"))
    d1 = delta(scores(1), "full", "drop:").group_by("name").agg(pl.col("loss_if_dropped").mean().alias("l"),
                                                              (pl.col("loss_if_dropped") > 0).sum().alias("c"))
    weak = sorted(r["name"] for r in d1.iter_rows(named=True) if FAM[r["name"]] not in CONDITIONERS and r["l"] <= 0 and r["c"] < 6)
    sel |= {"SA": [f for f in FEATS if f not in weak], "weak_amended": weak}
    json.dump(sel, open(STATE / "select.json", "w"))
    log("select_amended: removed", len(weak), weak)

if a.step == "finalize":
    sel = json.load(open(STATE / "select.json")); rep = sel["rep"]
    s1, s2, s3, s4 = scores(1), scores(2), scores(3), scores(4)
    final = sel["SA"] if agg(s4, "amended") >= agg(s4, "full") - 0.001 else FEATS
    pl.concat([s3, s4.filter(pl.col("model") == "amended")]).write_csv(OUT / "oos_final_models.csv")
    fam_tab = (delta(s1, "full", "dropfam:").group_by("name").agg(pl.col("loss_if_dropped").mean(), (pl.col("loss_if_dropped") > 0).sum().alias("cells_hurt_by_drop"))
               .join(delta(s1, "imb_only", "imbfam:").group_by("name").agg(pl.col("gain").mean().alias("gain_over_imb"), (pl.col("gain") > 0).sum().alias("cells_gain")), on="name")
               .rename({"name": "family"}).sort("loss_if_dropped", descending=True))
    fam_tab.write_csv(OUT / "ablation_family.csv")
    drop1 = delta(s1, "full", "drop:").group_by("name").agg(pl.col("loss_if_dropped").mean(), (pl.col("loss_if_dropped") > 0).sum().alias("cells_hurt"))
    add1 = delta(s1, "imb_only", "imb+:").group_by("name").agg(pl.col("gain").mean().alias("gain_over_imb"), (pl.col("gain") > 0).sum().alias("cells_gain"))
    d2 = delta(s2, "S1", "drop:").group_by("name").agg(pl.col("loss_if_dropped").mean().alias("loss2"), (pl.col("loss_if_dropped") > 0).sum().alias("cells_hurt2"))
    cell = pl.read_parquet(OUT / "cell_univariate.parquet")
    uni = cell.group_by("feature").agg(
        pl.col("ic").mean().alias("ic_mean"), pl.col("rank_ic").mean().alias("rank_ic_mean"),
        pl.col("t").abs().mean().alias("abs_t_mean"), (pl.col("t").abs() > 3).sum().alias("cells_abs_t_gt3"),
        pl.col("pos_share").mean().alias("pos_share"), pl.col("months_flipped").sum().alias("months_flipped"),
        pl.col("coverage").mean().alias("coverage"), pl.col("monotonic").mean().alias("monotonic"))
    cid = {f: i for i, c in enumerate(json.load(open(STATE / "clusters.json"))) for f in c}
    rep_df = (uni.join(add1.rename({"name": "feature"}), on="feature", how="left")
              .join(drop1.rename({"name": "feature"}), on="feature", how="left")
              .join(d2.rename({"name": "feature"}), on="feature", how="left")
              .with_columns(pl.col("feature").replace_strict(FAM).alias("family"), pl.col("feature").replace_strict(cid).alias("cluster"),
                            pl.col("feature").replace_strict(rep).alias("cluster_rep"), pl.col("feature").is_in(final).alias("kept")))
    reason = []
    for r in rep_df.iter_rows(named=True):
        if r["kept"]:
            reason.append("conditioner (kept for trees)" if r["family"] in CONDITIONERS else "kept")
        else:
            reason.append("no out-of-sample contribution (drop-one from full)")
    rep_df = rep_df.with_columns(pl.Series("decision", reason)).sort(["kept", "gain_over_imb"], descending=[True, True])
    rep_df.write_csv(OUT / "feature_report.csv")
    json.dump({"features": final, "removed": sel["weak_amended"], "rule": "amended drop-one (see script)",
               "rejected_preregistered": {"S1": sel["S1"], "S2": sel["S2"]}}, open(OUT / "pruned_features.json", "w"), indent=1)
    log("finalize:", {m: round(agg(s3, m), 4) for m in ["imb_only", "full", "S1", "pruned"]}, "amended", round(agg(s4, "amended"), 4), "final", len(final))
