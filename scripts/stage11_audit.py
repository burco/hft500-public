"""Stage 11 — leakage and causality audit of the full prediction pipeline (features + Stage 10 LightGBM).

  static                    grep the package for leak-prone constructs -> reports/stage11/static_scan.csv
  provenance                every fitted artifact's data window must lie inside the training period
  consistency               training-cache features == features rebuilt by the inference path
  access --date D           run predict_day under RestrictedRoot (date, date-1, no targets); list reads
  trunc --date D            official-style truncation test on the END-TO-END predictions (resumable)
  canary --date D           two deliberately leaky variants must FAIL the same test
  summary                   collect everything into reports/stage11/audit.json
"""
import argparse, datetime as dt, json, re, sys, time
from pathlib import Path
import numpy as np, polars as pl
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
from hft500 import align as A, causality as C, inventory as inv, models as MD, pipeline as P, spec

ap = argparse.ArgumentParser(); ap.add_argument("step"); ap.add_argument("--date"); ap.add_argument("--budget", type=float, default=140)
ap.add_argument("--cache-dir", default=str(Path.home() / "scratch/cache"))
a = ap.parse_args()
OUT = HERE / "reports/stage11"; OUT.mkdir(parents=True, exist_ok=True)
SCR = Path(a.cache_dir).parent / "stage11"; SCR.mkdir(exist_ok=True)
D = HERE.parent
log = lambda *s: print(f"[{time.strftime('%H:%M:%S')}]", *s, flush=True)


def root():
    chosen, _ = inv.pick_archives(sorted(D.glob("*.tar")))
    pepe = Path.home() / "scratch/data-PEPEUSDT_SPOT.tar"
    return A.ArchiveRoot(list(chosen.values()) + ([pepe] if pepe.exists() else []), fallback=A.DirRoot(D / "data"))


def ticks(date):
    m = pl.read_parquet(D / "data/meta/daily.parquet").filter(pl.col("date") == date)
    return dict(zip(m["instrument"], m["tick_size"]))


def models():
    return {i: MD.LGBModel.load(HERE / "models/stage10" / i / "lgb") for i in P.INSTRUMENTS}


def run_wide(R, date, M, feature_fn=P.features):
    return P.wide(P.predict_day(R, date, M, ticks(date), feature_fn))


# ------------------------------------------------------------------ static scan
if a.step == "static":
    pats = {"negative shift": r"shift\(\s*-|\[\s*1\s*:\s*\]\s*-|np\.roll\(", "backward fill": r"backward|bfill|fill_null\(strategy=.?backward",
            "centred window": r"center\s*=\s*True|centered", "interpolation": r"interpolate",
            "whole-array statistic": r"\.(mean|std|median|quantile|max|min)\(\)|np\.nan(mean|std|median|percentile|max|min)\(",
            "last timestamp / end of day": r"\[-1\]|\.last\(\)|max\(ts|ts\.max|DAY_MS\s*-", "targets access": r"targets|y_500ms|y_2s|y_8s",
            "sort without order guarantee": r"\.sort\("}
    rows = []
    for f in ["align.py", "features_book.py", "features_cross.py", "micro.py", "pipeline.py", "models.py"]:
        for n, line in enumerate((HERE / "src/hft500" / f).read_text().splitlines(), 1):
            if line.strip().startswith("#") or line.strip().startswith('"""'):
                continue
            for k, p in pats.items():
                if re.search(p, line):
                    rows.append(dict(file=f, line=n, pattern=k, code=line.strip()[:140]))
    pl.DataFrame(rows).write_csv(OUT / "static_scan.csv")
    print(pl.DataFrame(rows).group_by("file", "pattern").len().sort("file", "pattern"))

# ------------------------------------------------------------------ provenance
if a.step == "provenance":
    tr_end = spec.TRAIN_END.isoformat(); res = {}
    cache_dates = {i: sorted(p.stem for p in (Path(a.cache_dir) / i).glob("*.parquet")) for i in P.INSTRUMENTS}
    res["lgb_training_rows_dates"] = {i: [d[0], d[-1], len(d)] for i, d in cache_dates.items()}
    st9 = Path(a.cache_dir).parent / "stage9_state"
    res["linear_moments_dates"] = {i: (lambda z: [str(z[0]), str(z[-1]), len(z)])(np.load(st9 / f"mom_{i}.npz")["dates"]) for i in P.INSTRUMENTS}
    res["linear_clip_bounds_window"] = "training days < 2025-11-15 (stage9_report.py moments step)"
    res["target_clip_source"] = "meta/target_clip.parquet (official, training-set quantiles)"
    res["stage9_daily_stats_dates"] = (lambda s: [s[0], s[-1], len(s)])(sorted({p.stem for p in (HERE / "reports/stage9/daily").glob("*/*.parquet")}))
    res["tuning_folds"] = "monthly walk-forward inside 2025-12..2026-07 (training), 1-day embargo"
    vl = pl.read_csv(HERE / "reports/stage10/validation_log.csv"); res["validation_looks"] = vl.height
    ok = all(v[1] <= tr_end for v in res["lgb_training_rows_dates"].values()) and all(v[1] <= tr_end for v in res["linear_moments_dates"].values()) \
        and res["stage9_daily_stats_dates"][1] <= tr_end and vl.height == 1
    res["ok"] = bool(ok)
    json.dump(res, open(OUT / "provenance.json", "w"), indent=1); print(json.dumps(res, indent=1))

# ------------------------------------------------------------------ train/inference consistency
if a.step == "consistency":
    R = root(); rows = []
    for inst, d in [("BTCUSDT_PERP", "2025-09-16"), ("SOLUSDT_PERP", "2026-02-10"), ("1000PEPEUSDT_PERP", "2026-07-14")]:
        fresh = P.features(R, inst, d, ticks(d).get(inst))
        cache = pl.read_parquet(Path(a.cache_dir) / inst / f"{d}.parquet")
        feats = [c for c in cache.columns if c in fresh.columns and c != "ts"]
        f = fresh.gather_every(10).select("ts", *feats).with_columns(pl.exclude("ts").cast(pl.Float32))
        same_ts = np.array_equal(f["ts"].to_numpy(), cache["ts"].to_numpy())
        bad = [c for c in feats if not C._equal(f[c].to_numpy(), cache[c].to_numpy()).all()]
        rows.append(dict(instrument=inst, date=d, features=len(feats), same_ts=same_ts, mismatched=bad))
        log(rows[-1])
    json.dump(rows, open(OUT / "consistency.json", "w"), indent=1)

# ------------------------------------------------------------------ inference contract
if a.step == "access":
    rr = P.RestrictedRoot(root(), a.date); t = time.time()
    out = P.predict_day(rr, a.date, models(), ticks(a.date))
    reads = pl.DataFrame(rr.reads, schema=["table", "date", "instrument"], orient="row").unique()
    r = dict(date=a.date, rows=out.height, finite=bool(all(np.isfinite(out[c].to_numpy()).all() for c in P.PRED.values())),
             dates_read=sorted(reads["date"].unique().to_list()), tables=sorted(reads["table"].unique().to_list()),
             n_files=reads.height, seconds=round(time.time() - t, 1))
    json.dump(r, open(OUT / f"access_{a.date}.json", "w"), indent=1); log(r)

# ------------------------------------------------------------------ truncation on end-to-end predictions
def cuts_for(R, date):
    d0 = A.day_start(date); rng = np.random.default_rng(int(date.replace("-", "")))
    Ts = [d0, d0 + 500, d0 + A.DAY_MS - 500] + [int(d0 + rng.integers(0, A.DAY_MS)) for _ in range(5)]
    return Ts + C.gap_cuts(R, "BTCUSDT_PERP", date, k=1) + C.gap_cuts(R, "1000PEPEUSDT_PERP", date, k=1)


if a.step in ("trunc", "canary"):
    R = root(); M = models(); t0 = time.time()
    if a.step == "trunc":
        variants = {"honest": P.features}; Ts = cuts_for(R, a.date)
    else:
        def leak_future(r, inst, d, tick):
            return P.features(r, inst, d, tick).with_columns(pl.col("imb_l1").shift(-1))
        def leak_day_stat(r, inst, d, tick):
            return P.features(r, inst, d, tick).with_columns(pl.col("ret_2s") - pl.col("ret_2s").mean())
        variants = {"leak_future": leak_future, "leak_day_stat": leak_day_stat}
        Ts = [A.day_start(a.date) + 43_200_000]
    resf = OUT / f"{a.step}_{a.date}.json"
    res = json.load(open(resf)) if resf.exists() else []
    done = {(r["variant"], r["T"]) for r in res}
    for vn, fn in variants.items():
        fullf = SCR / f"full_{vn}_{a.date}.parquet"
        todo = [T for T in Ts if (vn, T) not in done]
        if not todo:
            continue
        if not fullf.exists():
            run_wide(R, a.date, M, fn).write_parquet(fullf); log("full run", vn, a.date)
        full = pl.read_parquet(fullf)
        for T in todo:
            if time.time() - t0 > a.budget:
                break
            r = C.compare(full, run_wide(A.TruncatedRoot(R, T), a.date, M, fn), T)
            r |= dict(variant=vn, date=a.date, T_utc=dt.datetime.utcfromtimestamp(T / 1000).isoformat())
            r = {k: (v if not isinstance(v, (np.integer, np.bool_)) else v.item()) for k, v in r.items()}
            res.append(r); json.dump(res, open(resf, "w"), indent=1)
            log(vn, r["T_utc"], "OK" if r["ok"] else f"MISMATCH {r.get('mismatches')} first {r.get('first_bad_ts')}")
    left = sum((vn, T) not in {(r["variant"], r["T"]) for r in res} for vn in variants for T in Ts)
    log(a.step, a.date, "left", left)

# ------------------------------------------------------------------ summary
if a.step == "summary":
    tr = [r for f in sorted(OUT.glob("trunc_*.json")) for r in json.load(open(f))]
    cn = [r for f in sorted(OUT.glob("canary_*.json")) for r in json.load(open(f))]
    acc = [json.load(open(f)) for f in sorted(OUT.glob("access_*.json"))]
    s = dict(truncation=dict(cuts=len(tr), days=sorted({r["date"] for r in tr}), failures=[r for r in tr if not r["ok"]]),
             canaries=[dict(variant=r["variant"], date=r["date"], detected=not r["ok"], mismatches=r.get("mismatches")) for r in cn],
             access=acc, provenance_ok=json.load(open(OUT / "provenance.json"))["ok"],
             consistency=json.load(open(OUT / "consistency.json")))
    s["verdict"] = ("PASS" if not s["truncation"]["failures"] and all(c["detected"] for c in s["canaries"]) and s["provenance_ok"]
                    and all(not c["mismatched"] and c["same_ts"] for c in s["consistency"])
                    and all(x["dates_read"] == sorted([A.prev_date(x["date"]), x["date"]]) or set(x["dates_read"]) <= {A.prev_date(x["date"]), x["date"]} for x in acc)
                    else "FAIL")
    json.dump(s, open(OUT / "audit.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in s.items() if k != "consistency"}, indent=1, default=str)[:3000])
