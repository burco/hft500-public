"""Stage 3 — aggregate the per-day quality metrics into tables, flags and charts.

    python scripts/stage3_report.py --out reports/stage3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hft500 import spec  # noqa: E402

VALID_START = spec.VALID_START.isoformat()
TARGETS = list(spec.TARGET_INSTRUMENTS)

# day-level flags: (source, name, expression, meaning)
FLAGS = [
    ("book", "book_gap", pl.col("missing_pts") > 100, "book missing > 100 grid points (> 50 s)"),
    ("book", "book_crossed", pl.col("crossed") > 0, "bid > ask in book"),
    ("book", "book_nan_values", pl.col("nonfinite_px_qty") + pl.col("bucket_nonfinite") > 0,
     "NaN prices/buckets (expected on stablecoin-depeg days: PDF null rule)"),
    ("book", "book_invalid_values", (pl.col("nonpos_px") + pl.col("bucket_negative")) > 0, "non-positive price / negative bucket"),
    ("book", "book_off_tick", pl.col("off_tick") > 0, "price not a multiple of the (float-cleaned) meta tick"),
    ("book", "book_jump", pl.col("moves_gt_50bps") > 0, "500 ms mid move > 50 bps"),
    ("bbo", "bbo_structure", (pl.col("on_500ms") + pl.col("dup_ts") + pl.col("non_monotonic")
                              + pl.col("outside_day") + pl.col("off_grid")) > 0, "bbo timestamp rule broken"),
    ("bbo", "bbo_crossed", pl.col("crossed") > 0, "bid > ask in bbo"),
    ("quotes", "prices_frozen", pl.col("longest_static_px_s") > 900, "best prices unchanged > 15 min (feed freeze?)"),
    ("trades", "trades_inconsistent", pl.col("consistency_violations") > 0, "trade-bar fields contradict each other"),
    ("trades", "trades_structure", (pl.col("dup_ts") + pl.col("non_monotonic") + pl.col("outside_day")
                                    + pl.col("off_grid")) > 0, "trade-bar timestamp rule broken"),
    ("targets", "targets_gap", (pl.col("missing_pts") + pl.col("y_500ms_nonfinite") + pl.col("y_2s_nonfinite")
                                + pl.col("y_8s_nonfinite")) > 0, "missing / non-finite targets"),
]


def load(out: Path, src: str) -> pl.DataFrame | None:
    files = sorted((out / "parts" / src).glob("*.parquet"))
    if not files:
        return None
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    keys = ["instrument", "date"] if "instrument" in df.columns else ["date", "pair"]
    return df.unique(subset=keys, keep="last").sort(keys)


def _dates(s: pl.Series):
    return [np.datetime64(x) for x in s.to_list()]


def vline(ax):
    ax.axvline(np.datetime64(VALID_START), color="grey", ls="--", lw=0.8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/stage3"))
    a = ap.parse_args(argv)
    pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_rows(60)
    D = {s: load(a.out, s) for s in ("book", "quotes", "bbo", "trades", "targets", "stable")}
    if D["book"] is None:
        print("no parts yet"); return 2
    rc = a.out / "tick_recheck.csv"     # off-tick recomputed with float-cleaned ticks (supersedes parts)
    if rc.exists():
        fix = pl.read_csv(rc).select("instrument", "date", pl.col("off_tick_clean").alias("_ot"))
        D["book"] = (D["book"].join(fix, on=["instrument", "date"], how="left")
                     .with_columns(pl.coalesce("_ot", "off_tick").alias("off_tick")).drop("_ot"))
    md = ["# Stage 3 — data quality report", "", "Generated from measured per-day metrics. Validation starts "
          f"{VALID_START} (dashed line in charts).", ""]

    # ---------------------------------------------------------------- per instrument
    b, qd, tr = D["book"], D["quotes"], D["trades"]
    inst = (b.group_by("instrument").agg(
        days=pl.len(), book_missing_total=pl.col("missing_pts").sum(),
        worst_day=pl.col("date").sort_by("missing_pts").last(), worst_missing=pl.col("missing_pts").max(),
        longest_book_gap_s=pl.col("longest_missing_s").max(), crossed=pl.col("crossed").sum(),
        off_tick=pl.col("off_tick").sum(), jumps_gt50=pl.col("moves_gt_50bps").sum(),
        max_move_bps=pl.col("max_abs_move_500ms_bps").max(), pct_1tick_med=pl.col("pct_1tick").median(),
        spread_bps_med=pl.col("spread_bps_med").median(), best_notional_med=pl.col("best_notional_med").median(),
        depth4_notional_med=pl.col("depth4_notional_med").median()))
    if qd is not None:
        inst = inst.join(qd.group_by("instrument").agg(longest_static_px_s=pl.col("longest_static_px_s").max(),
                                                       px_changes_day_med=pl.col("px_changes").median()), on="instrument", how="left")
    if tr is not None:
        inst = inst.join(tr.group_by("instrument").agg(
            notional_day_med=(pl.col("buy_notional") + pl.col("sell_notional")).median(),
            trades_day_med=(pl.col("buy_trades") + pl.col("sell_trades")).median(),
            buy_share_pct=(pl.col("buy_notional").sum() / (pl.col("buy_notional") + pl.col("sell_notional")).sum() * 100),
            trade_violations=pl.col("consistency_violations").sum()), on="instrument", how="left")
    inst = inst.sort("instrument")
    inst.write_csv(a.out / "summary_instruments.csv")
    print("=== PER INSTRUMENT ===\n", inst)

    # ---------------------------------------------------------------- targets
    if D["targets"] is not None:
        t = D["targets"].with_columns(split=pl.when(pl.col("date") < VALID_START).then(pl.lit("train")).otherwise(pl.lit("valid")))
        rows = []
        for (i, sp), g in t.group_by(["instrument", "split"], maintain_order=True):
            for h in ("y_500ms", "y_2s", "y_8s"):
                rows.append(dict(instrument=i, split=sp, horizon=h, days=g.height,
                                 zero_pct=g[f"{h}_zero_pct"].mean(), std_bps_med=g[f"{h}_std_bps"].median(),
                                 max_abs_bps=g[f"{h}_max_abs_bps"].max(),
                                 clipped_pct=g[f"{h}_clipped_pct"].mean() if f"{h}_clipped_pct" in g.columns else None,
                                 nonfinite=g[f"{h}_nonfinite"].sum(), missing_pts=g["missing_pts"].sum()))
        tt = pl.DataFrame(rows).sort("instrument", "split", "horizon")
        tt.write_csv(a.out / "summary_targets.csv")
        print("=== TARGETS ===\n", tt)

    # ---------------------------------------------------------------- stable
    if D["stable"] is not None:
        s = D["stable"]
        ss = s.group_by("pair").agg(days=pl.len(), missing_s=pl.col("missing_s").sum(), dev_bps_max=pl.col("dev_bps_max").max(),
                                    dev_bps_med=pl.col("dev_bps_med").median(), depeg_s=pl.col("depeg_s").sum(),
                                    depeg_days=(pl.col("depeg_s") > 0).sum(), stale_ms_max=pl.col("stale_ms_max").max(),
                                    stale_gt_10s=pl.col("stale_gt_10s").sum()).sort("pair")
        ss.write_csv(a.out / "summary_stable.csv")
        print("=== STABLE ===\n", ss)

    # ---------------------------------------------------------------- flags
    fl = []
    for src, name, expr, meaning in FLAGS:
        df = D.get(src)
        if df is None:
            continue
        hit = df.filter(expr)
        for r in hit.iter_rows(named=True):
            fl.append(dict(flag=name, instrument=r["instrument"], date=r["date"], meaning=meaning))
    if D["stable"] is not None:
        for r in D["stable"].filter(pl.col("depeg_s") > 0).iter_rows(named=True):
            fl.append(dict(flag="stable_depeg", instrument=r["pair"], date=r["date"], meaning=f"{r['depeg_s']} s beyond 50 bps"))
    flags = pl.DataFrame(fl, schema={"flag": pl.Utf8, "instrument": pl.Utf8, "date": pl.Utf8, "meaning": pl.Utf8})
    flags.write_csv(a.out / "flags.csv")
    fsum = flags.group_by("flag", "meaning").agg(days=pl.len(), instruments=pl.col("instrument").n_unique()).sort("days", descending=True)
    print("=== FLAGS ===\n", fsum)

    # ---------------------------------------------------------------- charts
    insts = sorted(b["instrument"].unique().to_list())
    dates = sorted(b["date"].unique().to_list())
    M = np.full((len(insts), len(dates)), np.nan)
    ix, dx = {k: n for n, k in enumerate(insts)}, {k: n for n, k in enumerate(dates)}
    for r in b.select("instrument", "date", "missing_pts").iter_rows():
        M[ix[r[0]], dx[r[1]]] = np.log10(1 + r[2])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(M, aspect="auto", cmap="magma_r", interpolation="nearest")
    ax.set_yticks(range(len(insts)), insts, fontsize=7)
    xt = [n for n, d in enumerate(dates) if d.endswith("-01")]
    ax.set_xticks(xt, [dates[n][:7] for n in xt], fontsize=7, rotation=45)
    fig.colorbar(im, ax=ax, label="log10(1 + missing book points)")
    ax.set_title("Missing 500 ms book snapshots per instrument-day"); fig.tight_layout()
    fig.savefig(a.out / "fig_book_missing.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in TARGETS:
        g = b.filter(pl.col("instrument") == i)
        axs[0].plot(_dates(g["date"]), g["spread_bps_med"], lw=1, label=i)
        if tr is not None:
            gt = tr.filter(pl.col("instrument") == i)
            axs[1].plot(_dates(gt["date"]), (gt["buy_notional"] + gt["sell_notional"]) / 1e9, lw=1, label=i)
        if D["targets"] is not None:
            gg = D["targets"].filter(pl.col("instrument") == i)
            axs[2].plot(_dates(gg["date"]), gg["y_500ms_std_bps"], lw=1, label=i)
    axs[0].set_ylabel("median spread (bps)"); axs[0].set_yscale("log")
    axs[1].set_ylabel("traded notional (bn USDT/day)")
    axs[2].set_ylabel("std of y_500ms (bps)")
    for ax in axs: vline(ax)
    axs[0].legend(fontsize=7, ncol=4, frameon=False); axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axs[0].set_title("Target perps: spread, activity and 500 ms volatility by day"); fig.tight_layout()
    fig.savefig(a.out / "fig_targets_daily.png", dpi=130); plt.close(fig)

    if D["targets"] is not None:
        fig, ax = plt.subplots(figsize=(10, 3.4))
        for i in TARGETS:
            gg = D["targets"].filter(pl.col("instrument") == i)
            ax.plot(_dates(gg["date"]), gg["y_500ms_zero_pct"], lw=1, label=i)
        vline(ax); ax.set_ylabel("% of y_500ms exactly 0"); ax.legend(fontsize=7, ncol=4, frameon=False)
        ax.set_title("Share of 500 ms targets with no mid change"); fig.tight_layout()
        fig.savefig(a.out / "fig_target_zeros.png", dpi=130); plt.close(fig)

    if D["stable"] is not None:
        fig, ax = plt.subplots(figsize=(10, 3.4))
        for p, g in D["stable"].group_by("pair", maintain_order=True):
            ax.plot(_dates(g["date"]), g["dev_bps_max"], lw=1, label=p[0] if isinstance(p, tuple) else p)
        ax.axhline(50, color="red", lw=0.8, ls=":"); vline(ax); ax.set_yscale("log")
        ax.set_ylabel("max |mid − 1| (bps)"); ax.legend(fontsize=7, frameon=False)
        ax.set_title("Stablecoin deviation from 1 (red: 50 bps null rule)"); fig.tight_layout()
        fig.savefig(a.out / "fig_stable_dev.png", dpi=130); plt.close(fig)

    md += ["## Flags", "", "| flag | meaning | days | instruments |", "|---|---|---|---|"]
    md += [f"| {r['flag']} | {r['meaning']} | {r['days']} | {r['instruments']} |" for r in fsum.iter_rows(named=True)]
    md += ["", "Files: summary_instruments.csv, summary_targets.csv, summary_stable.csv, flags.csv, fig_*.png"]
    (a.out / "report.md").write_text("\n".join(md) + "\n")
    print("\nwritten:", sorted(p.name for p in a.out.glob("*") if p.is_file()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
