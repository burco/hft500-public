"""Stage 1 — dataset inventory & schema audit.

Usage:
    uv run python scripts/stage1_inventory.py --data-dir /path/to/data --out reports/stage1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hft500 import inventory as inv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, help="extracted data/ folder")
    ap.add_argument("--archive", type=Path, action="append", default=[],
                    help="read directly from data-<INSTRUMENT>-NNN.tar archive(s) (repeatable)")
    ap.add_argument("--archive-dir", type=Path,
                    help="folder of data-*.tar archives; one copy per instrument is picked")
    ap.add_argument("--out", type=Path, default=Path("reports/stage1"))
    ap.add_argument("--repo-dir", type=Path, action="append", default=[],
                    help="extra folder(s) to search for common.py/baseline.py/predict.py")
    ap.add_argument("--sample-per-group", type=int, default=3,
                    help="files per source/instrument whose timestamp column is loaded")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)

    archives = list(a.archive)
    if a.archive_dir:
        chosen, groups = inv.pick_archives(sorted(a.archive_dir.glob("*.tar")))
        for k, v in sorted(groups.items()):
            logging.info("%-20s using %s  (%d copies)", k, chosen[k].name, len(v))
        archives += list(chosen.values())
    if not archives and not (a.data_dir and a.data_dir.is_dir()):
        logging.error("give --data-dir (existing folder) or --archive/--archive-dir")
        return 2
    a.out.mkdir(parents=True, exist_ok=True)

    parts = []
    if archives:
        parts.append(inv.discover_archives(archives))
    if a.data_dir and a.data_dir.is_dir():
        # meta/ is checked by name in readiness(); its ~6k tiny daily files need no footer scan
        parts.append(inv.discover(a.data_dir).filter(pl.col("source") != "meta"))
    files = pl.concat(parts)
    if a.data_dir is None:
        a.data_dir = a.out / "_no_extracted_data"   # readiness then reports meta/ as missing
    logging.info("found %d parquet files", files.height)
    if files.height == 0:
        logging.error("no .parquet files under %s", a.data_dir)
        return 2
    files = inv.add_footers(files)

    summary = inv.summarise(files)
    schemas = inv.schema_report(files)
    missing = inv.missing_dates(files)
    sampled = inv.sample_files(files, a.sample_per_group)
    ts = pl.DataFrame([asdict(inv.check_timestamps(r["source"], r["instrument"], r["date"], r["path"]))
                       for r in sampled.iter_rows(named=True)])
    repo_dirs = [a.data_dir, a.data_dir.parent, *a.repo_dir]
    ready = inv.readiness(a.data_dir, files, summary, ts, repo_dirs)

    # ---- write artefacts ---------------------------------------------------
    files.write_parquet(a.out / "inventory_files.parquet")
    summary.write_csv(a.out / "inventory_summary.csv")
    missing.write_csv(a.out / "missing_dates.csv")
    if ts.height:
        ts.write_csv(a.out / "ts_checks.csv")
    (a.out / "schema_report.json").write_text(json.dumps(schemas, indent=2))
    md = ["| item | status | detail |", "|---|---|---|"] + [f"| {i} | {s} | {d} |" for i, s, d in ready]
    (a.out / "readiness.md").write_text("\n".join(md) + "\n")

    # ---- console report ----------------------------------------------------
    print("\n=== INVENTORY SUMMARY ===")
    print(summary)
    print("\n=== SCHEMAS (first variant per group) ===")
    for k, v in schemas.items():
        cols = ", ".join(f"{c}:{t}" for c, t in v[0]["columns"])
        print(f"{k:40s} variants={len(v)}  {cols}")
    if ts.height:
        print("\n=== TIMESTAMP CHECKS (sampled files) ===")
        print(ts.select("source", "instrument", "date", "ts_col", "ts_unit", "rows", "monotonic",
                        "dup_ts", "pct_on_grid", "pct_on_500ms", "pct_inside_day", "max_gap_s", "coverage_vs_full_grid"))
    print("\n=== READINESS ===")
    for i, s, d in ready:
        print(f"[{s:7s}] {i}  {d}")
    print(f"\nartefacts written to {a.out.resolve()}")
    return 1 if any(s == "FAIL" for _, s, _ in ready) else 0


if __name__ == "__main__":
    raise SystemExit(main())
