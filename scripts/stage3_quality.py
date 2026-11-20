"""Stage 3 — full data-quality scan, resumable and time-budgeted.

Reads every row of every file once (book, bbo, trades, targets per instrument-day,
plus stable per day) and writes per-day metrics to <out>/parts/<source>/*.parquet.
Run it repeatedly: finished (instrument, day) units are skipped.

    python scripts/stage3_quality.py --archive-dir <DIR> --data-dir <DIR>/data \
        --archive <extra .tar> --out reports/stage3 --budget 150 --workers 2
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hft500 import inventory as inv, quality as q, spec  # noqa: E402
from hft500.tarsource import index_tar, open_any  # noqa: E402

SOURCES = ("book", "quotes", "bbo", "trades", "targets")


def _day0(d: str) -> int:
    return int(dt.datetime.fromisoformat(d).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def _read(vpath: str | None) -> pl.DataFrame | None:
    if vpath is None:
        return None
    src = open_any(vpath)
    try:
        return pl.from_arrow(pq.read_table(src))
    finally:
        if hasattr(src, "close"):
            src.close()


def run_unit(inst: str, day: str, paths: dict[str, str], tick: float | None, clip: dict | None) -> dict:
    """All metrics for one instrument-day. Returns {source: row-dict}."""
    d0 = _day0(day)
    key = dict(instrument=inst, date=day)
    out: dict[str, dict] = {}
    book = _read(paths.get("book"))
    if book is None:
        return out
    out["book"] = {**key, **q.book_metrics(book, d0, tick)}
    bbo = _read(paths.get("bbo"))
    if bbo is not None:
        out["bbo"] = {**key, **q.bbo_metrics(bbo, d0, tick)}
    out["quotes"] = {**key, **q.quote_metrics(q.quote_grid(book, bbo, d0))}
    tr = _read(paths.get("trades"))
    if tr is not None:
        b = out["book"]
        out["trades"] = {**key, **q.trades_metrics(tr, d0, b["mid_min"], b["mid_max"])}
    tg = _read(paths.get("targets"))
    if tg is not None:
        out["targets"] = {**key, **q.targets_metrics(tg, d0, clip)}
    return out


def run_stable(day: str, path: str) -> dict:
    st = pl.read_parquet(path)
    return {"stable": [{"date": day, **r} for r in q.stable_metrics(st, _day0(day))]}


def done_units(out: Path) -> set[tuple[str, str]]:
    done = set()
    for src in ("book", "stable"):
        for f in (out / "parts" / src).glob("*.parquet"):
            df = pl.read_parquet(f, columns=["instrument", "date"] if src == "book" else ["date"])
            if src == "book":
                done |= set(zip(df["instrument"], df["date"]))
            else:
                done |= {("__stable__", d) for d in df["date"]}
    return done


def flush(out: Path, rows: dict[str, list[dict]]) -> None:
    stamp = time.strftime("%Y%m%dT%H%M%S") + f"_{time.time_ns() % 10**6:06d}"
    for src, rs in rows.items():
        if rs:
            (out / "parts" / src).mkdir(parents=True, exist_ok=True)
            pl.DataFrame(rs, infer_schema_length=None).write_parquet(out / "parts" / src / f"{stamp}.parquet")
        rs.clear()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive-dir", type=Path, required=True)
    ap.add_argument("--archive", type=Path, action="append", default=[])
    ap.add_argument("--data-dir", type=Path, required=True, help="folder holding meta/ and stable/")
    ap.add_argument("--out", type=Path, default=Path("reports/stage3"))
    ap.add_argument("--budget", type=float, default=150, help="seconds before stopping cleanly")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--instruments", nargs="*", help="limit to these instruments")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.time()

    chosen, _ = inv.pick_archives(sorted(a.archive_dir.glob("*.tar")))
    archives = list(chosen.values()) + [p for p in a.archive if p.exists()]
    vfiles: dict[tuple[str, str], dict[str, str]] = {}
    for ar in archives:
        for r in index_tar(ar):
            d = inv.parse_date(Path(r["member"]))
            if d is not None:
                vfiles.setdefault((r["instrument"], d.isoformat()), {})[r["source"]] = r["path"]
    if a.instruments:
        vfiles = {k: v for k, v in vfiles.items() if k[0] in a.instruments}

    meta = pl.read_parquet(a.data_dir / "meta" / "daily.parquet")
    ticks = {(r["instrument"], r["date"]): r["tick_size"] for r in meta.iter_rows(named=True)}
    clips_df = pl.read_parquet(a.data_dir / "meta" / "target_clip.parquet")
    clips: dict[str, dict] = {}
    for r in clips_df.iter_rows(named=True):
        clips.setdefault(r["instrument"], {})[r["horizon"]] = (r["lo"], r["hi"])

    done = done_units(a.out)
    todo = [(i, d) for (i, d) in sorted(vfiles) if (i, d) not in done]
    stable_todo = [] if a.instruments else [
        (p.stem, str(p)) for p in sorted((a.data_dir / "stable").glob("*.parquet")) if ("__stable__", p.stem) not in done]
    total = len(vfiles) + (0 if a.instruments else len(list((a.data_dir / "stable").glob("*.parquet"))))
    logging.info("units: %d total, %d done, %d to do (+%d stable days)", total, len(done), len(todo), len(stable_todo))

    rows: dict[str, list[dict]] = {s: [] for s in SOURCES + ("stable",)}
    n_ok = n_err = 0
    stopped = False
    # 'spawn', not fork: forking a process that has used Polars' thread pool can deadlock
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        it = iter([("u", x) for x in todo] + [("s", x) for x in stable_todo])
        pending = {}

        def submit_next():
            try:
                kind, x = next(it)
            except StopIteration:
                return False
            if kind == "u":
                i, d = x
                fut = ex.submit(run_unit, i, d, vfiles[(i, d)], ticks.get((i, d)), clips.get(i))
            else:
                fut = ex.submit(run_stable, *x)
            pending[fut] = x
            return True

        for _ in range(a.workers * 2):
            submit_next()
        while pending:
            for fut in as_completed(list(pending)):
                x = pending.pop(fut)
                try:
                    res = fut.result()
                    for src, r in res.items():
                        rows[src].extend(r if isinstance(r, list) else [r])
                    n_ok += 1
                except Exception as exc:  # report, never hide
                    n_err += 1
                    logging.error("unit %s failed: %s: %s", x, type(exc).__name__, exc)
                if sum(len(v) for v in rows.values()) >= 400:
                    flush(a.out, rows)
                if time.time() - t0 < a.budget:
                    submit_next()
                else:
                    stopped = True
                break
    flush(a.out, rows)
    remaining = len(todo) + len(stable_todo) - n_ok - n_err
    logging.info("processed %d units (%d errors) in %.0fs; remaining %d", n_ok, n_err, time.time() - t0, remaining)
    print("STATUS:", "DONE" if remaining == 0 and n_err == 0 else ("PARTIAL" if stopped or remaining else "ERRORS"))
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
