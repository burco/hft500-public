"""Stage 1: file inventory, schema audit and timestamp sanity checks.

Design rules
- Footer-first: row counts, schemas and null counts come from Parquet
  metadata, so auditing hundreds of files reads kilobytes, not gigabytes.
- Only the timestamp column of a few sampled files is actually loaded.
- Nothing here interprets prices; that is Stage 3.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from . import spec
from .tarsource import open_any

log = logging.getLogger(__name__)

_DATE_RE = re.compile(r"(\d{4})-?(\d{2})-?(\d{2})")


# --------------------------------------------------------------------------- #
# 1. Discovery
# --------------------------------------------------------------------------- #
def parse_date(path: Path) -> date | None:
    """Take the LAST date-looking token in the path (file name wins over dirs)."""
    for m in reversed(list(_DATE_RE.finditer(str(path)))):
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            continue
    return None


def discover(data_dir: Path) -> pl.DataFrame:
    """One row per Parquet file: source, instrument, date, path, size."""
    rows = []
    for source in sorted(p.name for p in data_dir.iterdir() if p.is_dir()):
        src_dir = data_dir / source
        for f in sorted(src_dir.rglob("*.parquet")):
            rel = f.relative_to(src_dir).parts
            # data/<source>/<INSTRUMENT>/<file>  -> instrument = first sub-dir
            instrument = rel[0] if len(rel) > 1 else "(none)"
            rows.append(
                dict(source=source, instrument=instrument, date=parse_date(f),
                     path=str(f), size_bytes=f.stat().st_size)
            )
    schema = {"source": pl.Utf8, "instrument": pl.Utf8, "date": pl.Date,
              "path": pl.Utf8, "size_bytes": pl.Int64}
    return pl.DataFrame(rows, schema=schema)


_ARCHIVE_RE = re.compile(r"data-(?P<inst>[A-Za-z0-9]+_(?:PERP|SPOT)|meta|stable)(?:-\d+)?(?P<dup> \(\d+\)| - Copy)?\.tar$")


def pick_archives(paths: list[Path]) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """One archive per instrument. Prefer names without ' (1)' / ' - Copy', then lowest number."""
    groups: dict[str, list[Path]] = {}
    for p in paths:
        m = _ARCHIVE_RE.search(p.name)
        if m:
            groups.setdefault(m["inst"], []).append(p)
    chosen = {k: sorted(v, key=lambda p: (bool(_ARCHIVE_RE.search(p.name)["dup"]), p.name))[0]
              for k, v in groups.items()}
    return chosen, groups


def discover_archives(archives: list[Path]) -> pl.DataFrame:
    """Same columns as discover(), but every path is a virtual '<tar>::<member>' path."""
    from .tarsource import index_tar
    rows = []
    for a in archives:
        for r in index_tar(a):
            rows.append(dict(source=r["source"], instrument=r["instrument"],
                             date=parse_date(Path(r["member"])), path=r["path"],
                             size_bytes=r["size_bytes"]))
    schema = {"source": pl.Utf8, "instrument": pl.Utf8, "date": pl.Date,
              "path": pl.Utf8, "size_bytes": pl.Int64}
    return pl.DataFrame(rows, schema=schema)


# --------------------------------------------------------------------------- #
# 2. Footer metadata (cheap)
# --------------------------------------------------------------------------- #
def schema_fingerprint(schema: pa.Schema) -> str:
    text = "|".join(f"{f.name}:{f.type}" for f in schema)
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def read_footer(path: str) -> dict:
    try:
        src = open_any(path)
        pf = pq.ParquetFile(src)
    except Exception as exc:  # corrupt / truncated file
        return dict(readable=False, error=f"{type(exc).__name__}: {exc}",
                    num_rows=None, num_row_groups=None, schema_id=None,
                    schema_str=None, null_counts=None)
    md, schema = pf.metadata, pf.schema_arrow
    nulls: dict[str, int | None] = {}
    for ci in range(md.num_columns):
        name = md.schema.column(ci).path
        total: int | None = 0
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(ci).statistics
            if st is None or not st.has_null_count:
                total = None
                break
            total += st.null_count
        nulls[name] = total
    if hasattr(src, "close"):
        src.close()
    return dict(readable=True, error=None, num_rows=md.num_rows,
                num_row_groups=md.num_row_groups,
                schema_id=schema_fingerprint(schema),
                schema_str=json.dumps([[f.name, str(f.type)] for f in schema]),
                null_counts=json.dumps(nulls))


def add_footers(files: pl.DataFrame) -> pl.DataFrame:
    meta = [read_footer(p) for p in files["path"]]
    return files.hstack(pl.DataFrame(meta, infer_schema_length=None))


# --------------------------------------------------------------------------- #
# 3. Timestamp checks (reads ONE column of a few files)
# --------------------------------------------------------------------------- #
def find_ts_column(schema: pa.Schema) -> str | None:
    names = schema.names
    for c in spec.TS_CANDIDATES:
        if c in names:
            return c
    for f in schema:  # fall back to the first temporal column
        if pa.types.is_timestamp(f.type):
            return f.name
    return None


def to_epoch_ms(col: pa.ChunkedArray) -> tuple[np.ndarray, str]:
    """Return int64 epoch-ms and a description of the original unit."""
    t = col.type
    if pa.types.is_timestamp(t):
        ms = col.cast(pa.timestamp("ms", tz=t.tz)).cast(pa.int64())
        return ms.to_numpy(), f"timestamp[{t.unit}, tz={t.tz}]"
    if pa.types.is_integer(t):
        v = col.to_numpy().astype(np.int64)
        mag = np.nanmedian(np.abs(v)) if len(v) else 0
        # 2020..2030 in each unit: ms ~1.6e12-1.9e12, us ~1e15, ns ~1e18, s ~1e9
        for unit, div in (("s", 1e-3), ("ms", 1), ("us", 1e3), ("ns", 1e6)):
            if 1.5e12 <= mag / div <= 2.0e12:
                return (v * 1000 if unit == "s" else v // int(div)), f"int64 epoch-{unit} (inferred)"
        return v, f"int64 UNKNOWN unit (median={mag:.3g})"
    raise TypeError(f"Unsupported timestamp type {t}")


@dataclass
class TsCheck:
    source: str
    instrument: str
    date: date | None
    path: str
    ts_col: str | None
    ts_unit: str | None
    rows: int
    first_ts_utc: str | None
    last_ts_utc: str | None
    monotonic: bool | None
    dup_ts: int | None
    grid_ms: int | None
    pct_on_grid: float | None
    pct_on_500ms: float | None
    pct_inside_day: float | None
    max_gap_s: float | None
    full_grid_rows: int | None
    coverage_vs_full_grid: float | None
    note: str


def check_timestamps(source: str, instrument: str, d: date | None, path: str) -> TsCheck:
    grid = spec.SOURCES.get(source)
    base = dict(source=source, instrument=instrument, date=d, path=path, grid_ms=grid)
    empty = dict(ts_unit=None, rows=0, first_ts_utc=None, last_ts_utc=None,
                 monotonic=None, dup_ts=None, pct_on_grid=None, pct_inside_day=None,
                 max_gap_s=None, pct_on_500ms=None, full_grid_rows=None, coverage_vs_full_grid=None)
    src = open_any(path)
    schema = pq.ParquetFile(src).schema_arrow
    if hasattr(src, "close"):
        src.close()
    ts_col = find_ts_column(schema)
    if ts_col is None:
        return TsCheck(**base, ts_col=None, **empty, note="no timestamp column found")
    src = open_any(path)
    ms, unit = to_epoch_ms(pq.read_table(src, columns=[ts_col])[ts_col])
    if hasattr(src, "close"):
        src.close()
    n = len(ms)
    if n == 0:
        return TsCheck(**base, ts_col=ts_col, **{**empty, "ts_unit": unit}, note="empty file")

    fmt = lambda x: datetime.fromtimestamp(x / 1000, tz=timezone.utc).isoformat(timespec="milliseconds")
    diffs = np.diff(ms)
    full_grid = 86_400_000 // grid if grid else None
    inside = None
    if d is not None:
        day0 = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)
        if source == "trades":   # bar (ts-100, ts]: the day's last bar is stamped next-day 00:00:00.000
            inside = float(np.mean((ms > day0) & (ms <= day0 + 86_400_000)) * 100)
        else:
            inside = float(np.mean((ms >= day0) & (ms < day0 + 86_400_000)) * 100)
    uniq = int(np.unique(ms).size)
    return TsCheck(
        **base, ts_col=ts_col, ts_unit=unit, rows=n,
        first_ts_utc=fmt(ms.min()), last_ts_utc=fmt(ms.max()),
        monotonic=bool(np.all(diffs >= 0)) if n > 1 else True,
        dup_ts=n - uniq,
        pct_on_grid=float(np.mean(ms % grid == 0) * 100) if grid else None,
        pct_on_500ms=float(np.mean(ms % 500 == 0) * 100),
        pct_inside_day=inside,
        max_gap_s=float(diffs.max() / 1000) if n > 1 else None,
        full_grid_rows=full_grid,
        coverage_vs_full_grid=(uniq / full_grid * 100) if full_grid else None,
        note="",
    )


def sample_files(files: pl.DataFrame, per_group: int) -> pl.DataFrame:
    """First, middle and last dated file of every (source, instrument)."""
    ok = files.filter(pl.col("readable"))
    out = []
    for _, g in ok.sort("date").group_by(["source", "instrument"], maintain_order=True):
        idx = sorted(set(np.linspace(0, g.height - 1, min(per_group, g.height)).round().astype(int)))
        out.append(g[idx])
    return pl.concat(out) if out else ok.head(0)


# --------------------------------------------------------------------------- #
# 4. Summaries
# --------------------------------------------------------------------------- #
def _missing_days(dates: list[date], start: date, end: date) -> int:
    have = set(dates)
    return sum((start + timedelta(i)) not in have for i in range((end - start).days + 1))


def missing_columns(source: str, readable: pl.DataFrame) -> list[str]:
    """Columns the PDF requires for this source that are absent from ANY file's schema."""
    req = spec.REQUIRED_COLUMNS.get(source)
    if not req or readable.height == 0:
        return []
    miss: set[str] = set()
    for sch in readable["schema_str"].unique().to_list():
        have = {c for c, _ in json.loads(sch)}
        miss |= set(req) - have
    return sorted(miss)


def summarise(files: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (src, inst), g in files.group_by(["source", "instrument"], maintain_order=True):
        dates = [x for x in g["date"].to_list() if x is not None]
        readable = g.filter(pl.col("readable"))
        rows.append(dict(
            source=src, instrument=inst, files=g.height,
            unreadable=g.height - readable.height,
            undated=g["date"].null_count(),
            first_date=min(dates) if dates else None,
            last_date=max(dates) if dates else None,
            dup_dates=len(dates) - len(set(dates)),
            missing_train_days=_missing_days(dates, spec.TRAIN_START, spec.TRAIN_END) if src in spec.DATED_SOURCES else None,
            missing_valid_days=_missing_days(dates, spec.VALID_START, spec.VALID_END) if src in spec.DATED_SOURCES else None,
            missing_columns=",".join(missing_columns(src, readable)) or None,
            size_gb=round(g["size_bytes"].sum() / 1e9, 3),
            median_rows=int(readable["num_rows"].median()) if readable.height else None,
            schema_variants=readable["schema_id"].n_unique(),
            target_instrument=inst in spec.TARGET_INSTRUMENTS,
        ))
    return pl.DataFrame(rows).sort(["source", "instrument"])


def schema_report(files: pl.DataFrame) -> dict:
    rep: dict = {}
    ok = files.filter(pl.col("readable"))
    for (src, inst), g in ok.group_by(["source", "instrument"], maintain_order=True):
        variants = (g.group_by("schema_id", "schema_str")
                     .agg(pl.len().alias("files"), pl.col("date").min().alias("from"),
                          pl.col("date").max().alias("to"))
                     .sort("from"))
        rep[f"{src}/{inst}"] = [
            dict(schema_id=r["schema_id"], files=r["files"], first=str(r["from"]),
                 last=str(r["to"]), columns=json.loads(r["schema_str"]))
            for r in variants.iter_rows(named=True)
        ]
    return rep


def missing_dates(files: pl.DataFrame) -> pl.DataFrame:
    """Every calendar day absent inside the official train+valid window."""
    rows = []
    for (src, inst), g in files.group_by(["source", "instrument"], maintain_order=True):
        if src not in spec.DATED_SOURCES:
            continue
        have = set(g["date"].drop_nulls().to_list())
        d = spec.TRAIN_START
        while d <= spec.VALID_END:
            if d not in have:
                rows.append(dict(source=src, instrument=inst, missing_date=d))
            d += timedelta(1)
    return pl.DataFrame(rows, schema={"source": pl.Utf8, "instrument": pl.Utf8, "missing_date": pl.Date})


def readiness(data_dir: Path, files: pl.DataFrame, summary: pl.DataFrame,
              ts: pl.DataFrame, repo_dirs: list[Path]) -> list[tuple[str, str, str]]:
    """(item, status, detail). status in PASS / WARN / FAIL / MISSING."""
    out: list[tuple[str, str, str]] = []
    present = set(files["source"].unique().to_list())
    inst_of = lambda src: set(files.filter(pl.col("source") == src)["instrument"].to_list())

    # --- directories & instrument coverage --------------------------------
    for src in ("book", "bbo", "trades"):
        if src not in present:
            out.append((f"dir {src}/", "FAIL", "not found")); continue
        have = inst_of(src)
        lack_t = [i for i in spec.TARGET_INSTRUMENTS if i not in have]
        lack_f = [i for i in spec.FEATURE_INSTRUMENTS if i not in have and i not in spec.TARGET_INSTRUMENTS]
        unexpected = sorted(have - set(spec.FEATURE_INSTRUMENTS))
        status = "FAIL" if lack_t else ("WARN" if lack_f else "PASS")
        detail = f"{len(have)} instruments"
        if lack_t: detail += f"; missing TARGETS {lack_t}"
        if lack_f: detail += f"; missing feature instruments {lack_f}"
        if unexpected: detail += f"; unrecognised names {unexpected} (spot naming is inferred)"
        out.append((f"dir {src}/", status, detail))

    if "targets" in present:
        lack = [i for i in spec.TARGET_INSTRUMENTS if i not in inst_of("targets")]
        out.append(("dir targets/", "FAIL" if lack else "PASS",
                    f"missing {lack}" if lack else "all 4 perps"))
    else:
        out.append(("dir targets/", "MISSING", "training and official scoring BLOCKED"))

    out.append(("dir stable/", "PASS" if "stable" in present else "MISSING",
                "flat daily files (pair column)" if "stable" in present else "cross-stablecoin features blocked"))

    meta_dir = data_dir / "meta"
    stems = {p.stem for p in meta_dir.rglob("*") if p.is_file()} if meta_dir.is_dir() else set()
    lack_meta = [m for m in spec.META_FILES if m not in stems]
    out.append(("meta/ instruments, daily, target_clip",
                "MISSING" if not stems else ("WARN" if lack_meta else "PASS"),
                f"missing {lack_meta}; found {sorted(stems)}" if lack_meta else "all present"))

    for name in spec.CODE_FILES:
        hits = [str(d / name) for d in repo_dirs if (d / name).exists()]
        out.append((name, "PASS" if hits else "MISSING", hits[0] if hits else "not found"))

    # --- file & schema health ---------------------------------------------
    bad = summary.filter(pl.col("unreadable") > 0)
    out.append(("all files readable", "FAIL" if bad.height else "PASS",
                f"{bad['unreadable'].sum()} unreadable" if bad.height else ""))
    cols = summary.filter(pl.col("missing_columns").is_not_null())
    out.append(("PDF-required columns present", "FAIL" if cols.height else "PASS",
                "; ".join(f"{r['source']}/{r['instrument']}: {r['missing_columns']}"
                          for r in cols.iter_rows(named=True))[:400]))
    drift = summary.filter(pl.col("schema_variants") > 1)
    out.append(("one schema per source/instrument", "WARN" if drift.height else "PASS",
                ", ".join(f"{r['source']}/{r['instrument']}" for r in drift.iter_rows(named=True))))

    # --- coverage of train + valid ----------------------------------------
    core = summary.filter(pl.col("source").is_in(["book", "bbo", "trades", "targets"])
                          & pl.col("target_instrument"))
    gaps = core.filter((pl.col("missing_train_days") > 0) | (pl.col("missing_valid_days") > 0))
    out.append(("target perps cover full train+valid window", "WARN" if gaps.height or core.height == 0 else "PASS",
                f"{gaps.height} groups with gaps (see missing_dates.csv)" if gaps.height else ""))

    # --- timestamp conventions --------------------------------------------
    if ts.height:
        per_row = pl.col("source") != "stable"   # stable has 3 pairs per ts, so dups are expected
        issues = ts.filter((per_row & ((pl.col("monotonic") == False) | (pl.col("dup_ts") > 0)))  # noqa: E712
                           | (pl.col("pct_on_grid") < 100) | (pl.col("pct_inside_day") < 100)
                           | pl.col("ts_col").is_null()
                           | ((pl.col("source") == "bbo") & (pl.col("pct_on_500ms") > 0))
                           | ~pl.col("ts_unit").fill_null("").str.starts_with("int64 epoch-ms"))
        out.append(("sampled ts: int64 ms, sorted, unique, on grid, inside UTC day, bbo off 500 ms points",
                    "WARN" if issues.height else "PASS",
                    f"{issues.height} sampled files flagged (see ts_checks.csv)" if issues.height else ""))
    return out
