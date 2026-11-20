"""Stage 13 fast features + evaluate(): known answers and leakage tests on a SYNTHETIC market.

The market is built so the right answers are known in advance:
* ETH and BTC perps quote on the 100 ms grid (book at 500 ms points, bbo in between);
* BTC copies ETH's log-mid move exactly ONE 100 ms step later (a planted 100 ms lead);
* SOL and PEPE have no files at all (absent instruments must give NaN, not crash).
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hft500 import align as A, causality as C, features_fast as F, incremental as I  # noqa: E402

D1, D2 = "2025-09-15", "2025-09-16"
BTC, ETH = "BTCUSDT_PERP", "ETHUSDT_PERP"
STEPS = A.DAY_MS // 100


def _write_quotes(r: Path, inst: str, day: str, ts: np.ndarray, mid: np.ndarray, rng, tick: float):
    n = len(ts)
    q = {"ts": ts, "bid_px": mid - tick / 2, "ask_px": mid + tick / 2,
         "bid_qty": (rng.random(n) * 5 + 0.1).astype(np.float32), "ask_qty": (rng.random(n) * 5 + 0.1).astype(np.float32)}
    on500 = ts % 500 == 0
    book = {k: v[on500] for k, v in q.items()}
    for c in A.BUCKETS:
        book[c] = np.full(on500.sum(), 1e5, np.float32)
    for table, cols in (("book", book), ("bbo", {k: v[~on500] for k, v in q.items()})):
        p = r / table / inst / f"{day}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(cols).write_parquet(p)


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    r = tmp_path_factory.mktemp("fast")
    rng = np.random.default_rng(13)
    mids = {}
    eth_last = btc_last = None
    for day in (D1, D2):
        t0 = A.day_start(day)
        ts = np.arange(t0, t0 + A.DAY_MS, 100, dtype=np.int64)
        d_eth = rng.normal(0, 2e-5, len(ts))                          # ETH log-mid increments
        l_eth = (np.log(3000.0) if eth_last is None else eth_last) + np.cumsum(d_eth)
        d_btc = np.concatenate([[0.0], d_eth[:-1]])                   # BTC copies ETH one step later
        l_btc = (np.log(100_000.0) if btc_last is None else btc_last) + np.cumsum(d_btc)
        eth_last, btc_last = l_eth[-1], l_btc[-1]
        _write_quotes(r, ETH, day, ts, np.exp(l_eth), rng, 0.0)
        _write_quotes(r, BTC, day, ts, np.exp(l_btc), rng, 0.0)
        mids[day] = dict(ts=ts, eth=l_eth, btc=l_btc)
    return A.DirRoot(r), mids


@pytest.fixture(scope="module")
def btc_day2(market):
    root, _ = market
    return F.build(root, BTC, D2)


# --------------------------------------------------------------------------- shape and NaN rules
def test_shape_and_registry(btc_day2):
    assert btc_day2.height == 172_800
    assert btc_day2["ts"][0] == A.day_start(D2) and (np.diff(btc_day2["ts"].to_numpy()) == 500).all()
    assert set(F.FEATURES) <= set(btc_day2.columns)


def test_own_and_absent_coins_are_nan(btc_day2):
    for c in ("xr_btc_0.1s", "xr_btc_0.2s", "xofi_btc_0.5s",          # own coin
              "xr_sol_0.1s", "xr_pepe_0.2s", "xofi_sol_0.5s"):          # no files at all
        assert btc_day2[c].is_nan().all(), c
    for c in ("xr_eth_0.1s", "r_0.1s", "ofi_0.2s", "xofi_eth_0.5s", "gap_z_0.5s"):
        assert btc_day2[c].is_finite().mean() > 0.99, c


# --------------------------------------------------------------------------- known answers
def test_fast_return_known_answer(market, btc_day2):
    _, mids = market
    m = mids[D2]
    rng = np.random.default_rng(1)
    rows = rng.integers(10, 172_800, 50)
    t = btc_day2["ts"].to_numpy()[rows]
    k = (t - m["ts"][0]) // 100
    want01 = (m["eth"][k] - m["eth"][k - 1]) * 1e4
    want02 = (m["eth"][k] - m["eth"][k - 2]) * 1e4
    np.testing.assert_allclose(btc_day2["xr_eth_0.1s"].to_numpy()[rows], want01, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(btc_day2["xr_eth_0.2s"].to_numpy()[rows], want02, rtol=1e-6, atol=1e-6)


def test_planted_lead_is_found(market, btc_day2):
    """BTC's next 100 ms move IS ETH's last 100 ms move: the feature must see it."""
    _, mids = market
    m = mids[D2]
    t = btc_day2["ts"].to_numpy()[:-1]
    k = (t - m["ts"][0]) // 100
    btc_next = m["btc"][k + 1] - m["btc"][k]
    x = btc_day2["xr_eth_0.1s"].to_numpy()[:-1]
    assert np.corrcoef(x, btc_next)[0, 1] > 0.99
    # and the gap feature points the right way (ETH moved, BTC not yet -> BTC follows)
    g = btc_day2["gap_z_0.5s"].to_numpy()[:-1]
    ok = np.isfinite(g)
    assert np.corrcoef(g[ok], btc_next[ok])[0, 1] > 0.05


# --------------------------------------------------------------------------- leakage
def _fn(root, date):
    return F.build(root, BTC, date)


def test_truncation_bit_identical(market):
    root, _ = market
    d0 = A.day_start(D2)
    res = C.truncation_check(_fn, root, D2, n=3, seed=5,
                             Ts=[d0, d0 + 100, d0 + 500, d0 + 43_200_050, d0 + A.DAY_MS - 500])
    bad = [r for r in res if not r["ok"]]
    assert not bad, bad


def test_truncation_first_day_without_warmup(market):
    root, _ = market
    d0 = A.day_start(D1)
    res = C.truncation_check(_fn, root, D1, n=2, seed=6, Ts=[d0, d0 + 1_000])
    assert all(r["ok"] for r in res), res


def test_leaky_canary_is_caught(market):
    """A one-step look-ahead must FAIL the same check (proves the test can see leaks)."""
    root, _ = market

    def leaky(r, date):
        df = F.build(r, BTC, date)
        return df.with_columns(pl.col("xr_eth_0.1s").shift(-1))
    d0 = A.day_start(D2)
    res = C.truncation_check(leaky, root, D2, n=0, Ts=[d0 + 43_200_000])
    assert not res[0]["ok"] and "xr_eth_0.1s" in res[0]["bad_columns"]


def test_static_scan_feature_code():
    """No look-ahead idioms and no target access anywhere in the feature module."""
    src = inspect.getsource(F)
    code = "\n".join(l.split("#")[0] for l in src.splitlines() if not l.strip().startswith(('"', "'")))
    for pat in (r"shift\(\s*-", r"bfill", r"backward_fill", r"center\s*=\s*True", r"interpolate",
                r"read\(\s*['\"]targets", r"\[\s*k\s*\+\s*[2-9]", r"\.mean\(\)\s*$"):
        assert not re.search(pat, code, re.M), pat
    assert "targets" not in code.replace("TARGETS", "")


# --------------------------------------------------------------------------- evaluate()
def _frame(n_days=12, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_days):
        d = f"2026-01-{i + 1:02d}"
        b1, b2, good, noise = (rng.normal(size=n) for _ in range(4))
        y = 0.3 * b1 + 0.5 * good + rng.normal(size=n)
        rows.append(pl.DataFrame({"date": d, "ts": np.arange(n), "b1": b1, "b2": b2, "good": good, "noise": noise,
                                  "spread_ticks": rng.integers(1, 3, n).astype(float), "rv_5min": rng.random(n),
                                  "y_500ms": y, "y_2s": y, "y_8s": y}))
    return pl.concat(rows)


def test_evaluate_finds_planted_feature():
    r = I.evaluate_frame(_frame(), base=["b1", "b2"], candidates=["good", "noise"])
    f = pl.DataFrame(r["features"])
    good = f.filter(pl.col("feature") == "good")
    noise = f.filter(pl.col("feature") == "noise")
    assert (good["addone_gain"] > 0.2).all() and (good["resid_ic"] > 0.3).all()
    assert (noise["addone_gain"].abs() < 0.01).all() and (noise["resid_ic"].abs() < 0.03).all()
    c = pl.DataFrame(r["cells"])
    assert (c["gain"] > 0.2).all() and c["n_test_days"][0] == 4 and c["n_train_days"][0] == 7   # 12 = 7 + 1 embargo + 4
    assert {x["regime"] for x in r["regimes"]} == {"spread>1tick", "spread=1tick", "vol_high", "vol_low"}


def test_evaluate_refuses_validation_days_and_overlap():
    df = _frame().with_columns(pl.lit("2026-08-01").alias("date"))
    with pytest.raises(ValueError, match="validation"):
        I.evaluate_frame(df, ["b1"], ["good"])
    with pytest.raises(ValueError, match="already in the base"):
        I.evaluate_frame(_frame(), ["b1", "good"], ["good"])


def test_evaluate_train_test_never_overlap():
    tr, te = I.split_days([f"2026-02-{i:02d}" for i in range(1, 21)], 0.35, 1)
    assert max(tr) < min(te) and len(set(tr) & set(te)) == 0
    assert len(tr) + len(te) == 19      # one embargo day dropped
