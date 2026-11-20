"""Stage 2 primitives, checked against values computed BY HAND from a real
BTCUSDT_PERP book row (2026-04-06 03:00:00.000 UTC)."""
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hft500 import micro as m  # noqa: E402

ROW = pl.DataFrame({  # real row; buckets 3..11 set to 0 to keep hand maths short
    "bid_px": [69183.6], "ask_px": [69183.7], "bid_qty": [1.371], "ask_qty": [6.428],
    **{f"bid_v{i:02d}": [0.0] for i in range(1, 12)}, **{f"ask_v{i:02d}": [0.0] for i in range(1, 12)},
}).with_columns(bid_v01=pl.lit(154554.375), ask_v01=pl.lit(792020.9375),
                bid_v02=pl.lit(50914.984375), ask_v02=pl.lit(253093.25))


def one(expr):
    return ROW.select(expr.alias("x"))["x"][0]


def test_prices():
    assert one(m.mid()) == pytest.approx(69183.65)
    assert one(m.spread()) == pytest.approx(0.1)
    assert one(m.spread_ticks(0.1)) == 1
    assert one(m.spread_bps()) == pytest.approx(0.1 / 69183.65 * 1e4)      # ~0.01445 bps


def test_imbalance_and_microprice():
    imb = (1.371 - 6.428) / (1.371 + 6.428)                                  # -0.64842
    assert one(m.queue_imbalance()) == pytest.approx(imb)
    mp = (69183.6 * 6.428 + 69183.7 * 1.371) / (1.371 + 6.428)               # 69183.61758
    assert one(m.microprice()) == pytest.approx(mp)
    # identity: microprice - mid = imb * spread / 2
    assert one(m.microprice() - m.mid()) == pytest.approx(imb * 0.1 / 2)
    assert one(m.microprice_dev_ticks(0.1)) == pytest.approx(imb / 2)


def test_depth():
    b1, a1 = 154554.375, 792020.9375
    assert one(m.depth_imbalance(1)) == pytest.approx((b1 - a1) / (b1 + a1))  # -0.67344
    b2, a2 = b1 + 50914.984375, a1 + 253093.25
    assert one(m.depth_imbalance(2)) == pytest.approx((b2 - a2) / (b2 + a2))
    with pytest.raises(ValueError):
        m.depth_imbalance(12)


def test_empty_sides_are_zero_not_nan():
    df = pl.DataFrame({"bid_qty": [0.0], "ask_qty": [0.0], "buy_qty": [0.0], "sell_qty": [0.0]})
    r = df.select(q=m.queue_imbalance(), t=m.trade_imbalance())
    assert r.row(0) == (0.0, 0.0)


def test_trade_flow():
    df = pl.DataFrame({"buy_qty": [0.0, 5.098], "sell_qty": [11.514, 0.04]})  # two real 100 ms bars
    r = df.select(sv=m.signed_volume(), ti=m.trade_imbalance())
    assert r["sv"].to_list() == pytest.approx([-11.514, 5.058])
    assert r["ti"].to_list() == pytest.approx([-1.0, 5.058 / 5.138])


def test_log_return_bps():
    r = pl.DataFrame({"p0": [69183.65], "p1": [69180.05]}).select(m.log_return_bps(pl.col("p0"), pl.col("p1")))
    assert r.item() == pytest.approx(-0.520368, abs=1e-5)   # matches the supplied y_500ms target
