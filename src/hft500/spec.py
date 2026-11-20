"""Assignment constants — checked against Short_Term_Perpetual_Predictions_EN_1.pdf (2026-11-12).

Items marked INFERRED are not stated in the PDF and must be confirmed on disk.
"""
from __future__ import annotations

from datetime import date

TARGET_INSTRUMENTS: tuple[str, ...] = (
    "BTCUSDT_PERP",
    "ETHUSDT_PERP",
    "SOLUSDT_PERP",
    "1000PEPEUSDT_PERP",
)

# 20-instrument feature universe = 4 targets + 1000PEPEUSDC_PERP + 12 spot pairs + 3 stablecoin rates.
# Spot naming "<BASE><QUOTE>_SPOT" is INFERRED from the PDF example "BTCUSDC_SPOT".
EXTRA_PERPS: tuple[str, ...] = ("1000PEPEUSDC_PERP",)
SPOT_INSTRUMENTS: tuple[str, ...] = tuple(
    f"{b}{q}_SPOT" for b in ("BTC", "ETH", "SOL") for q in ("USDT", "USDC", "FDUSD")
) + ("BTCUSD1_SPOT", "ETHUSD1_SPOT", "PEPEUSDT_SPOT")
STABLE_PAIRS: tuple[str, ...] = ("USDCUSDT", "FDUSDUSDT", "USD1USDT")  # rows of stable/<date>.parquet
USD1_START = {"BTCUSD1_SPOT": date(2025, 12, 8), "ETHUSD1_SPOT": date(2025, 12, 15)}
FEATURE_INSTRUMENTS = TARGET_INSTRUMENTS + EXTRA_PERPS + SPOT_INSTRUMENTS   # 17 with book/bbo/trades

HORIZONS_MS: dict[str, int] = {"pred_500ms": 500, "pred_2s": 2_000, "pred_8s": 8_000}
LATENCY_OFFSET_MS = 10
PRED_STEP_MS = 500
PRED_POINTS_PER_DAY = 86_400_000 // PRED_STEP_MS               # 172,800
OUTPUT_ROWS_PER_DAY = PRED_POINTS_PER_DAY * len(TARGET_INSTRUMENTS)  # 691,200
OUTPUT_COLUMNS = ("ts", "instrument", "pred_500ms", "pred_2s", "pred_8s")

TRAIN_START, TRAIN_END = date(2025, 9, 15), date(2026, 7, 14)   # inclusive
VALID_START, VALID_END = date(2026, 7, 15), date(2026, 9, 14)   # inclusive
TEST_START = date(2026, 9, 15)                                  # no data provided

# Book depth buckets: (lo, hi] in bps from mid; never narrower than one tick.
# Actual daily bounds live in meta/daily.parquet.
BUCKET_BOUNDS_BPS = ((0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 10), (10, 20),
                     (20, 30), (30, 40), (40, 50), (50, 75), (75, 100))

# Timestamp grid per source (ms). ts = int64 UTC ms, collector RECEIVE time.
SOURCES: dict[str, int | None] = {
    "book": 500,      # one row per grid point
    "bbo": 100,       # change-only; ts % 100 == 0 and NEVER a multiple of 500 (those are in book)
    "trades": 100,    # bar (ts-100, ts]; empty bars not recorded
    "stable": 1_000,  # flat: stable/<date>.parquet, dense, one row per pair per second
    "targets": 500,   # perps only, train + valid only
    "meta": None,     # static: instruments, daily, target_clip
}
DATED_SOURCES = ("book", "bbo", "trades", "stable", "targets")
FLAT_SOURCES = ("stable",)        # no instrument sub-folder
CONFIRMED_SOURCES = ("book", "bbo", "trades")
CODE_FILES = ("common.py", "baseline.py", "predict.py")
META_FILES = ("instruments", "daily", "target_clip")   # stems; extension not stated in full

_SIDE = ("bid", "ask")
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "book": ("ts", "bid_px", "ask_px", "bid_qty", "ask_qty")
            + tuple(f"{s}_v{k:02d}" for s in _SIDE for k in range(1, 12)),
    "bbo": ("ts", "bid_px", "ask_px", "bid_qty", "ask_qty"),
    "trades": ("ts",) + tuple(f"{s}_{f}" for f in ("qty", "vwap", "n", "max_qty", "last_ms")
                              for s in ("buy", "sell")),
    "stable": ("ts", "pair", "bid_px", "ask_px", "bid_qty", "ask_qty", "buy_qty", "sell_qty", "stale_ms"),
    "targets": ("ts", "y_500ms", "y_2s", "y_8s"),
}

TS_CANDIDATES = ("ts", "timestamp", "time")
