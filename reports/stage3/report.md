# Stage 3 — data quality report

Generated from measured per-day metrics. Validation starts 2026-07-15 (dashed line in charts).

## Flags

| flag | meaning | days | instruments |
|---|---|---|---|
| targets_gap | missing / non-finite targets | 1460 | 4 |
| book_jump | 500 ms mid move > 50 bps | 463 | 17 |
| prices_frozen | best prices unchanged > 15 min (feed freeze?) | 308 | 14 |
| book_off_tick | price not a multiple of the (float-cleaned) meta tick | 18 | 3 |
| book_nan_values | NaN prices/buckets (expected on stablecoin-depeg days: PDF null rule) | 17 | 9 |
| book_gap | book missing > 100 grid points (> 50 s) | 14 | 10 |
| trades_inconsistent | trade-bar fields contradict each other | 10 | 8 |
| stable_depeg | 4353 s beyond 50 bps | 1 | 1 |
| stable_depeg | 508 s beyond 50 bps | 1 | 1 |
| stable_depeg | 7656 s beyond 50 bps | 1 | 1 |
| stable_depeg | 384 s beyond 50 bps | 1 | 1 |
| stable_depeg | 271 s beyond 50 bps | 1 | 1 |

Files: summary_instruments.csv, summary_targets.csv, summary_stable.csv, flags.csv, fig_*.png
