# Stage 3 — data-quality findings and the decisions they force

Scan: all 6,403 instrument-days (17 instruments) + 365 stable days, every row read, 0 errors.
Source tables: summary_*.csv, flags.csv, tick_recheck.csv, fig_*.png in this folder.

| # | Finding (measured) | Decision | Stage |
|---|---|---|---|
| 1 | Book is never complete: every day misses its first ~1-2 s (00:00:00.000...) and targets are NaN at exactly those points (median 3/day per perp). 14 instrument-days miss >100 points; worst 2026-07-28 (all perps, BTC 2,228 pts) and 2026-03-26 (BTC, ETH). | Build the 500 ms grid from the date; carry state from the previous day for warm-up; drop NaN targets in training/scoring like baseline.py does. | 4, 5 |
| 2 | Frozen-price episodes: best prices unchanged for 15-50 min while sizes, bbo updates and trades keep flowing (e.g. BTC perp 2026-08-16 08:44-09:34: 16,545 bbo rows, 164 BTC traded, all at the frozen prices; BTC spot frozen too). Hits all 4 perps together on 2026-05-21, 06-12, 07-14, 07-28; 21 perp-days in total. Targets ~all zero inside. | Treat as a feed artefact, not market behaviour. Build a CAUSAL "price unchanged for N s" feature/indicator; consider down-weighting these windows in training; never use them to tune. | 7, 10 |
| 3 | meta tick_size has float noise (0.01 stored as 0.00999999999476131) -> naive tick checks fail. | Always clean ticks to 8 significant digits (quality.clean_tick). | all |
| 4 | meta tick_size for SOLUSDT_PERP is wrong on 164/365 days (0.001-0.004; real tick 0.01 every day). Bucket edges are still consistent with the real tick. USD1 first days have nonsense ticks (5.05, 0.84). | Use the empirical tick (quality.empirical_tick) for anything measured in ticks; keep meta bucket edges. | 7 |
| 5 | NaN prices/buckets on 17 spot-days = the PDF's stablecoin null rule; dates match meta/depeg_intervals (FDUSD 2025-09-24/10-10/10-11, USDC 10-10, USD1 2026-02-23). FDUSD sits 20-30 bps off 1 for months. | Expected. Cross-stablecoin features must handle nulls; convert with the stable table, never with price ratios. | 8 |
| 6 | Real stress events: 2025-10-10 crash (BTC perp 316 bps in 500 ms, PEPE 900 bps), 2026-08-22 (SOL 608 bps). 822-1,059 >50 bps jumps on PEPE perps. | Keep (real). Clip targets with meta/target_clip; winsorise features; Pearson is outlier-sensitive. | 5, 7 |
| 7 | Regime drift: share of zero y_500ms rises from ~80% (train) to ~88% (valid) on BTC; target std falls (BTC 0.43 -> 0.32 bps). Spreads in bps rise all year as prices fall with a fixed tick. | Normalise features by recent vol/spread; validate on later months; expect test to differ again. | 7, 9, 10 |
| 8 | Clip bounds remove exactly 0.20% of train targets (as designed) and 0.07-0.23% of valid. | Metric implementation check passes. | 6 |
| 9 | Trade bars: only 10 days with a VWAP outside the day's mid range (event days); no structural violations. trades.vwap is NaN (not null) when a side is empty. | Use fill_nan before null logic. | 8 |
