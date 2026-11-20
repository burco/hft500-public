# Stage 7: order-book features

32 causal features in 6 families (src/hft500/features_book.py; dictionary: feature_dictionary.csv).
Build time ~0.7 s per instrument-day. Truncation test: 100/100 cuts passed (4 perps x 6 days incl.
first day, crash, outage, last training day).

Exploratory daily Pearson with clipped targets: TRAINING days only, every 14th day (22/perp).
NOT feature selection (Stage 9). Mean over the 12 cells, then min/max cell:

| feature | mean IC | range | note |
|---|---|---|---|
| imb_l1 (= baseline imbalance) | 0.200 | 0.110-0.260 | identical to the book-row imbalance at every grid point that has a book row (book holds the 500 ms points) |
| mp_dev_ticks | 0.199 | 0.108-0.263 | same information as imb_l1 at a 1-tick spread |
| dimb_0.5bps / 1bps / 2bps / 10bps | 0.199 / 0.191 / 0.165 / 0.063 | | deeper = weaker; dimb_1bps corr 0.87 with imb_l1 |
| imb_l1_chg_1s | 0.130 | 0.062-0.190 | only 0.41 correlated with imb_l1 -> likely new information |
| imb_l1_ewm_5s / 30s | 0.075 / 0.029 | | slow imbalance carries less |
| ret_0.5s ... ret_60s | 0.059 ... 0.005 | | momentum, decays with lookback; corr 0.25 with imb_l1 |
| retz_* (vol-scaled) | 0.060 / 0.052 / 0.031 | | about the same as raw returns |
| spread, volatility, activity, staleness | ~0.00 | | directionless by design: useful only as CONDITIONERS (e.g. imbalance matters less when prices are frozen) -> trees, interactions |

Lessons
1. A whole-day tick estimate leaks (it changes under truncation); the tick comes from the previous day.
   SOL 2025-12-18 (meta tick 0.003): previous-day tick 0.01, 99.86% one-tick spreads.
2. "Nothing known yet" must be NaN whichever code path runs, or the first day of data fails the check.
3. The empirical tick estimator must ignore float noise between equal prices.
4. Hypotheses for Stage 9: imb_l1_chg_1s and ret_0.5s add information beyond imb_l1; depth beyond 1 bps adds little;
   staleness/volatility help only through interactions.
