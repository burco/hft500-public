# Next 500 ms — short-horizon return prediction for Binance perpetuals

```bash
uv sync --frozen                       # numpy 2.2.6, polars 2.0.0, pyarrow 25.0.1, lightgbm 4.7.0 (Python 3.10–3.12)
uv run python predict.py --data-dir <DIR> --date <YYYY-MM-DD> --out <PATH>.parquet
```

- `<DIR>` uses the official layout without `targets/` and holds the requested day plus the previous day. `meta/` is used if present but is optional.
- Output: 691,200 rows (4 perps × 172,800 grid points) with columns `ts` (int64 ms UTC), `instrument`, `pred_500ms`, `pred_2s` and `pred_8s` (float64, all finite).
- The script keeps no state between calls.

## Approach
1. **Quote reconstruction.** The quote grid is rebuilt the way `common.py` does it: book rows give the 500 ms points, bbo changes fill in between, and values are only ever carried forward. The prediction grid is generated from the date, never from the data. The previous day is used only to warm up the features.
2. **46 causal features per perp.** Every feature at t uses only rows with `ts ≤ t`.
   - *Own book:* best-level queue imbalance with its 1 s change and 5 s / 30 s EWMAs; depth imbalance within 0.5 / 1 / 2 / 10 bps; spread in ticks and bps; microprice − mid.
   - *Own price:* past returns over 0.5–30 s plus a volatility-scaled 10 s return; realised volatility over 30 s and 5 min and their ratio; mid-change counts; staleness ages and a frozen-feed flag.
   - *Trade flow:* buy/sell imbalance over 0.5 / 2 / 10 s; signed notional relative to its usual size; side of the last trade.
   - *Order-flow imbalance (OFI):* sampled from the 100 ms quote grid, over 0.5 / 1 / 2 / 5 s.
   - *Cross-market:* the other perps' past 0.5 s and 2 s returns; perp–spot basis minus its 10 s EWMA; the USDC-quoted twin converted with the `stable` USDCUSDT mid (never with price ratios), as a level and as a deviation from its 60 s EWMA. Depeg-nulled prices stay null.
3. **Models.** One LightGBM regressor per (perp, horizon), 12 in total. Settings: 127 leaves, 100 trees, learning rate 0.05, at least 1,000 rows per leaf, L2 = 10, feature fraction 0.8, bagging 0.5. Trained on clipped targets, using every 40th grid point. Missing values go to the trees as-is. Any non-finite prediction becomes 0, which the metric treats like a missing prediction.
4. **Selection was causal throughout.** Features were evaluated and pruned with monthly walk-forward folds inside the training period (1-day embargo). Model settings were tuned on internal folds. The validation period was scored **once**, with everything fixed beforehand. The frozen model was then refit once on train + validation (365 days), which the brief allows.

## Results (official metric: daily Pearson on clipped targets, mean over days, then over the 12 cells)
| model | validation 2026-07-15…09-14 |
|---|---|
| official baseline (OLS on imbalance) | 0.1838 |
| ridge on the same 46 features | 0.2076 |
| **this model (before the refit)** | **0.2247** |

- LightGBM beats ridge in all 12 cells.
- The gain is largest at 500 ms (+0.03 to +0.045) and smallest at 8 s (+0.002 to +0.005).
- The submitted weights include the validation days, so no unbiased score exists for them. The 0.2247 above is the honest estimate.
- One caveat: timestamps are the collector's receive times, so the cross-market lead-lag features assume collector latency in the test period is similar to the training period.

## Causality and robustness checks (all passed)
- **Official-style truncation test of this package.** `predict.py` was run as a separate process on full days and on copies of the data folder cut at T in every file. 15/15 cuts on 2026-08-16 (which has a BTC feed freeze) and 2026-09-14 matched bit-for-bit for all t ≤ T, including T = 00:00:00.000 / 00:00:00.500 / 23:59:59.500.
- **Research-repo audit.** 41 more end-to-end cuts on 4 days, and 2 deliberately leaky features (a one-step look-ahead and a whole-day statistic) that the test correctly **failed**.
- **Data access.** A full-day run reads only D and D−1, and never `targets/`.
- **Regression check.** Running this package with the pre-refit weights reproduced the research pipeline's validation score for 2026-08-16 exactly (0.221747).
- **Robustness.** With the previous day missing, or with a perp's book file missing and `meta/` absent, the script still writes 691,200 finite rows.

## Cost (2-core / 4 GB Linux VM)
- One day for all 4 perps: about 11.5 s wall-clock, peak memory 0.94 GB.
- Training: about 45 s per perp (3 boosters).
- Weights: 15 MB in total.

## Files
- `predict.py`: the command-line entry point.
- `models/<PERP>/y_{500ms,2s,8s}.txt`: the LightGBM weights. `meta.json` lists the feature order and `fit_info.json` the training window and parameters.
- `hft500/`: helper modules (quote alignment, feature builders, model wrapper, pipeline). They are copied unchanged from the research repository, where they have 68 unit tests.
