# hft500 — Predicting the next 500 ms on Binance perpetual futures

A leak-free pipeline that forecasts log returns **500 ms, 2 s and 8 s ahead** for four Binance perpetual futures (BTCUSDT, ETHUSDT, SOLUSDT, 1000PEPEUSDT) from raw order-book, best-bid/offer and trade data, on a 500 ms prediction grid.

Built for the short-horizon challenge "Can You Predict the Next 500ms?". Scoring is the per-day Pearson correlation between predicted and realised returns, averaged over days and over the 12 instrument × horizon cells.

## Results (validation, 62 days, 2026-07-15 → 2026-09-14, scored once)

| Model | Score |
|---|---|
| Official baseline (imbalance OLS) | 0.1838 |
| Ridge, 46 features | 0.2076 |
| **LightGBM, 46 features** | **0.2247** (+22% vs baseline) |

| LightGBM | 500 ms | 2 s | 8 s |
|---|---|---|---|
| BTCUSDT | 0.2889 | 0.2630 | 0.2133 |
| ETHUSDT | 0.2728 | 0.2320 | 0.1718 |
| SOLUSDT | 0.2594 | 0.2503 | 0.2033 |
| 1000PEPEUSDT | 0.2093 | 0.1937 | 0.1390 |

- LightGBM beats ridge in all 12 cells, in every month, and on about 91% of cell-days. Most of the gain is at 500 ms.
- The validation period was looked at exactly once, with every model and hyperparameter fixed beforehand.

## How leakage is prevented
- **Causal alignment:** every feature is built with backward as-of joins from data at or before the prediction time; targets start 10 ms later to model order-submission delay.
- **Truncation test:** each day is predicted, then re-predicted with every record after time T hidden in every table and instrument. **41 / 41** end-to-end cuts are bit-identical, and two deliberately leaky features are caught (`reports/stage11/`).
- **Inference contract:** prediction code can only read day D and D−1 and can never read the targets table (`RestrictedRoot`).
- **Feature selection on training data only:** 46 of 61 candidate features, chosen with walk-forward folds and a 1-day embargo (`reports/stage9/`).

## Submission
Refit once on 365 days; predicts a full day for all four perps in **11.5 s** using **0.94 GB** on a 2-core VM; passes **15 / 15** official-style truncation checks (`reports/stage12/FINDINGS.md`, `submission/`).

## Stages
| Stage | What it does | Findings |
|---|---|---|
| 1 | Dataset inventory and schema audit | `reports/stage1` |
| 3 | Data quality | `reports/stage3` |
| 5 | Targets and their statistics | `reports/stage5` |
| 6 | Reproduce the official baseline (0.1838) | `reports/stage6` |
| 7 | Order-book features (32) | `reports/stage7` |
| 8 | Trade-flow and cross-market features (29), lead-lag | `reports/stage8` |
| 9 | Feature evidence and selection (46 kept) | `reports/stage9` |
| 10 | Model comparison on validation | `reports/stage10` |
| 11 | Causality and leakage audit | `reports/stage11` |
| 12 | Final refit and packaged submission | `reports/stage12` |
| 13 | 100 ms features experiment (+0.005, training folds only) | `reports/stage13` |

## Layout
```
src/hft500/     library: alignment, features, models, evaluation, causality checks
tests/          pytest suite (runs on a synthetic fixture, no market data needed)
scripts/        one script per stage
notebooks/      exploratory analysis
reports/        findings and summary tables per stage
submission/     predict.py entry point
```

## Run
```bash
uv sync
uv run pytest -q
```
Market data and fitted model weights are not included in this repository.

Stack: Python, Polars, NumPy, LightGBM, pytest, uv.
