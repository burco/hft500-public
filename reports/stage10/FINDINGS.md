# Stage 10 — Model comparison

Code: `src/hft500/models.py` (tests in `tests/test_models.py`; the full suite has 63 passing), `scripts/stage10_tune.py` and `scripts/stage10_final.py`.

## Protocol
1. **Internal folds (training period only).**
   - Linear models were scored on 8 monthly walk-forward folds (Dec–Jul) with a 1-day embargo.
   - LightGBM settings were tuned on the May 2026 fold: two leaf sizes (31 and 127), tree counts from 100 to 800, and a pruned-vs-full feature-set check on 6 cells. The chosen settings were then confirmed on the Mar and Jul folds.
   - Training data is every 40th grid point. Targets are clipped.
2. **Final fits** on all 303 training days.
3. **One validation run** (2026-07-15 to 2026-09-14, 62 days × 4 coins), logged in `validation_log.csv`.
   - Features were built on the full 500 ms grid and scored with the official metric (clipped targets, missing prediction = 0).
   - All models were fixed before this run. Nothing was changed afterwards.

## Internal folds (mean of 12 cells)
| model | Mar | May (tuning month) | Jul | 8-fold mean |
|---|---|---|---|---|
| imb_only (≈ official OLS) | 0.189 | 0.215 | 0.201 | 0.1965 |
| ridge, 46 features (λ≈0, the same as OLS) | 0.203 | 0.238 | 0.219 | 0.2137 |
| LightGBM, 127 leaves × 100 trees | 0.215 | 0.256 | 0.241 | — |

- **Ridge needs no shrinkage.** The best penalty was essentially zero; λ=10 dropped the score to 0.201.
- **LightGBM settings were nearly tied.** 127 leaves × 100 trees scored 0.2558; 31 leaves × 200 trees scored 0.2554.
- **Feature set doesn't matter for trees.** Full vs pruned scored 0.2515 vs 0.2513.

## Validation (official metric, full grid) — single look
| model | score | train time (4 coins) | size | prediction time per coin-day |
|---|---|---|---|---|
| zero | 0.0000 | — | 0 | — |
| imb_only (≈ official OLS) | **0.1838** | <1 s | 12 weights | 0.8 s |
| OLS, all 61 features | 0.2073 | <1 s* | 777 | 0.8 s |
| ridge, 46 features | 0.2076 | <1 s* | 570 | 0.8 s |
| **LightGBM, 46 features** | **0.2247** | 145 s | 1,200 trees (≈305k leaf params) | 14.9 s** |

\* Fitted from the per-day sums computed in Stage 9 (about 60 s per coin for that pass).
\*\* With 1 thread while 3–4 workers shared 2 cores. Feature building takes about 5.2 s per coin-day (max 11 s). Peak memory was 1.3 GB per worker process; 4 workers on this 4 GB VM ran out of memory once.

- **The imb_only model reproduces the official baseline exactly** (Stage 6 official: 0.183844).
- **Scoring with baseline.py's 10,000-point day filter gives identical numbers.**
- **Every prediction was finite and non-constant.**

### Per cell (validation)
| coin | horizon | imb_only | ridge | LightGBM | LightGBM − ridge |
|---|---|---|---|---|---|
| BTC | 500 ms | 0.1945 | 0.2440 | 0.2889 | +0.045 |
| BTC | 2 s | 0.2155 | 0.2463 | 0.2630 | +0.017 |
| BTC | 8 s | 0.1914 | 0.2084 | 0.2133 | +0.005 |
| ETH | 500 ms | 0.2004 | 0.2337 | 0.2728 | +0.039 |
| ETH | 2 s | 0.2019 | 0.2211 | 0.2320 | +0.011 |
| ETH | 8 s | 0.1583 | 0.1688 | 0.1718 | +0.003 |
| SOL | 500 ms | 0.1958 | 0.2250 | 0.2594 | +0.034 |
| SOL | 2 s | 0.2212 | 0.2391 | 0.2503 | +0.011 |
| SOL | 8 s | 0.1912 | 0.1998 | 0.2033 | +0.004 |
| PEPE | 500 ms | 0.1550 | 0.1823 | 0.2093 | +0.027 |
| PEPE | 2 s | 0.1617 | 0.1861 | 0.1937 | +0.008 |
| PEPE | 8 s | 0.1191 | 0.1367 | 0.1390 | +0.002 |

- **LightGBM beats ridge in all 12 cells and in each month** (Jul / Aug / Sep: 0.249 / 0.217 / 0.213 vs 0.229 / 0.200 / 0.198). It wins on 90.9% of individual cell-days.
- **The gain is mostly at 500 ms** (+0.027 to +0.045) and small at 8 s (+0.002 to +0.005). The nonlinearity sits in the fastest signals.
- **Validation is below the internal folds** (0.2247 vs about 0.24 for LightGBM). This matches the calmer validation period found in Stage 5. The ranking of models is unchanged.

## Decision for Stage 11–12
- **Candidate model:** LightGBM, configuration B (127 leaves, 100 trees, learning rate 0.05), on the 46 pruned features.
- **Final step:** freeze these hyperparameters and refit on train + validation, as the brief allows.
- **Ridge stays as a fallback** (0.2076): cheap and transparent.
- **Not tried** (possible later work, all must use internal folds only): blending LightGBM with ridge, one model per horizon pooled across coins, and deep learning.
