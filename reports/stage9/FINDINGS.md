# Stage 9 — Feature evaluation (training days only)

This stage covers 61 features (32 order-book + 29 trade/cross-market) on all 303 training days for each of the 4 perps, 1,212 instrument-days in total. The validation period (Jul 15 – Sep 14) was **not used**.

Code is in `src/hft500/evaluation.py` (tests: `tests/test_evaluation.py`; the full suite has 60 passing), plus `scripts/stage9_build.py` and `scripts/stage9_report.py`.

## Method
- **Univariate statistics** are computed on the full 500 ms grid for each day. They include Pearson and Spearman correlation with the clipped target, decile means of the target by feature value, and coverage. Across days I report the mean, a t-statistic, the share of positive days, sign flips by month, and decile monotonicity. Caveat: daily correlations are autocorrelated, so the t-statistics overstate certainty. They are used for ranking only.
- **Out-of-sample evidence** comes from a standardised ridge regression on walk-forward folds. Each test month (Dec through Jul, 8 folds) is predicted from a model trained on all earlier days, leaving one day out before the test month (the embargo).
  - Linear preparation: clip each feature at its 0.1/99.9% bounds estimated from Sep–Nov (inside every fold's training data), fill NaN with 0, and add a missing-value indicator.
  - The model is fitted from per-day sums, so the whole training set never has to sit in memory.
  - Scores use the official daily Pearson metric on every 10th grid point (17,280 points per day).
- **Clusters** group features whose |Spearman| is at least 0.9, averaged over the 4 instruments.

## Out-of-sample results (mean of the 12 cells)
| model | score |
|---|---|
| imb_l1 only (official-baseline feature) | 0.1965 |
| all 61 features | 0.2136 |
| pre-registered prune, cluster step (53 features) | 0.2064, **rejected** |
| pre-registered prune, cluster + drop-one (43 features) | 0.2066, **rejected** |
| **amended prune (46 features) — selected** | **0.2137** |

- **Every cell improves.** The amended set beats imb_l1 alone in all 12 cells, by +0.006 (SOL 8 s) up to +0.040 (BTC 500 ms).
- **Every month improves.** The gain is positive in all 8 folds, ranging from +0.011 (Feb) to +0.022 (May).
- **The gain shrinks with horizon:** about +0.026 at 500 ms, +0.016 at 2 s and +0.008 at 8 s.

### Pruning rule change (disclosed)
The rule I fixed in advance first reduced each cluster to one member, choosing the member with the largest gain when added on its own to imb_l1. That rule had two flaws:
1. imb_l1's gain over itself is zero by construction, so it could never be chosen as the representative.
2. The imbalance variants (imb_l1, dimb_0.5bps, dimb_1bps, mp_dev_ticks) are correlated above 0.9, but they still complement each other in a linear model. Removing them cost 0.007.

The amended rule makes a single drop-one pass from the full model. A non-conditioner feature is removed if dropping it does not hurt on average **and** it helps in fewer than 6 of the 12 cells. Pruning is accepted only if the result scores at least the full model minus 0.001, which it does (0.2137 vs 0.2136).

This second choice reuses the same training folds, so it is slightly optimistic. Stage 10 will confirm it on the validation period.

**Removed (15):** basis_bps, basis_dev_60s, depth2bps_rel, depth_near_share, intensity_10s, ret_1s, ret_5s, ret_60s, retz_0.5s, retz_2s, spot_minus_perp_0.5s, sret_0.5s, sret_2s, svn_2s, tim_30s.

**Conditioners kept for trees:** volatility, activity and staleness features. They have about zero linear correlation but may matter through interactions.

## Family ablation (from the full model)
| family | loss if removed | gain when added to imb_l1 alone | cells |
|---|---|---|---|
| imbalance (chg / ewm) | 0.0043 | 0.0081 | 12/12 |
| depth | 0.0016 | 0.0029 | 9/12 |
| leadlag | 0.0015 | 0.0028 | 12/12 |
| ofi | 0.0015 | 0.0036 | 12/12 |
| trades | 0.0011 | 0.0026 | 12/12 |
| stable (usdc gap) | 0.0009 | 0.0020 | 12/12 |
| spread | 0.0003 | 0.0005 | 12/12 |
| momentum, spot | 0.0001 | 0.0003 / 0.0013 | — |
| staleness, activity, volatility | ≈0 | ≈0 | — |

- **Spot overlaps with other features.** Spot features add 0.0013 alone but only 0.0001 once the others are present: spot returns largely duplicate the lead-lag and OFI information.

## Stability and nonlinearity
- **Stable signals.** The strongest features keep their sign in every month, but they weaken from autumn to summer. At 500 ms, imb_l1 falls from 0.25 to 0.21, ofi_0.5s from 0.17 to 0.13 and xret_eth_0.5s from 0.13 to 0.10. This matches Stage 5's finding that later months are calmer.
- **Possibly nonlinear signals.** Spread and staleness features have about zero linear correlation but a wider decile range (spread_bps about 0.12 bps), so they are candidates for tree models. The basis has low monotonicity (0.47), a U-shape hint.

## Outputs
`feature_report.csv` (ranked, with decision and reason), `pruned_features.json`, `ablation_family.csv`, `clusters.csv`, `abs_spearman.parquet`, `cell_univariate.parquet`, `ic_by_month.parquet`, `oos_round1-4.parquet`, `oos_final_models.csv`, and `daily/` (per-day statistics).

The cache (every 10th grid point, float32, about 2.6 MB per instrument-day) is in the VM scratch area, not in OneDrive. Stage 10 reuses it.
