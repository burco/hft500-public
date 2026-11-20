# Stage 8 — Trade-flow, OFI and cross-market features

Module: `src/hft500/features_cross.py`, with 29 features in `REGISTRY` (see feature_dictionary.csv). Tests: `tests/test_features_cross.py`; the full suite has 54 passing.

## Validation
- **Causality on real data:** 32 truncation cuts with 0 fails. Cuts used every instrument (including the other perps, spot and stable), on 4 targets × 1 day each. The days included the first day's midnight (BTC 2025-09-15) and the PEPE freeze day 2026-08-16, plus cuts at d0, d0+250 and d0+500 and inside the largest book gaps.
- **Build time:** about 1.5–2 s per instrument-day.
- **Coverage:** 0% NaN on a normal day, apart from the own-coin `xret_*` columns, which are NaN by design.
- **PEPE spot is very coarse.** `sret_0.5s` is zero at p1–p99, and one tick is about 28 bps. Spot features are therefore weak for PEPE.

## Exploratory IC (training days only, every 14th day, 22 days × 4 instruments)
"resid IC" is the correlation with the target after removing that day's linear fit on `imb_l1`. It tests whether a feature adds anything the baseline lacks. Values are means over the 12 cells (9 cells for xret).

| feature | IC | resid IC | days > 0 |
|---|---|---|---|
| xret_eth_0.5s | 0.088 | 0.038 | 98% |
| xret_sol_0.5s / xret_pepe_0.5s | 0.081 / 0.084 | 0.033 / 0.033 | 98% |
| imb_l1_chg_1s (Stage 7) | 0.130 | 0.033 | 97% |
| ofi_0.5s | 0.114 | 0.032 | 97% |
| last_side | 0.061 | 0.027 | 100% |
| usdc_gap_dev_60s | 0.047 | 0.027 | 98% |
| xret_btc_0.5s | 0.060 | 0.023 | 94% |
| tim_0.5s / svn_0.5s | 0.060 / 0.031 | 0.017 / 0.016 | 96% / 94% |
| basis_dev_10s | 0.010 | −0.008 | 31% (consistently negative) |
| tim_30s, svn_10s, intensity_10s, ofi_5s | ≈0 | ≈0 | — |

- **Fast signals decay with horizon.** For ofi_0.5s, resid IC is 0.054 at 500 ms, 0.030 at 2 s and 0.011 at 8 s. `usdc_gap_dev_60s` is flat at about 0.025–0.028 across all three horizons, so it is the only new feature that helps at 8 s.
- **Basis:** perp rich versus spot predicts that the perp falls. The sign is consistent, but the effect is small.
- **Longer windows** (2 s+ trade imbalance, 5 s OFI) add nothing once imbalance is known.

## Lead-lag and the latency caveat (leadlag_latency.csv)
| target ← lead | IC at lag 0 | lag 0.5 s | lag 1 s |
|---|---|---|---|
| BTC ← ETH | 0.164 | 0.060 | ≈0.01 |
| ETH ← BTC | 0.083 | 0.025 | 0.010 |
| SOL ← BTC | 0.096 | 0.030 | 0.010 |
| PEPE ← ETH | 0.110 | 0.038 | 0.014 |

- ETH "leads" BTC more strongly than the reverse. Almost all of the effect is gone within 1 s.
- Timestamps are the collector's receive time, so part of this lead may be data latency (one feed arriving later) rather than price discovery. We cannot separate the two with this data.
- For this challenge, scoring uses the same timestamps, so the effect is usable if the test period has the same latency structure. Stage 10 must check that it holds out of sample, i.e. on the validation months.

## Decisions carried to Stage 9
- **Candidates:** xret_*_0.5s, ofi_0.5s/1s, last_side, tim_0.5s, svn_0.5s, usdc_gap_dev_60s, usdc_gap_bps, basis_dev_10s, sret_0.5s, plus the Stage 7 set.
- **Likely redundant:** ofi_0.5s versus imb_l1_chg_1s (both measure queue change). Stage 9 measures the overlap.
- All IC here is exploratory and in-sample on training days. Selection happens in Stage 9 with walk-forward folds.
