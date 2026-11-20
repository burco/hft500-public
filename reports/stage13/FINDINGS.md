# Stage 13 — Fast (100 ms) features and the evaluate() harness

Code:
- Features: `src/hft500/features_fast.py` (17 features, `REGISTRY` like Stages 7–8).
- Evaluation: `src/hft500/incremental.py` (`evaluate()`, `evaluate_frame()`).
- Scripts: `scripts/stage13_build.py`, `scripts/stage13_eval.py`, `scripts/stage13_lgb_check.py`.
- Tests: `tests/test_features_fast.py` (11 new tests; the full suite has 79 passing).

## Question
Stage 8 measured lead-lag only on the 500 ms grid, and it found the lead gone within 1 s. Quotes arrive every 100 ms. Do the freshest 100–200 ms of other perps' moves, cross-perp OFI and a faster own OFI add anything to the frozen 46-feature model?

## Protocol (fixed before looking)
- **Days:** every 14th training day (22 per perp, 2025-09-15 … 2026-07-06), stride 10. Validation days were not touched, and `evaluate()` refuses them by default.
- **Split:** walk-forward. Train on 13 days, embargo 1 day, test on the last 8 sampled days (2026-03-30 … 2026-07-06).
- **Linear probe:** the Stage 9 preparation (clip at train 0.1/99.9%, standardise, NaN→0 plus an indicator), then ridge.
- **Production check:** LightGBM config B (Stage 10/12 settings, 100 trees) on the same split.
- **Metric:** official daily Pearson with clipped targets, averaged over days, then over cells.

## Results
| model | base 46 | base + fast | gain | cells improved |
|---|---|---|---|---|
| ridge (linear probe), 17 candidates | 0.2238 | 0.2287 | **+0.0049** | 12/12 |
| LightGBM config B, 15 candidates (gap_z dropped) | 0.2347 | 0.2393 | **+0.0046** | 12/12 |

LightGBM gain by cell:

| coin | 500 ms | 2 s | 8 s |
|---|---|---|---|
| BTC | +0.0050 (t 6.4) | +0.0029 | +0.0013 |
| ETH | +0.0040 (t 2.8) | +0.0018 | +0.0006 (t 0.7) |
| SOL | +0.0109 (t 6.0) | +0.0042 | +0.0038 |
| PEPE | +0.0111 (t 5.7) | +0.0055 | +0.0040 |

- **The gain sits at 500 ms,** as expected for information that is 100–200 ms old.
- **ETH gains least.** This is consistent with Stage 8: ETH tends to lead, so there is less for it to follow.
- **ETH 8 s is not significant.**
- **t-statistics come from 8 test days,** and daily correlations are autocorrelated. Treat them as ranking evidence, not proof.

## Which features carry it (ridge, add-one to base, mean of 4 coins at 500 ms)
| feature | add-one gain at 500 ms |
|---|---|
| xr_eth_0.1s | +0.0040 |
| ofi_0.2s | +0.0032 |
| xr_btc_0.1s | +0.0026 |
| xr_sol_0.1s | +0.0021 |
| xr_eth_0.2s | +0.0021 |
| xr_pepe_0.1s | +0.0020 |

- **The last 100 ms beats the last 200 ms** for every leader.
- **Cross-OFI helps less,** but xofi_eth_0.5s is in the top 3 by residual IC.
- **gap_z_0.5s and gap_z_2s add nothing (≈0).** The trees already combine xret and ret, so these are dropped.

## Regimes (ridge, gain of base + all vs base)
- **Volatility:** at 500 ms the gain is larger in low-volatility periods (+0.010) than in high-volatility periods (+0.006). This suggests the fast lead matters most when the own book is quiet, which is a reason to keep the volatility conditioners in the trees.
- **Spread:** spreads wider than 1 tick occur on 0.14% of points, too rare to measure. The spread regime is not a useful split for these perps.

## Caveats
1. **Receive-time latency.** As in Stage 8, a 100 ms "lead" may be feed latency rather than price discovery. It is usable only if the test period has the same latency structure. ETH leading BTC more than the reverse is a hint of this.
2. **Small, recent sample.** The test block is 8 sampled days in spring and summer, the calm months (Stage 5).
3. **Optimistic selection.** Choosing features by add-one gain on this split would bias the result. Freeze the candidate list first, then confirm.

## Next steps
1. **Monthly folds:** confirm on the 8 Stage 9 monthly walk-forward folds, with all training days and the candidate list frozen as above minus gap_z.
2. **One validation look:** LightGBM, base 46 + 15 fast features, compared with Stage 10's 0.2247.
3. **Re-run the audits:** if it holds, re-run the Stage 11 audit and the Stage 12 truncation test with the extended feature builder before refitting the submission.
