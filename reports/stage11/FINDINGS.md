# Stage 11 — Leakage and causality audit: **PASS**

What was audited: the full prediction path (`src/hft500/pipeline.py`), i.e. Stage 7 + Stage 8 features feeding the Stage 10 LightGBM models for all 4 instruments. This is the same code `predict.py` will run in Stage 12.

Tests: `tests/test_pipeline_audit.py` (synthetic data). The full suite has 68 passing. Real-data runner: `scripts/stage11_audit.py`. Machine-readable verdict: `audit.json`.

## 1. Official-style truncation test on end-to-end predictions — 41 cuts, 0 mismatches
- **Procedure:** predict the full day. Then hide every row with ts > T in **every** source and instrument (book, bbo, trades, stable, all 20 instruments) and predict again. All 12 prediction columns for t ≤ T must be bit-identical.
- **Days tested:**
  - 2025-09-15: first day, no previous day exists.
  - 2026-03-03: an ordinary training day.
  - 2026-08-16: validation day with a BTC feed freeze from 08:44 to 09:34.
  - 2026-09-14: the last validation day.
- **Cut times on each day:**
  - The three day edges: 00:00:00.000, 00:00:00.500 and 23:59:59.500.
  - 5 random times.
  - Times inside the largest BTC and PEPE book gaps.
- **Extra cuts:** 2 at 09:00 and 09:34 inside the BTC freeze.
- Stage 4 (reconstruction, 180 cuts), Stage 7 (book features, 100 cuts) and Stage 8 (cross features, 32 cuts) already passed at the component level.

## 2. Deliberately leaky canaries — both caught
| canary | mismatched values (t ≤ T) | first mismatch |
|---|---|---|
| `imb_l1` taken from t + 500 ms (`shift(-1)`) | 12 | exactly at T (12:00:00) |
| `ret_2s` minus its **whole-day** mean | 97,691 | 00:00:01, i.e. everywhere |

The test can see both kinds of leak: a one-step look-ahead shows up only at the cut point, and a whole-day statistic contaminates the entire day. The same two canaries also FAIL in the synthetic unit tests, while the honest pipeline passes there.

## 3. Inference contract
`RestrictedRoot` raises an error on any read of `targets` and on any date other than D and D−1. A full-day prediction for 2026-08-16 under it:
- read only 2026-08-15 and 2026-08-16, from book, bbo, trades and stable (58 files);
- returned 691,200 rows, all finite;
- took 11.4 s for all 4 instruments.

The model never needs the next day's file. (The day's last trade bar is stamped next-day midnight but lives in the day's own file; this was checked in Stage 1.)

## 4. Train / inference consistency (warm-up mismatch check)
- Features rebuilt by the inference path were compared with the training cache rows for BTC 2025-09-16, SOL 2026-02-10 and PEPE 2026-07-14: all 61 features are **bit-identical** (as float32) with identical timestamps.
- Training and inference use one builder with the same two-day context. LightGBM is trained on float32 and receives float32 at inference.

## 5. Provenance of every fitted quantity (`provenance.json`)
| artifact | data window |
|---|---|
| LightGBM training rows | 2025-09-15 … 2026-07-14 (303 days, training only) |
| linear-model sums | same 303 training days |
| linear clip bounds | training days before 2025-11-15 |
| Stage 9 feature statistics / pruning | training days only (walk-forward folds Dec–Jul, 1-day embargo) |
| target clip bounds | official `meta/target_clip.parquet` |
| validation period | looked at **once** (Stage 10, `validation_log.csv`); nothing was tuned on it |

## 6. Static scan of the code (`static_scan.csv`)
I searched for negative shifts, backward fill, centred windows, interpolation, whole-array statistics, last-timestamp logic and target access:
- **Every hit is harmless:** it is a comment that states the rule, an order-preserving `sort(maintain_order=True)`, or the explicitly guarded `load_day(with_targets=…)`, which feature code never calls.
- **Rolling sums use prefix sums.** `np.cumsum` is applied to the past part of the array and differenced with a lag, so it only looks back.
- **Last-trade ages are safe.** They sort trade times taken from bars with ts ≤ t. A bar stamped after t only contains trades after t, so the as-of lookup can't pick them up.

## Checklist (from the master prompt)
| check | result |
|---|---|
| negative shifts / centred windows / backward fill | none (static scan + truncation) |
| whole-day statistics / last-timestamp keys | none (the day-statistic canary proves the test would catch one) |
| cross-asset misalignment | all instruments are truncated together; 41/41 OK |
| target leakage | targets cannot be read at inference (`RestrictedRoot`) |
| preprocessing fitted on validation or test | none (provenance) |
| train/inference warm-up mismatch | none (bit-identical features) |
| leaky feature must FAIL | 2/2 caught |

No critical issues are open. Remaining caveat (not a leak): timestamps are receive times, so the lead-lag features rely on the test period having similar collector latency (Stage 8).
