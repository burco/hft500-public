# Stage 12 — Final model and submission

The deliverable is the `../submission/` folder: `predict.py`, `models/` (12 LightGBM weight files), `hft500/` (helper modules), `pyproject.toml`, `uv.lock` and `README.md`.

## What was frozen
- **Features:** the 46 selected in Stage 9 (`reports/stage9/pruned_features.json`).
- **Model:** LightGBM configuration B from Stage 10: 127 leaves, 100 trees, learning rate 0.05, at least 1,000 rows per leaf, L2 = 10, feature fraction 0.8, bagging 0.5, `max_bin` 63, seed 7, deterministic.
- **Refit:** done **once** on train + validation (2025-09-15 … 2026-09-14, 365 days), using every 40th grid point. That is 1,576,800 rows per perp, about 45 s per perp. Script: `scripts/stage12_refit.py`.
- **Honest score estimate:** the validation score before the refit, **0.2247** (official baseline 0.1838; ridge 0.2076). The refit weights have seen the validation days, so no unbiased score exists for them.

## Verification of the package as shipped (`scripts/stage12_verify.py`, `run12c.sh`)
| check | result |
|---|---|
| clean environment from `uv.lock` (`uv sync --frozen`) | OK (Python 3.10, numpy 2.2.6, polars 2.0.0, pyarrow 25.0.1, lightgbm 4.7.0) |
| data folder holds only D and D−1 (17 book instruments, stable, meta; no targets) | OK: 691,200 rows, all finite, grid exact |
| regression: package + pre-refit weights vs Stage 10 research pipeline (2026-08-16) | identical day score, 0.221747 |
| official-style truncation, `predict.py` run as a separate process on truncated copies of the data folder | **15/15 bit-identical** (2026-08-16 ×10 incl. 00:00:00, 00:00:00.5, 23:59:59.5 and 09:00 in the BTC freeze; 2026-09-14 ×5) |
| previous day missing | runs; 691,200 finite rows |
| BTC book file missing **and** `meta/` absent | runs; 691,200 finite rows |
| helper modules identical to the research package (68 tests) | `cmp` OK |

## Cost (2-core / 4 GB VM, one process)
- **Run time:** 11.5 s wall-clock per day for all 4 perps.
- **Memory:** peak 0.94 GB (`profile_2026-08-16.txt`).
- **Size:** weights 15 MB; submission 15 MB in total.

## Known limits
- **Lead-lag features use receive times,** so they assume collector latency in the test period is similar to before.
- **Constant output is possible only in a degenerate case.** If nothing at all is known for a perp on a day, its predictions are all 0 (constant), which scores 0 either way. `predict.py` warns but deliberately does not add artificial noise: any data-independent jitter could break the truncation check.
- **Signal strength drifts down over time.** It weakened from autumn to summer (Stage 9). A periodic refit would be the natural improvement, but it is outside the one-shot submission format.
