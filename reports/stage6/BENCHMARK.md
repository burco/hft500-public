# Benchmark (official metric, validation 2026-07-15 .. 2026-09-14, 62 days)

| model | features | BTC 500ms | BTC 2s | BTC 8s | ETH 500ms | ETH 2s | ETH 8s | SOL 500ms | SOL 2s | SOL 8s | PEPE 500ms | PEPE 2s | PEPE 8s | **score** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zero | none | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0.0000** |
| official OLS | best-level imbalance | 0.1946 | 0.2156 | 0.1914 | 0.2004 | 0.2019 | 0.1583 | 0.1958 | 0.2212 | 0.1912 | 0.1550 | 0.1617 | 0.1191 | **0.1838** |

Reproduction: our code vs the organisers' baseline.py -> identical coefficients and identical per-day
correlations (max abs diff 0.0) on all 12 cells. Official-metric score 0.183844 vs baseline.py-style 0.183847
(the 3e-6 gap = the few grid points without a book row, which predict 0).
Per-cell standard error across days ~0.004-0.006.
