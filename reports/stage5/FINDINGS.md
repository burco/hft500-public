# Stage 5: target findings and the decisions they force

Sample: every 7th day, 53 days per perp (44 train / 9 valid), from scripts/stage5_analyze.py.

| # | Finding (measured) | Decision | Stage |
|---|---|---|---|
| 1 | All horizons come from ONE label-mid series M(t) = mid(t+10 ms) on the 500 ms grid: y_2s = sum of 4 y_500ms, y_8s = sum of 16 (max error ~1e-15, every sampled day). | Treat targets as one series; horizon models can share structure. | 9, 10 |
| 2 | The 10 ms offset has no effect on our 100 ms data (offsets 0 and 10 give identical rebuilds). M(t) equals our visible mid at t on 98.5-99.3% of points; the rest moved inside (t, t+10 ms]. Rebuild exact 97-99%, corr 0.986-0.989 (500 ms), > 0.999 (8 s). | Supplied labels are ground truth; never train on our rebuild. The hidden component is unpredictable noise, mostly at 500 ms. | 6, 10 |
| 3 | Very discrete: y_500ms is 0 on 66-80% of points; half-tick moves are ~0.01%; SOL is 78% zero, 11% ±1 tick, 2% > 2 ticks; BTC/ETH/PEPE move > 2 ticks on 18-27% of points. | Keep regression (Pearson rewards size), but check models per instrument; consider tick-normalised features. | 7, 10 |
| 4 | The clipped 0.2% hold 17-20% of the 500 ms target variance (8-10% at 8 s). Kurtosis 54-84 raw, 17-25 clipped. | Train on clipped targets (meta/target_clip); winsorise features. | 6, 7, 10 |
| 5 | Overlap: acf of y_8s follows (16-k)/16 (0.95 at lag 1, ~0.52 at lag 8, ~0.02 at lag 16). | Split by whole UTC days; significance on days, not rows; purge >= 8.01 s. | 10 |
| 6 | Labels cross midnight: the last 16 y_8s of a day use the next day's prices (finite on all but one sampled day). First 2-10 points of a day are NaN. | 1-day embargo (or drop the last 16 training rows before a validation day); drop NaN targets. | 10 |
| 7 | Momentum hint: acf of consecutive y_500ms = +0.06 to +0.10. | HYPOTHESIS: past observable 500 ms return as a feature; test on training days only. | 7, 9 |
| 8 | Validation is calmer on every perp: std -15% to -32%, zero share +9 to +11 points. | Normalise features by recent activity; expect another shift in test. | 7, 9, 10 |
