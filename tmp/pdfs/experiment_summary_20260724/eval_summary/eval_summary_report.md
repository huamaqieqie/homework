# Original vs Residual vs Factored

## Summary

| group | n | success mean ± std | min | max | time/episode | eval budget | num eval |
|---|---:|---:|---:|---:|---:|---|---|
| PushT Original 15 | 5 | 88.80 ± 3.03 | 86.00 | 94.00 | 6.42s | 300 | 50 |
| PushT Residual 15 | 5 | 90.00 ± 3.74 | 86.00 | 94.00 | 7.50s | 300 | 50 |
| PushT Factored 96-96 15 | 5 | 79.60 ± 4.77 | 74.00 | 86.00 | 5.65s | 300 | 50 |
| PushT Factored 64-128 15 | 5 | 68.40 ± 8.65 | 56.00 | 80.00 | 7.63s | 300 | 50 |
| Reacher Original 15 | 5 | 62.00 ± 8.00 | 52.00 | 72.00 | 4.16s | 50 | 50 |
| Reacher Residual 15 | 5 | 57.60 ± 10.14 | 48.00 | 72.00 | 4.12s | 50 | 50 |

## Figures

![Success Rate Summary](success_rate_summary.png)

![Success Rate by Seed](success_rate_by_seed.png)

![Time per Episode](time_per_episode.png)
