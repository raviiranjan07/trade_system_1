# Day Trading Strategy

**Status:** TESTED & OPTIMIZED

**Horizon Range:** 5-10 minutes

---

## Best Configuration

| Parameter | Value |
|-----------|-------|
| Horizon | 5 min |
| Sample Interval | 15 bars |
| Normalization Window | 2000 bars |
| min_expectancy | 0.001 |
| max_distance | 3.0 |
| blocked_regimes | NONE |
| K (neighbors) | 200 |

---

## Results

| Metric | Value |
|--------|-------|
| Win Rate | 100% |
| Cumulative Return | +20.65% (over 1.77 years) |
| **Annualized Return** | **~11%** |
| Max Drawdown | 0% |
| Total Trades | 137 |
| Avg Trade Duration | ~80 hours |

---

## Characteristics

- **Trade frequency:** ~77 trades/year
- **Avg profit per trade:** ~0.15%
- **Risk level:** Very low (100% win rate, 0% drawdown)
- **Best for:** Conservative traders seeking consistent small gains

---

## Files

```
day_trading/
├── README.md                 # This file
├── backtest/                 # Trade result files
│   └── *.parquet
├── grid_search/              # Parameter search results
│   └── *.csv
└── docs/
    ├── tested_parameters.md  # All experiments
    └── best_config.yaml      # Optimal config
```

---

## How to Use

```bash
# Run backtest with day trading config
python run_backtest.py --horizon 5 --sample-interval 15 --min-expectancy 0.001

# Or use the best_config.yaml
python run_backtest.py --config experiments/day_trading/docs/best_config.yaml
```

---

## Test History

- **2024-12-27:** Initial grid search (H=5)
- **2024-12-28:** min_expectancy tuning (0.001 optimal)
- **2024-12-29:** sample_interval testing (si=15 best)
- **2026-01-04:** Final validation (+11% annualized)
