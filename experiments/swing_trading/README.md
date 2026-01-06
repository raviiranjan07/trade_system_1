# Swing Trading Style

**Status:** PARTIALLY TESTED (Not Working Yet)

## Overview

Swing trading aims for longer holds (30+ minutes to hours) capturing larger price movements.

---

## Horizon Range

| Parameter | Value |
|-----------|-------|
| Horizon | 30-240 minutes |
| Hold Time | 30 min - 4 hours |

---

## Tested Results (NOT WORKING)

| Horizon | Result | Win Rate | Status |
|---------|--------|----------|--------|
| H=15 | -$40.62 | 51% | LOSS |
| H=30 | -$1,457 | 99% | LOSS |

**Analysis:** Longer horizons show consistent losses despite high win rates at H=30, suggesting the model may not generalize well to longer time frames.

---

## Recommended Parameters to Test

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `default_horizon` | 60-240 | Much longer than day trading |
| `sample_interval` | 30-60 | Less frequent checks |
| `normalization_window` | 5000-10000 | Longer historical context |
| `min_expectancy` | 0.002+ | Higher threshold for bigger moves |
| `k` (neighbors) | 200-300 | More neighbors for stability |

---

## Why Current Model May Not Work

1. **Training data mismatch**: Model trained for short-term patterns
2. **Different market dynamics**: Long-term trends have different drivers
3. **Regime relevance**: 30-bar smoothing may be too short
4. **Feature time scales**: Current features optimized for 5-min horizon

---

## Potential Improvements

1. **Longer feature lookbacks**: EMA500 instead of EMA200
2. **Daily-level features**: Include daily OHLCV patterns
3. **Different regime classification**: Higher volatility thresholds
4. **Separate model training**: Train specifically for swing trading

---

## Directory Structure

```
swing_trading/
├── README.md                    # This file
├── backtest/                    # Test result files
├── grid_search/                 # Grid search CSVs
└── docs/
    └── tested_parameters.md     # What's been tested
```

---

## Testing Priority

1. H=60 with current setup (baseline)
2. H=120 with window=5000
3. H=240 with window=10000
4. Different EMA configurations

---

## Notes

Current system is optimized for Day Trading (H=5). Swing trading requires:
- Potential model retraining
- Different feature engineering
- Longer historical data
- Possibly different similarity search approach
