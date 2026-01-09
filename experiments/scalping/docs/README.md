# Scalping Trading Style

**Status:** NOT YET TESTED

## Overview
Scalping aims for very short-term trades (1-2 minutes) with frequent entries and quick exits.

## Horizon Range

| Parameter | Value |
|-----------|-------|
| Horizon | 1-2 minutes |
| Hold Time | < 2 minutes |

## Recommended Parameters to Test

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `default_horizon` | 1-2 | Very short-term predictions |
| `sample_interval` | 1 | Check every bar (real-time) |
| `normalization_window` | 200-500 | Faster adaptation to recent conditions |
| `min_expectancy` | 0.002+ | Stricter filter for quick trades |
| `min_consensus` | 0.75+ | Higher confidence for real-time safety |
| `k` (neighbors) | 100-150 | Fewer neighbors for speed |

## Expected Characteristics
- High frequency: many trades per day
- Small per-trade profit: ~0.1-0.2% per trade
- Quick execution: requires low latency
- Spread sensitive: needs tight spreads

## Risk Considerations
1. Slippage: very sensitive to execution speed
2. Spread cost: can eat into small profits
3. Overfitting: short horizons are noisy
4. High activity: more trades = more commission

## Directory Structure

```
scalping/
  docs/                       # This file + recommended_parameters.md
  scripts/                    # Run scripts (grid search, batch, outcomes)
  outcome/
    grid_search/              # Grid search CSV results
    backtest/                 # Per-run backtest outputs
  scalping_grid_search_colab.ipynb (if present)
```

## Running Grid Search on Colab
1. Upload `trade_system_1` folder to Google Drive.
2. Open `scalping_grid_search_colab.ipynb` in Colab.
3. Set runtime to **T4 GPU**.
4. Change `BATCH_NUMBER` (1-7) for each session.
5. Run all cells (~45 min per batch).
6. After all 7 batches, set `MERGE_ALL_BATCHES = True` to combine results.

Total combinations: 648; total time: ~5 hours (across 7 sessions).

## Testing Priority
1. H=1 with si=1, min_exp=0.002
2. H=2 with si=1, min_exp=0.002
3. Vary normalization window (200, 300, 500)
4. Test blocked_regimes impact

## Notes
Scalping requires:
- Fast data feed
- Low latency execution
- Tight spreads
- Potentially different exchange/API setup

Current system is optimized for Day Trading (H=5); scalping parameters are untested.
