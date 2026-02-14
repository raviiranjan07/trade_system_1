# Scalping Smart Agent - Requirements & Vision

## Objective

> **To build a state-aware, data-driven trading engine that can consistently extract small, repeatable edges from BTC price action — and compound them through high trade frequency and controlled risk.**

## User Vision

> "My objective was to capture the state of each candle, draw regime and output labeling,
> do similarity search on test sample and find best params for SCALPING.
> In real trading, I'm gonna test in real-time, so it's like a smart agent which
> will monitor each candle and will take trades based on patterns it is trained on."

---

## Target Performance

```
Current: 0.07%/day, 1.8 trades/day
Target:  1-2%/day, 15-30 trades/day
Gap:     ~15x improvement needed
```

| Metric | Target | Current | Gap |
|--------|--------|---------|-----|
| Daily return | 1-2% | 0.07% | 15-30x |
| Trades/day | 15-30 | 1.8 | 8-17x |
| Win rate | >90% | 100% | OK |

**Strategy to close gap:**
1. Lower `min_mfe` threshold → more signals
2. Higher `max_distance` → accept more neighbors
3. `sample_interval=1` → check every bar
4. Keep `max_bars_in_trade=0` → wait for TP (critical for 100% WR)

---

## Core Concept

This is a **pattern-recognition trading bot** that:
1. Monitors each candle in real-time
2. Captures market state (10 features + regime)
3. Finds similar historical patterns via KNN
4. Takes trades based on learned outcomes
5. Optimized for **SCALPING** (high frequency, small profits)

---

## Validated Configuration

From grid search (2026-01-09):

```yaml
# Best performing config
horizon: 3                    # 3-minute forward prediction
sample_interval: 3            # Check every 3 bars
normalization_window: 180     # Fast adaptation (3 hours)
min_mfe: 0.0008              # Quality trade filter
max_distance: 2.0             # Strict similarity threshold
k: 200                        # Number of neighbors
blocked_regimes: []           # Trade ALL regimes
```

### Performance Metrics
| Metric | Value |
|--------|-------|
| Total PnL | +7.22% |
| Win Rate | 100% |
| Total Trades | 188 |
| Trades/Day | ~1.8 |
| Long Trades | 56 (30%) → +2.13% |
| Short Trades | 132 (70%) → +5.10% |
| Test Period | ~104 days |

Source: `experiments/scalping/grid_search/scalping_BATCH_1_20260109_180612.csv`

---

## State Vector (10 Features)

Each candle is represented by 10 normalized features:

| Feature | Type | Description |
|---------|------|-------------|
| ema50_slope_z | Trend | EMA(50) slope, z-normalized |
| ema200_slope_z | Trend | EMA(200) slope, z-normalized |
| trend_alignment | Trend | Sign of EMA50-EMA200 |
| return_5m_z | Momentum | 5-bar return, z-normalized |
| return_15m_z | Momentum | 15-bar return, z-normalized |
| rsi_z | Momentum | RSI(14), z-normalized |
| atr_percentile | Volatility | ATR as percentile |
| volume_z | Volume | Volume, z-normalized |
| vwap_distance_z | Location | Distance from VWAP |
| range_position | Location | Position in 50-bar range |

---

## Trade Logic

```
Every 3 bars (sample_interval=3):
  1. Compute state vector from current candle
  2. Query KNN for k=200 similar historical states
  3. If neighbors found with distance <= 2.0:
     - Get mean_mfe, mean_mae from neighbors
     - If mean_mfe >= 0.0008 (quality filter):
       - Direction = LONG if long_mfe > short_mfe else SHORT
       - TP = mean_mfe (take profit target)
       - SL = mae_5pct (5th percentile MAE for stop loss)
       - Enter trade
  4. Exit when TP or SL hit
```

---

## What Doesn't Work (ABANDONED APPROACHES)

### Adaptive Horizon (Multi-Horizon Selection)

**Concept:** Query multiple horizons (H=2,3,5,10,15,30) per state and dynamically select the best one.

**Grid Search Results:** 144 configurations tested - ALL LOST MONEY

```
experiments/adaptive_horizon/grid_search/adaptive_FINAL_20260110_002647.csv
- Best config: -$0.02 (still negative)
- ~0.5 trades/day (too few)
- Even 82% win rate configs lost money (TP too small)
```

**Why it failed:**
1. Too complex - multiple horizon queries slow down decisions
2. TP targets too small (mean MFE unrealistic)
3. Losses bigger than wins despite high win rate
4. Not enough trade frequency

**Lesson learned:** Simple single-horizon (H=3) outperforms complex multi-horizon selection.

### Timeout-Based Exit (max_bars_in_trade > 0)

**Concept:** Exit trades after N bars if TP not hit.

**Grid Search Results:** All configs with `max_bars_in_trade=3` lost money.

```
max_bars=0 (no timeout): +7.22%, 100% WR
max_bars=3 (3-bar timeout): -17% to -94%, ~45-50% WR
```

**Why it failed:**
1. Trades need time to reach TP
2. Timeout forces exit at bad prices
3. Converts winning trades into losers

**Lesson learned:** `max_bars_in_trade=0` is CRITICAL. Let trades run until TP hit.

---

## Pending Test: Adaptive Horizon V2

**Hypothesis:** Previous adaptive horizon test failed because `normalization_window` was not set (defaulted to 2000 instead of 180).

### Parameters to Test (V2)

```python
PARAM_GRID = {
    # Horizon combinations
    "adaptive_horizons": [
        [2, 3, 5],              # Short only
        [3, 5, 10],             # Short-medium
        [2, 3, 5, 10, 15, 30],  # All
    ],

    # CRITICAL - was missing in V1!
    "normalization_window": [180, 300],

    # MFE thresholds (per-bar, scaled by horizon)
    "adaptive_min_mfe_per_bar": [0.0001, 0.00015, 0.0002, 0.00025],

    # Selection strategy
    "adaptive_selection_strategy": ["max_mfe", "max_expectancy", "max_quality"],

    # KNN params
    "k": [150, 200, 250],
    "max_distance": [2.0, 2.5, 3.0, 4.0],  # EXPANDED
    "adaptive_min_neighbors": [30, 50],

    # Sampling
    "sample_interval": [1, 2],
}
# Total: ~1,728 combinations
```

### Script Location
`experiments/adaptive_horizon/run_adaptive_grid_search_v2.py`

### Output Location
`experiments/scalping/grid_search/adaptive_v2_batch_*.csv` (per batch)
`experiments/scalping/grid_search/adaptive_v2_final.csv` (combined)

### Metadata Included
Each result includes: experiment name, timestamp, sample_size, train_ratio, batch number

---
