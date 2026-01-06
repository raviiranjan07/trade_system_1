# Feature Documentation

## Overview

The trading system extracts **10 features** from OHLCV (Open, High, Low, Close, Volume) data to create a state vector that represents the current market condition.

---

## The 10 Features

| # | Feature | What it measures | Why we need it |
|---|---------|------------------|----------------|
| 1 | `ema50_slope_z` | Is short-term trend going up or down? | Detect momentum direction |
| 2 | `ema200_slope_z` | Is long-term trend going up or down? | Detect bigger trend |
| 3 | `trend_alignment` | Is EMA50 above or below EMA200? | Bullish (+1) or Bearish (-1) |
| 4 | `return_5m_z` | Price change in last 5 bars | Short-term momentum |
| 5 | `return_15m_z` | Price change in last 15 bars | Medium-term momentum |
| 6 | `rsi_z` | Overbought or oversold? | Reversal signals |
| 7 | `atr_percentile` | How volatile is market now? | Risk assessment |
| 8 | `volume_z` | Is volume high or low? | Confirms price moves |
| 9 | `vwap_distance_z` | Price above or below VWAP? | Fair value reference |
| 10 | `range_position` | Where is price in recent range? | Near high (1) or low (0) |

---

## Feature Categories

### Trend Features (trend.py)

| Feature | Formula | Description |
|---------|---------|-------------|
| `ema50_slope_z` | zscore(EMA50[now] - EMA50[5 bars ago]) | Rate of change of 50-period EMA |
| `ema200_slope_z` | zscore(EMA200[now] - EMA200[20 bars ago]) | Rate of change of 200-period EMA |
| `trend_alignment` | sign(EMA50 - EMA200) | +1 bullish, -1 bearish, 0 neutral |

### Momentum Features (momentum.py)

| Feature | Formula | Description |
|---------|---------|-------------|
| `return_5m_z` | zscore((close - close[5]) / close[5]) | 5-bar percentage return |
| `return_15m_z` | zscore((close - close[15]) / close[15]) | 15-bar percentage return |
| `rsi_z` | zscore(RSI(14)) | Relative Strength Index (0-100 scale) |

### Volatility Features (volatility.py)

| Feature | Formula | Description |
|---------|---------|-------------|
| `atr_percentile` | percentile(ATR(14)) | Average True Range as percentile (0-1) |

**ATR (Average True Range):** Measures average price movement per bar.
```
True Range = max(high-low, |high-prev_close|, |low-prev_close|)
ATR = rolling_mean(True Range, 14)
```

### Volume Features (volume.py)

| Feature | Formula | Description |
|---------|---------|-------------|
| `volume_z` | zscore(volume) | Normalized trading volume |

### Location Features (location.py)

| Feature | Formula | Description |
|---------|---------|-------------|
| `vwap_distance_z` | zscore((close - VWAP) / VWAP) | Distance from volume-weighted average price |
| `range_position` | (close - low50) / (high50 - low50) | Position in 50-bar range (0 = bottom, 1 = top) |

**VWAP:** Volume-Weighted Average Price
```
VWAP = cumsum(typical_price * volume) / cumsum(volume)
typical_price = (high + low + close) / 3
```

---

## Normalization

Features are normalized using a **rolling window of 2000 bars**:

### Z-Score (most features)
```
z = (value - rolling_mean) / rolling_std
```
- Output: typically -3 to +3
- Meaning: How many standard deviations from recent average

### Percentile (ATR only)
```
percentile = rank(value) / count
```
- Output: 0 to 1
- Meaning: How current value compares to recent history

---

## Storage

**File:** `data/state_vectors/{PAIR}_1m_state.parquet`

**Format:** Parquet (compressed columnar format)

**Structure:**
```
| time (index)        | ema50_slope_z | ema200_slope_z | trend_alignment | ... | range_position |
|---------------------|---------------|----------------|-----------------|-----|----------------|
| 2024-01-01 00:00:00 | 0.52          | 0.31           | 1               | ... | 0.75           |
| 2024-01-01 00:01:00 | 0.48          | 0.32           | 1               | ... | 0.72           |
```

Each row = 1 minute = 10 feature values

---

## Why These 10 Features?

These features capture the **market state** from different perspectives:

| Category | Features | Purpose |
|----------|----------|---------|
| **Trend** | 3 | Where is market going? |
| **Momentum** | 3 | How fast is it moving? |
| **Volatility** | 1 | How risky is it now? |
| **Volume** | 1 | Is the move backed by volume? |
| **Location** | 2 | Where is price relative to history? |

The KNN algorithm finds historical moments with **similar feature values** and checks what happened next (MFE/MAE outcomes).

---

## File Structure

```
features/
├── __init__.py
├── feature.md          # This documentation
├── trend.py            # EMA slopes, trend alignment
├── momentum.py         # Returns, RSI
├── volatility.py       # ATR
├── volume.py           # Volume
└── location.py         # VWAP distance, range position
```
