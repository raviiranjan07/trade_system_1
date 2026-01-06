# Normalization Documentation

## What is Normalization?

Normalization transforms raw feature values into a **standard scale** so the KNN algorithm can compare them fairly.

**Problem without normalization:**
- ATR might be 500 (price units)
- RSI is 0-100
- Volume might be 1,000,000

KNN uses distance. Without normalization, large numbers dominate the comparison.

**Solution:** Convert all features to similar scales (typically -3 to +3 or 0 to 1).

---

## Two Normalization Methods

### 1. Z-Score (Most features)

```python
z = (value - mean) / std
```

| Input | Output | Meaning |
|-------|--------|---------|
| value = mean | z = 0 | Average |
| value > mean | z > 0 | Above average |
| value < mean | z < 0 | Below average |
| z = +2 | - | 2 standard deviations above mean |
| z = -2 | - | 2 standard deviations below mean |

**Example:**
```
RSI values over last 2000 bars: mean=50, std=15
Current RSI = 80

z = (80 - 50) / 15 = 2.0

Meaning: RSI is 2 standard deviations above recent average (overbought)
```

**Used for:** ema50_slope, ema200_slope, return_5m, return_15m, rsi, volume, vwap_distance

---

### 2. Percentile (ATR only)

```python
percentile = rank(value) / count
```

| Output | Meaning |
|--------|---------|
| 0.0 | Lowest in recent history |
| 0.5 | Median (middle) |
| 1.0 | Highest in recent history |

**Example:**
```
ATR values over last 2000 bars
Current ATR ranks #1800 out of 2000

percentile = 1800 / 2000 = 0.90

Meaning: Current volatility is higher than 90% of recent history
```

**Used for:** atr_percentile

---

## Rolling Window

Normalization uses a **rolling window of 2000 bars** (not all historical data).

```
Bar 1    Bar 2000   Bar 2001   Bar 4000
|--------|          |----------|
 Window 1            Window 2
```

**Why rolling?**
- Market conditions change over time
- A "high" RSI in 2020 might be "normal" in 2024
- Rolling window adapts to current market context

---

## Code Implementation

```python
class RollingNormalizer:
    def __init__(self, window: int = 2000):
        self.window = window

    def zscore(self, series: pd.Series) -> pd.Series:
        mean = series.rolling(self.window).mean()
        std = series.rolling(self.window).std()
        return (series - mean) / (std + 1e-9)  # 1e-9 prevents division by zero

    def percentile(self, series: pd.Series) -> pd.Series:
        return series.rolling(self.window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
```

---

## How Features Are Normalized

| Raw Feature | Normalization | Output Feature |
|-------------|---------------|----------------|
| ema50_slope | z-score | ema50_slope_z |
| ema200_slope | z-score | ema200_slope_z |
| return_5m | z-score | return_5m_z |
| return_15m | z-score | return_15m_z |
| rsi_14 | z-score | rsi_z |
| volume | z-score | volume_z |
| vwap_distance | z-score | vwap_distance_z |
| atr_14 | percentile | atr_percentile |
| trend_alignment | none (already -1, 0, +1) | trend_alignment |
| range_position | none (already 0-1) | range_position |

---

## Storage

Normalized features are stored in: `data/state_vectors/BTCUSDT_1m_state.parquet`

| Column | Type | Range |
|--------|------|-------|
| ema50_slope_z | float | typically -3 to +3 |
| ema200_slope_z | float | typically -3 to +3 |
| trend_alignment | int | -1, 0, +1 |
| return_5m_z | float | typically -3 to +3 |
| return_15m_z | float | typically -3 to +3 |
| rsi_z | float | typically -3 to +3 |
| atr_percentile | float | 0 to 1 |
| volume_z | float | typically -3 to +3 |
| vwap_distance_z | float | typically -3 to +3 |
| range_position | float | 0 to 1 |

---

## Visual Example

```
Raw Data:                          Normalized Data:

RSI: 80                            rsi_z: +2.0
ATR: 1500                   →      atr_percentile: 0.90
Volume: 50,000,000                 volume_z: +1.5
EMA50 Slope: 0.0003                ema50_slope_z: +0.8

All values now comparable for KNN distance calculation
```

---

## File Structure

```
normalization/
├── __init__.py          # Exports RollingNormalizer
├── normalizer.py        # RollingNormalizer class
└── normalization.md     # This documentation
```
