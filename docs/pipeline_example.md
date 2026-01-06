# Complete Pipeline Documentation

A structured guide showing each step of the pipeline, what it does, and how data is stored.

---

# Step 1: Feature Extraction

## What Happens?

Raw OHLCV data is transformed into meaningful features that describe market conditions.

## Input

**Source:** Database or `data/ohlcv/BTCUSDT_1m_ohlcv.parquet`

| Column | Description |
|--------|-------------|
| time | Timestamp (index) |
| open | Opening price |
| high | Highest price |
| low | Lowest price |
| close | Closing price |
| volume | Trading volume |

**Example Row:**
```
time: 2024-06-15 14:30:00
open: 65,320
high: 65,400
low: 65,250
close: 65,380
volume: 1,500,000
```

---

## Features Extracted

### A. Trend Features (trend.py)

| Feature | Formula | Example |
|---------|---------|---------|
| `ema50` | EMA(close, 50) | 65,100 |
| `ema200` | EMA(close, 200) | 64,500 |
| `ema50_slope` | ema50[now] - ema50[5 bars ago] | 200 |
| `ema200_slope` | ema200[now] - ema200[20 bars ago] | 400 |
| `trend_alignment` | sign(ema50 - ema200) | +1 |

### B. Momentum Features (momentum.py)

| Feature | Formula | Example |
|---------|---------|---------|
| `return_5m` | (close - close[5]) / close[5] | 0.0043 |
| `return_15m` | (close - close[15]) / close[15] | 0.0090 |
| `rsi_14` | RSI calculation (14 period) | 70.6 |

### C. Volatility Features (volatility.py)

| Feature | Formula | Example |
|---------|---------|---------|
| `atr_14` | Average True Range (14 period) | 180 |

### D. Volume Features (volume.py)

| Feature | Formula | Example |
|---------|---------|---------|
| `volume_raw` | volume (unchanged) | 1,500,000 |

### E. Location Features (location.py)

| Feature | Formula | Example |
|---------|---------|---------|
| `vwap` | cumsum(typical_price × volume) / cumsum(volume) | 64,800 |
| `vwap_distance` | (close - vwap) / vwap | 0.0090 |
| `range_position` | (close - low50) / (high50 - low50) | 0.91 |

---

## Output After Step 1

**DataFrame with 13 new columns:**
```
| time       | open   | high   | low    | close  | volume    | ema50  | ema200 | ema50_slope | ema200_slope | trend_alignment | return_5m | return_15m | rsi_14 | atr_14 | volume_raw | vwap   | vwap_distance | range_position |
|------------|--------|--------|--------|--------|-----------|--------|--------|-------------|--------------|-----------------|-----------|------------|--------|--------|------------|--------|---------------|----------------|
| 2024-06-15 | 65,320 | 65,400 | 65,250 | 65,380 | 1,500,000 | 65,100 | 64,500 | 200         | 400          | +1              | 0.0043    | 0.0090     | 70.6   | 180    | 1,500,000  | 64,800 | 0.0090        | 0.91           |
```

## Storage After Step 1

**Not saved separately.** Features are computed in memory and passed to Step 2.

---

# Step 2: Normalization

## What Happens?

Raw features have different scales (RSI: 0-100, volume: millions). Normalization converts them to a common scale so KNN can compare them fairly.

## Input

Raw features from Step 1.

---

## Normalization Methods

### Method 1: Z-Score

**Formula:**
```
z = (value - rolling_mean) / rolling_std
```

**Window:** 2000 bars (rolling)

**Output Range:** Typically -3 to +3

**Used For:** 7 features

| Raw Feature | → | Normalized Feature |
|-------------|---|-------------------|
| ema50_slope | → | ema50_slope_z |
| ema200_slope | → | ema200_slope_z |
| return_5m | → | return_5m_z |
| return_15m | → | return_15m_z |
| rsi_14 | → | rsi_z |
| volume_raw | → | volume_z |
| vwap_distance | → | vwap_distance_z |

**Example Calculation:**
```
ema50_slope = 200
rolling_mean(last 2000 bars) = 50
rolling_std(last 2000 bars) = 150

ema50_slope_z = (200 - 50) / 150 = 1.0
```

---

### Method 2: Percentile

**Formula:**
```
percentile = rank(value) / count
```

**Window:** 2000 bars (rolling)

**Output Range:** 0 to 1

**Used For:** 1 feature

| Raw Feature | → | Normalized Feature |
|-------------|---|-------------------|
| atr_14 | → | atr_percentile |

**Example Calculation:**
```
atr_14 = 180
In last 2000 bars, 180 ranks at position 1600

atr_percentile = 1600 / 2000 = 0.80
```

---

### Method 3: No Normalization

**Already in correct range.**

| Raw Feature | → | Normalized Feature |
|-------------|---|-------------------|
| trend_alignment | → | trend_alignment (unchanged, -1/0/+1) |
| range_position | → | range_position (unchanged, 0-1) |

---

## Output After Step 2

**10 Normalized Features:**

| # | Feature | Example Value | Range |
|---|---------|---------------|-------|
| 1 | ema50_slope_z | 1.0 | -3 to +3 |
| 2 | ema200_slope_z | 1.0 | -3 to +3 |
| 3 | trend_alignment | +1 | -1, 0, +1 |
| 4 | return_5m_z | 1.9 | -3 to +3 |
| 5 | return_15m_z | 2.0 | -3 to +3 |
| 6 | rsi_z | 1.37 | -3 to +3 |
| 7 | atr_percentile | 0.80 | 0 to 1 |
| 8 | volume_z | 1.25 | -3 to +3 |
| 9 | vwap_distance_z | 1.6 | -3 to +3 |
| 10 | range_position | 0.91 | 0 to 1 |

---

## Storage After Step 2

**File:** `data/state_vectors/BTCUSDT_1m_state.parquet`

**Format:** Parquet (compressed columnar)

**Columns:**
```
| time (index)        | ema50_slope_z | ema200_slope_z | trend_alignment | return_5m_z | return_15m_z | rsi_z | atr_percentile | volume_z | vwap_distance_z | range_position | pair    |
|---------------------|---------------|----------------|-----------------|-------------|--------------|-------|----------------|----------|-----------------|----------------|---------|
| 2024-06-15 14:30:00 | 1.0           | 1.0            | 1               | 1.9         | 2.0          | 1.37  | 0.80           | 1.25     | 1.6             | 0.91           | BTCUSDT |
```

**Size:** ~3 million rows (2020-2025)

---

# Step 3: Regime Labeling

## What Happens?

Each row is classified into one of 4 market regimes based on trend and volatility.

## Input

State vectors from Step 2.

**Required Columns:**
- `ema200_slope_z` (trend strength)
- `atr_percentile` (volatility level)
- `trend_alignment` (trend direction)

---

## The 4 Regimes

| Regime | Code | Description |
|--------|------|-------------|
| RANGE_LOW_VOL | 0 | Sideways, calm market |
| TREND_LOW_VOL | 1 | Clear trend, smooth moves |
| TREND_HIGH_VOL | 2 | Clear trend, volatile moves |
| HIGH_VOL | 3 | Extreme volatility, risky |

---

## Classification Thresholds

```
TREND_SLOPE_THRESHOLD = 0.7
HIGH_VOL_THRESHOLD = 0.85
LOW_VOL_THRESHOLD = 0.35
```

---

## Classification Logic

```
IF atr_percentile >= 0.85:
    → HIGH_VOL

ELSE IF |ema200_slope_z| >= 0.7 AND trend_alignment != 0:
    IF atr_percentile <= 0.35:
        → TREND_LOW_VOL
    ELSE:
        → TREND_HIGH_VOL

ELSE:
    → RANGE_LOW_VOL
```

---

## Example Classification

**Input:**
```
ema200_slope_z = 1.0
atr_percentile = 0.80
trend_alignment = +1
```

**Logic:**
```
Step 1: atr_percentile (0.80) >= 0.85?  → NO
Step 2: |ema200_slope_z| (1.0) >= 0.7?  → YES
        trend_alignment (+1) != 0?      → YES
        → This is TRENDING
Step 3: atr_percentile (0.80) <= 0.35?  → NO
        → Not low volatility

Result: TREND_HIGH_VOL
```

---

## Regime Smoothing

**Problem:** Raw labels can flip every bar (noisy).

**Solution:** Rolling majority vote over 30 bars.

**How It Works:**
```
Look at last 30 bars:
  TREND_HIGH_VOL: 22 bars
  TREND_LOW_VOL: 8 bars

Winner: TREND_HIGH_VOL (majority)
```

**Window:** 30 bars

---

## Output After Step 3

| time | regime_raw | regime (smoothed) |
|------|------------|-------------------|
| 2024-06-15 14:30:00 | TREND_HIGH_VOL | TREND_HIGH_VOL |

---

## Storage After Step 3

**File:** `data/regimes/BTCUSDT_1m_regimes.parquet`

**Format:** Parquet

**Columns:**
```
| time (index)        | regime         |
|---------------------|----------------|
| 2024-06-15 14:30:00 | TREND_HIGH_VOL |
| 2024-06-15 14:31:00 | TREND_HIGH_VOL |
| 2024-06-15 14:32:00 | TREND_HIGH_VOL |
```

**Size:** ~3 million rows (matches state vectors)

---

# Summary: Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: FEATURE EXTRACTION                                      │
├─────────────────────────────────────────────────────────────────┤
│ Input:  OHLCV (5 columns)                                       │
│ Output: 13 raw features                                         │
│ Stored: NOT SAVED (in memory)                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: NORMALIZATION                                           │
├─────────────────────────────────────────────────────────────────┤
│ Input:  13 raw features                                         │
│ Output: 10 normalized features (state vector)                   │
│ Stored: data/state_vectors/BTCUSDT_1m_state.parquet            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: REGIME LABELING                                         │
├─────────────────────────────────────────────────────────────────┤
│ Input:  3 features (ema200_slope_z, atr_percentile, alignment) │
│ Output: 1 regime label (smoothed)                               │
│ Stored: data/regimes/BTCUSDT_1m_regimes.parquet                │
└─────────────────────────────────────────────────────────────────┘
```

---

# File Storage Summary

| Step | File | Columns | Rows |
|------|------|---------|------|
| Step 1 | (not saved) | - | - |
| Step 2 | `data/state_vectors/BTCUSDT_1m_state.parquet` | 10 features + time + pair | ~3M |
| Step 3 | `data/regimes/BTCUSDT_1m_regimes.parquet` | regime + time | ~3M |

---

# Code Location

| Step | File |
|------|------|
| Step 1: Feature Extraction | `src/trade_system/features/*.py` |
| Step 2: Normalization | `src/trade_system/normalization/normalizer.py` |
| Step 2: State Building | `src/trade_system/state/state_builder.py` |
| Step 2: State Storage | `src/trade_system/state/state_store.py` |
| Step 3: Regime Labeling | `src/trade_system/regime/regime_labeler.py` |
