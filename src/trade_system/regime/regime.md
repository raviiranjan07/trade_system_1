# Regime Labeling Documentation

## What is a Regime?

A **regime** is the current market condition/environment. Markets behave differently in different regimes:
- Trending markets → momentum strategies work
- Range-bound markets → mean-reversion strategies work
- High volatility → increased risk, wider stops needed

The system classifies each minute into one of **4 regimes**.

---

## The 4 Regimes

| Regime | Code | Description | Characteristics |
|--------|------|-------------|-----------------|
| `RANGE_LOW_VOL` | 0 | Sideways, calm | No trend, low volatility, choppy price action |
| `TREND_LOW_VOL` | 1 | Trending, calm | Clear direction, smooth moves, ideal for trading |
| `TREND_HIGH_VOL` | 2 | Trending, volatile | Clear direction but with large swings |
| `HIGH_VOL` | 3 | Volatility shock | Extreme volatility, directionless, risky |

---

## Inputs for Regime Classification

Three features from the state vector are used:

| Feature | Source | Purpose |
|---------|--------|---------|
| `ema200_slope_z` | State vector | Measures long-term trend strength |
| `atr_percentile` | State vector | Measures current volatility level |
| `trend_alignment` | State vector | Confirms trend direction (-1, 0, +1) |

---

## Thresholds

```python
TREND_SLOPE_THRESHOLD = 0.7    # abs(ema200_slope_z) >= 0.7 = trending
HIGH_VOL_THRESHOLD = 0.85      # atr_percentile >= 0.85 = high volatility
LOW_VOL_THRESHOLD = 0.35       # atr_percentile <= 0.35 = low volatility
```

---

## Classification Logic

```python
def label_regime_row(row) -> str:
    trend_strength = abs(row["ema200_slope_z"])
    vol = row["atr_percentile"]
    alignment = row["trend_alignment"]

    # Step 1: Check for volatility shock (highest priority)
    if vol >= 0.85:
        return "HIGH_VOL"

    # Step 2: Check for trending market
    if trend_strength >= 0.7 and alignment != 0:
        if vol <= 0.35:
            return "TREND_LOW_VOL"
        else:
            return "TREND_HIGH_VOL"

    # Step 3: Default to range-bound
    return "RANGE_LOW_VOL"
```

---

## Decision Tree

```
                        START
                          │
                          ▼
                ┌─────────────────┐
                │ atr_percentile  │
                │    >= 0.85?     │
                └────────┬────────┘
                    YES  │  NO
                    ▼    │
              ┌──────────┐    │
              │ HIGH_VOL │    │
              └──────────┘    │
                              ▼
                    ┌─────────────────┐
                    │ |ema200_slope_z|│
                    │    >= 0.7?      │
                    │      AND        │
                    │ alignment != 0? │
                    └────────┬────────┘
                        YES  │  NO
                        ▼    │
              ┌─────────────────┐    │
              │ atr_percentile  │    │
              │    <= 0.35?     │    │
              └────────┬────────┘    │
                  YES  │  NO         │
                  ▼    ▼             ▼
         ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
         │TREND_LOW_VOL │ │TREND_HIGH_VOL│ │RANGE_LOW_VOL │
         └──────────────┘ └──────────────┘ └──────────────┘
```

---

## Regime Smoothing

Raw regime labels can be noisy (flipping every bar). Smoothing stabilizes them.

### Method: Rolling Majority Vote

```python
def smooth_regime(regime_series, window=30):
    # For each bar, look at last 30 bars
    # The most common regime in those 30 bars wins
    return rolling_majority_vote(regime_series, window=30)
```

### Example:

```
Bar #  | Raw Regime     | Smoothed Regime
-------|----------------|----------------
1      | RANGE_LOW_VOL  | RANGE_LOW_VOL
2      | RANGE_LOW_VOL  | RANGE_LOW_VOL
3      | TREND_LOW_VOL  | RANGE_LOW_VOL  ← minority, smoothed out
4      | RANGE_LOW_VOL  | RANGE_LOW_VOL
5      | RANGE_LOW_VOL  | RANGE_LOW_VOL
...
25     | TREND_LOW_VOL  | TREND_LOW_VOL  ← now majority
26     | TREND_LOW_VOL  | TREND_LOW_VOL
```

### Why 30 bars?
- 30 minutes of data
- Filters out momentary spikes
- Keeps regime stable for trading decisions

---

## Complete Pipeline

### Step 1: Load State Vectors

```python
df = pd.read_parquet("data/state_vectors/BTCUSDT_1m_state.parquet")
```

**Required columns:** `ema200_slope_z`, `atr_percentile`, `trend_alignment`

---

### Step 2: Label Each Row (Raw)

```python
df["regime_raw"] = df.apply(label_regime_row, axis=1)
```

**Result:**
```
| time                | ema200_slope_z | atr_percentile | trend_alignment | regime_raw    |
|---------------------|----------------|----------------|-----------------|---------------|
| 2024-01-01 00:00:00 | 0.31           | 0.45           | 1               | RANGE_LOW_VOL |
| 2024-01-01 00:01:00 | 0.85           | 0.30           | 1               | TREND_LOW_VOL |
| 2024-01-01 00:02:00 | 0.90           | 0.88           | 1               | HIGH_VOL      |
```

---

### Step 3: Smooth Regimes

```python
df["regime"] = smooth_regime(df["regime_raw"], window=30)
```

**Result:** Noisy labels become stable.

---

### Step 4: Save to Parquet

```python
df[["regime"]].to_parquet("data/regimes/BTCUSDT_1m_regimes.parquet")
```

**Storage location:** `data/regimes/BTCUSDT_1m_regimes.parquet`

**Final format:**
```
| time (index)        | regime        |
|---------------------|---------------|
| 2024-01-01 00:00:00 | RANGE_LOW_VOL |
| 2024-01-01 00:01:00 | RANGE_LOW_VOL |
| 2024-01-01 00:02:00 | RANGE_LOW_VOL |
```

---

## Visual Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STATE VECTORS                                      │
│            (ema200_slope_z, atr_percentile, trend_alignment)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       REGIME CLASSIFICATION                                 │
│                                                                             │
│   Input:                          Output:                                   │
│   • ema200_slope_z = 0.85         • regime_raw = "TREND_LOW_VOL"           │
│   • atr_percentile = 0.30                                                   │
│   • trend_alignment = 1                                                     │
│                                                                             │
│   Logic:                                                                    │
│   1. vol (0.30) < 0.85         → not HIGH_VOL                              │
│   2. trend (0.85) >= 0.7       → trending                                  │
│   3. alignment (1) != 0        → confirmed                                 │
│   4. vol (0.30) <= 0.35        → low volatility                            │
│   Result: TREND_LOW_VOL                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SMOOTHING (30-bar window)                          │
│                                                                             │
│   Raw:      [R, R, T, R, R, R, T, T, T, T, T, T, T, T, T, T, ...]          │
│   Smoothed: [R, R, R, R, R, R, R, R, T, T, T, T, T, T, T, T, ...]          │
│                                                                             │
│   (R = RANGE_LOW_VOL, T = TREND_LOW_VOL)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PARQUET STORAGE                                    │
│                                                                             │
│   File: data/regimes/BTCUSDT_1m_regimes.parquet                             │
│   Columns: time (index), regime (string)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Regime Examples

### Example 1: RANGE_LOW_VOL
```
ema200_slope_z = 0.3   (weak trend, below 0.7)
atr_percentile = 0.25  (low volatility)
trend_alignment = 0    (no clear direction)

Result: RANGE_LOW_VOL (sideways, calm market)
```

### Example 2: TREND_LOW_VOL
```
ema200_slope_z = 1.2   (strong trend, above 0.7)
atr_percentile = 0.30  (low volatility, below 0.35)
trend_alignment = 1    (bullish)

Result: TREND_LOW_VOL (smooth uptrend, ideal for trading)
```

### Example 3: TREND_HIGH_VOL
```
ema200_slope_z = 0.9   (strong trend, above 0.7)
atr_percentile = 0.60  (medium volatility)
trend_alignment = -1   (bearish)

Result: TREND_HIGH_VOL (volatile downtrend)
```

### Example 4: HIGH_VOL
```
ema200_slope_z = 0.5   (doesn't matter)
atr_percentile = 0.92  (extreme volatility, above 0.85)
trend_alignment = 1    (doesn't matter)

Result: HIGH_VOL (volatility shock, risky)
```

---

## Why Regimes Matter for Trading

| Regime | Trading Implication |
|--------|---------------------|
| `RANGE_LOW_VOL` | Low opportunity, choppy, avoid or use mean-reversion |
| `TREND_LOW_VOL` | Best for trading, smooth trends, high win rate |
| `TREND_HIGH_VOL` | Tradeable but volatile, wider stops needed |
| `HIGH_VOL` | Dangerous, consider blocking trades in this regime |

**In backtests:**
- `blocked_regimes: ["HIGH_VOL"]` can filter out risky periods
- Most profitable trades occur in `TREND_LOW_VOL`

---

## Regime Distribution (Typical)

```
RANGE_LOW_VOL   ████████████████████████████████  ~50%
TREND_LOW_VOL   ██████████████                    ~25%
TREND_HIGH_VOL  ████████                          ~15%
HIGH_VOL        ████                              ~10%
```

---

## File Structure

```
regime/
├── __init__.py           # Exports label_regime_row, smooth_regime
├── regime_labeler.py     # Core logic: thresholds, classification, smoothing
├── run_regime_labeling.py# Pipeline script
└── regime.md             # This documentation
```

---

## Usage

**Run regime labeling:**
```bash
python -m trade_system.regime.run_regime_labeling --pair BTCUSDT
```

**Load saved regimes:**
```python
import pandas as pd
regimes = pd.read_parquet("data/regimes/BTCUSDT_1m_regimes.parquet")
```

**Use in similarity engine:**
```python
# Only search for neighbors in the same regime
neighbors = similarity_engine.query(state_vector, regime="TREND_LOW_VOL", horizon=5)
```
