# State Vector Documentation

## What is a State Vector?

A **state vector** is a 10-dimensional representation of the current market condition at any given moment. It captures trend, momentum, volatility, volume, and price location in a normalized format that can be compared using KNN.

---

## The 10 Dimensions

| # | Dimension | Type | Range | Description |
|---|-----------|------|-------|-------------|
| 1 | `ema50_slope_z` | float | -3 to +3 | Short-term trend direction |
| 2 | `ema200_slope_z` | float | -3 to +3 | Long-term trend direction |
| 3 | `trend_alignment` | int | -1, 0, +1 | EMA50 vs EMA200 position |
| 4 | `return_5m_z` | float | -3 to +3 | 5-bar momentum |
| 5 | `return_15m_z` | float | -3 to +3 | 15-bar momentum |
| 6 | `rsi_z` | float | -3 to +3 | Overbought/oversold level |
| 7 | `atr_percentile` | float | 0 to 1 | Volatility level |
| 8 | `volume_z` | float | -3 to +3 | Volume activity |
| 9 | `vwap_distance_z` | float | -3 to +3 | Distance from fair value |
| 10 | `range_position` | float | 0 to 1 | Position in price range |

---

## Complete Pipeline: OHLCV → State Vector

### Step 1: Load OHLCV Data

```python
loader = OHLCVLoader()
df = loader.fetch_ohlcv(pair="BTCUSDT", start_time="2020-01-01", end_time="2025-12-31")
```

**Input DataFrame:**
```
| time                | open    | high    | low     | close   | volume     |
|---------------------|---------|---------|---------|---------|------------|
| 2024-01-01 00:00:00 | 42000.0 | 42100.0 | 41900.0 | 42050.0 | 1500000.0  |
| 2024-01-01 00:01:00 | 42050.0 | 42150.0 | 42000.0 | 42100.0 | 1200000.0  |
```

---

### Step 2: Compute Raw Features

```python
df = compute_trend_features(df)      # ema50, ema200, ema50_slope, ema200_slope, trend_alignment
df = compute_momentum_features(df)   # return_5m, return_15m, rsi_14
df = compute_volatility_features(df) # atr_14
df = compute_volume_features(df)     # volume_raw
df = compute_location_features(df)   # vwap, vwap_distance, range_position
```

**After feature computation:**
```
| time       | close   | ema50   | ema200  | ema50_slope | rsi_14 | atr_14 | ... |
|------------|---------|---------|---------|-------------|--------|--------|-----|
| 2024-01-01 | 42050.0 | 41800.0 | 41500.0 | 0.00025     | 55.3   | 450.0  | ... |
| 2024-01-01 | 42100.0 | 41820.0 | 41510.0 | 0.00028     | 57.1   | 455.0  | ... |
```

---

### Step 3: Normalize Features

```python
norm = RollingNormalizer(window=2000)

df["ema50_slope_z"] = norm.zscore(df["ema50_slope"])
df["ema200_slope_z"] = norm.zscore(df["ema200_slope"])
df["return_5m_z"] = norm.zscore(df["return_5m"])
df["return_15m_z"] = norm.zscore(df["return_15m"])
df["rsi_z"] = norm.zscore(df["rsi_14"])
df["volume_z"] = norm.zscore(df["volume_raw"])
df["vwap_distance_z"] = norm.zscore(df["vwap_distance"])
df["atr_percentile"] = norm.percentile(df["atr_14"])
```

**Normalization mapping:**

| Raw Feature | Method | Normalized Feature |
|-------------|--------|-------------------|
| ema50_slope | z-score | ema50_slope_z |
| ema200_slope | z-score | ema200_slope_z |
| return_5m | z-score | return_5m_z |
| return_15m | z-score | return_15m_z |
| rsi_14 | z-score | rsi_z |
| volume_raw | z-score | volume_z |
| vwap_distance | z-score | vwap_distance_z |
| atr_14 | percentile | atr_percentile |
| trend_alignment | none | trend_alignment |
| range_position | none | range_position |

**After normalization:**
```
| time       | ema50_slope_z | rsi_z | atr_percentile | volume_z | ... |
|------------|---------------|-------|----------------|----------|-----|
| 2024-01-01 | 0.52          | 0.31  | 0.75           | 1.20     | ... |
| 2024-01-01 | 0.58          | 0.45  | 0.78           | 0.95     | ... |
```

---

### Step 4: Build State Object

```python
state_df["state"] = state_df.apply(build_state, axis=1)
```

**build_state function:**
```python
def build_state(row) -> MarketState:
    return MarketState(
        ema50_slope_z=row["ema50_slope_z"],
        ema200_slope_z=row["ema200_slope_z"],
        trend_alignment=row["trend_alignment"],
        return_5m_z=row["return_5m_z"],
        return_15m_z=row["return_15m_z"],
        rsi_z=row["rsi_z"],
        atr_percentile=row["atr_percentile"],
        volume_z=row["volume_z"],
        vwap_distance_z=row["vwap_distance_z"],
        range_position=row["range_position"],
    )
```

**MarketState dataclass:**
```python
@dataclass(frozen=True)
class MarketState:
    ema50_slope_z: float
    ema200_slope_z: float
    trend_alignment: int      # -1, 0, +1
    return_5m_z: float
    return_15m_z: float
    rsi_z: float
    atr_percentile: float
    volume_z: float
    vwap_distance_z: float
    range_position: float
```

---

### Step 5: Save to Parquet

```python
save_state_vectors_parquet(df=state_df, pair="BTCUSDT", timeframe="1m")
```

**Storage location:** `data/state_vectors/BTCUSDT_1m_state.parquet`

**Final stored format:**
```
| time (index)        | ema50_slope_z | ema200_slope_z | trend_alignment | ... | range_position | pair    |
|---------------------|---------------|----------------|-----------------|-----|----------------|---------|
| 2024-01-01 00:00:00 | 0.52          | 0.31           | 1               | ... | 0.75           | BTCUSDT |
| 2024-01-01 00:01:00 | 0.58          | 0.45           | 1               | ... | 0.72           | BTCUSDT |
```

---

## Visual Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OHLCV DATA                                     │
│                    (open, high, low, close, volume)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE COMPUTATION                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ ┌───────────┐ │
│  │   TREND     │ │  MOMENTUM   │ │ VOLATILITY  │ │ VOLUME  │ │ LOCATION  │ │
│  │ ema50       │ │ return_5m   │ │ atr_14      │ │ volume  │ │ vwap      │ │
│  │ ema200      │ │ return_15m  │ │             │ │         │ │ range_pos │ │
│  │ slopes      │ │ rsi_14      │ │             │ │         │ │           │ │
│  │ alignment   │ │             │ │             │ │         │ │           │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ └───────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NORMALIZATION                                     │
│                                                                             │
│   Z-Score (7 features):        Percentile (1 feature):    Raw (2 features): │
│   • ema50_slope_z              • atr_percentile           • trend_alignment │
│   • ema200_slope_z                                        • range_position  │
│   • return_5m_z                                                             │
│   • return_15m_z                                                            │
│   • rsi_z                                                                   │
│   • volume_z                                                                │
│   • vwap_distance_z                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STATE VECTOR (10-D)                                │
│                                                                             │
│   [ ema50_slope_z,  ema200_slope_z,  trend_alignment,  return_5m_z,        │
│     return_15m_z,   rsi_z,           atr_percentile,   volume_z,            │
│     vwap_distance_z, range_position ]                                       │
│                                                                             │
│   Example: [ 0.52, 0.31, 1, 0.15, 0.22, -0.45, 0.75, 1.20, 0.08, 0.65 ]    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PARQUET STORAGE                                    │
│                                                                             │
│   File: data/state_vectors/BTCUSDT_1m_state.parquet                         │
│   Rows: ~3,000,000 (one per minute from 2020 to 2025)                       │
│   Columns: 10 features + time index + pair                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example: Single State Vector

**Time:** 2024-06-15 14:30:00

| Dimension | Value | Interpretation |
|-----------|-------|----------------|
| ema50_slope_z | +1.5 | Strong uptrend (short-term) |
| ema200_slope_z | +0.8 | Moderate uptrend (long-term) |
| trend_alignment | +1 | Bullish (EMA50 > EMA200) |
| return_5m_z | +0.3 | Slight positive momentum |
| return_15m_z | +0.7 | Good positive momentum |
| rsi_z | +1.2 | Approaching overbought |
| atr_percentile | 0.85 | High volatility (85th percentile) |
| volume_z | +2.0 | Very high volume |
| vwap_distance_z | +0.5 | Slightly above fair value |
| range_position | 0.90 | Near top of recent range |

**Interpretation:** Strong bullish conditions with high volume and volatility. Price near recent highs.

---

## Why 10 Dimensions?

| Category | Dimensions | Purpose |
|----------|------------|---------|
| **Trend** | 3 | Direction of market (up/down) |
| **Momentum** | 3 | Speed of price movement |
| **Volatility** | 1 | Risk level |
| **Volume** | 1 | Confirmation of moves |
| **Location** | 2 | Context within price range |

These 10 dimensions capture the essential market state for KNN similarity matching.

---

## File Structure

```
state/
├── __init__.py           # Exports MarketState, build_state, save_state_vectors_parquet
├── state_schema.py       # MarketState dataclass definition
├── state_builder.py      # build_state() function
├── state_store.py        # save_state_vectors_parquet() function
├── state_validator.py    # (empty - for future validation)
├── run_state_pipeline.py # Complete pipeline orchestrator
└── state.md              # This documentation
```

---

## Usage

**Build state vectors from database:**
```bash
python -m trade_system.state.run_state_pipeline --pair BTCUSDT --start 2020-01-01 --end 2025-12-31
```

**Load saved state vectors:**
```python
import pandas as pd
states = pd.read_parquet("data/state_vectors/BTCUSDT_1m_state.parquet")
```
