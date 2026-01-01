# Trading System - Complete Documentation

A **state-based quantitative trading system** that trades only when historical market conditions show statistical edge. The system avoids prediction and overtrading by relying on **market memory, regimes, and expectancy**.

---

## Table of Contents

1. [Core Philosophy](#1-core-philosophy)
2. [System Architecture](#2-system-architecture)
3. [Installation & Setup](#3-installation--setup)
4. [Configuration](#4-configuration)
5. [Pipeline Stages](#5-pipeline-stages)
6. [Market State Vector](#6-market-state-vector)
7. [Market Regimes](#7-market-regimes)
8. [Outcome Labels (MFE/MAE)](#8-outcome-labels-mfemae)
9. [Similarity Engine](#9-similarity-engine)
10. [Decision Engine](#10-decision-engine)
11. [Backtesting Framework](#11-backtesting-framework)
12. [Grid Search & Hyperparameter Tuning](#12-grid-search--hyperparameter-tuning)
13. [Running Experiments](#13-running-experiments)
14. [Using BEST_PARAMS.yaml](#14-using-best_paramsyaml)
15. [Project Structure](#15-project-structure)
16. [Troubleshooting](#16-troubleshooting)
17. [Key Concepts Explained](#17-key-concepts-explained)
18. [Glossary](#18-glossary)

---

## 1. Core Philosophy

> **We do not predict price. We recognize market states and act only when history supports an asymmetric edge.**

### Principles

1. **Markets are probabilistic, not deterministic**
   - We don't try to predict exact price movements
   - We calculate probabilities based on similar historical situations

2. **Capital preservation comes first**
   - Never risk more than configured amount per trade
   - Filter out low-quality signals

3. **Fewer high-quality trades > frequent trades**
   - Quality over quantity
   - Only trade when statistical edge exists

4. **Decisions are statistics-driven, not indicator-driven**
   - No subjective interpretation of charts
   - Pure mathematical expectancy calculations

---

## 2. System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  PostgreSQL/TimescaleDB (1m OHLCV) OR Local Parquet Files       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                           │
├─────────────────────────────────────────────────────────────────┤
│  EMA Slopes │ RSI │ ATR │ Volume │ VWAP │ Range Position        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STATE VECTOR ENGINE                            │
├─────────────────────────────────────────────────────────────────┤
│  10-Dimensional Normalized Market State Representation           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REGIME DETECTION                              │
├─────────────────────────────────────────────────────────────────┤
│  HIGH_VOL │ TREND_HIGH_VOL │ TREND_LOW_VOL │ RANGE_LOW_VOL      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OUTCOME LABELING                               │
├─────────────────────────────────────────────────────────────────┤
│  MFE (Max Favorable) │ MAE (Max Adverse) for each horizon       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SIMILARITY ENGINE                              │
├─────────────────────────────────────────────────────────────────┤
│  KNN Search: Find K similar historical states (bruteforce/FAISS)│
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DECISION ENGINE                                │
├─────────────────────────────────────────────────────────────────┤
│  Expectancy Calculation │ Trade Filters │ Position Sizing       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TRADE EXECUTION                                │
├─────────────────────────────────────────────────────────────────┤
│  Entry │ Stop Loss │ Take Profit │ Trailing Stop │ Exit         │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Explained

| Step | Component | Input | Output |
|------|-----------|-------|--------|
| 1 | Data Layer | Database/Files | Raw OHLCV candles |
| 2 | Features | OHLCV | Technical indicators |
| 3 | State Vector | Indicators | 10D normalized vector |
| 4 | Regime | State vectors | Market classification |
| 5 | Outcomes | OHLCV | MFE/MAE labels |
| 6 | Similarity | Current state | K similar historical states |
| 7 | Decision | Similar states | TRADE or NO_TRADE signal |
| 8 | Execution | Signal | Open/close positions |

---

## 3. Installation & Setup

### Step 1: System Requirements

- **Python**: 3.9 or higher
- **OS**: Windows, Linux, or macOS
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for data files

### Step 2: Clone/Download the Project

```bash
cd c:\Users\infra\Desktop\ml_pro
# Your project should be in trade_system_1/
```

### Step 3: Create Virtual Environment

**Windows:**
```bash
cd trade_system_1
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
cd trade_system_1
python -m venv .venv
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- pandas
- numpy
- scikit-learn
- pyyaml
- pyarrow (for parquet files)
- psycopg2 (for PostgreSQL)
- faiss-cpu (optional, for fast similarity search)
- tqdm (for progress bars)

### Step 5: Configure Data Source

**Option A: Use Local Parquet Files**

Place your OHLCV data in:
```
data/ohlcv/BTCUSDT_1m_ohlcv.parquet
```

Required columns: `open`, `high`, `low`, `close`, `volume`
Index: DateTimeIndex (timestamp)

**Option B: Use PostgreSQL Database**

Create `.env` file in project root:
```bash
DATABASE_URL=postgresql://username:password@host:5432/crypto_data
```

### Step 6: Verify Installation

```bash
python scripts/run_pipeline.py --dry-run
```

Expected output:
```
Pipeline would execute: state_vectors -> regime_labeling -> outcome_labeling
```

---

## 4. Configuration

### Configuration File Location

```
config/config.yaml
```

### Complete Configuration Reference

```yaml
# =============================================================================
# DATA SETTINGS
# =============================================================================
data:
  pair: "BTCUSDT"              # Trading pair symbol
  timeframe: "1m"              # Candle timeframe (1m, 5m, 15m, 1h)
  start_date: "2020-01-01"     # Start date for data fetch
  end_date: "2025-12-15"       # End date for data fetch
  database_url: "postgresql://localhost/crypto_data"  # Can override with .env
  max_gap_tolerance: 5000      # Max missing candles allowed
  fill_small_gaps: false       # Auto-fill gaps <= 100 candles

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
features:
  ema_fast_period: 50          # Fast EMA period (trend)
  ema_slow_period: 200         # Slow EMA period (trend)
  rsi_period: 14               # RSI period (momentum)
  return_periods:              # Return calculation periods
    - 5                        # 5-minute returns
    - 15                       # 15-minute returns
  atr_period: 14               # ATR period (volatility)
  range_lookback: 50           # High/low range lookback

# =============================================================================
# NORMALIZATION
# =============================================================================
normalization:
  window: 2000                 # Rolling window for z-score calculations
                               # Prevents look-ahead bias

# =============================================================================
# REGIME DETECTION
# =============================================================================
regime:
  high_vol_threshold: 0.85     # ATR percentile for HIGH_VOL regime
  low_vol_threshold: 0.35      # ATR percentile for LOW_VOL variants
  trend_strength_threshold: 0.7 # Min trend strength for TREND regimes
  smoothing_window: 30         # Majority vote smoothing (bars)

# =============================================================================
# OUTCOME LABELING
# =============================================================================
outcomes:
  horizons:                    # Forward-looking windows (minutes)
    - 5                        # 5-minute outcomes
    - 10                       # 10-minute outcomes
    - 15                       # 15-minute outcomes
    - 30                       # 30-minute outcomes
    - 120                      # 2-hour outcomes

# =============================================================================
# SIMILARITY ENGINE (KNN)
# =============================================================================
similarity:
  k: 200                       # Number of similar states to find
  default_horizon: 30          # Default outcome horizon for queries
  max_distance: 1.5            # Max distance threshold (reject if too far)
  backend: "bruteforce"        # "bruteforce" (exact) or "faiss" (fast)
  faiss_nlist: 100             # FAISS: Number of IVF clusters
  faiss_nprobe: 10             # FAISS: Clusters to search at query time

# =============================================================================
# DECISION ENGINE
# =============================================================================
decision:
  capital: 10000               # Starting capital in USD
  risk_per_trade: 0.005        # Risk per trade (0.5% = 0.005)
  max_leverage: 1.0            # Maximum leverage multiplier
  min_expectancy: 0.001        # Minimum expected return to take trade
  max_distance: 3.0            # Max avg distance to neighbors
  blocked_regimes: []          # Regimes to skip trading

# =============================================================================
# BACKTEST SETTINGS
# =============================================================================
backtest:
  train_ratio: 0.70            # Train/test split (70% train, 30% test)
  slippage_pct: 0.0005         # Slippage per trade (0.05%)
  commission_pct: 0.0004       # Commission per trade (0.04%)
  max_bars_in_trade: 120       # Force exit after N bars (0 = disabled)
  trailing_stop_pct: 0.0       # Trailing stop % (0 = disabled)
  trailing_stop_activation_pct: 0.0  # Activate after X% profit
  sample_interval: 60          # Check signals every N bars
  save_trades: true            # Save trade log to parquet
  output_dir: "backtest"       # Subdirectory for results

# =============================================================================
# PIPELINE SETTINGS
# =============================================================================
pipeline:
  stages:
    state_vectors: true
    regime_labeling: true
    outcome_labeling: true
    similarity: true
    decision: true
  n_jobs: 1                    # Parallel processing (future)

# =============================================================================
# OUTPUT PATHS
# =============================================================================
paths:
  data_dir: "data"
  state_vectors_dir: "state_vectors"
  regimes_dir: "regimes"
  outcomes_dir: "outcomes"
  logs_dir: "logs"

# =============================================================================
# LOGGING
# =============================================================================
logging:
  level: "INFO"                # DEBUG, INFO, WARNING, ERROR
  console_output: true
  file_output: true
  log_file: "pipeline.log"
```

---

## 5. Pipeline Stages

### Running the Pipeline

**Run all stages:**
```bash
python scripts/run_pipeline.py
```

**Run specific stages:**
```bash
python scripts/run_pipeline.py --stages state_vectors regime_labeling
```

**Override parameters:**
```bash
python scripts/run_pipeline.py --pair ETHUSDT --start 2023-01-01 --end 2023-12-31
```

**Dry run (preview without executing):**
```bash
python scripts/run_pipeline.py --dry-run
```

### Stage Details

#### Stage 1: State Vectors

**Purpose:** Transform raw OHLCV data into normalized 10D state vectors.

**Process:**
1. Fetch OHLCV data from database or local file
2. Calculate technical indicators (EMA, RSI, ATR, etc.)
3. Normalize indicators using rolling z-scores
4. Save to `data/state_vectors/`

**Output:** `data/state_vectors/BTCUSDT_1m_state.parquet`

#### Stage 2: Regime Labeling

**Purpose:** Classify each timestamp into one of 4 market regimes.

**Process:**
1. Load state vectors
2. Calculate trend strength and volatility metrics
3. Apply classification rules
4. Smooth using majority vote
5. Save to `data/regimes/`

**Output:** `data/regimes/BTCUSDT_1m_regimes.parquet`

#### Stage 3: Outcome Labeling

**Purpose:** Calculate forward-looking MFE/MAE for each timestamp.

**Process:**
1. Load OHLCV data
2. For each timestamp, look forward N bars
3. Calculate maximum favorable excursion (MFE)
4. Calculate maximum adverse excursion (MAE)
5. Save to `data/outcomes/`

**Output:** `data/outcomes/BTCUSDT_1m_outcomes.parquet`

#### Stage 4: Similarity (In-Memory)

**Purpose:** Build the KNN search engine for finding similar states.

**Process:**
1. Load state vectors and outcomes
2. Build index (bruteforce or FAISS)
3. Ready for queries

**Output:** In-memory engine

#### Stage 5: Decision (In-Memory)

**Purpose:** Generate trading signals based on similarity results.

**Process:**
1. Query similarity engine
2. Calculate expectancy from neighbors
3. Apply trade filters
4. Generate TRADE or NO_TRADE signal

**Output:** Trading signal with direction, size, stops

---

## 6. Market State Vector

### What is a State Vector?

A **10-dimensional normalized representation** of market conditions at a specific timestamp. Think of it as a "fingerprint" of the market at that moment.

### The 10 Dimensions

| # | Name | Description | Normalization | Range |
|---|------|-------------|---------------|-------|
| 1 | `ema50_slope_z` | 5-bar rate of change of EMA(50) | Z-score | -3 to +3 typical |
| 2 | `ema200_slope_z` | 20-bar rate of change of EMA(200) | Z-score | -3 to +3 typical |
| 3 | `trend_alignment` | Direction of EMA50 vs EMA200 | Sign | -1, 0, +1 |
| 4 | `return_5m_z` | 5-minute percentage return | Z-score | -3 to +3 typical |
| 5 | `return_15m_z` | 15-minute percentage return | Z-score | -3 to +3 typical |
| 6 | `rsi_z` | RSI(14) normalized | Z-score | -3 to +3 typical |
| 7 | `atr_percentile` | ATR(14) as percentile | Percentile | 0 to 1 |
| 8 | `volume_z` | Volume relative to average | Z-score | -3 to +3 typical |
| 9 | `vwap_distance_z` | Price distance from VWAP | Z-score | -3 to +3 typical |
| 10 | `range_position` | Position in 50-bar high-low range | Ratio | 0 to 1 |

### Normalization Methods

**Z-Score:**
```
z = (value - rolling_mean) / rolling_std
```
- Uses rolling window of 2000 bars
- Measures how many standard deviations from mean
- Prevents look-ahead bias

**Percentile:**
```
percentile = rank(value) / count
```
- Position within historical distribution
- 0 = lowest, 1 = highest

### Example State Vector

```
Timestamp: 2024-03-15 10:30:00
State: [0.45, -0.12, 1.0, 1.23, 0.89, -0.34, 0.72, 1.56, 0.23, 0.65]

Interpretation:
- ema50_slope_z = 0.45  → EMA50 rising slightly above average
- ema200_slope_z = -0.12 → EMA200 nearly flat
- trend_alignment = 1.0  → EMA50 > EMA200 (bullish)
- return_5m_z = 1.23     → Strong 5m return (1.23 std above mean)
- return_15m_z = 0.89    → Above average 15m return
- rsi_z = -0.34          → RSI slightly below average
- atr_percentile = 0.72  → Volatility at 72nd percentile (elevated)
- volume_z = 1.56        → High volume (1.56 std above mean)
- vwap_distance_z = 0.23 → Price slightly above VWAP
- range_position = 0.65  → Price at 65% of recent range (upper half)
```

---

## 7. Market Regimes

### What is a Regime?

A **market regime** is a classification of market behavior into distinct states. The system identifies 4 regimes:

### Regime Definitions

#### HIGH_VOL
- **Criteria:** ATR percentile >= 0.85
- **Description:** Extreme volatility, no clear direction
- **Characteristics:**
  - Fast, large price swings
  - Often during news events or market stress
  - Can be profitable for short horizons
- **Typical occurrence:** ~15% of the time

#### TREND_HIGH_VOL
- **Criteria:** Trend strength >= 0.7 AND ATR percentile > 0.35
- **Description:** Strong directional move with high volatility
- **Characteristics:**
  - Clear direction (up or down)
  - Large candles in trend direction
  - Good for momentum strategies
- **Typical occurrence:** ~25% of the time

#### TREND_LOW_VOL
- **Criteria:** Trend strength >= 0.7 AND ATR percentile <= 0.35
- **Description:** Gradual trend with low volatility
- **Characteristics:**
  - Slow, grinding price movement
  - Small candles
  - Often difficult to trade profitably
- **Typical occurrence:** ~20% of the time

#### RANGE_LOW_VOL
- **Criteria:** Everything else
- **Description:** Consolidation, sideways movement
- **Characteristics:**
  - Price bouncing between support/resistance
  - Mean reversion opportunities
  - Choppy price action
- **Typical occurrence:** ~40% of the time

### Regime Smoothing

Raw regime labels can flip frequently. To prevent whipsaws:

```
Smoothed regime = Majority vote over last 30 bars
```

Example:
```
Raw:      HIGH_VOL, RANGE, HIGH_VOL, RANGE, RANGE, ...
Smoothed: RANGE (majority of last 30)
```

### Why Regimes Matter

1. **Similar states in same regime perform similarly**
   - A pattern in HIGH_VOL may not work in RANGE_LOW_VOL

2. **Some regimes are unprofitable**
   - TREND_LOW_VOL often has low edge for momentum strategies

3. **Regime-filtered similarity search**
   - Only compare to historical states in the same regime

---

## 8. Outcome Labels (MFE/MAE)

### What is MFE?

**Maximum Favorable Excursion (MFE)** = The best possible gain you could have achieved within a time horizon.

```
For LONG:  MFE = max((high - entry) / entry) over horizon
For SHORT: MFE = max((entry - low) / entry) over horizon
```

### What is MAE?

**Maximum Adverse Excursion (MAE)** = The worst possible drawdown within a time horizon.

```
For LONG:  MAE = min((low - entry) / entry) over horizon  [negative]
For SHORT: MAE = min((entry - high) / entry) over horizon [negative]
```

### Example

```
Entry price: $100
Horizon: 15 minutes
Price path: $100 → $99 → $98 → $101 → $102 → $100

MFE = ($102 - $100) / $100 = +2.0%  (best case)
MAE = ($98 - $100) / $100 = -2.0%   (worst case)
```

### Horizons

The system computes outcomes for multiple horizons:
- **5 minutes** - Very short-term
- **10 minutes** - Short-term
- **15 minutes** - Short-term
- **30 minutes** - Medium-term
- **120 minutes** - Longer-term

### Expectancy

```
Expectancy = mean(MFE) + mean(MAE)
```

Since MAE is negative:
- **Positive expectancy** = Edge exists (MFE > |MAE|)
- **Negative expectancy** = No edge (|MAE| > MFE)

### Example Expectancy Calculation

Over 200 similar historical states:
```
mean(MFE) = +0.15%  (average best case gain)
mean(MAE) = -0.08%  (average worst case loss)

Expectancy = +0.15% + (-0.08%) = +0.07% (profitable)
```

---

## 9. Similarity Engine

### Purpose

Find **K most similar historical states** to the current market state.

### How It Works

1. **Input:** Current 10D state vector
2. **Filter:** Only states from same regime
3. **Distance:** Calculate Euclidean distance in 10D space
4. **Select:** Return K=200 nearest neighbors
5. **Aggregate:** Extract outcomes from neighbors

### Distance Calculation

```
distance = sqrt(Σ(current[i] - historical[i])²)
```

Where i = 1 to 10 dimensions.

### Backends

| Backend | Algorithm | Accuracy | Speed | Memory |
|---------|-----------|----------|-------|--------|
| bruteforce | Exact search | 100% | ~50ms | Low |
| faiss | Approximate (IVF) | ~95-99% | ~0.5ms | Higher |

### When to Use Each Backend

**Bruteforce:**
- Backtesting (need exact results)
- Small datasets (<100K samples)
- When max_timestamp filtering is needed

**FAISS:**
- Realtime trading (speed critical)
- Large datasets (>500K samples)
- When slight approximation is acceptable

### Why Bruteforce for Backtesting?

During backtesting, we must prevent **look-ahead bias** by only searching states BEFORE the current timestamp.

FAISS doesn't efficiently support this `max_timestamp` filtering, so bruteforce is used.

### Similarity Query Result

```python
{
    "neighbors": 200,           # Number of similar states found
    "avg_distance": 1.23,       # Average distance to neighbors
    "mean_mfe": 0.0015,         # 0.15% average best case
    "mean_mae": -0.0008,        # -0.08% average worst case
    "expectancy": 0.0007,       # 0.07% expected return
    "std_mfe": 0.002,           # Standard deviation of MFE
    "std_mae": 0.001,           # Standard deviation of MAE
    "regime": "HIGH_VOL"        # Current regime
}
```

---

## 10. Decision Engine

### Decision Flow

```
Query similarity engine
        │
        ▼
Calculate expectancy
        │
        ▼
┌───────┴───────┐
│ Apply Filters │
├───────────────┤
│ min_expectancy│
│ max_distance  │
│ blocked_regime│
│ min_neighbors │
└───────┬───────┘
        │
        ▼
   Pass all? ──No──► NO_TRADE
        │
       Yes
        │
        ▼
Determine direction
        │
        ▼
Calculate position size
        │
        ▼
Set stop loss & take profit
        │
        ▼
      TRADE
```

### Trade Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| Expectancy | min_expectancy | 0.001 | Skip if expected return < 0.1% |
| Distance | max_distance | 3.0 | Skip if neighbors too dissimilar |
| Regime | blocked_regimes | [] | Skip if in blocked regime |
| Neighbors | (hardcoded) | 50 | Skip if too few similar states |

### Direction Selection

```python
if mean_mfe > abs(mean_mae):
    direction = "LONG"   # Upside potential > Downside risk
else:
    direction = "SHORT"  # Downside potential > Upside risk
```

### Position Sizing

```python
# Calculate stop loss from historical MAE distribution
stop_loss_pct = percentile(MAE_values, 5)  # 5th percentile

# Calculate risk amount
risk_amount = capital * risk_per_trade  # e.g., $10,000 * 0.5% = $50

# Calculate position size
position_size = risk_amount / abs(stop_loss_pct)

# Cap at max leverage
max_position = capital * max_leverage
position_size = min(position_size, max_position)
```

### Take Profit Calculation

```python
take_profit_pct = mean(MFE_values)  # Average best case from history
```

### Decision Output

```python
{
    "action": "TRADE",
    "direction": "LONG",
    "position_size": 5000.0,
    "stop_loss_pct": -0.003,     # -0.3%
    "take_profit_pct": 0.0015,   # +0.15%
    "expectancy": 0.0007,
    "confidence": 0.85
}
```

---

## 11. Backtesting Framework

### What is Walk-Forward Backtesting?

A methodology that simulates real trading by:
1. Using only past data to make decisions
2. Walking forward through time bar-by-bar
3. Executing trades with realistic costs

### Train/Test Split

```
|------------ TRAINING (70%) ------------|---- TEST (30%) ----|
start_date                          split_point           end_date

Training period: Build similarity database
Test period: Walk forward and make decisions
```

### Running Backtests

**Basic backtest:**
```bash
python scripts/run_backtest.py
```

**With custom capital:**
```bash
python scripts/run_backtest.py --capital 50000
```

**With custom split:**
```bash
python scripts/run_backtest.py --train-ratio 0.80
```

**Using BEST_PARAMS.yaml:**
```bash
python scripts/run_backtest.py --params-file data/grid_search/h5/BEST_PARAMS.yaml
```

**Save trade log:**
```bash
python scripts/run_backtest.py --save-trades
```

### Execution Model

```
Signal at time T
        │
        ▼
Entry at T+1 open price
        │
        ▼
Apply slippage (0.05%)
        │
        ▼
Deduct commission (0.04%)
        │
        ▼
Monitor position...
        │
        ▼
Exit on: TP hit, SL hit, Trailing Stop, Timeout, or End of Test
```

### Exit Conditions

| Exit Type | Trigger | Description |
|-----------|---------|-------------|
| TP_HIT | Price >= entry * (1 + take_profit_pct) | Target reached |
| SL_HIT | Price <= entry * (1 + stop_loss_pct) | Stop loss triggered |
| TRAILING_STOP | Price fell from peak by X% | Profit protection |
| TIMEOUT | Bars in trade >= max_bars_in_trade | Time limit |
| FORCED | End of test period | Close all positions |

### Performance Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| Win Rate | wins / total_trades | % of profitable trades |
| Profit Factor | gross_profit / gross_loss | Ratio of gains to losses |
| Expectancy | avg_profit / trade | Expected profit per trade |
| Max Drawdown | peak_to_trough | Worst losing streak |
| Sharpe Ratio | mean_return / std_return | Risk-adjusted return |
| Sortino Ratio | mean_return / downside_std | Downside risk-adjusted return |

### Sample Backtest Report

```
======================================================================
                         BACKTEST REPORT
======================================================================

  Pair: BTCUSDT
  Test Period: 2024-03-02 to 2025-12-14 (652 days)
  Starting Capital: $200.00

----------------------------------------------------------------------
  TRADE SUMMARY
----------------------------------------------------------------------
  Total Trades:      93
  Winning Trades:    92 (98.9%)
  Losing Trades:     1 (1.1%)

----------------------------------------------------------------------
  PERFORMANCE
----------------------------------------------------------------------
  Total P&L:         $+12.65 (+6.32%)
  Avg Win:           $0.14
  Avg Loss:          $-0.50
  Profit Factor:     25.94
  Expectancy:        $0.14 per trade

----------------------------------------------------------------------
  RISK METRICS
----------------------------------------------------------------------
  Max Drawdown:      $1.40 (0.70%)
  Sharpe Ratio:      2.33
  Sortino Ratio:     3.45

----------------------------------------------------------------------
  TRADE DURATION
----------------------------------------------------------------------
  Avg Duration:      45.2 bars (minutes)
  Avg Bars to Win:   44.1
  Avg Bars to Loss:  145.0

----------------------------------------------------------------------
  EXIT REASONS
----------------------------------------------------------------------
  TP_HIT         92 trades  +$13.15
  TIMEOUT         1 trades  -$0.50

----------------------------------------------------------------------
  BY REGIME
----------------------------------------------------------------------
  HIGH_VOL         55 trades   100% win  +$8.23
  RANGE_LOW_VOL    15 trades   100% win  +$2.12
  TREND_HIGH_VOL   23 trades   95.7% win +$2.30
======================================================================
```

---

## 12. Grid Search & Hyperparameter Tuning

### What is Grid Search?

Systematically testing **all combinations** of parameters to find optimal settings.

### What is Hyperparameter Tuning?

Fine-tuning parameters **around a known good value** to find the precise optimum.

### Difference

| Aspect | Grid Search | Hyperparameter Tuning |
|--------|-------------|----------------------|
| Purpose | Explore broadly | Refine precisely |
| Range | Wide (0.0 to 0.005) | Narrow (0.0008 to 0.0012) |
| Combinations | Many | Few |
| When | Initial exploration | After finding approximate best |

### Parameters to Optimize

| Parameter | Description | Impact |
|-----------|-------------|--------|
| min_expectancy | Minimum expected return | Filters low-quality signals |
| max_distance | Maximum neighbor distance | Filters dissimilar states |
| blocked_regimes | Regimes to skip | Avoids unprofitable conditions |
| sample_interval | Signal check frequency | Trade frequency |

### Why min_expectancy is Most Important

Without filtering (min_expectancy=0):
- Takes every signal
- Many marginal trades
- Costs eat profits
- **Result: Losing strategy**

With min_expectancy=0.001:
- Only high-conviction signals
- Filters 90%+ of trades
- Quality over quantity
- **Result: Profitable strategy**

**H=5m Example:**

| min_expectancy | Trades | Win Rate | P&L |
|----------------|--------|----------|-----|
| 0.000 | 2,057 | 21% | -$4,253 |
| 0.001 | 42 | 100% | +$373 |
| 0.002 | 18 | 100% | +$152 |

---

## 13. Running Experiments

### Directory Structure

```
tests/grid_search/
├── base.py                     # Shared logic
├── h5/                         # 5-minute horizon
│   ├── exp1_min_expectancy.py
│   ├── exp2_max_distance.py
│   ├── exp3_blocked_regimes.py
│   ├── exp4_combined.py
│   └── exp5_hyperparameter_tuning.py
├── h10/                        # 10-minute horizon
│   ├── exp1_min_expectancy.py
│   ├── exp2_max_distance.py
│   └── exp3_blocked_regimes.py
├── h15/                        # 15-minute horizon
└── h30/                        # 30-minute horizon
```

### Running Experiments

**Step 1: Activate virtual environment**
```bash
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

**Step 2: Run experiment**
```bash
python -m tests.grid_search.h5.exp1_min_expectancy
```

**Step 3: Run multiple in parallel** (open separate terminals)
```bash
# Terminal 1
python -m tests.grid_search.h5.exp1_min_expectancy

# Terminal 2
python -m tests.grid_search.h5.exp2_max_distance

# Terminal 3
python -m tests.grid_search.h5.exp3_blocked_regimes
```

### Experiment Types

| Experiment | Tests | Parameters |
|------------|-------|------------|
| exp1 | min_expectancy | 0.0, 0.001, 0.002, 0.003, 0.004, 0.005 |
| exp2 | max_distance | 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 |
| exp3 | blocked_regimes | Various combinations |
| exp4 | combined | Best from exp1 + exp2 + exp3 |
| exp5 | hyperparameter | Fine-tune around best value |

### Workflow

1. **Run exp1, exp2, exp3** for target horizon
2. **Analyze results** - identify best values
3. **Create exp4** - combine best parameters
4. **Run exp5** - fine-tune most impactful parameter
5. **Update BEST_PARAMS.yaml** - document optimal settings

### Results Location

```
data/grid_search/
├── h5/
│   ├── exp1_min_expectancy_BTCUSDT_20251228.csv
│   ├── exp2_max_distance_BTCUSDT_20251228.csv
│   ├── exp3_blocked_regimes_BTCUSDT_20251228.csv
│   ├── exp4_combined_BTCUSDT_20251228.csv
│   └── BEST_PARAMS.yaml
├── h10/
├── h15/
└── h30/
```

### Time Estimates

| Experiment | Combinations | Time per Combo | Total Time |
|------------|--------------|----------------|------------|
| exp1 | 6 | ~10 min | ~1 hour |
| exp2 | 6 | ~10 min | ~1 hour |
| exp3 | 7 | ~10 min | ~1.2 hours |
| exp4 | 6-12 | ~10 min | ~1-2 hours |
| exp5 | 10 | ~10 min | ~1.7 hours |

---

## 14. Using BEST_PARAMS.yaml

### Purpose

Stores the **optimal configuration** discovered through experiments.

### File Structure

```yaml
# =============================================================================
# H=5m BEST PARAMETERS (Updated: 2025-12-29)
# =============================================================================
horizon: 5

# Best Configuration (from experiments)
best_params:
  min_expectancy: 0.001        # From exp1/exp5
  max_distance: 3.0            # From exp2
  blocked_regimes: []          # From exp3/exp4

# Backtest Settings (must match grid search!)
backtest:
  capital: 200
  max_bars_in_trade: 0         # 0 = no timeout
  trailing_stop_pct: 0.0       # No trailing stop
  sample_interval: 60          # Check every 60 bars

# Performance Metrics (from experiments)
performance:
  total_pnl_pct: 6.32
  win_rate: 98.9
  total_trades: 93
  profit_factor: 890.27
  max_drawdown_pct: 0.7
  sharpe_ratio: 2.33
```

### Using in Backtest

```bash
python scripts/run_backtest.py --params-file data/grid_search/h5/BEST_PARAMS.yaml
```

This loads:
- min_expectancy, max_distance, blocked_regimes
- horizon
- backtest settings (capital, sample_interval, etc.)

### Important: sample_interval Must Match

Grid search results are only valid for the **same sample_interval**.

| If Grid Search Used | Backtest Must Use |
|---------------------|-------------------|
| sample_interval: 60 | sample_interval: 60 |
| sample_interval: 15 | sample_interval: 15 |

Using different sample_interval = **different results!**

### Scaling Capital

Results scale linearly with capital:

| Capital | Expected P&L |
|---------|--------------|
| $200 | +$12.65 (+6.32%) |
| $1,000 | +$63.25 (+6.32%) |
| $10,000 | +$632.50 (+6.32%) |

---

## 15. Project Structure

```
trade_system_1/
│
├── config/                      # Configuration
│   ├── __init__.py              # Config loader
│   └── config.yaml              # Central settings
│
├── data/                        # DATA FILES ONLY
│   ├── ohlcv/                   # Raw OHLCV data
│   │   └── BTCUSDT_1m_ohlcv.parquet
│   ├── state_vectors/           # Generated states
│   │   └── BTCUSDT_1m_state.parquet
│   ├── regimes/                 # Regime labels
│   │   └── BTCUSDT_1m_regimes.parquet
│   ├── outcomes/                # MFE/MAE labels
│   │   └── BTCUSDT_1m_outcomes.parquet
│   ├── backtest/                # Trade logs
│   │   └── BTCUSDT_h5_trades_*.parquet
│   ├── grid_search/             # Experiment results
│   │   ├── h5/
│   │   │   ├── BEST_PARAMS.yaml
│   │   │   └── *.csv
│   │   ├── h10/
│   │   ├── h15/
│   │   └── h30/
│   ├── raw/                     # (Legacy code location)
│   │   └── ohlcv_loader.py
│   └── validators/              # (Legacy code location)
│       └── data_integrity.py
│
├── features/                    # Feature engineering
│   ├── __init__.py
│   ├── trend.py                 # EMA, trend alignment
│   ├── momentum.py              # RSI, returns
│   ├── volatility.py            # ATR
│   ├── volume.py                # Volume analysis
│   └── location.py              # VWAP, range position
│
├── state/                       # State vector engine
│   ├── __init__.py
│   ├── state_schema.py          # MarketState dataclass
│   ├── normalizer.py            # Z-score normalization
│   ├── state_builder.py         # Build state vectors
│   ├── state_store.py           # Parquet I/O
│   └── state_validator.py       # Validation
│
├── regime/                      # Regime detection
│   ├── regime_labeler.py
│   └── run_regime_labeling.py
│
├── outcomes/                    # Outcome labeling
│   ├── __init__.py
│   ├── outcome_labeler.py
│   └── run_outcome_labeling.py
│
├── similarity/                  # Similarity engine
│   ├── similarity_engine.py     # KNN search
│   └── run_similarity_test.py
│
├── decision/                    # Decision engine
│   ├── __init__.py
│   ├── decision_engine.py
│   └── run_decision_test.py
│
├── pipeline/                    # Pipeline orchestration
│   ├── __init__.py
│   └── orchestrator.py
│
├── backtest/                    # Backtesting framework
│   ├── __init__.py
│   ├── backtester.py            # Main backtester
│   ├── trade_simulator.py       # Trade execution
│   └── metrics.py               # Performance metrics
│
├── tests/                       # Test files
│   ├── __init__.py
│   └── grid_search/             # Grid search experiments
│       ├── __init__.py
│       ├── base.py              # Shared logic
│       ├── h5/                  # 5-min experiments
│       ├── h10/                 # 10-min experiments
│       ├── h15/                 # 15-min experiments
│       └── h30/                 # 30-min experiments
│
├── visualizations/              # Chart generation
│   ├── __init__.py
│   ├── plot_regimes.py
│   ├── plot_outcomes.py
│   └── plot_states.py
│
├── run_pipeline.py              # Main pipeline CLI
├── run_backtest.py              # Backtesting CLI
├── run_grid_search.py           # Grid search CLI
├── run_visualizations.py        # Visualization CLI
│
├── exceptions.py                # Custom exceptions
├── requirements.txt             # Dependencies
├── .env                         # Database URL (gitignored)
├── DOCUMENTATION.md             # This file
├── EXPERIMENTS.txt              # Experiment tracking
└── RESTRUCTURE_PLAN.md          # Future refactoring plan
```

---

## 16. Troubleshooting

### Issue: Database Connection Error

**Error:**
```
ERROR: Unable to connect to database
```

**Solutions:**
1. Check PostgreSQL is running
2. Verify DATABASE_URL in `.env`
3. Use local parquet files instead:
   - Place data in `data/ohlcv/BTCUSDT_1m_ohlcv.parquet`

### Issue: No Outcome/Regime Files

**Error:**
```
ERROR: No outcome files found for BTCUSDT
```

**Solution:**
```bash
python scripts/run_pipeline.py
```

### Issue: FAISS Not Found

**Error:**
```
ImportError: faiss not found
```

**Solution:**
```bash
pip install faiss-cpu
```

### Issue: Results Don't Match Grid Search

**Causes:**
- Different `sample_interval`
- Different `max_bars_in_trade`
- Different `min_expectancy`

**Solution:**
Check BEST_PARAMS.yaml backtest section matches grid search settings.

### Issue: Slow Backtesting

**Cause:** Bruteforce search on large dataset

**Solutions:**
1. Increase `sample_interval` (60 instead of 5)
2. Reduce training data size
3. Use FAISS for realtime (not backtest)

### Issue: 0% Win Rate or Losing Money

**Causes:**
- Trading costs exceed expected returns
- Wrong parameters for market conditions
- min_expectancy too low

**Solution:**
Run grid search to find optimal parameters.

### Issue: Memory Error

**Cause:** Dataset too large for RAM

**Solutions:**
1. Reduce date range
2. Use chunked processing
3. Add more RAM

---

## 17. Key Concepts Explained

### Why min_expectancy Matters

The most important parameter. Without it:
- System takes every signal
- Many low-quality trades
- Costs overwhelm small gains
- **Result: Loss**

With proper min_expectancy:
- Filters out 90%+ of signals
- Only high-conviction trades
- Quality > quantity
- **Result: Profit**

### Why Bruteforce for Backtesting

FAISS is fast but doesn't support `max_timestamp` filtering.

During backtesting, we must prevent **look-ahead bias**:
- Only search states BEFORE current time
- FAISS can't do this efficiently
- Bruteforce handles it correctly

### Why sample_interval Affects Results

Higher sample_interval = fewer signal checks = fewer trades

| sample_interval | Checks/Day | Character |
|-----------------|------------|-----------|
| 5 | 288 | Very active |
| 60 | 24 | Hourly |
| 1440 | 1 | Daily |

**Critical:** Use same sample_interval for grid search and backtest!

### Why Regime Blocking Can Help or Hurt

Each regime has different characteristics:

| Regime | H=5m | H=30m |
|--------|------|-------|
| HIGH_VOL | Often good | May hurt |
| TREND_LOW_VOL | Often bad | May be good |

**Always test empirically!**

### Position Sizing Formula

```python
risk_amount = capital * risk_per_trade
stop_loss_pct = percentile(MAE, 5)
position_size = risk_amount / abs(stop_loss_pct)
position_size = min(position_size, capital * max_leverage)
```

---

## 18. Glossary

| Term | Definition |
|------|------------|
| **ATR** | Average True Range - volatility indicator |
| **EMA** | Exponential Moving Average |
| **Expectancy** | Expected profit/loss per trade |
| **FAISS** | Facebook AI Similarity Search - fast KNN library |
| **Grid Search** | Testing all parameter combinations |
| **Horizon** | Forward-looking time window (5m, 10m, etc.) |
| **Hyperparameter** | Configuration setting (not learned from data) |
| **KNN** | K-Nearest Neighbors algorithm |
| **Look-ahead Bias** | Using future information (invalid in backtest) |
| **MAE** | Maximum Adverse Excursion - worst case loss |
| **MFE** | Maximum Favorable Excursion - best case gain |
| **Regime** | Market state classification |
| **RSI** | Relative Strength Index - momentum indicator |
| **State Vector** | 10D representation of market conditions |
| **VWAP** | Volume Weighted Average Price |
| **Walk-Forward** | Testing by moving through time sequentially |
| **Z-Score** | Standard deviations from mean |

---

## Changelog

### 2025-12-31
- Created comprehensive DOCUMENTATION.md

### 2025-12-29
- Added grid search framework
- Created BEST_PARAMS.yaml support
- Added --params-file to run_backtest.py

### 2025-12-28
- Fixed MFE/MAE calculation bug
- Fixed regime blocking bug
- Added hyperparameter tuning experiments

### 2025-12-27
- Initial release
