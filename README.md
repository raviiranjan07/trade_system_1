# Trading System - State-Driven Quantitative Trading

A **state-based quantitative trading system** that trades only when historical market conditions show statistical edge. The system avoids prediction and overtrading by relying on **market memory, regimes, and expectancy**.

---

## Core Philosophy

> **We do not predict price. We recognize market states and act only when history supports an asymmetric edge.**

- Markets are probabilistic, not deterministic
- Capital preservation comes first
- Fewer high-quality trades > frequent trades
- Decisions are statistics-driven, not indicator-driven

---

## Architecture

```
PostgreSQL / TimescaleDB (1m OHLCV)
         |
         v
   Market State Vector Engine (10D normalized features)
         |
         v
     Regime Detection (4 market regimes)
         |
         v
   Outcome Labeling (MFE / MAE for multiple horizons)
         |
         v
   Similarity Search (KNN Market Memory)
         |
         v
   Decision Engine (Expectancy-based signals)
         |
         v
   Risk & Exit Management (Stop Loss, Take Profit, Trailing Stop)
```

---

## Quick Start

### 1. Prerequisites

- Python 3.9+
- PostgreSQL with OHLCV data (or use parquet files)
- Required packages: `pip install -r requirements.txt`

### 2. Configuration

Create a `.env` file in the project root:

```bash
DATABASE_URL=postgresql://user@host:5432/crypto_data
```

All settings are in `config/config.yaml`. Key settings:

```yaml
data:
  pair: "BTCUSDT"
  timeframe: "1m"
  start_date: "2020-01-01"
  end_date: "2025-12-15"

similarity:
  k: 200                      # Number of similar states to find
  backend: "faiss"       # or "faiss" for speed

decision:
  capital: 10000
  risk_per_trade: 0.005       # 0.5% risk per trade
  min_expectancy: -0.002      # Minimum expectancy to trade
  blocked_regimes: ["TREND_LOW_VOL"]  # Skip unprofitable regimes
```

### 3. Run the Pipeline

```bash
# Run all stages
python run_pipeline.py

# Run specific stages
python run_pipeline.py --stages state_vectors regime_labeling

# Override pair and dates
python run_pipeline.py --pair ETHUSDT --start 2023-06-01 --end 2023-09-01

# Dry run (show plan without executing)
python run_pipeline.py --dry-run
```

### 4. Run Backtest

```bash
# Default 70/30 train/test split
python run_backtest.py

# Custom split
python run_backtest.py --train-ratio 0.80

# Save trade log
python run_backtest.py --save-trades
```

### 5. Run Grid Search (Parameter Optimization)

```bash
python run_grid_search.py
```

---

## Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| **state_vectors** | Fetch OHLCV, compute features, normalize, build 10D state vectors | `data/state_vectors/*.parquet` |
| **regime_labeling** | Classify market regimes using trend strength + volatility | `data/regimes/*.parquet` |
| **outcome_labeling** | Compute MFE/MAE outcomes for 10m, 15m, 30m, 120m horizons | `data/outcomes/*.parquet` |
| **similarity** | Find K similar historical states using KNN | In-memory result |
| **decision** | Generate trading decision based on expectancy | Trading signal |

---

## Market State Vector

A **10-dimensional normalized representation** of market conditions at each timestamp:

| Dimension | Description | Normalization |
|-----------|-------------|---------------|
| `ema50_slope_z` | 5-bar momentum of EMA(50) | Z-score |
| `ema200_slope_z` | 20-bar momentum of EMA(200) | Z-score |
| `trend_alignment` | Sign(EMA50 - EMA200) | {-1, 0, +1} |
| `return_5m_z` | 5-minute percent return | Z-score |
| `return_15m_z` | 15-minute percent return | Z-score |
| `rsi_z` | RSI(14) normalized | Z-score |
| `atr_percentile` | ATR(14) volatility level | Percentile (0-1) |
| `volume_z` | Volume relative to history | Z-score |
| `vwap_distance_z` | Distance from VWAP | Z-score |
| `range_position` | Position in 50-bar range | Ratio (0-1) |

Normalization uses a **rolling window of 2000 bars** to prevent look-ahead bias.

---

## Market Regimes

The system classifies markets into 4 regimes:

| Regime | Criteria | Description |
|--------|----------|-------------|
| `HIGH_VOL` | ATR percentile >= 0.85 | Volatility shock, no clear direction |
| `TREND_HIGH_VOL` | Trend strength >= 0.7, ATR > 0.35 | Strong directional move with high volatility |
| `TREND_LOW_VOL` | Trend strength >= 0.7, ATR <= 0.35 | Gradual trend with low volatility |
| `RANGE_LOW_VOL` | Everything else | Consolidation, choppy markets |

Regimes are **smoothed using 30-bar rolling majority vote** to prevent whipsaws.

---

## Outcome Labels (MFE/MAE)

For each historical state, the system computes forward-looking outcomes:

- **MFE (Maximum Favorable Excursion)**: Best possible gain within horizon
- **MAE (Maximum Adverse Excursion)**: Worst possible drawdown within horizon

Horizons: **10, 15, 30, 120 minutes** (configurable)

**Expectancy** = mean(MFE) + mean(MAE) *(MAE is negative)*

---

## Similarity Engine

Finds K similar historical states using KNN with two backends:

| Backend | Accuracy | Speed | Use Case |
|---------|----------|-------|----------|
| `bruteforce` | 100% exact | ~50ms/query on 700K samples | Accuracy verification, smaller datasets |
| `faiss` | ~95-99% approx | ~0.5ms/query | Production speed, large datasets |

**FAISS requires**: `pip install faiss-cpu` (or `faiss-gpu`)

Query process:
1. Filter by current regime (prevents cross-regime matching)
2. Enforce time boundary (backtesting: only use past data)
3. Calculate Euclidean distance in 10D state space
4. Return K=200 nearest neighbors
5. Aggregate outcomes to compute expectancy

---

## Decision Engine

Generates trading signals based on expectancy analysis:

### Trade Filters (no trade if any trigger)

- Expectancy < `min_expectancy` (default: -0.002)
- Current regime in `blocked_regimes` (default: TREND_LOW_VOL)
- Average distance to neighbors > `max_distance` (default: 3.0)
- Insufficient historical data

### Direction Selection

- **LONG** if mean_mfe > |mean_mae|
- **SHORT** otherwise

### Risk Sizing

- Stop loss: 5th percentile of historical MAE outcomes
- Take profit: Mean MFE outcome
- Position size: `(capital * risk_per_trade) / stop_loss_pct`
- Capped at `max_leverage` (default: 1.0x)

---

## Backtesting Framework

Implements **walk-forward backtesting** with proper train/test split:

```
|------------ TRAINING (70%) ------------|---- TEST (30%) ----|
start_date                          split_point           end_date

Training: Build similarity database from historical states
Test: Walk forward bar-by-bar, making decisions using ONLY past data
```

### Key Features

- **No Look-Ahead Bias**: Similarity engine only searches states before current time
- **Realistic Execution**: Entry at next bar's open with slippage (0.05%)
- **Commission Modeling**: Configurable taker fee (0.04%)
- **Trade Management**: Stop loss, take profit, trailing stop, timeout exits
- **Performance Sampling**: Check for signals every N bars (configurable) for speed

### Exit Reasons

| Exit | Description |
|------|-------------|
| `TAKE_PROFIT` | Price reached target profit level |
| `STOP_LOSS` | Price hit stop loss level |
| `TRAILING_STOP` | Price fell from peak by trailing % |
| `TIMEOUT` | Max bars in trade reached (default: 120 bars / 2 hours) |

### Backtest Configuration

```yaml
backtest:
  train_ratio: 0.70
  slippage_pct: 0.0005        # 0.05%
  commission_pct: 0.0004      # 0.04%
  max_bars_in_trade: 120      # Force exit after 2 hours
  trailing_stop_pct: 0.0      # Disabled by default
  sample_interval: 60         # Check every 60 bars (hourly)
```

### Sample Output

```
================================================================================
                         BACKTEST REPORT
================================================================================

  Pair: BTCUSDT
  Test Period: 2023-02-01 to 2023-03-01 (28 days)
  Starting Capital: $10,000.00

--------------------------------------------------------------------------------
  TRADE SUMMARY
--------------------------------------------------------------------------------
  Total Trades:      47
  Winning Trades:    28 (59.6%)
  Losing Trades:     19 (40.4%)

--------------------------------------------------------------------------------
  PERFORMANCE
--------------------------------------------------------------------------------
  Total P&L:         +$847.32 (+8.47%)
  Avg Win:           $62.15
  Avg Loss:          $-41.23
  Profit Factor:     1.89
  Expectancy:        $18.03 per trade

--------------------------------------------------------------------------------
  RISK METRICS
--------------------------------------------------------------------------------
  Max Drawdown:      $312.45 (3.12%)
  Sharpe Ratio:      1.24
  Sortino Ratio:     1.67
================================================================================
```

---

## Grid Search (Parameter Optimization)

`run_grid_search.py` performs exhaustive search over parameter combinations:

- **k values**: 50, 100, 150, 200, 250, 300
- **min_expectancy**: -0.01 to +0.01
- **max_distance**: 0.5 to 3.0
- **blocked_regimes**: Various combinations

Reports: Total trades, P&L, win rate, profit factor, Sharpe ratio for each combo.

---

## Project Structure

```
trade_system_1/
|
├── config/
│   ├── __init__.py           # Config loader with validation
│   └── config.yaml           # Central configuration
|
├── data/
│   ├── raw/
│   │   └── ohlcv_loader.py   # Database fetch layer
│   ├── validators/
│   │   └── data_integrity.py # OHLCV validation
│   ├── state_vectors/        # Generated state vectors (parquet)
│   ├── regimes/              # Generated regime labels (parquet)
│   └── outcomes/             # Generated outcomes (parquet)
|
├── features/                  # Feature computation
│   ├── trend.py              # EMA slopes, trend alignment
│   ├── momentum.py           # RSI, returns
│   ├── volatility.py         # ATR
│   ├── volume.py             # Volume analysis
│   └── location.py           # VWAP distance, range position
|
├── state/                     # Market State Vector Engine
│   ├── state_schema.py       # MarketState dataclass
│   ├── normalizer.py         # Rolling z-score normalization
│   ├── state_builder.py      # State vector construction
│   ├── state_store.py        # Parquet persistence
│   └── state_validator.py    # State validation
|
├── regime/
│   └── regime_labeler.py     # Market regime classification
|
├── outcomes/
│   └── outcome_labeler.py    # MFE/MAE computation
|
├── similarity/
│   └── similarity_engine.py  # KNN similarity search (bruteforce/FAISS)
|
├── decision/
│   └── decision_engine.py    # Expectancy-based decisions
|
├── pipeline/
│   └── orchestrator.py       # Unified pipeline runner
|
├── backtest/
│   ├── backtester.py         # Walk-forward backtester
│   ├── trade_simulator.py    # Trade execution simulation
│   └── metrics.py            # Performance metrics
|
├── visualizations/
│   ├── plot_regimes.py       # Regime visualizations
│   ├── plot_outcomes.py      # MFE/MAE visualizations
│   └── plot_states.py        # State vector visualizations
|
├── run_pipeline.py           # Main pipeline CLI
├── run_backtest.py           # Backtesting CLI
├── run_grid_search.py        # Parameter optimization
├── run_visualizations.py     # Visualization generator
├── debug_outcomes.py         # Data analysis script
├── requirements.txt
└── .env                      # Database URL (gitignored)
```

---

## Visualizations

```bash
# Generate all visualizations
python run_visualizations.py

# Generate specific chart types
python run_visualizations.py --type regimes
python run_visualizations.py --type outcomes
python run_visualizations.py --type states
```

### Available Charts

| Category | Chart | Description |
|----------|-------|-------------|
| **Regimes** | Regime Distribution | Pie/bar chart of regime proportions |
| | Regime Transitions | Transition probability matrix |
| **Outcomes** | MFE/MAE Distribution | Histograms for each horizon |
| | Expectancy by Regime | Which regimes have positive edge |
| | Horizon Comparison | Compare 10m, 15m, 30m, 120m outcomes |
| **States** | State Heatmap | State vectors over time |
| | State Correlation | Correlation between dimensions |
| | PCA Projection | 2D visualization of state space |

Charts are saved to `output/charts/` by default.

---

## Database Schema

```sql
CREATE TABLE ohlcv_data (
    time TIMESTAMPTZ NOT NULL,
    pair TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    num_trades INTEGER
);

-- Recommended: Use TimescaleDB hypertable for performance
SELECT create_hypertable('ohlcv_data', 'time');
CREATE INDEX ON ohlcv_data (pair, time DESC);
```

---

## Configuration Reference

### Data Settings

| Key | Default | Description |
|-----|---------|-------------|
| `data.pair` | BTCUSDT | Trading pair |
| `data.timeframe` | 1m | Candle timeframe |
| `data.start_date` | 2020-01-01 | Data start date |
| `data.end_date` | 2025-12-15 | Data end date |

### Feature Settings

| Key | Default | Description |
|-----|---------|-------------|
| `features.ema_fast_period` | 50 | Fast EMA period |
| `features.ema_slow_period` | 200 | Slow EMA period |
| `features.rsi_period` | 14 | RSI period |
| `features.atr_period` | 14 | ATR period |
| `features.range_lookback` | 50 | High/low range lookback |

### Normalization

| Key | Default | Description |
|-----|---------|-------------|
| `normalization.window` | 2000 | Rolling window for z-scores |

### Regime Detection

| Key | Default | Description |
|-----|---------|-------------|
| `regime.high_vol_threshold` | 0.85 | ATR percentile for HIGH_VOL |
| `regime.low_vol_threshold` | 0.35 | ATR percentile for LOW_VOL |
| `regime.trend_strength_threshold` | 0.7 | Min trend strength for TREND regimes |
| `regime.smoothing_window` | 30 | Majority vote smoothing |

### Similarity Engine

| Key | Default | Description |
|-----|---------|-------------|
| `similarity.k` | 200 | Number of similar states |
| `similarity.default_horizon` | 30 | Default outcome horizon (minutes) |
| `similarity.max_distance` | 1.5 | Max distance to neighbors |
| `similarity.backend` | bruteforce | bruteforce or faiss |

### Decision Engine

| Key | Default | Description |
|-----|---------|-------------|
| `decision.capital` | 10000 | Trading capital (USD) |
| `decision.risk_per_trade` | 0.005 | Risk per trade (0.5%) |
| `decision.min_expectancy` | -0.002 | Minimum expectancy to trade |
| `decision.max_distance` | 3.0 | Max avg distance filter |
| `decision.blocked_regimes` | [TREND_LOW_VOL] | Regimes to skip |

### Backtest Settings

| Key | Default | Description |
|-----|---------|-------------|
| `backtest.train_ratio` | 0.70 | Train/test split ratio |
| `backtest.slippage_pct` | 0.0005 | Slippage per trade (0.05%) |
| `backtest.commission_pct` | 0.0004 | Commission per trade (0.04%) |
| `backtest.max_bars_in_trade` | 120 | Force exit after N bars |
| `backtest.trailing_stop_pct` | 0.0 | Trailing stop % (0=disabled) |
| `backtest.sample_interval` | 60 | Check signals every N bars |

---

## Error Handling

The system provides user-friendly error messages:

- **DatabaseConnectionError**: Connection issues with troubleshooting steps
- **ConfigurationError**: Invalid config with specific field errors
- **DataValidationError**: Data quality issues
- **MissingDataError**: Required data not found

---

## Key Design Decisions

### Look-Ahead Bias Prevention
The similarity engine enforces `max_timestamp` during backtesting - only searches historical states BEFORE current time, preventing future information leakage.

### Regime Isolation
Similarity searches within the current regime only - prevents cross-regime pattern matching that might break down during regime transitions.

### Blocking Unprofitable Regimes
Default blocks `TREND_LOW_VOL` (empirically unprofitable) - this filter can be adjusted via configuration.

### Performance Optimization
- `sample_interval: 60` checks for signals every 60 bars (hourly) instead of every bar
- FAISS backend reduces query time from ~50ms to ~0.5ms on 700K+ samples
- Per-regime indexing reduces search space significantly

---

## License

MIT
