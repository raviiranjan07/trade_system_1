# Trading Agent - State-Driven Quant Trading System

A **state-based quantitative trading system** designed to trade **only when historical market conditions show statistical edge**.
This system avoids prediction, emotion, and overtrading by relying on **market memory, regimes, and expectancy**.

---

## Core Philosophy

> **We do not predict price.
> We recognize market states and act only when history supports an asymmetric edge.**

Key principles:
- Markets are probabilistic, not deterministic
- Capital preservation comes first
- Fewer high-quality trades > frequent trades
- Structure protects survival
- Decisions are statistics-driven, not indicator-driven

---

## Architecture

```
PostgreSQL / TimescaleDB (1m OHLCV)
         |
         v
   Market State Vector Engine
         |
         v
     Regime Detection
         |
         v
   Outcome Labeling (MFE / MAE)
         |
         v
   Similarity Search (Market Memory)
         |
         v
   Decision Engine (Expected Value)
         |
         v
   Risk & Exit Management
```

---

## Quick Start

### 1. Prerequisites

- Python 3.9+
- PostgreSQL with OHLCV data
- Required packages: `pip install -r requirements.txt`

### 2. Configuration

Create a `.env` file in the project root:

```bash
# Database connection
DATABASE_URL=postgresql://user@host:5432/crypto_data
```

Configuration is managed via `config/config.yaml`:

```yaml
data:
  pair: "BTCUSDT"
  timeframe: "1m"
  start_date: "2023-01-01"
  end_date: "2023-06-01"

similarity:
  k: 200                    # Number of similar states to find

decision:
  capital: 10000            # Trading capital
  risk_per_trade: 0.005     # 0.5% risk per trade
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

# Verbose output
python run_pipeline.py -v
```

---

## Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| **state_vectors** | Fetch OHLCV, compute features, normalize, build state vectors | `data/state_vectors/*.parquet` |
| **regime_labeling** | Classify market regimes (TREND_HIGH_VOL, RANGE_LOW_VOL, etc.) | `data/regimes/*.parquet` |
| **outcome_labeling** | Compute MFE/MAE outcomes for 10m, 30m, 120m horizons | `data/outcomes/*.parquet` |
| **similarity** | Find K similar historical states using KNN | In-memory result |
| **decision** | Generate trading decision based on expectancy | Trading signal |

---

## Project Structure

```
trading_agent/
|
├── config/
│   ├── __init__.py          # Config loader with validation
│   └── config.yaml          # Central configuration
|
├── data/
│   ├── raw/
│   │   └── ohlcv_loader.py  # Database fetch layer
│   └── validators/
│       └── data_integrity.py # OHLCV validation
|
├── features/                 # Feature computation
│   ├── trend.py             # EMA slopes, trend alignment
│   ├── momentum.py          # RSI, returns
│   ├── volatility.py        # ATR
│   ├── volume.py            # Volume analysis
│   └── location.py          # VWAP distance, range position
|
├── state/                    # Market State Vector Engine
│   ├── state_schema.py      # MarketState dataclass
│   ├── normalizer.py        # Rolling z-score normalization
│   ├── state_builder.py     # State vector construction
│   ├── state_store.py       # Parquet persistence
│   └── state_validator.py   # State validation
|
├── regime/                   # Regime detection
│   └── regime_labeler.py    # Market regime classification
|
├── outcomes/                 # Outcome labeling
│   └── outcome_labeler.py   # MFE/MAE computation
|
├── similarity/               # Market memory
│   └── similarity_engine.py # KNN similarity search
|
├── decision/                 # Trading decisions
│   └── decision_engine.py   # Expectancy-based decisions
|
├── pipeline/                 # Orchestration
│   └── orchestrator.py      # Unified pipeline runner
|
├── visualizations/           # Charts and plots
│   ├── plot_regimes.py      # Regime visualizations
│   ├── plot_outcomes.py     # MFE/MAE visualizations
│   └── plot_states.py       # State vector visualizations
|
├── exceptions.py             # Custom exception classes
├── run_pipeline.py           # CLI entry point
├── run_visualizations.py     # Visualization generator
├── debug_outcomes.py         # Data analysis script
├── requirements.txt
└── .env                      # Database URL (gitignored)
```

---

## Market State Vector

A **Market State Vector** is a compact, normalized representation of the market at a specific moment.

### 10 Dimensions

| Dimension | Description |
|-----------|-------------|
| `ema50_slope_z` | Short-term trend direction (z-score) |
| `ema200_slope_z` | Long-term trend direction (z-score) |
| `trend_alignment` | Trend agreement (-1, 0, +1) |
| `return_5m_z` | 5-minute momentum (z-score) |
| `return_15m_z` | 15-minute momentum (z-score) |
| `rsi_z` | RSI normalized (z-score) |
| `atr_percentile` | Volatility percentile (0-1) |
| `volume_z` | Volume relative to history (z-score) |
| `vwap_distance_z` | Distance from VWAP (z-score) |
| `range_position` | Position in recent range (0-1) |

Each minute produces **one state vector**. Over years, this becomes **market memory**.

---

## Market Regimes

The system classifies market into 4 regimes:

| Regime | Description |
|--------|-------------|
| `TREND_HIGH_VOL` | Strong directional move with high volatility |
| `TREND_LOW_VOL` | Gradual trend with low volatility |
| `RANGE_LOW_VOL` | Consolidation, low volatility |
| `HIGH_VOL` | High volatility without clear direction |

---

## Outcome Labels (MFE/MAE)

For each state, the system computes forward-looking outcomes:

- **MFE (Maximum Favorable Excursion)**: Best possible gain within horizon
- **MAE (Maximum Adverse Excursion)**: Worst possible drawdown within horizon

Horizons: 10 minutes, 30 minutes, 120 minutes

These outcomes are used to calculate **expectancy** when similar historical states are found.

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

## Debug Tools

```bash
# Analyze outcomes and regimes
python debug_outcomes.py
```

---

## Visualizations

Generate charts to analyze the trading system:

```bash
# Generate all visualizations
python run_visualizations.py

# Generate specific chart types
python run_visualizations.py --type regimes    # Regime analysis
python run_visualizations.py --type outcomes   # MFE/MAE analysis
python run_visualizations.py --type states     # State vector analysis

# For a specific pair
python run_visualizations.py --pair ETHUSDT
```

### Available Charts

| Category | Chart | Description |
|----------|-------|-------------|
| **Regimes** | Regime Distribution | Pie/bar chart of regime proportions |
| | Regime Transitions | Transition probability matrix |
| **Outcomes** | MFE/MAE Distribution | Histograms for each horizon |
| | Expectancy by Regime | Which regimes have positive edge |
| | Horizon Comparison | Compare 10m, 30m, 120m outcomes |
| | Outcome Over Time | Rolling expectancy chart |
| **States** | State Heatmap | State vectors over time |
| | State Correlation | Correlation between dimensions |
| | State Distributions | Histogram per dimension |
| | State by Regime | Average state per regime |
| | PCA Projection | 2D visualization of state space |

Charts are saved to `output/charts/` by default.

---

## Error Handling

The system provides user-friendly error messages:

- **DatabaseConnectionError**: Connection issues with troubleshooting steps
- **ConfigurationError**: Invalid config with specific field errors
- **DataValidationError**: Data quality issues
- **MissingDataError**: Required data not found

---

## Configuration Reference

See `config/config.yaml` for all available options:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| data | pair | BTCUSDT | Trading pair |
| data | timeframe | 1m | Candle timeframe |
| normalization | window | 2000 | Rolling window for z-scores |
| similarity | k | 200 | Number of similar states |
| decision | capital | 10000 | Trading capital |
| decision | risk_per_trade | 0.005 | Risk per trade (0.5%) |
| regime | high_vol_threshold | 0.7 | ATR percentile for high vol |
| regime | low_vol_threshold | 0.3 | ATR percentile for low vol |

---

## License

MIT
