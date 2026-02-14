# AI Coding Instructions for Trade System

## System Overview

This is a **state-driven quantitative trading system** that trades cryptocurrency (BTC/ETH) at 1-minute timeframes using KNN similarity search to find historical market patterns with statistical edge.

**Core Philosophy:** We do not predict price direction. We recognize market states, find historical analogues via similarity search, and act only when historical outcomes show expectancy > 0.

### Architecture Pipeline

```
OHLCV Data (PostgreSQL/Parquet)
    ↓
State Vector Engine (normalized 10D features)
    ↓
Regime Detection (4 market regimes)
    ↓
Outcome Labeling (MFE/MAE calculations)
    ↓
Similarity Search (KNN find K=200 similar states)
    ↓
Decision Engine (expectancy-based signals)
    ↓
Risk Management (stops, profit targets)
```

## Critical Rules

### Rule #1: Minimum Profitable Move = 12bp
- Fees: 8bp round-trip
- Minimum profitable target: 12bp (4bp net gain)
- **Anything < 12bp is noise**

### Rule #2: Economics First, Code Later
- Validate with data analysis before coding
- Reference: `docs/analysis_findings.md` (3.15M candles analyzed)
- Key finding: Direction is 50/50 even at H=600; edge comes only from **selective entry**

## Essential Workflows

```bash
# Full pipeline: state vectors → regimes → outcomes → similarity
python scripts/run_pipeline.py

# Backtest (70/30 train/test split)
python scripts/run_backtest.py

# Specific stages only
python scripts/run_pipeline.py --stages state_vectors regime_labeling

# Override pair/dates
python scripts/run_pipeline.py --pair ETHUSDT --start 2023-06-01 --end 2023-09-01

# Paper trading (live)
python scripts/run_paper_trade.py
```

## Code Structure

- **src/trade_system/state/** - Market state vectors (10D normalized features)
- **src/trade_system/regime/** - Regime classification (4 regimes)
- **src/trade_system/outcomes/** - MFE/MAE outcome labeling
- **src/trade_system/similarity/** - KNN similarity search (FAISS or brute-force)
- **src/trade_system/decision/** - Expectancy-based signals
- **src/trade_system/backtest/** - Walk-forward validation
- **src/trade_system/pipeline/orchestrator.py** - Stage orchestration

## Key Patterns

1. **MFE/MAE Framework** - Measure max favorable/adverse excursion at multiple horizons (H=[3,5,10,15,30,60,...])
2. **4 Regimes** - TREND_HIGH_VOL, TREND_LOW_VOL, MEAN_REVERT_UP, MEAN_REVERT_DOWN
3. **Parquet Storage** - All outputs stored as parquet files with timestamps in `data/` directory
4. **Walk-forward validation only** - Never train on future data
5. **Normalized state vectors** - All features normalized to [-1, 1] range per 100-bar window

## Key Files
- [README.md](../README.md) - Quick start
- [CLAUDE.md](../CLAUDE.md) - Development rules
- [analysis_findings.md](../docs/analysis_findings.md) - Analysis of 3.15M candles (foundational)
- [config/config.yaml](../config/config.yaml) - Configuration (pair, horizons, K, risk)

## Important Constraints
- All targets must be ≥ 12bp
- Direction is 50/50 - edge comes only from selective entry
- State vector alone has no predictive power
- Only trade complete OHLCV bars (never forming candles)
