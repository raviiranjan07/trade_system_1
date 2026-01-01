# Paper Trading System

Real-time paper trading system for the KNN-based trading strategy.

## Overview

This module connects to Binance for live market data and executes simulated trades based on the trained similarity engine. No real money is at risk - all trades are paper trades.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PAPER TRADING SYSTEM                           │
└─────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────┐
                        │ Binance WebSocket│
                        │  (Real-time 1m)  │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Candle Buffer   │
                        │  (2500 bars)    │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  State Builder  │───────────────┐
                        │ (Features + Z)  │               │
                        └────────┬────────┘               │
                                 │                        │
                                 ▼                        ▼
                        ┌─────────────────┐      ┌──────────────┐
                        │ Regime Labeler  │      │  Current     │
                        │                 │      │  State (10D) │
                        └────────┬────────┘      └──────┬───────┘
                                 │                      │
                                 ▼                      ▼
                        ┌─────────────────────────────────────┐
                        │         Similarity Engine           │
                        │  (Query 200 similar historical      │
                        │   states from 3M+ training data)    │
                        └──────────────────┬──────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────┐
                        │          Decision Engine            │
                        │  - Check expectancy > 0.001         │
                        │  - Check distance < 3.0             │
                        │  - Check regime not blocked         │
                        └──────────────────┬──────────────────┘
                                           │
                              ┌────────────┴────────────┐
                              │                         │
                              ▼                         ▼
                     ┌───────────────┐         ┌───────────────┐
                     │   NO TRADE    │         │    TRADE      │
                     │   (Wait)      │         │  LONG/SHORT   │
                     └───────────────┘         └───────┬───────┘
                                                       │
                                                       ▼
                                              ┌───────────────┐
                                              │Paper Executor │
                                              │ - Entry price │
                                              │ - TP/SL levels│
                                              │ - Slippage    │
                                              └───────┬───────┘
                                                       │
                                                       ▼
                                              ┌───────────────┐
                                              │Position Mgr   │
                                              │ - Monitor P&L │
                                              │ - Check exits │
                                              │ - Save state  │
                                              └───────────────┘
```

## Components

### 1. BinanceConnector (`binance_connector.py`)

Handles real-time data from **Binance USD-M Futures** (real, not testnet):

- **WebSocket Connection**: Subscribes to `btcusdt@kline_1m` stream on `fstream.binance.com`
- **Historical Bootstrap**: Fetches 2500 candles on startup via `/fapi/v1/klines`
- **Auto-Reconnect**: Automatically reconnects on disconnect
- **Candle Buffer**: Rolling buffer of closed candles

**Endpoints used:**
- WebSocket: `wss://fstream.binance.com/ws`
- REST API: `https://fapi.binance.com`

```python
connector = BinanceConnector(
    symbol="BTCUSDT",
    interval="1m",
    buffer_size=2500,
    on_candle_close=callback_function,
)
await connector.start()
```

### 2. RealtimeStateBuilder (`state_builder.py`)

Converts live candles to state vectors:

- **Feature Computation**: EMA slopes, RSI, ATR, volume, etc.
- **Normalization**: Rolling z-scores and percentiles (2000-bar window)
- **Regime Detection**: Classifies current market regime
- **Output**: 10-dimensional state vector matching training format

```python
builder = RealtimeStateBuilder(normalization_window=2000)
builder.initialize(historical_df)  # 2500 bars from connector

# On each new candle:
state = builder.update(candle)
regime = builder.get_current_regime()
```

**State Vector (10 dimensions):**
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | ema50_slope_z | Short-term trend slope (z-score) |
| 1 | ema200_slope_z | Long-term trend slope (z-score) |
| 2 | trend_alignment | EMA alignment (-1, 0, +1) |
| 3 | return_5m_z | 5-minute returns (z-score) |
| 4 | return_15m_z | 15-minute returns (z-score) |
| 5 | rsi_z | RSI (z-score) |
| 6 | atr_percentile | Volatility (percentile 0-1) |
| 7 | volume_z | Volume (z-score) |
| 8 | vwap_distance_z | Distance from VWAP (z-score) |
| 9 | range_position | Position in range (0-1) |

### 3. PaperExecutor (`paper_executor.py`)

Simulates order execution:

- **Market Orders**: Simulates market order fills
- **Slippage**: Applies configurable slippage (default 0.05%)
- **Commission**: Calculates commission (default 0.04%)
- **Position Tracking**: Tracks open position with TP/SL levels

```python
executor = PaperExecutor(
    capital=200.0,
    slippage_pct=0.0005,
    commission_pct=0.0004,
)

order = executor.open_position(
    side="LONG",
    size_usd=5.0,
    current_price=97000.0,
    tp_pct=0.005,
    sl_pct=0.003,
)
```

### 4. PositionManager (`position_manager.py`)

High-level position management:

- **Trade Lifecycle**: Open, monitor, close positions
- **Timeout Exits**: Force close after N bars
- **Session Persistence**: Save/load session state to JSON
- **Statistics**: Track wins, losses, P&L

```python
manager = PositionManager(
    executor=executor,
    max_bars_in_trade=120,
    session_file=Path("data/paper_trading/session_state.json"),
)

# On new bar:
trade = manager.on_new_bar(current_price)  # Checks TP/SL/timeout

# Open trade from decision:
manager.open_trade(decision, current_price, regime)
```

### 5. Web Dashboard (`web/`)

Real-time browser-based dashboard. See [WEB_DASHBOARD.md](WEB_DASHBOARD.md) for details.

Features:
- Real-time price updates via WebSocket
- Position tracking with unrealized P&L
- Session statistics (trades, win rate, P&L)
- Trade history
- Access from any device on network

### 6. LiveOrchestrator (`live_orchestrator.py`)

Main coordination loop:

```
Every 1 minute (on candle close):
  1. Update state builder with new candle
  2. Check position exit conditions (TP/SL/timeout)
  3. Update dashboard display

Every 60 minutes (sample_interval from config):
  4. If no position open:
     a. Get current state vector
     b. Query similarity engine for 200 neighbors
     c. Get decision from decision engine
     d. If TRADE signal → open paper position
  5. Log status
```

## Usage

### Quick Start

```bash
# 1. Install dependencies
pip install websockets aiohttp

# 2. Validate setup
python run_paper_trade.py --dry-run

# 3. Start paper trading
python run_paper_trade.py
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--capital` | from config | Starting capital in USD |
| `--horizon` | from config | Outcome horizon in minutes |
| `--sample-interval` | from config | Check signals every N bars |
| `--min-expectancy` | from config | Minimum expectancy to trade |
| `--max-distance` | from config | Maximum similarity distance |
| `--blocked-regimes` | from config | Regimes to avoid |
| `--verbose` | False | Enable debug logging |
| `--dry-run` | False | Validate without trading |
| `--web-port` | 8080 | Port for web dashboard |

### Examples

```bash
# Default settings (H=5, si=15, $200 capital)
python run_paper_trade.py

# Higher capital
python run_paper_trade.py --capital 1000

# More frequent signals
python run_paper_trade.py --sample-interval 5

# Verbose logging
python run_paper_trade.py --verbose

# Block high volatility regime
python run_paper_trade.py --blocked-regimes HIGH_VOL
```

## Data Flow

### Signal Generation (Every 15 minutes)

```
1. Current State Vector (10D)
   └─→ Query Similarity Engine
       └─→ Find 200 nearest neighbors from training data
           └─→ Calculate:
               - Mean MFE (Maximum Favorable Excursion)
               - Mean MAE (Maximum Adverse Excursion)
               - Expectancy = MFE - |MAE|
               - Average distance to neighbors

2. Decision Engine Filters:
   ├─ Expectancy > 0.001? ──→ If NO: NO_TRADE (NEGATIVE_EXPECTANCY)
   ├─ Distance < 3.0?     ──→ If NO: NO_TRADE (LOW_SIMILARITY)
   ├─ Regime not blocked? ──→ If NO: NO_TRADE (BLOCKED_REGIME)
   └─ All pass?           ──→ TRADE signal

3. If TRADE:
   ├─ Direction: LONG if MFE > |MAE|, else SHORT
   ├─ Position size: Based on risk_per_trade (0.5% of capital)
   ├─ Stop loss: MAE 5th percentile
   └─ Take profit: Mean MFE
```

### Position Monitoring (Every 1 minute)

```
On each closed candle:
├─ Check if price >= TP ──→ Close with TAKE_PROFIT
├─ Check if price <= SL ──→ Close with STOP_LOSS
├─ Check bars >= max    ──→ Close with TIMEOUT
└─ Otherwise            ──→ Continue holding
```

## Session Persistence

The system automatically saves state to `data/paper_trading/session_state.json`:

```json
{
  "capital": 205.45,
  "initial_capital": 200.0,
  "total_commission": 0.32,
  "trade_history": [
    {
      "trade_id": "a1b2c3d4",
      "entry_time": "2026-01-01T10:15:00+00:00",
      "exit_time": "2026-01-01T10:23:00+00:00",
      "side": "LONG",
      "size_usd": 5.0,
      "entry_price": 97100.0,
      "exit_price": 97250.0,
      "pnl": 0.77,
      "pnl_pct": 0.0154,
      "exit_reason": "TP",
      "regime": "RANGE_LOW_VOL"
    }
  ],
  "saved_at": "2026-01-01T10:23:01+00:00"
}
```

On restart, the system loads previous session and continues tracking.

## Logging

Logs are written to:
- **Console**: Real-time status and signals
- **`logs/paper_trading.log`**: Full debug log

Log format:
```
2026-01-01 10:15:00 | INFO     | [SIGNAL] LONG | exp=0.150% | regime=RANGE_LOW_VOL | price=97100.00
2026-01-01 10:15:00 | INFO     | [TRADE] Opened LONG | size=$5.00 | TP=0.50% | SL=0.30%
2026-01-01 10:23:00 | INFO     | [CLOSED] LONG | PnL: +$0.77 (+1.54%) | reason=TP
2026-01-01 11:00:00 | INFO     | [STATUS] Uptime: 0:45:00 | Capital: $205.45 | Trades: 3 | Win Rate: 100%
```

## Configuration

All settings come from two sources:
1. **`config/config.yaml`** - Trading parameters, thresholds, execution settings
2. **`.env`** - Secrets, URLs, database connection

### Config File (`config/config.yaml`)

The paper trading system reads ALL settings from config:

```yaml
# Data settings
data:
  pair: "BTCUSDT"           # Trading pair
  timeframe: "1m"           # Candle timeframe

# State vector normalization
normalization:
  window: 2000              # Rolling window for z-scores

# Similarity engine
similarity:
  k: 200                    # Number of neighbors
  default_horizon: 5        # Outcome horizon in minutes

# Decision engine
decision:
  capital: 200              # Starting capital in USD
  min_expectancy: 0.001     # Minimum expectancy to trade
  max_distance: 3.0         # Maximum similarity distance
  blocked_regimes: []       # Regimes to avoid
  risk_per_trade: 0.005     # 0.5% risk per trade

# Backtesting / Paper trading
backtest:
  sample_interval: 15       # Check signals every N bars
  slippage_pct: 0.0005      # Applied to paper trades
  commission_pct: 0.0004    # Taker fee
  max_bars_in_trade: 120    # Force exit after N bars

# Paths
paths:
  data_dir: "data"          # Base directory for data files
```

### Environment Variables (`.env`)

Secrets and URLs are managed via `.env`:

```env
# Database Connection
DATABASE_URL=postgresql://user@host:5432/crypto_data

# Binance USD-M Futures API (uses real endpoints by default)
# Optional: Override default endpoints
# BINANCE_FUTURES_WS_URL=wss://fstream.binance.com/ws
# BINANCE_FUTURES_REST_URL=https://fapi.binance.com

# API keys (only needed for live trading with real orders)
# BINANCE_API_KEY=
# BINANCE_API_SECRET=
```

**Important:**
- Uses **real Binance Futures** endpoints (not testnet)
- Paper trading uses public WebSocket streams and does NOT require API keys
- The `.env` file is gitignored and should never be committed
- Copy `.env` file when deploying to new environments

## Safety Features

1. **Paper Mode Only**: No real orders ever sent to exchange
2. **Single Position**: Maximum one position at a time
3. **Session Persistence**: State saved on each trade close
4. **Graceful Shutdown**: Ctrl+C triggers clean shutdown
5. **Auto-Reconnect**: WebSocket reconnects on disconnect

## Troubleshooting

### "Missing required data files"
Run the pipeline first to generate training data:
```bash
python run_pipeline.py
```

### "websockets package required"
Install dependencies:
```bash
pip install websockets aiohttp
```

### "Connection refused"
- Check internet connection
- Binance WebSocket may be temporarily unavailable
- System will auto-reconnect

### "Not enough historical data"
- Wait for connector to fetch 2500 candles (~42 hours of data)
- Or the system will bootstrap from Binance REST API on startup

## Transitioning to Live Trading

After validating paper trading for 1-2 weeks:

1. **Compare Results**: Paper vs backtest expectations
2. **Small Size**: Start with 10% of intended capital
3. **Monitor 24/7**: Watch for unexpected behavior
4. **Gradual Scale**: Increase size if consistent

Live trading requires:
- Separate `live_executor.py` with real order submission
- Binance API keys with trading permissions
- Balance verification before each trade
- Rate limiting for API calls
