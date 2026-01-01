# Paper Trading Implementation Guide

## Overview

This document describes the implementation of a paper trading system for the H=5m strategy on Binance. The system uses real-time market data but executes simulated trades only (no real money at risk).

## Optimized Parameters (from Grid Search)

Based on extensive backtesting, the following parameters achieved **100% win rate** with **+12.63% return** and **0% drawdown**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Horizon | 5 minutes | Short-term momentum |
| Sample Interval | 15 bars | Check signals every 15 minutes |
| Min Expectancy | 0.001 | Filters marginal trades |
| Max Distance | 3.0 | Allows more similar states |
| Blocked Regimes | None | Volatility helps 5m trades |
| Capital | $200 | Starting paper capital |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PAPER TRADING SYSTEM                            │
└─────────────────────────────────────────────────────────────────────────┘

  Binance WebSocket ──► State Builder ──► Regime Labeler
         │                                      │
         │                                      ▼
         │                              Similarity Engine
         │                                      │
         │                                      ▼
         │                              Decision Engine
         │                                      │
         ▼                                      ▼
  OHLCV Buffer ◄─────────────────────► Paper Executor
  (2500 bars)                                   │
                                               ▼
                                        Position Manager
                                               │
                                               ▼
                                     Dashboard / Logging
```

---

## File Structure

```
live/
├── __init__.py              # Package init
├── binance_connector.py     # WebSocket + REST API client
├── state_builder.py         # Real-time state vector calculation
├── live_orchestrator.py     # Main loop coordinator
├── paper_executor.py        # Simulated order execution
├── position_manager.py      # Track positions & P&L
└── dashboard.py             # Console display

run_paper_trade.py           # Entry point script
```

---

## Component Specifications

### 1. Binance Connector (`live/binance_connector.py`)

**Purpose**: Connect to Binance for real-time and historical data.

**Features**:
- WebSocket subscription to `btcusdt@kline_1m` stream
- REST API for historical candle bootstrap
- Automatic reconnection on disconnect
- Rolling buffer of last 2500 candles

**Key Methods**:
```python
class BinanceConnector:
    async def connect()           # Start WebSocket connection
    async def disconnect()        # Clean shutdown
    def get_historical(n: int)    # Fetch n historical candles
    def get_latest_candle()       # Get current candle
    def get_candle_buffer()       # Get full buffer as DataFrame
```

**WebSocket Message Format**:
```json
{
  "e": "kline",
  "k": {
    "t": 1234567890000,  // Open time
    "o": "50000.00",     // Open
    "h": "50100.00",     // High
    "l": "49900.00",     // Low
    "c": "50050.00",     // Close
    "v": "100.5",        // Volume
    "x": true            // Is candle closed?
  }
}
```

---

### 2. State Builder (`live/state_builder.py`)

**Purpose**: Convert live candles to state vectors matching backtest format.

**Features**:
- Incremental calculation (not full recalc each bar)
- Rolling windows for all indicators
- Normalization using 2000-bar lookback

**Indicators Calculated**:
| Indicator | Window | Description |
|-----------|--------|-------------|
| EMA_20 | 20 | Short-term trend |
| EMA_50 | 50 | Medium-term trend |
| EMA_200 | 200 | Long-term trend |
| ATR_14 | 14 | Volatility measure |
| RSI_14 | 14 | Momentum oscillator |
| BB_upper/lower | 20 | Bollinger Bands |
| MACD | 12/26/9 | Trend momentum |

**Key Methods**:
```python
class StateBuilder:
    def update(candle: dict)      # Process new candle
    def get_current_state()       # Get latest state vector
    def get_state_history(n: int) # Get last n states
```

---

### 3. Live Orchestrator (`live/live_orchestrator.py`)

**Purpose**: Coordinate all components in the main trading loop.

**Flow**:
```
Every 1 minute (on candle close):
  1. Receive new candle from WebSocket
  2. Update state builder
  3. Update position manager (check TP/SL)

Every 15 minutes (sample_interval):
  4. Calculate current regime
  5. Query similarity engine
  6. Get trading decision
  7. Execute paper trade if signal
  8. Update dashboard
```

**Key Methods**:
```python
class LiveOrchestrator:
    async def start()             # Begin trading loop
    async def stop()              # Graceful shutdown
    def get_status()              # Current system state
```

---

### 4. Paper Executor (`live/paper_executor.py`)

**Purpose**: Simulate order execution without real orders.

**Features**:
- Market order simulation at current price
- Slippage modeling (0.05% default)
- Commission calculation (0.04% default)
- Order logging

**Key Methods**:
```python
class PaperExecutor:
    def execute_order(
        side: str,           # "BUY" or "SELL"
        size: float,         # Position size in BTC
        price: float,        # Current market price
        tp_price: float,     # Take profit price
        sl_price: float      # Stop loss price
    ) -> Order
```

**Order Object**:
```python
@dataclass
class Order:
    order_id: str
    timestamp: datetime
    side: str
    size: float
    entry_price: float      # After slippage
    tp_price: float
    sl_price: float
    commission: float
    status: str             # "OPEN", "CLOSED", "CANCELLED"
```

---

### 5. Position Manager (`live/position_manager.py`)

**Purpose**: Track open positions and calculate P&L.

**Features**:
- Single position at a time (no pyramiding)
- Real-time unrealized P&L
- TP/SL monitoring
- Trade history logging

**Key Methods**:
```python
class PositionManager:
    def open_position(order: Order)
    def close_position(exit_price: float, reason: str)
    def check_exit_conditions(current_price: float)
    def get_unrealized_pnl()
    def get_realized_pnl()
    def get_trade_history() -> List[Trade]
```

**Trade Object**:
```python
@dataclass
class Trade:
    trade_id: str
    entry_time: datetime
    exit_time: datetime
    side: str
    size: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_reason: str        # "TP", "SL", "SIGNAL", "MANUAL"
```

---

### 6. Dashboard (`live/dashboard.py`)

**Purpose**: Real-time console display of trading status.

**Display Format**:
```
╔════════════════════════════════════════════════════════════════════════╗
║                    PAPER TRADING - BTCUSDT H=5m                        ║
╠════════════════════════════════════════════════════════════════════════╣
║ Status: RUNNING          │ Uptime: 2h 34m 12s                          ║
║ Current Price: $97,234   │ Regime: trending_up                         ║
╠════════════════════════════════════════════════════════════════════════╣
║ POSITION                                                               ║
║ Side: LONG               │ Size: 0.00205 BTC                           ║
║ Entry: $97,100           │ Current: $97,234                            ║
║ TP: $97,600              │ SL: $96,800                                  ║
║ Unrealized P&L: +$0.27 (+0.14%)                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║ SESSION STATS                                                          ║
║ Capital: $200.00         │ Realized P&L: +$2.45 (+1.23%)               ║
║ Trades: 3                │ Win Rate: 100%                              ║
║ Last Signal: 12:45 UTC   │ Next Check: 13:00 UTC                       ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

### 7. Entry Point (`run_paper_trade.py`)

**Usage**:
```bash
# Start paper trading with default params
python scripts/run_paper_trade.py

# Start with custom params
python scripts/run_paper_trade.py --capital 500 --horizon 5

# Verbose mode
python scripts/run_paper_trade.py --verbose
```

**Command Line Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--capital` | 200 | Starting capital in USD |
| `--horizon` | 5 | Outcome horizon in minutes |
| `--pair` | BTCUSDT | Trading pair |
| `--verbose` | False | Enable debug logging |
| `--dry-run` | False | Validate setup without trading |

---

## Configuration

### Environment Variables (`.env`)

```env
# Binance API (required for WebSocket)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Optional
BINANCE_TESTNET=false
```

### Config File Updates (`config/config.yaml`)

```yaml
# Paper Trading Section
paper_trading:
  enabled: true
  capital: 200

  # Risk Management
  max_position_pct: 1.0      # Max % of capital per trade
  daily_loss_limit_pct: 10   # Stop trading if daily loss exceeds

  # Execution
  slippage_pct: 0.0005       # 0.05% slippage
  commission_pct: 0.0004     # 0.04% commission (Binance spot)

  # Data
  buffer_size: 2500          # Candles to keep in memory
  reconnect_delay: 5         # Seconds before reconnect attempt
```

---

## Safety Features

### 1. Paper Mode Lock
- No real orders ever sent
- API keys only used for data access
- Clear "PAPER" label in all logs

### 2. Position Limits
- Maximum 1 position at a time
- Position size capped at `max_position_pct` of capital

### 3. Daily Loss Limit
- Auto-stops trading if daily loss exceeds threshold
- Requires manual restart

### 4. Graceful Shutdown
- Ctrl+C triggers clean shutdown
- Saves state to `data/paper_trading/session_state.json`
- Can resume from saved state

### 5. Comprehensive Logging
```
logs/
├── paper_trading.log       # All events
├── paper_signals.log       # Signals only
└── paper_trades.log        # Trades only
```

---

## Testing Checklist

### Before Going Live
- [ ] Verify WebSocket connection stable for 1 hour
- [ ] Confirm state vectors match backtest format
- [ ] Check regime detection matches historical
- [ ] Validate signals against backtest expectations
- [ ] Run for 24 hours without crashes
- [ ] Compare paper results to backtest projections

### Monitoring
- [ ] Set up alerts for connection drops
- [ ] Monitor memory usage (buffer size)
- [ ] Track signal frequency vs expected
- [ ] Compare win rate to backtest

---

## Dependencies

Add to `requirements.txt`:
```
python-binance>=1.0.19
websockets>=12.0
aiohttp>=3.9.0
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install python-binance websockets aiohttp

# 2. Set up environment
cp .env.example .env
# Edit .env with your Binance API keys

# 3. Verify setup
python scripts/run_paper_trade.py --dry-run

# 4. Start paper trading
python scripts/run_paper_trade.py

# 5. Monitor logs
tail -f logs/paper_trading.log
```

---

## Transition to Live Trading

Once paper trading proves consistent:

1. **Validation Period**: Run paper for minimum 1 week
2. **Performance Check**: Verify metrics match backtest
3. **Small Size**: Start live with 10% of intended capital
4. **Monitoring**: 24/7 monitoring for first week
5. **Scale Up**: Gradually increase position size

**Live trading will require**:
- Separate `live_executor.py` with real order submission
- Additional safety checks (balance verification, order confirmation)
- Rate limiting for API calls
- Emergency stop functionality
