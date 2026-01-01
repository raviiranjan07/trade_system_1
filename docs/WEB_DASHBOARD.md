# Web Dashboard Documentation

## Overview

The web dashboard provides a real-time browser-based interface for monitoring paper trading. It displays price, position, P&L, and trade history with live updates via WebSocket.

## Quick Start

```bash
# Start paper trading with web dashboard
python run_paper_trade.py --web-port 8080

# Access the dashboard
# Same machine: http://localhost:8080
# Other devices: http://<your-ip>:8080
```

## Access URLs

| Location | URL |
|----------|-----|
| Same machine | `http://localhost:8080` or `http://127.0.0.1:8080` |
| Other devices on network | `http://<your-computer-ip>:8080` |

To find your IP address:
```bash
# Windows
ipconfig

# Linux/Mac
ifconfig
```

## Features

- **Real-time price** - Price updates every 500ms from Binance WebSocket ticks
- **Live dashboard** - Regime, position, and stats update on each candle close (1 min)
- **Price & Regime** - Current BTC price and detected market regime
- **Position tracking** - Entry, TP, SL, and unrealized P&L
- **Session stats** - Capital, total P&L, win rate, commission
- **Trade history** - List of recent closed trades
- **Dark theme** - Easy on the eyes for extended monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      WEB DASHBOARD                           │
└─────────────────────────────────────────────────────────────┘

  Browser (React)                    Server (FastAPI)
 ┌─────────────────┐               ┌─────────────────┐
 │  React Frontend │◄──WebSocket──►│  FastAPI Server │
 │                 │               │                 │
 │  - Status Panel │◄───REST API──►│  /api/status    │
 │  - Position     │               │  /api/position  │
 │  - Stats        │               │  /api/stats     │
 │  - Trades List  │               │  /ws (realtime) │
 └─────────────────┘               └────────┬────────┘
                                            │
                                   ┌────────▼────────┐
                                   │ LiveOrchestrator│
                                   │  (shared state) │
                                   └─────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Current trading status |
| `/api/position` | GET | Open position details |
| `/api/stats` | GET | Session statistics |
| `/api/trades` | GET | Recent trade history |
| `/api/config` | GET | Trading configuration |
| `/api/all` | GET | All dashboard data |
| `/ws` | WebSocket | Real-time updates |

### Example API Response

```json
// GET /api/all
{
  "status": {
    "status": "RUNNING",
    "uptime_seconds": 3600,
    "price": 97234.50,
    "regime": "RANGE_LOW_VOL",
    "bar_count": 154,
    "next_check_in": 12
  },
  "position": {
    "has_position": true,
    "side": "LONG",
    "size_btc": 0.000052,
    "entry_price": 97100.00,
    "tp_price": 97600.00,
    "sl_price": 96800.00,
    "unrealized_pnl": 0.27,
    "unrealized_pnl_pct": 0.0014
  },
  "stats": {
    "capital": 5.12,
    "total_pnl": 0.12,
    "total_pnl_pct": 0.024,
    "total_trades": 3,
    "win_rate": 1.0
  },
  "trades": [...]
}
```

## File Structure

```
web/
├── __init__.py              # Package init
├── state.py                 # Shared state manager (thread-safe)
├── server.py                # FastAPI app with REST + WebSocket
└── frontend/
    ├── package.json         # npm dependencies
    ├── vite.config.js       # Build configuration
    ├── index.html           # HTML template
    ├── src/
    │   ├── main.jsx         # React entry point
    │   ├── App.jsx          # Main component
    │   ├── index.css        # Styling (dark theme)
    │   └── components/
    │       ├── StatusPanel.jsx   # Price/regime cards
    │       ├── PositionCard.jsx  # Position display
    │       ├── StatsSection.jsx  # Session stats
    │       └── TradesList.jsx    # Trade history
    └── dist/                # Built files (served by FastAPI)
```

## Dependencies

### Python (Backend)
```bash
pip install fastapi uvicorn[standard]
```

### Node.js (Frontend - for development only)
```bash
cd web/frontend
npm install
npm run build
```

The frontend is pre-built, so Node.js is only needed if you want to modify the React code.

## Configuration

The web dashboard is enabled via command-line:

```bash
# Enable web dashboard on port 8080
python run_paper_trade.py --web-port 8080

# Disable terminal dashboard, use only web
python run_paper_trade.py --web-port 8080 --no-dashboard

# Custom capital with web dashboard
python run_paper_trade.py --capital 100 --web-port 8080
```

## Troubleshooting

### Dashboard not loading?
1. Check if port is in use: `netstat -an | findstr 8080`
2. Try a different port: `--web-port 8081`
3. Check firewall settings for the port

### Can't access from other devices?
1. Make sure you're using your computer's IP, not `localhost`
2. Check Windows Firewall allows incoming connections on the port
3. Ensure devices are on the same network

### WebSocket disconnecting?
- The dashboard auto-reconnects after 3 seconds
- Check network stability
- Ensure the trading system is still running

## Development

To modify the frontend:

```bash
cd web/frontend

# Install dependencies
npm install

# Start dev server (with hot reload)
npm run dev

# Build for production
npm run build
```

The dev server proxies API requests to `localhost:8080`, so run the paper trading system first.
