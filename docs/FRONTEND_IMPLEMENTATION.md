# Frontend Implementation Documentation

## Overview

The web dashboard frontend is a React application built with Vite. It provides a real-time interface for monitoring paper trading activity via WebSocket connections to the FastAPI backend.

## Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.2.0 | UI framework |
| Vite | 5.0.0 | Build tool & dev server |
| WebSocket | Native | Real-time data streaming |
| CSS | Custom | Dark theme styling |

## File Structure

```
web/frontend/
├── package.json           # Dependencies and scripts
├── vite.config.js         # Vite configuration
├── index.html             # HTML entry point
├── src/
│   ├── main.jsx           # React entry point
│   ├── App.jsx            # Main application component
│   ├── index.css          # Global styles (dark theme)
│   └── components/
│       ├── StatusPanel.jsx    # Price/regime/bars display
│       ├── PositionCard.jsx   # Open position details
│       ├── StatsSection.jsx   # Session statistics
│       └── TradesList.jsx     # Trade history table
└── dist/                  # Built files (served by FastAPI)
```

---

## Core Files

### 1. package.json

```json
{
  "name": "paper-trading-dashboard",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.1",
    "vite": "^5.0.0"
  }
}
```

**Scripts:**
- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production (output to `dist/`)
- `npm run preview` - Preview production build

---

### 2. vite.config.js

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8080',
      '/ws': {
        target: 'ws://localhost:8080',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  },
})
```

**Configuration:**
- Proxies `/api` and `/ws` to backend during development
- Outputs built files to `dist/` directory
- Assets placed in `dist/assets/`

---

### 3. index.html

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Paper Trading Dashboard</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

Minimal HTML template. React mounts to `#root` div.

---

### 4. main.jsx

```javascript
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

React entry point. Renders `App` component with StrictMode enabled.

---

## Main Application (App.jsx)

### State Management

```javascript
const [data, setData] = useState(null)      // Dashboard data from API
const [connected, setConnected] = useState(false)  // WebSocket connection status
const wsRef = useRef(null)                  // WebSocket reference
const reconnectTimeoutRef = useRef(null)    // Reconnection timer
```

### WebSocket Connection

```javascript
const connectWebSocket = () => {
  // Determine protocol (ws or wss based on page protocol)
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//${window.location.host}/ws`

  wsRef.current = new WebSocket(wsUrl)

  wsRef.current.onopen = () => {
    setConnected(true)
  }

  wsRef.current.onmessage = (event) => {
    const message = JSON.parse(event.data)
    if (message.type !== 'ping') {
      setData(message)  // Update dashboard with new data
    }
  }

  wsRef.current.onclose = () => {
    setConnected(false)
    // Auto-reconnect after 3 seconds
    reconnectTimeoutRef.current = setTimeout(connectWebSocket, 3000)
  }

  wsRef.current.onerror = (error) => {
    wsRef.current.close()
  }
}
```

**Key Features:**
- Automatic protocol detection (ws/wss)
- Ignores ping messages (used for keepalive)
- Auto-reconnects on disconnect (3 second delay)
- Cleans up on component unmount

### Initial Data Fetch

```javascript
useEffect(() => {
  // Fetch initial data via REST API
  fetch('/api/all')
    .then(res => res.json())
    .then(setData)
    .catch(console.error)

  // Start WebSocket connection
  connectWebSocket()

  // Cleanup on unmount
  return () => {
    if (wsRef.current) wsRef.current.close()
    if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current)
  }
}, [])
```

### Data Structure

The `data` state contains:

```javascript
{
  status: {
    status: "RUNNING",           // STARTING, RUNNING, STOPPED
    uptime_seconds: 3600,        // Seconds since start
    price: 97234.50,             // Current BTC price
    regime: "RANGE_LOW_VOL",     // Market regime
    bar_count: 154,              // Candles processed
    next_check_in: 12,           // Bars until next signal check
    last_update: "2025-01-01T..."  // ISO timestamp
  },
  position: {
    has_position: true,
    side: "LONG",
    size_btc: 0.000052,
    entry_price: 97100.00,
    current_price: 97234.50,
    tp_price: 97600.00,
    sl_price: 96800.00,
    unrealized_pnl: 0.27,
    unrealized_pnl_pct: 0.0014
  },
  stats: {
    capital: 5.12,
    initial_capital: 5.00,
    total_pnl: 0.12,
    total_pnl_pct: 0.024,
    total_trades: 3,
    wins: 3,
    losses: 0,
    win_rate: 1.0,
    total_commission: 0.01
  },
  trades: [
    {
      trade_id: "abc123",
      side: "LONG",
      entry_price: 97050.00,
      exit_price: 97150.00,
      entry_time: "2025-01-01T12:00:00Z",
      exit_time: "2025-01-01T12:15:00Z",
      pnl: 0.05,
      pnl_pct: 0.001,
      exit_reason: "TP",
      status: "CLOSED"
    }
  ],
  config: {
    pair: "BTCUSDT",
    horizon: 15,
    sample_interval: 15,
    min_expectancy: 0.001,
    initial_capital: 5.0
  }
}
```

### Uptime Formatter

```javascript
const formatUptime = (seconds) => {
  if (!seconds) return '0s'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  if (hours > 0) return `${hours}h ${minutes}m ${secs}s`
  if (minutes > 0) return `${minutes}m ${secs}s`
  return `${secs}s`
}
```

### Component Layout

```jsx
<div className="dashboard">
  {/* Header */}
  <header className="header">
    <h1>
      Paper Trading Dashboard
      <span className="pair">{config?.pair}</span>
    </h1>
    <div>
      <span className="uptime">Uptime: {formatUptime(...)}</span>
      <div className="connection-status">
        <span className="connection-dot connected|disconnected"></span>
        Live | Reconnecting...
      </div>
      <span className="status-badge running|stopped|starting">
        {status}
      </span>
    </div>
  </header>

  {/* Status Cards - 3 columns */}
  <div className="grid grid-3">
    <StatusPanel label="Price" value={price} format="price" />
    <StatusPanel label="Regime" value={regime} />
    <StatusPanel label="Bars Processed" value={bar_count} format="number" />
  </div>

  {/* Position Card */}
  <PositionCard position={position} />

  {/* Stats Section */}
  <StatsSection stats={stats} />

  {/* Trades List */}
  <TradesList trades={trades} />
</div>
```

---

## Components

### StatusPanel.jsx

**Purpose:** Display a single metric with optional formatting.

**Props:**
| Prop | Type | Description |
|------|------|-------------|
| label | string | Header text (uppercase) |
| value | any | Value to display |
| format | string | `"price"`, `"number"`, `"percent"`, or none |

**Formatting Logic:**
```javascript
switch (format) {
  case 'price':
    return `$${Number(value).toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    })}`  // e.g., "$97,234.50"

  case 'number':
    return Number(value).toLocaleString('en-US')  // e.g., "1,234"

  case 'percent':
    return `${(Number(value) * 100).toFixed(1)}%`  // e.g., "5.4%"

  default:
    return value  // Raw value
}
```

**Output:**
```html
<div class="card status-panel">
  <div class="card-header">PRICE</div>
  <div class="card-value">$97,234.50</div>
</div>
```

---

### PositionCard.jsx

**Purpose:** Display current open position details.

**Props:**
| Prop | Type | Description |
|------|------|-------------|
| position | object | Position data from API |

**Empty State:**
```jsx
<div className="card position-card no-position">
  <div className="card-header">Position</div>
  <div className="position-empty">No open position</div>
</div>
```

**With Position:**
```jsx
<div className="card position-card">
  <div className="position-header">
    <div>
      <div className="card-header">Position</div>
      <span className="side-badge long|short">LONG</span>
    </div>
    <div className="pnl-display">
      <div className="amount positive|negative">+$0.27</div>
      <div className="percent positive|negative">+0.14%</div>
    </div>
  </div>

  <div className="position-grid">
    <div className="position-item">
      <div className="label">Size</div>
      <div className="value">0.000052 BTC</div>
    </div>
    <div className="position-item">
      <div className="label">Entry Price</div>
      <div className="value">$97,100.00</div>
    </div>
    <div className="position-item">
      <div className="label">Take Profit</div>
      <div className="value positive">$97,600.00</div>
    </div>
    <div className="position-item">
      <div className="label">Stop Loss</div>
      <div className="value negative">$96,800.00</div>
    </div>
  </div>
</div>
```

**Helper Functions:**
```javascript
const formatPrice = (price) => {
  return `$${Number(price).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  })}`
}

const formatPnl = (pnl) => {
  const sign = pnl >= 0 ? '+' : ''
  return `${sign}$${pnl.toFixed(2)}`
}

const formatPnlPct = (pct) => {
  const sign = pct >= 0 ? '+' : ''
  return `${sign}${(pct * 100).toFixed(2)}%`
}
```

---

### StatsSection.jsx

**Purpose:** Display session statistics in a 5-column grid.

**Props:**
| Prop | Type | Description |
|------|------|-------------|
| stats | object | Stats data from API |

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ SESSION STATISTICS                                           │
├───────────┬───────────┬───────────┬───────────┬─────────────┤
│  Capital  │ Total P&L │  Trades   │ Win Rate  │ Commission  │
│   $5.12   │ +$0.12    │     3     │   100%    │    $0.01    │
│           │  (2.40%)  │           │           │             │
└───────────┴───────────┴───────────┴───────────┴─────────────┘
```

**Stats Displayed:**
| Stat | Source | Format |
|------|--------|--------|
| Capital | `stats.capital` | `$X.XX` |
| Total P&L | `stats.total_pnl` + `stats.total_pnl_pct` | `+$X.XX (X.XX%)` |
| Trades | `stats.total_trades` | Integer |
| Win Rate | `stats.win_rate` | `X%` |
| Commission | `stats.total_commission` | `$X.XX` |

---

### TradesList.jsx

**Purpose:** Display recent trades in a table.

**Props:**
| Prop | Type | Description |
|------|------|-------------|
| trades | array | Array of trade objects |

**Table Columns:**
| Column | Source | Format |
|--------|--------|--------|
| # | Index (reversed) | Integer |
| Side | `trade.side` | LONG (green) / SHORT (red) |
| Entry | `trade.entry_price` | `$X.XX` |
| Exit | `trade.exit_price` | `$X.XX` |
| P&L | `trade.pnl` | `+$X.XX` / `-$X.XX` |
| Reason | `trade.exit_reason` | TP, SL, TIMEOUT |
| Time | `trade.exit_time` | HH:MM (24h) |

**Empty State:**
```html
<div class="trades-empty">No trades yet</div>
```

**Time Formatter:**
```javascript
const formatTime = (isoString) => {
  if (!isoString) return '---'
  const date = new Date(isoString)
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false
  })
}
```

---

## CSS Styling (index.css)

### Color Variables

```css
:root {
  --bg-primary: #0d1117;      /* Page background */
  --bg-secondary: #161b22;    /* Card backgrounds */
  --bg-card: #21262d;         /* Inner card sections */
  --border-color: #30363d;    /* Borders */
  --text-primary: #f0f6fc;    /* Main text */
  --text-secondary: #8b949e;  /* Labels, muted text */
  --text-muted: #6e7681;      /* Very muted text */
  --accent-green: #3fb950;    /* Positive values, LONG */
  --accent-red: #f85149;      /* Negative values, SHORT */
  --accent-blue: #58a6ff;     /* Links, pair name */
  --accent-yellow: #d29922;   /* Warnings, starting status */
}
```

### Responsive Breakpoints

```css
@media (max-width: 900px) {
  .grid-3, .grid-2 {
    grid-template-columns: 1fr;  /* Stack all columns */
  }
  .stats-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 768px) {
  .position-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 600px) {
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
```

### Key CSS Classes

| Class | Purpose |
|-------|---------|
| `.dashboard` | Main container (max-width: 1200px, centered) |
| `.header` | Top header bar with title and status |
| `.card` | Standard card container with border |
| `.card-header` | Uppercase label text |
| `.card-value` | Large value text |
| `.grid-3` | 3-column grid layout |
| `.grid-2` | 2-column grid layout |
| `.status-badge` | Status pill (running/stopped/starting) |
| `.side-badge` | Position side pill (long/short) |
| `.positive` | Green text for positive values |
| `.negative` | Red text for negative values |
| `.connection-dot` | Live connection indicator (green glow) |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA FLOW DIAGRAM                        │
└─────────────────────────────────────────────────────────────┘

  Binance WebSocket                 FastAPI Server
        │                                 │
        ▼                                 │
  BinanceConnector ──────────────────────►│
        │                                 │
        │ on_candle_update (every tick)   │
        │ on_candle_close (every 1 min)   │
        ▼                                 │
  LiveOrchestrator ──────────────────────►│
        │                                 │
        │ web_state.update_status()       │
        │ web_state.update_position()     │
        │ web_state.update_stats()        │
        │ web_state.add_trade()           │
        ▼                                 │
  DashboardState ────────────────────────►│
        │                                 │
        │ _broadcast_update()             │
        ▼                                 ▼
  WebSocket Queue ──────► FastAPI /ws ──────► Browser
                              │
                              │ JSON message
                              ▼
                         React App
                              │
                              │ setData(message)
                              ▼
                         Re-render UI
```

### Update Frequency

| Data | Update Rate | Trigger |
|------|-------------|---------|
| Price | ~500ms (throttled) | Every WebSocket tick |
| Uptime | ~500ms | Every price broadcast |
| Regime | 1 minute | Candle close |
| Bar Count | 1 minute | Candle close |
| Position | 1 minute | Candle close |
| Stats | 1 minute | Candle close |
| Trades | On trade close | TP/SL/Timeout hit |

---

## Development

### Prerequisites

```bash
# Node.js 18+ required
node --version  # v18.x.x or higher

# Navigate to frontend directory
cd web/frontend
```

### Install Dependencies

```bash
npm install
```

### Development Mode

```bash
# Start the paper trading backend first
python scripts/run_paper_trade.py

# In another terminal, start the Vite dev server
cd web/frontend
npm run dev
```

The dev server runs on http://localhost:5173 and proxies API/WebSocket to port 8080.

### Build for Production

```bash
npm run build
```

Output goes to `web/frontend/dist/`, which FastAPI serves automatically.

### Preview Production Build

```bash
npm run preview
```

---

## Troubleshooting

### WebSocket Not Connecting

1. Check that the backend is running on port 8080
2. Look for CORS errors in browser console
3. Verify WebSocket URL is correct (`ws://localhost:8080/ws`)

### Data Not Updating

1. Check WebSocket connection indicator (should show "Live")
2. Verify backend is receiving candle data from Binance
3. Check browser console for errors

### Styles Not Loading

1. Ensure `index.css` is imported in `main.jsx`
2. Check for CSS syntax errors
3. Clear browser cache

### Build Fails

1. Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`
2. Check for import errors in components
3. Verify all dependencies are in `package.json`

---

## Future Improvements

1. **Chart Integration** - Add price chart with TradingView or Chart.js
2. **Audio Alerts** - Sound on trade open/close
3. **Trade Annotations** - Mark trades on price chart
4. **Performance Metrics** - Sharpe ratio, max drawdown
5. **Multi-Pair Support** - Monitor multiple trading pairs
6. **Dark/Light Theme Toggle** - User preference for theme
7. **Mobile Optimization** - Better touch controls for mobile
