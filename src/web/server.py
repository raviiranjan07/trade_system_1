"""
FastAPI Web Server for Paper Trading Dashboard.

Provides REST API endpoints and WebSocket for real-time updates.
Serves static React frontend files.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from sqlalchemy import create_engine, text
import uvicorn

from .state import DashboardState, dashboard_state
from .portfolio import PortfolioClient

load_dotenv()

logger = logging.getLogger(__name__)


def create_app(state: Optional[DashboardState] = None) -> FastAPI:
    """
    Create FastAPI application with all routes.

    Args:
        state: Dashboard state instance (uses global if None)

    Returns:
        FastAPI application
    """
    if state is None:
        state = dashboard_state

    app = FastAPI(
        title="Paper Trading Dashboard",
        description="Real-time monitoring for paper trading system",
        version="1.0.0",
    )

    # CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =========================================================================
    # Persistence files for drawings & indicators
    # =========================================================================
    PERSIST_DIR = Path("data/trades")
    DRAWINGS_FILE = PERSIST_DIR / "drawings.json"
    INDICATORS_FILE = PERSIST_DIR / "indicators.json"

    # Portfolio client (lazy init on first request)
    portfolio_client = PortfolioClient()

    # =========================================================================
    # REST API Endpoints
    # =========================================================================

    @app.get("/api/drawings")
    async def get_drawings():
        """Load saved drawings from disk."""
        if DRAWINGS_FILE.exists():
            try:
                return JSONResponse(content=json.loads(DRAWINGS_FILE.read_text()))
            except Exception:
                return JSONResponse(content=[])
        return JSONResponse(content=[])

    @app.post("/api/drawings")
    async def save_drawings(request: Request):
        """Save drawings to disk."""
        body = await request.json()
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        DRAWINGS_FILE.write_text(json.dumps(body))
        return JSONResponse(content={"ok": True})

    @app.get("/api/indicators")
    async def get_indicators():
        """Load saved indicators from disk."""
        if INDICATORS_FILE.exists():
            try:
                return JSONResponse(content=json.loads(INDICATORS_FILE.read_text()))
            except Exception:
                return JSONResponse(content=[])
        return JSONResponse(content=[])

    @app.post("/api/indicators")
    async def save_indicators(request: Request):
        """Save indicators to disk."""
        body = await request.json()
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        INDICATORS_FILE.write_text(json.dumps(body))
        return JSONResponse(content={"ok": True})

    @app.get("/api/status")
    async def get_status():
        """Get current trading status."""
        return JSONResponse(content=state.get_status())

    @app.get("/api/position")
    async def get_position():
        """Get current position details."""
        return JSONResponse(content=state.get_position())

    @app.get("/api/stats")
    async def get_stats():
        """Get session statistics."""
        return JSONResponse(content=state.get_stats())

    @app.get("/api/trades")
    async def get_trades(limit: int = 10):
        """Get recent trade history."""
        return JSONResponse(content=state.get_trades(limit=limit))

    @app.get("/api/config")
    async def get_config():
        """Get trading configuration."""
        return JSONResponse(content=state.get_config())

    @app.get("/api/candles")
    async def get_candles():
        """Get candle history for chart (recent candles from bot memory)."""
        return JSONResponse(content=state.get_candles())

    @app.get("/api/candles/history")
    async def get_candles_history(
        before: int = Query(..., description="Unix timestamp (seconds) — fetch candles older than this"),
        limit: int = Query(1000, ge=100, le=5000, description="Number of candles to return"),
    ):
        """Lazy-load older candles from TimescaleDB (paginated, with indicators)."""
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            return JSONResponse(content={"candles": [], "has_more": False})

        # Fetch extra bars for SMA200 and RSI14 warm-up
        warmup = 220  # 200 for SMA + 20 buffer for RSI
        fetch_limit = limit + warmup + 1  # +1 to detect has_more

        try:
            engine = create_engine(db_url, connect_args={"connect_timeout": 10})
            before_dt = datetime.fromtimestamp(before, tz=timezone.utc)
            query = text("""
                SELECT
                    time_bucket('15 minutes', time) AS time,
                    first(open, time) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, time) AS close,
                    sum(volume) AS volume
                FROM ohlcv_data
                WHERE pair = :pair
                  AND time < :before
                GROUP BY 1
                ORDER BY 1 DESC
                LIMIT :lim
            """)
            df = pd.read_sql(query, engine, params={
                "pair": "BTCUSDT", "before": before_dt, "lim": fetch_limit,
            })
            engine.dispose()
        except Exception as e:
            logger.error("Failed to load candle history from DB: %s", e)
            return JSONResponse(content={"candles": [], "has_more": False})

        if df.empty:
            return JSONResponse(content={"candles": [], "has_more": False})

        # Reverse so oldest first (ascending time)
        df = df.iloc[::-1].reset_index(drop=True)

        # Compute indicators on full dataset (including warm-up bars)
        df["sma"] = df["close"].rolling(window=200).mean()

        # RSI14 calculation
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Trim warm-up bars — keep only the requested window
        # The last `limit + 1` rows are the actual requested candles
        actual_count = len(df) - warmup
        if actual_count <= 0:
            # Not enough data for full warm-up, return what we have
            actual_count = len(df)
            has_more = False
        else:
            df = df.iloc[warmup:]
            has_more = len(df) > limit

        df = df.head(limit).reset_index(drop=True)

        candles = []
        for _, row in df.iterrows():
            ts = row["time"]
            if hasattr(ts, 'timestamp'):
                t = int(ts.timestamp())
            else:
                t = int(pd.Timestamp(ts).timestamp())
            sma_val = round(float(row["sma"]), 2) if pd.notna(row["sma"]) else None
            rsi_val = round(float(row["rsi"]), 1) if pd.notna(row["rsi"]) else None
            candles.append({
                "time": t,
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": round(float(row["volume"]), 2),
                "sma": sma_val,
                "rsi": rsi_val,
            })

        return JSONResponse(content={"candles": candles, "has_more": has_more})

    @app.get("/api/all")
    async def get_all():
        """Get all dashboard data."""
        return JSONResponse(content=state.get_all())

    @app.get("/api/decisions")
    async def get_decisions(limit: int = Query(50, ge=1, le=500)):
        """Get recent risk decisions from CSV logs (V1.4 + ML combined)."""
        KEEP_COLS = [
            "timestamp", "wallet_usd", "btc_price", "signal_type",
            "action", "qty", "risk_pct", "health_multiplier",
            "pnl_bps", "pnl_usd",
        ]
        frames = []
        for csv_path in [
            Path("data/trades/risk_logs/decisions.csv"),
            Path("data/trades/risk_logs/ml/decisions.csv"),
        ]:
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    frames.append(df)
                except Exception:
                    pass

        if not frames:
            return JSONResponse(content=[])

        combined = pd.concat(frames, ignore_index=True)
        # Keep only columns that exist in the data
        keep = [c for c in KEEP_COLS if c in combined.columns]
        combined = combined[keep]
        combined = combined.sort_values("timestamp", ascending=False).head(limit)
        # Replace NaN with None for clean JSON
        combined = combined.where(combined.notna(), None)
        return JSONResponse(content=combined.to_dict(orient="records"))

    # =========================================================================
    # Portfolio Endpoints (Binance authenticated)
    # =========================================================================

    @app.get("/api/portfolio/status")
    async def get_portfolio_status():
        """Check if Binance API keys are configured."""
        return JSONResponse(content={"configured": portfolio_client.is_configured})

    @app.get("/api/portfolio")
    async def get_portfolio():
        """Get live Binance Futures portfolio data."""
        if not portfolio_client.is_configured:
            return JSONResponse(content={
                "error": "not_configured",
                "message": "Set BINANCE_API_KEY and BINANCE_API_SECRET in .env",
            })
        data = await portfolio_client.get_portfolio_snapshot()
        return JSONResponse(content=data)

    @app.get("/api/portfolio/pnl")
    async def get_portfolio_pnl(period: str = Query("week")):
        """Get PNL breakdown for a time period (week/month/year/all)."""
        if not portfolio_client.is_configured:
            return JSONResponse(content={"error": "not_configured"})
        if period not in ("week", "month", "year", "all"):
            return JSONResponse(content={"error": "invalid period"}, status_code=400)
        data = await portfolio_client.get_pnl_summary(period)
        return JSONResponse(content=data)

    # =========================================================================
    # WebSocket Endpoint
    # =========================================================================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time updates."""
        await websocket.accept()
        queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        state.register_ws_client(queue)

        try:
            # Send initial state
            await websocket.send_json(state.get_all())

            # Stream updates
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    await websocket.send_json(data)
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await websocket.send_json({"type": "ping"})
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            state.unregister_ws_client(queue)

    # =========================================================================
    # Static Files (React Frontend)
    # =========================================================================

    # Path to frontend build directory
    frontend_dir = Path(__file__).parent / "frontend" / "dist"

    if frontend_dir.exists():
        # Serve static files
        app.mount(
            "/assets",
            StaticFiles(directory=frontend_dir / "assets"),
            name="assets",
        )

        @app.get("/")
        async def serve_index():
            """Serve React app index (no-cache so browser always gets latest build)."""
            return FileResponse(
                frontend_dir / "index.html",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        @app.get("/{path:path}")
        async def serve_static(path: str):
            """Serve static files or fallback to index."""
            file_path = frontend_dir / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(
                frontend_dir / "index.html",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
    else:
        # Serve simple status page if no frontend build
        @app.get("/")
        async def serve_fallback():
            """Fallback page when frontend not built."""
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Paper Trading Dashboard</title>
                <style>
                    body {
                        font-family: monospace;
                        background: #1a1a2e;
                        color: #eee;
                        padding: 20px;
                    }
                    .card {
                        background: #16213e;
                        padding: 20px;
                        margin: 10px 0;
                        border-radius: 8px;
                    }
                    h1 { color: #00d4aa; }
                    .label { color: #888; }
                    .value { color: #fff; font-weight: bold; }
                    .positive { color: #00d4aa; }
                    .negative { color: #ff6b6b; }
                </style>
            </head>
            <body>
                <h1>Paper Trading Dashboard</h1>
                <p class="label">API Endpoints:</p>
                <ul>
                    <li><a href="/api/status">/api/status</a> - Current status</li>
                    <li><a href="/api/position">/api/position</a> - Position info</li>
                    <li><a href="/api/stats">/api/stats</a> - Session stats</li>
                    <li><a href="/api/trades">/api/trades</a> - Trade history</li>
                    <li><a href="/api/all">/api/all</a> - All data</li>
                </ul>
                <div id="data" class="card">Loading...</div>
                <script>
                    async function fetchData() {
                        const res = await fetch('/api/all');
                        const data = await res.json();
                        document.getElementById('data').innerHTML =
                            '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    }
                    fetchData();
                    setInterval(fetchData, 5000);
                </script>
            </body>
            </html>
            """
            return HTMLResponse(content=html)

    return app


async def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    state: Optional[DashboardState] = None,
) -> None:
    """
    Run the FastAPI server.

    Args:
        host: Host to bind to (0.0.0.0 for network access)
        port: Port to listen on
        state: Dashboard state instance
    """
    app = create_app(state)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    await server.serve()


def start_server_background(
    host: str = "0.0.0.0",
    port: int = 8080,
    state: Optional[DashboardState] = None,
) -> asyncio.Task:
    """
    Start server as background task.

    Args:
        host: Host to bind to
        port: Port to listen on
        state: Dashboard state instance

    Returns:
        Asyncio task running the server
    """
    return asyncio.create_task(run_server(host, port, state))


if __name__ == "__main__":
    # Run standalone for testing
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    asyncio.run(run_server(port=port))
