"""
FastAPI Web Server for Paper Trading Dashboard.

Provides REST API endpoints and WebSocket for real-time updates.
Serves static React frontend files.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
import uvicorn

from .state import DashboardState, dashboard_state

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
    # REST API Endpoints
    # =========================================================================

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

    @app.get("/api/all")
    async def get_all():
        """Get all dashboard data."""
        return JSONResponse(content=state.get_all())

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
            """Serve React app index."""
            return FileResponse(frontend_dir / "index.html")

        @app.get("/{path:path}")
        async def serve_static(path: str):
            """Serve static files or fallback to index."""
            file_path = frontend_dir / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(frontend_dir / "index.html")
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
