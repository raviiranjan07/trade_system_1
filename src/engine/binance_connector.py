"""
Binance Futures WebSocket Connector for Real-Time OHLCV Data.

Provides:
- WebSocket subscription to Binance Futures kline streams
- REST API for historical candle bootstrap
- Automatic reconnection on disconnect
- Rolling buffer of candles

Uses Binance USD-M Futures (real, not testnet):
- WebSocket: wss://fstream.binance.com/ws
- REST API: https://fapi.binance.com

Environment Variables (optional overrides):
- BINANCE_FUTURES_WS_URL: WebSocket URL
- BINANCE_FUTURES_REST_URL: REST API URL
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Callable, Optional, Dict, List
from collections import deque

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use os.environ directly

try:
    import websockets
except ImportError:
    websockets = None

try:
    import aiohttp
except ImportError:
    aiohttp = None


logger = logging.getLogger(__name__)


# Binance USD-M Futures endpoints (REAL - not testnet)
DEFAULT_FUTURES_WS_URL = "wss://fstream.binance.com/ws"
DEFAULT_FUTURES_REST_URL = "https://fapi.binance.com"


def get_binance_urls() -> tuple:
    """Get Binance Futures API URLs from environment or use defaults."""
    # Use real Binance Futures endpoints (can be overridden via .env)
    ws_url = os.getenv("BINANCE_FUTURES_WS_URL", DEFAULT_FUTURES_WS_URL)
    rest_url = os.getenv("BINANCE_FUTURES_REST_URL", DEFAULT_FUTURES_REST_URL)

    logger.info(f"Using Binance Futures: {ws_url}")
    return ws_url, rest_url


class BinanceConnector:
    """
    Binance USD-M Futures WebSocket connector for real-time kline data.

    Uses REAL Binance Futures endpoints (not testnet):
    - WebSocket: wss://fstream.binance.com/ws
    - REST API: https://fapi.binance.com

    Optional environment variable overrides:
    - BINANCE_FUTURES_WS_URL: Custom WebSocket endpoint
    - BINANCE_FUTURES_REST_URL: Custom REST API endpoint

    Usage:
        connector = BinanceConnector(
            symbol="BTCUSDT",
            interval="1m",
            buffer_size=2500,
            on_candle_close=my_callback
        )
        await connector.start()
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        buffer_size: int = 2500,
        on_candle_close: Optional[Callable] = None,
        on_candle_update: Optional[Callable] = None,
        reconnect_delay: int = 5,
    ):
        """
        Initialize the Binance connector.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (e.g., "1m", "5m", "1h")
            buffer_size: Number of candles to keep in memory
            on_candle_close: Callback when a candle closes
            on_candle_update: Callback on each candle update (tick)
            reconnect_delay: Seconds to wait before reconnecting
        """
        if websockets is None:
            raise ImportError("websockets package required. Install: pip install websockets")
        if aiohttp is None:
            raise ImportError("aiohttp package required. Install: pip install aiohttp")

        self.symbol = symbol.upper()
        self.interval = interval
        self.buffer_size = buffer_size
        self.on_candle_close = on_candle_close
        self.on_candle_update = on_candle_update
        self.reconnect_delay = reconnect_delay

        # Get URLs from environment
        self._ws_base_url, self._rest_base_url = get_binance_urls()

        # Internal state
        self._candle_buffer: deque = deque(maxlen=buffer_size)
        self._current_candle: Optional[Dict] = None
        self._ws = None
        self._running = False
        self._connected = False
        self._last_candle_time: Optional[datetime] = None

    @property
    def stream_name(self) -> str:
        """WebSocket stream name."""
        return f"{self.symbol.lower()}@kline_{self.interval}"

    @property
    def ws_url(self) -> str:
        """Full WebSocket URL."""
        return f"{self._ws_base_url}/{self.stream_name}"

    async def start(self) -> None:
        """Start the WebSocket connection and begin receiving data."""
        if self._running:
            logger.warning("Connector already running")
            return

        self._running = True
        logger.info(f"Starting Binance connector for {self.symbol} {self.interval}")

        # Bootstrap with historical data
        await self._fetch_historical()

        # Start WebSocket loop
        await self._ws_loop()

    async def stop(self) -> None:
        """Stop the WebSocket connection gracefully."""
        logger.info("Stopping Binance connector...")
        self._running = False
        if self._ws:
            await self._ws.close()
        self._connected = False

    async def _fetch_historical(self) -> None:
        """Fetch historical candles to bootstrap the buffer."""
        logger.info(f"Fetching {self.buffer_size} historical candles from Binance Futures...")

        # Binance Futures uses /fapi/v1/klines endpoint
        url = f"{self._rest_base_url}/fapi/v1/klines"
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": min(self.buffer_size, 1000),  # API limit is 1000
        }

        async with aiohttp.ClientSession() as session:
            # Fetch in batches if needed
            all_candles = []
            end_time = None

            while len(all_candles) < self.buffer_size:
                if end_time:
                    params["endTime"] = end_time

                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise Exception(f"Failed to fetch historical data: {error}")

                    data = await response.json()

                if not data:
                    break

                # Prepend older candles
                all_candles = data + all_candles

                # Get timestamp of oldest candle for next batch
                end_time = data[0][0] - 1

                if len(data) < 1000:
                    break  # No more data available

            # Parse and store candles
            for kline in all_candles[-self.buffer_size:]:
                candle = self._parse_rest_kline(kline)
                self._candle_buffer.append(candle)

            if self._candle_buffer:
                self._last_candle_time = self._candle_buffer[-1]["close_time"]

        logger.info(f"Loaded {len(self._candle_buffer)} historical candles")

    def _parse_rest_kline(self, kline: list) -> Dict:
        """Parse REST API kline format to dict."""
        return {
            "open_time": datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
            "open": float(kline[1]),
            "high": float(kline[2]),
            "low": float(kline[3]),
            "close": float(kline[4]),
            "volume": float(kline[5]),
            "close_time": datetime.fromtimestamp(kline[6] / 1000, tz=timezone.utc),
            "quote_volume": float(kline[7]),
            "num_trades": int(kline[8]),
            "is_closed": True,
        }

    def _parse_ws_kline(self, data: Dict) -> Dict:
        """Parse WebSocket kline message to dict."""
        k = data["k"]
        return {
            "open_time": datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "close_time": datetime.fromtimestamp(k["T"] / 1000, tz=timezone.utc),
            "quote_volume": float(k["q"]),
            "num_trades": int(k["n"]),
            "is_closed": k["x"],
        }

    async def _ws_loop(self) -> None:
        """Main WebSocket loop with reconnection logic."""
        while self._running:
            try:
                logger.info(f"Connecting to {self.ws_url}")
                async with websockets.connect(self.ws_url) as ws:
                    self._ws = ws
                    self._connected = True
                    logger.info("WebSocket connected")

                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(message)

            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self._connected = False
                self._ws = None

            if self._running:
                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)

    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            if "e" not in data or data["e"] != "kline":
                return

            candle = self._parse_ws_kline(data)
            self._current_candle = candle

            # Call update callback (every tick)
            if self.on_candle_update:
                try:
                    if asyncio.iscoroutinefunction(self.on_candle_update):
                        await self.on_candle_update(candle)
                    else:
                        self.on_candle_update(candle)
                except Exception as e:
                    logger.error(f"Error in on_candle_update callback: {e}")

            # Handle candle close
            if candle["is_closed"]:
                # Add to buffer
                self._candle_buffer.append(candle)
                self._last_candle_time = candle["close_time"]

                logger.debug(
                    f"Candle closed: {candle['open_time']} "
                    f"O={candle['open']:.2f} H={candle['high']:.2f} "
                    f"L={candle['low']:.2f} C={candle['close']:.2f}"
                )

                # Call close callback
                if self.on_candle_close:
                    try:
                        if asyncio.iscoroutinefunction(self.on_candle_close):
                            await self.on_candle_close(candle)
                        else:
                            self.on_candle_close(candle)
                    except Exception as e:
                        logger.error(f"Error in on_candle_close callback: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def get_candle_buffer(self) -> pd.DataFrame:
        """
        Get the candle buffer as a DataFrame.

        Returns:
            DataFrame with OHLCV data indexed by open_time
        """
        if not self._candle_buffer:
            return pd.DataFrame()

        df = pd.DataFrame(list(self._candle_buffer))
        df.set_index("open_time", inplace=True)
        return df

    def get_latest_candle(self) -> Optional[Dict]:
        """Get the most recent closed candle."""
        if self._candle_buffer:
            return self._candle_buffer[-1]
        return None

    def get_current_candle(self) -> Optional[Dict]:
        """Get the current (potentially incomplete) candle."""
        return self._current_candle

    def get_current_price(self) -> Optional[float]:
        """Get the current price (last close)."""
        if self._current_candle:
            return self._current_candle["close"]
        elif self._candle_buffer:
            return self._candle_buffer[-1]["close"]
        return None

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected

    @property
    def buffer_length(self) -> int:
        """Number of candles in buffer."""
        return len(self._candle_buffer)


async def main():
    """Test the connector."""
    logging.basicConfig(level=logging.INFO)

    def on_close(candle):
        print(f"Candle closed: {candle['close_time']} Close={candle['close']:.2f}")

    connector = BinanceConnector(
        symbol="BTCUSDT",
        interval="1m",
        buffer_size=100,
        on_candle_close=on_close,
    )

    try:
        await connector.start()
    except KeyboardInterrupt:
        await connector.stop()


if __name__ == "__main__":
    asyncio.run(main())
