"""
Shared State Manager for Web Dashboard.

Thread-safe state container that bridges the trading orchestrator
with the web API endpoints. Updated by orchestrator, read by API.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
import threading


@dataclass
class StatusData:
    """Current trading status."""
    status: str = "STARTING"
    uptime_seconds: float = 0
    price: Optional[float] = None
    regime: Optional[str] = None
    bar_count: int = 0
    next_check_in: int = 0
    last_update: Optional[str] = None


@dataclass
class PositionData:
    """Current position information."""
    has_position: bool = False
    side: Optional[str] = None
    size_btc: float = 0
    entry_price: Optional[float] = None
    current_price: Optional[float] = None
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    unrealized_pnl: float = 0
    unrealized_pnl_pct: float = 0


@dataclass
class StatsData:
    """Session statistics."""
    capital: float = 0
    initial_capital: float = 0
    total_pnl: float = 0
    total_pnl_pct: float = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0
    total_commission: float = 0


@dataclass
class TradeData:
    """Single trade record."""
    trade_id: str = ""
    side: str = ""
    entry_price: float = 0
    exit_price: Optional[float] = None
    entry_time: str = ""
    exit_time: Optional[str] = None
    pnl: float = 0
    pnl_pct: float = 0
    exit_reason: Optional[str] = None
    status: str = "OPEN"


@dataclass
class SignalData:
    """Single signal record."""
    id: str = ""
    action: str = "SKIP"  # TRADE or SKIP
    side: Optional[str] = None
    expectancy: float = 0
    distance: Optional[float] = None
    reason: Optional[str] = None
    regime: Optional[str] = None
    time: str = ""


@dataclass
class ConfigData:
    """Trading configuration."""
    pair: str = "BTCUSDT"
    horizon: int = 5
    sample_interval: int = 15
    min_expectancy: float = 0.001
    initial_capital: float = 5.0


class DashboardState:
    """
    Thread-safe shared state for dashboard data.

    Updated by the trading orchestrator, read by FastAPI endpoints.
    Supports WebSocket broadcast to connected clients.
    """

    def __init__(self, price_throttle_ms: int = 500):
        self._lock = threading.Lock()
        self._status = StatusData()
        self._position = PositionData()
        self._stats = StatsData()
        self._config = ConfigData()
        self._trades: List[TradeData] = []
        self._signals: List[SignalData] = []
        self._start_time: Optional[datetime] = None

        # WebSocket clients
        self._ws_clients: List[asyncio.Queue] = []
        self._ws_lock = threading.Lock()

        # Throttling for real-time price updates
        self._price_throttle_ms = price_throttle_ms
        self._last_price_broadcast: float = 0

    def start(self) -> None:
        """Mark trading session as started."""
        with self._lock:
            self._start_time = datetime.now(timezone.utc)
            self._status.status = "RUNNING"

    def stop(self) -> None:
        """Mark trading session as stopped."""
        with self._lock:
            self._status.status = "STOPPED"

    def set_config(
        self,
        pair: str = "BTCUSDT",
        horizon: int = 5,
        sample_interval: int = 15,
        min_expectancy: float = 0.001,
        initial_capital: float = 5.0,
    ) -> None:
        """Set trading configuration."""
        with self._lock:
            self._config = ConfigData(
                pair=pair,
                horizon=horizon,
                sample_interval=sample_interval,
                min_expectancy=min_expectancy,
                initial_capital=initial_capital,
            )
            self._stats.initial_capital = initial_capital
            self._stats.capital = initial_capital

    def update_status(
        self,
        price: Optional[float] = None,
        regime: Optional[str] = None,
        bar_count: Optional[int] = None,
        next_check_in: Optional[int] = None,
    ) -> None:
        """Update current status."""
        # Check if this is a price-only update (real-time tick)
        is_price_only = (
            price is not None and
            regime is None and
            bar_count is None and
            next_check_in is None
        )

        # Throttle price-only updates to avoid overwhelming WebSocket
        if is_price_only:
            now_ms = time.time() * 1000
            if now_ms - self._last_price_broadcast < self._price_throttle_ms:
                # Still update price internally but don't broadcast
                with self._lock:
                    self._status.price = price
                    self._position.current_price = price
                return
            self._last_price_broadcast = now_ms

        with self._lock:
            if price is not None:
                self._status.price = price
                self._position.current_price = price
            if regime is not None:
                self._status.regime = regime
            if bar_count is not None:
                self._status.bar_count = bar_count
            if next_check_in is not None:
                self._status.next_check_in = next_check_in

            # Always update uptime on every broadcast
            if self._start_time:
                delta = datetime.now(timezone.utc) - self._start_time
                self._status.uptime_seconds = delta.total_seconds()

            self._status.last_update = datetime.now(timezone.utc).isoformat()

        # Broadcast to WebSocket clients
        self._broadcast_update()

    def update_position(
        self,
        has_position: bool = False,
        side: Optional[str] = None,
        size_btc: float = 0,
        entry_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        unrealized_pnl: float = 0,
        unrealized_pnl_pct: float = 0,
    ) -> None:
        """Update position information."""
        with self._lock:
            self._position.has_position = has_position
            self._position.side = side
            self._position.size_btc = size_btc
            self._position.entry_price = entry_price
            self._position.tp_price = tp_price
            self._position.sl_price = sl_price
            self._position.unrealized_pnl = unrealized_pnl
            self._position.unrealized_pnl_pct = unrealized_pnl_pct

        self._broadcast_update()

    def update_stats(
        self,
        capital: Optional[float] = None,
        total_pnl: Optional[float] = None,
        total_pnl_pct: Optional[float] = None,
        total_trades: Optional[int] = None,
        wins: Optional[int] = None,
        losses: Optional[int] = None,
        win_rate: Optional[float] = None,
        total_commission: Optional[float] = None,
    ) -> None:
        """Update session statistics."""
        with self._lock:
            if capital is not None:
                self._stats.capital = capital
            if total_pnl is not None:
                self._stats.total_pnl = total_pnl
            if total_pnl_pct is not None:
                self._stats.total_pnl_pct = total_pnl_pct
            if total_trades is not None:
                self._stats.total_trades = total_trades
            if wins is not None:
                self._stats.wins = wins
            if losses is not None:
                self._stats.losses = losses
            if win_rate is not None:
                self._stats.win_rate = win_rate
            if total_commission is not None:
                self._stats.total_commission = total_commission

        self._broadcast_update()

    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Add a completed trade to history."""
        with self._lock:
            trade_data = TradeData(
                trade_id=trade.get("trade_id", ""),
                side=trade.get("side", ""),
                entry_price=trade.get("entry_price", 0),
                exit_price=trade.get("exit_price"),
                entry_time=trade.get("entry_time", ""),
                exit_time=trade.get("exit_time"),
                pnl=trade.get("pnl", 0),
                pnl_pct=trade.get("pnl_pct", 0),
                exit_reason=trade.get("exit_reason"),
                status=trade.get("status", "CLOSED"),
            )
            self._trades.insert(0, trade_data)
            # Keep last 50 trades
            self._trades = self._trades[:50]

        self._broadcast_update()

    def add_signal(self, signal: Dict[str, Any]) -> None:
        """Add a signal to the log."""
        import uuid
        with self._lock:
            signal_data = SignalData(
                id=signal.get("id", str(uuid.uuid4())[:8]),
                action=signal.get("action", "SKIP"),
                side=signal.get("side"),
                expectancy=signal.get("expectancy", 0),
                distance=signal.get("distance"),
                reason=signal.get("reason"),
                regime=signal.get("regime"),
                time=signal.get("time", datetime.now(timezone.utc).isoformat()),
            )
            self._signals.insert(0, signal_data)
            # Keep last 50 signals
            self._signals = self._signals[:50]

        self._broadcast_update()

    def get_status(self) -> Dict[str, Any]:
        """Get current status as dict."""
        with self._lock:
            return asdict(self._status)

    def get_position(self) -> Dict[str, Any]:
        """Get current position as dict."""
        with self._lock:
            return asdict(self._position)

    def get_stats(self) -> Dict[str, Any]:
        """Get session stats as dict."""
        with self._lock:
            return asdict(self._stats)

    def get_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades as list of dicts."""
        with self._lock:
            return [asdict(t) for t in self._trades[:limit]]

    def get_signals(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent signals as list of dicts."""
        with self._lock:
            return [asdict(s) for s in self._signals[:limit]]

    def get_config(self) -> Dict[str, Any]:
        """Get trading config as dict."""
        with self._lock:
            return asdict(self._config)

    def get_all(self) -> Dict[str, Any]:
        """Get all dashboard data."""
        return {
            "status": self.get_status(),
            "position": self.get_position(),
            "stats": self.get_stats(),
            "trades": self.get_trades(),
            "signals": self.get_signals(),
            "config": self.get_config(),
        }

    def register_ws_client(self, queue: asyncio.Queue) -> None:
        """Register a WebSocket client for updates."""
        with self._ws_lock:
            self._ws_clients.append(queue)

    def unregister_ws_client(self, queue: asyncio.Queue) -> None:
        """Unregister a WebSocket client."""
        with self._ws_lock:
            if queue in self._ws_clients:
                self._ws_clients.remove(queue)

    def _broadcast_update(self) -> None:
        """Broadcast update to all WebSocket clients."""
        data = self.get_all()
        with self._ws_lock:
            for queue in self._ws_clients:
                try:
                    queue.put_nowait(data)
                except asyncio.QueueFull:
                    pass  # Skip if client is slow


# Global state instance
dashboard_state = DashboardState()
