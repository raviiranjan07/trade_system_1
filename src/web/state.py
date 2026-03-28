"""
Shared State Manager for V1.3 Dashboard.

Thread-safe state container that bridges the V1.3 trading bot
with the web API endpoints. Updated by bot, read by API.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
import threading


@dataclass
class StatusData:
    """Current trading status + indicators."""
    status: str = "STARTING"
    uptime_seconds: float = 0
    price: Optional[float] = None
    regime: Optional[str] = None
    bar_count: int = 0
    # V1.2 indicators
    rsi: Optional[float] = None
    atr_percentile: Optional[float] = None
    ema_separation: Optional[float] = None
    last_update: Optional[str] = None


@dataclass
class PositionData:
    """Current position information (V1.2 trailing stop model)."""
    has_position: bool = False
    side: Optional[str] = None
    entry_price: Optional[float] = None
    current_price: Optional[float] = None
    # V1.2 specific
    trailing_stop_bps: float = 0
    highest_profit_bps: float = 0
    current_pnl_bps: float = 0
    bars_held: int = 0
    max_bars: int = 10
    mfe_bps: float = 0
    mae_bps: float = 0
    is_reentry: bool = False


@dataclass
class StatsData:
    """Session statistics (bps-based for V1.2)."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0
    total_bps: float = 0
    avg_bps: float = 0
    profit_factor: float = 0
    config_hash: str = ""


@dataclass
class TradeData:
    """Single trade record (V1.2 format)."""
    trade_id: str = ""
    direction: str = ""
    signal_type: str = ""
    entry_price: float = 0
    exit_price: float = 0
    net_profit_bps: float = 0
    mfe_bps: float = 0
    mae_bps: float = 0
    exit_bar: int = 0
    exit_reason: str = ""
    is_reentry: bool = False
    exit_time: Optional[str] = None
    entry_time: Optional[str] = None
    qty: float = 0.0
    pnl_usd: float = 0.0


@dataclass
class SignalData:
    """Single signal record (V1.2 format)."""
    id: str = ""
    action: str = "SKIP"  # ENTRY, REENTRY, or SKIP
    direction: Optional[str] = None
    rsi: float = 0
    atr_ok: Optional[bool] = None
    ema_ok: Optional[bool] = None
    reason: Optional[str] = None
    time: str = ""


@dataclass
class RiskData:
    """Risk management state for dashboard display."""
    wallet_usd: float = 0.0
    drawdown_pct: float = 0.0
    health_multiplier: float = 1.0
    consecutive_losses: int = 0
    recent_winrate: float = 1.0
    peak_usd: float = 0.0
    last_qty: float = 0.0
    last_pnl_usd: float = 0.0
    total_skips: int = 0


@dataclass
class MLData:
    """ML model state for dashboard display."""
    ml_wallet_usd: float = 0.0
    ml_total_trades: int = 0
    ml_wins: int = 0
    ml_losses: int = 0
    ml_win_rate: float = 0.0
    ml_total_bps: float = 0.0
    ml_avg_bps: float = 0.0
    ml_has_position: bool = False
    ml_position_side: Optional[str] = None
    ml_position_pnl_bps: float = 0.0
    ml_last_prob: float = 0.5
    ml_model_loaded: bool = False
    ml_drawdown_pct: float = 0.0
    ml_health_multiplier: float = 1.0
    ml_consecutive_losses: int = 0
    ml_recent_winrate: float = 1.0
    ml_peak_usd: float = 0.0
    ml_last_pnl_usd: float = 0.0
    ml_last_qty: float = 0.0
    ml_total_skips: int = 0


@dataclass
class ConfigData:
    """V1.2 trading configuration."""
    pair: str = "BTCUSDT"
    timeframe: str = "15m"
    mode: str = "paper"
    trailing_stop_long: int = 20
    trailing_stop_short: int = 30
    max_bars: int = 10
    reentry_enabled: bool = True
    config_hash: str = ""


@dataclass
class CandleData:
    """Single candle for chart display."""
    time: int = 0           # Unix timestamp in SECONDS (LWC requirement)
    open: float = 0
    high: float = 0
    low: float = 0
    close: float = 0
    volume: float = 0
    sma: Optional[float] = None
    rsi: Optional[float] = None


class DashboardState:
    """
    Thread-safe shared state for V1.2 dashboard data.

    Updated by the V1.2 bot, read by FastAPI endpoints.
    Supports WebSocket broadcast to connected clients.
    """

    def __init__(self, price_throttle_ms: int = 500):
        self._lock = threading.Lock()
        self._status = StatusData()
        self._position = PositionData()
        self._stats = StatsData()
        self._config = ConfigData()
        self._risk = RiskData()
        self._ml = MLData()
        self._ml_trades: List[TradeData] = []
        self._trades: List[TradeData] = []
        self._signals: List[SignalData] = []
        self._start_time: Optional[datetime] = None

        # Signal proximity data (V1.3)
        self._proximity: Dict[str, Any] = {}

        # Chart candle data
        self._candles: List[CandleData] = []
        self._current_candle: Optional[CandleData] = None

        # WebSocket clients
        self._ws_clients: List[asyncio.Queue] = []
        self._ws_lock = threading.Lock()

        # Throttling for real-time price updates
        self._price_throttle_ms = price_throttle_ms
        self._last_price_broadcast: float = 0
        self._last_candle_broadcast: float = 0

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
        timeframe: str = "15m",
        mode: str = "paper",
        trailing_stop_long: int = 20,
        trailing_stop_short: int = 30,
        max_bars: int = 10,
        reentry_enabled: bool = True,
        config_hash: str = "",
    ) -> None:
        """Set V1.2 trading configuration."""
        with self._lock:
            self._config = ConfigData(
                pair=pair,
                timeframe=timeframe,
                mode=mode,
                trailing_stop_long=trailing_stop_long,
                trailing_stop_short=trailing_stop_short,
                max_bars=max_bars,
                reentry_enabled=reentry_enabled,
                config_hash=config_hash,
            )

    def update_status(
        self,
        price: Optional[float] = None,
        regime: Optional[str] = None,
        bar_count: Optional[int] = None,
        rsi: Optional[float] = None,
        atr_percentile: Optional[float] = None,
        ema_separation: Optional[float] = None,
    ) -> None:
        """Update current status and indicators."""
        is_price_only = (
            price is not None and
            regime is None and
            bar_count is None and
            rsi is None
        )

        if is_price_only:
            now_ms = time.time() * 1000
            if now_ms - self._last_price_broadcast < self._price_throttle_ms:
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
            if rsi is not None:
                self._status.rsi = round(rsi, 1)
            if atr_percentile is not None:
                self._status.atr_percentile = round(atr_percentile, 1)
            if ema_separation is not None:
                self._status.ema_separation = round(ema_separation, 4)

            if self._start_time:
                delta = datetime.now(timezone.utc) - self._start_time
                self._status.uptime_seconds = delta.total_seconds()

            self._status.last_update = datetime.now(timezone.utc).isoformat()

        self._broadcast_update()

    def update_position(
        self,
        has_position: bool = False,
        side: Optional[str] = None,
        entry_price: Optional[float] = None,
        trailing_stop_bps: float = 0,
        highest_profit_bps: float = 0,
        current_pnl_bps: float = 0,
        bars_held: int = 0,
        max_bars: int = 10,
        mfe_bps: float = 0,
        mae_bps: float = 0,
        is_reentry: bool = False,
    ) -> None:
        """Update position information."""
        with self._lock:
            self._position.has_position = has_position
            self._position.side = side
            self._position.entry_price = entry_price
            self._position.trailing_stop_bps = trailing_stop_bps
            self._position.highest_profit_bps = highest_profit_bps
            self._position.current_pnl_bps = current_pnl_bps
            self._position.bars_held = bars_held
            self._position.max_bars = max_bars
            self._position.mfe_bps = mfe_bps
            self._position.mae_bps = mae_bps
            self._position.is_reentry = is_reentry

        self._broadcast_update()

    def clear_position(self) -> None:
        """Clear position (after exit)."""
        with self._lock:
            self._position = PositionData()
        self._broadcast_update()

    def update_stats(
        self,
        total_trades: Optional[int] = None,
        wins: Optional[int] = None,
        losses: Optional[int] = None,
        win_rate: Optional[float] = None,
        total_bps: Optional[float] = None,
        avg_bps: Optional[float] = None,
        profit_factor: Optional[float] = None,
    ) -> None:
        """Update session statistics."""
        with self._lock:
            if total_trades is not None:
                self._stats.total_trades = total_trades
            if wins is not None:
                self._stats.wins = wins
            if losses is not None:
                self._stats.losses = losses
            if win_rate is not None:
                self._stats.win_rate = win_rate
            if total_bps is not None:
                self._stats.total_bps = total_bps
            if avg_bps is not None:
                self._stats.avg_bps = avg_bps
            if profit_factor is not None:
                self._stats.profit_factor = profit_factor

        self._broadcast_update()

    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Add a completed trade to history."""
        with self._lock:
            trade_data = TradeData(
                trade_id=trade.get("trade_id", ""),
                direction=trade.get("direction", ""),
                signal_type=trade.get("signal_type", ""),
                entry_price=trade.get("entry_price", 0),
                exit_price=trade.get("exit_price", 0),
                net_profit_bps=trade.get("net_profit_bps", 0),
                mfe_bps=trade.get("mfe_bps", 0),
                mae_bps=trade.get("mae_bps", 0),
                exit_bar=trade.get("exit_bar", 0),
                exit_reason=trade.get("exit_reason", ""),
                is_reentry=trade.get("is_reentry", False),
                exit_time=trade.get("exit_time", ""),
                entry_time=trade.get("entry_time", ""),
                qty=trade.get("qty", 0.0),
                pnl_usd=trade.get("pnl_usd", 0.0),
            )
            self._trades.insert(0, trade_data)
            self._trades = self._trades[:50]

        self._broadcast_update()

    def add_signal(self, signal: Dict[str, Any]) -> None:
        """Add a signal to the log."""
        import uuid
        with self._lock:
            signal_data = SignalData(
                id=signal.get("id", str(uuid.uuid4())[:8]),
                action=signal.get("action", "SKIP"),
                direction=signal.get("direction"),
                rsi=signal.get("rsi", 0),
                atr_ok=signal.get("atr_ok"),
                ema_ok=signal.get("ema_ok"),
                reason=signal.get("reason"),
                time=signal.get("time", datetime.now(timezone.utc).isoformat()),
            )
            self._signals.insert(0, signal_data)
            self._signals = self._signals[:50]

        self._broadcast_update()

    def set_candle_history(self, candles: List[Dict[str, Any]]) -> None:
        """Set initial candle history for chart (called once on startup)."""
        with self._lock:
            self._candles = [
                CandleData(
                    time=c["time"], open=c["open"], high=c["high"],
                    low=c["low"], close=c["close"], volume=c["volume"],
                    sma=c.get("sma"), rsi=c.get("rsi"),
                )
                for c in candles
            ]

    def update_current_candle(self, candle: Dict[str, Any]) -> None:
        """Update the live (incomplete) candle on each tick."""
        now_ms = time.time() * 1000
        if now_ms - self._last_candle_broadcast < self._price_throttle_ms:
            with self._lock:
                self._current_candle = CandleData(
                    time=candle["time"], open=candle["open"], high=candle["high"],
                    low=candle["low"], close=candle["close"], volume=candle["volume"],
                )
            return
        self._last_candle_broadcast = now_ms

        with self._lock:
            self._current_candle = CandleData(
                time=candle["time"], open=candle["open"], high=candle["high"],
                low=candle["low"], close=candle["close"], volume=candle["volume"],
            )
        self._broadcast_update(msg_type="candle_update")

    def close_candle(self, candle: Dict[str, Any]) -> None:
        """Append a closed candle with indicators to history."""
        with self._lock:
            cd = CandleData(
                time=candle["time"], open=candle["open"], high=candle["high"],
                low=candle["low"], close=candle["close"], volume=candle["volume"],
                sma=candle.get("sma"), rsi=candle.get("rsi"),
            )
            self._candles.append(cd)
            self._current_candle = None
        self._broadcast_update(msg_type="candle_close")

    def get_candles(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "candles": [asdict(c) for c in self._candles],
                "current_candle": asdict(self._current_candle) if self._current_candle else None,
            }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return asdict(self._status)

    def get_position(self) -> Dict[str, Any]:
        with self._lock:
            return asdict(self._position)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return asdict(self._stats)

    def get_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return [asdict(t) for t in self._trades[:limit]]

    def get_signals(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return [asdict(s) for s in self._signals[:limit]]

    def get_config(self) -> Dict[str, Any]:
        with self._lock:
            return asdict(self._config)

    def update_proximity(self, proximity: Dict[str, Any]) -> None:
        """Update signal proximity data (V1.3)."""
        with self._lock:
            self._proximity = proximity

    def get_proximity(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._proximity)

    def update_risk(
        self,
        wallet_usd: float = 0.0,
        drawdown_pct: float = 0.0,
        health_multiplier: float = 1.0,
        consecutive_losses: int = 0,
        recent_winrate: float = 1.0,
        peak_usd: float = 0.0,
        last_qty: float = 0.0,
        last_pnl_usd: float = 0.0,
        total_skips: int = 0,
    ) -> None:
        """Update risk management state."""
        with self._lock:
            self._risk = RiskData(
                wallet_usd=round(wallet_usd, 2),
                drawdown_pct=round(drawdown_pct, 4),
                health_multiplier=round(health_multiplier, 2),
                consecutive_losses=consecutive_losses,
                recent_winrate=round(recent_winrate, 2),
                peak_usd=round(peak_usd, 2),
                last_qty=last_qty,
                last_pnl_usd=round(last_pnl_usd, 4),
                total_skips=total_skips,
            )
        self._broadcast_update()

    def get_risk(self) -> Dict[str, Any]:
        with self._lock:
            return asdict(self._risk)

    def update_ml(self, **kwargs) -> None:
        """Update ML model state."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._ml, k):
                    setattr(self._ml, k, v)
        self._broadcast_update()

    def get_ml(self) -> Dict[str, Any]:
        with self._lock:
            return asdict(self._ml)

    def add_ml_trade(self, trade_dict: Dict[str, Any]) -> None:
        """Add a completed ML trade."""
        with self._lock:
            td = TradeData(
                trade_id=str(len(self._ml_trades) + 1),
                direction=trade_dict.get("direction", ""),
                signal_type=trade_dict.get("signal_type", ""),
                entry_price=trade_dict.get("entry_price", 0),
                exit_price=trade_dict.get("exit_price", 0),
                net_profit_bps=trade_dict.get("net_profit_bps", 0),
                mfe_bps=trade_dict.get("mfe_bps", 0),
                mae_bps=trade_dict.get("mae_bps", 0),
                exit_bar=trade_dict.get("exit_bar", 0),
                exit_reason=trade_dict.get("exit_reason", ""),
                is_reentry=trade_dict.get("is_reentry", False),
                exit_time=trade_dict.get("exit_time", ""),
                entry_time=trade_dict.get("entry_time", ""),
            )
            self._ml_trades.insert(0, td)
            if len(self._ml_trades) > 100:
                self._ml_trades = self._ml_trades[:100]
        self._broadcast_update()

    def get_ml_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            return [asdict(t) for t in self._ml_trades[:limit]]

    def get_all(self) -> Dict[str, Any]:
        """Get all dashboard data. Excludes candles (too large for WS broadcast)."""
        return {
            "type": "full",
            "status": self.get_status(),
            "position": self.get_position(),
            "stats": self.get_stats(),
            "trades": self.get_trades(limit=50),
            "signals": self.get_signals(),
            "config": self.get_config(),
            "proximity": self.get_proximity(),
            "risk": self.get_risk(),
            "ml": self.get_ml(),
            "ml_trades": self.get_ml_trades(limit=20),
        }

    def register_ws_client(self, queue: asyncio.Queue) -> None:
        with self._ws_lock:
            self._ws_clients.append(queue)

    def unregister_ws_client(self, queue: asyncio.Queue) -> None:
        with self._ws_lock:
            if queue in self._ws_clients:
                self._ws_clients.remove(queue)

    def _broadcast_update(self, msg_type: str = "full") -> None:
        if msg_type == "candle_update":
            with self._lock:
                data = {
                    "type": "candle_update",
                    "candle": asdict(self._current_candle) if self._current_candle else None,
                }
        elif msg_type == "candle_close":
            with self._lock:
                data = {
                    "type": "candle_close",
                    "candle": asdict(self._candles[-1]) if self._candles else None,
                }
        else:
            data = self.get_all()

        with self._ws_lock:
            for queue in self._ws_clients:
                try:
                    queue.put_nowait(data)
                except asyncio.QueueFull:
                    pass


# Global state instance
dashboard_state = DashboardState()
