"""Shared State Manager for the trading dashboard.

Thread-safe state container bridging the trading bot with the web API.
Updated by the bot, read by FastAPI endpoints + WebSocket clients.

Model panels are UNIFIED: one ModelPanelData dataclass, one dict keyed by
track name. Adding a model to the dashboard is a single register_model()
call — no new dataclass, no new update methods.

Write path: register_model() / update_model() / add_model_trade()
(driven by engine.dashboard_bridge).
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
import threading


# =====================================================================
# Data models
# =====================================================================

@dataclass
class StatusData:
    """Current trading status + shared market indicators."""
    status: str = "STARTING"
    uptime_seconds: float = 0
    price: Optional[float] = None
    regime: Optional[str] = None
    bar_count: int = 0
    rsi: Optional[float] = None
    atr_percentile: Optional[float] = None
    ema_separation: Optional[float] = None
    last_update: Optional[str] = None


@dataclass
class TradeData:
    """Single trade record."""
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
    liq_price: float = 0.0


@dataclass
class ConfigData:
    """Trading configuration."""
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


@dataclass
class ModelPanelData:
    """ONE model's dashboard card — same shape for every model.

    Replaces the old MLData/MLAttnData/MLV3Data/RiskData/StatsData zoo.
    Field names carry no model prefix; the panel's identity is the dict
    key it is registered under.
    """
    # Identity / display (set once at register_model)
    name: str = ""
    title: str = ""
    accent_color: str = "#8892a0"
    exit_version: str = ""
    sort_order: int = 0
    model_loaded: bool = False

    # Money / risk
    wallet_usd: float = 0.0
    drawdown_pct: float = 0.0
    health_multiplier: float = 1.0
    consecutive_losses: int = 0
    recent_winrate: float = 1.0
    peak_usd: float = 0.0
    last_qty: float = 0.0
    last_pnl_usd: float = 0.0
    total_skips: int = 0

    # Performance stats
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_bps: float = 0.0
    avg_bps: float = 0.0
    profit_factor: float = 0.0

    # Open position
    has_position: bool = False
    position_side: Optional[str] = None
    position_pnl_bps: float = 0.0
    position_entry_price: float = 0.0
    position_trailing_stop_bps: float = 0.0
    position_highest_profit_bps: float = 0.0
    position_bars_held: int = 0
    position_max_bars: int = 10
    position_mfe_bps: float = 0.0
    position_mae_bps: float = 0.0
    position_liq_price: float = 0.0
    is_reentry: bool = False

    # Prediction display
    last_prob: float = 0.5
    long_pct: float = 50.0
    short_pct: float = 50.0
    signal_status: str = ""

    # Expected-vs-actual monitoring (baseline set at register_model)
    daily_expected_bps: float = 0.0
    days_elapsed: float = 0.0
    d7_actual_bps: float = 0.0
    d7_expected_bps: float = 0.0
    d7_delta_pct: float = 0.0
    d7_status: str = "pending"
    cum_actual_bps: float = 0.0
    cum_expected_bps: float = 0.0
    cum_delta_pct: float = 0.0
    cum_status: str = "pending"


# Models are registered dynamically by the orchestrator's bridge at
# startup (bridge.register). Pre-seed rows here only if the dashboard
# must show a model before the bot connects.
DEFAULT_MODELS = [
    # (name, title, accent, exit_version, daily_expected_bps)
]


class DashboardState:
    """Thread-safe shared state for the dashboard.

    Updated by the bot, read by FastAPI endpoints.
    Supports WebSocket broadcast to connected clients.
    """

    def __init__(self, price_throttle_ms: int = 500):
        self._lock = threading.Lock()
        self._status = StatusData()
        self._config = ConfigData()
        self._start_time: Optional[datetime] = None

        # Unified model panels + their trade lists, keyed by track name.
        self._models: Dict[str, ModelPanelData] = {}
        self._model_trades: Dict[str, List[TradeData]] = {}
        for i, (name, title, accent, exit_v, daily_exp) in enumerate(DEFAULT_MODELS):
            self.register_model(name, title=title, accent_color=accent,
                                exit_version=exit_v, daily_expected_bps=daily_exp,
                                sort_order=i, _broadcast=False)

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

    # ------------------------------------------------------------ session

    def start(self) -> None:
        with self._lock:
            self._start_time = datetime.now(timezone.utc)
            self._status.status = "RUNNING"

    def stop(self) -> None:
        with self._lock:
            self._status.status = "STOPPED"

    def get_session_start(self) -> Optional[datetime]:
        return self._start_time

    def set_config(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._config, k):
                    setattr(self._config, k, v)

    # ------------------------------------------------------------ models

    def register_model(
        self,
        name: str,
        title: str = "",
        accent_color: str = "#8892a0",
        exit_version: str = "",
        daily_expected_bps: float = 0.0,
        sort_order: int = 99,
        _broadcast: bool = True,
    ) -> None:
        """Create (or re-brand) a model panel. One call per track."""
        with self._lock:
            panel = self._models.get(name) or ModelPanelData(name=name)
            panel.title = title or name
            panel.accent_color = accent_color
            panel.exit_version = exit_version
            panel.daily_expected_bps = daily_expected_bps
            panel.sort_order = sort_order
            self._models[name] = panel
            self._model_trades.setdefault(name, [])
        if _broadcast:
            self._broadcast_update()

    def update_model(self, name: str, **kwargs) -> None:
        """Update any fields of one model's panel (unknown fields ignored)."""
        with self._lock:
            panel = self._models.get(name)
            if panel is None:
                return
            for k, v in kwargs.items():
                if hasattr(panel, k):
                    setattr(panel, k, v)
            self._refresh_monitoring(name)
        self._broadcast_update()

    def add_model_trade(self, name: str, trade: Dict[str, Any]) -> None:
        """Add a completed trade to one model's history (newest first)."""
        with self._lock:
            if name not in self._model_trades:
                return
            td = TradeData(
                trade_id=trade.get("trade_id", str(len(self._model_trades[name]) + 1)),
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
                liq_price=trade.get("liq_price", 0.0),
            )
            self._model_trades[name].insert(0, td)
            self._refresh_monitoring(name)
        self._broadcast_update()

    def get_model(self, name: str) -> Dict[str, Any]:
        with self._lock:
            panel = self._models.get(name)
            return asdict(panel) if panel else {}

    def get_models(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            ordered = sorted(self._models.values(), key=lambda p: p.sort_order)
            return {p.name: asdict(p) for p in ordered}

    def get_model_trades(self, name: str, limit: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            trades = self._model_trades.get(name, [])
            if limit > 0:
                trades = trades[:limit]
            return [asdict(t) for t in trades]

    # ---------------------------------------------------- monitoring math

    def _status_from_delta(self, delta_pct: float, pending: bool) -> str:
        if pending:
            return "pending"
        abs_p = abs(delta_pct)
        if abs_p <= 30:
            return "on_track"
        if abs_p <= 60:
            return "drift"
        return "diverged"

    def _sum_bps_since(self, trades: List[TradeData], cutoff: datetime) -> float:
        total = 0.0
        for t in trades:
            if not t.exit_time:
                continue
            try:
                ts = datetime.fromisoformat(t.exit_time.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                total += t.net_profit_bps or 0.0
        return total

    def _earliest_trade_time(self, trades: List[TradeData]) -> Optional[datetime]:
        earliest: Optional[datetime] = None
        for t in trades:
            if not t.exit_time:
                continue
            try:
                ts = datetime.fromisoformat(t.exit_time.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if earliest is None or ts < earliest:
                earliest = ts
        return earliest

    def _refresh_monitoring(self, name: str) -> None:
        """Recompute 7-day + cumulative drift vs backtest baseline.

        The ONE copy of what used to be three identical _refresh_*_monitoring
        methods. Caller must hold self._lock. No-op when the model has no
        baseline (daily_expected_bps == 0, e.g. the rule-based track).
        """
        panel = self._models.get(name)
        if panel is None or panel.daily_expected_bps <= 0:
            return
        trades = self._model_trades.get(name, [])

        now = datetime.now(timezone.utc)
        anchor = self._earliest_trade_time(trades) or self._start_time
        days_elapsed = ((now - anchor).total_seconds() / 86400.0) if anchor else 0.0
        days_elapsed = max(days_elapsed, 0.0)
        panel.days_elapsed = round(days_elapsed, 2)
        daily_exp = panel.daily_expected_bps

        # 7-day window
        cutoff_7d = now - timedelta(days=7)
        actual_7d = self._sum_bps_since(trades, cutoff_7d)
        window_days = min(days_elapsed, 7.0)
        expected_7d = window_days * daily_exp
        panel.d7_actual_bps = round(actual_7d, 1)
        panel.d7_expected_bps = round(expected_7d, 1)
        if expected_7d <= 0 or days_elapsed < 1.0:
            panel.d7_delta_pct = 0.0
            panel.d7_status = "pending"
        else:
            delta_7d = (actual_7d - expected_7d) / abs(expected_7d) * 100.0
            panel.d7_delta_pct = round(delta_7d, 1)
            panel.d7_status = self._status_from_delta(delta_7d, pending=False)

        # Cumulative since first trade
        actual_cum = panel.total_bps
        expected_cum = days_elapsed * daily_exp
        panel.cum_actual_bps = round(actual_cum, 1)
        panel.cum_expected_bps = round(expected_cum, 1)
        if expected_cum <= 0 or days_elapsed < 1.0:
            panel.cum_delta_pct = 0.0
            panel.cum_status = "pending"
        else:
            delta_cum = (actual_cum - expected_cum) / abs(expected_cum) * 100.0
            panel.cum_delta_pct = round(delta_cum, 1)
            panel.cum_status = self._status_from_delta(delta_cum, pending=False)

    # ------------------------------------------------------------ status

    def update_status(
        self,
        price: Optional[float] = None,
        regime: Optional[str] = None,
        bar_count: Optional[int] = None,
        rsi: Optional[float] = None,
        atr_percentile: Optional[float] = None,
        ema_separation: Optional[float] = None,
    ) -> None:
        """Update current status and shared indicators."""
        is_price_only = (
            price is not None and regime is None and
            bar_count is None and rsi is None
        )
        if is_price_only:
            now_ms = time.time() * 1000
            if now_ms - self._last_price_broadcast < self._price_throttle_ms:
                with self._lock:
                    self._status.price = price
                return
            self._last_price_broadcast = now_ms

        with self._lock:
            if price is not None:
                self._status.price = price
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

    # ------------------------------------------------------------ candles

    def set_candle_history(self, candles: List[Dict[str, Any]]) -> None:
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
        now_ms = time.time() * 1000
        throttled = now_ms - self._last_candle_broadcast < self._price_throttle_ms
        with self._lock:
            self._current_candle = CandleData(
                time=candle["time"], open=candle["open"], high=candle["high"],
                low=candle["low"], close=candle["close"], volume=candle["volume"],
            )
        if throttled:
            return
        self._last_candle_broadcast = now_ms
        self._broadcast_update(msg_type="candle_update")

    def close_candle(self, candle: Dict[str, Any]) -> None:
        with self._lock:
            cd = CandleData(
                time=candle["time"], open=candle["open"], high=candle["high"],
                low=candle["low"], close=candle["close"], volume=candle["volume"],
                sma=candle.get("sma"), rsi=candle.get("rsi"),
            )
            self._candles.append(cd)
            self._current_candle = None
        self._broadcast_update(msg_type="candle_close")

    # ------------------------------------------------------------ getters

    def get_candles(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "candles": [asdict(c) for c in self._candles],
                "current_candle": asdict(self._current_candle) if self._current_candle else None,
            }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return asdict(self._status)

    def get_config(self) -> Dict[str, Any]:
        with self._lock:
            return asdict(self._config)

    def get_all(self) -> Dict[str, Any]:
        """All dashboard data. Excludes candles (too large for WS broadcast)."""
        return {
            "type": "full",
            "status": self.get_status(),
            "config": self.get_config(),
            "models": self.get_models(),
            "model_trades": {
                name: self.get_model_trades(name, limit=0) for name in self.get_models()
            },
        }

    # ------------------------------------------------------------ websocket

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
