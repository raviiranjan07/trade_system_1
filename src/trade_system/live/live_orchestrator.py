"""
Live Trading Orchestrator.

Coordinates all components for paper/live trading:
- Binance WebSocket for real-time data
- State builder for feature computation
- Similarity engine for pattern matching
- Decision engine for trade signals
- Paper executor for simulated execution
"""

import asyncio
import logging
import signal
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict
import sys

import pandas as pd

from trade_system.config import get_config
from trade_system.similarity.similarity_engine import SimilarityEngine
from trade_system.decision.decision_engine import DecisionEngine

from .binance_connector import BinanceConnector
from .state_builder import RealtimeStateBuilder
from .paper_executor import PaperExecutor
from .position_manager import PositionManager

# Web dashboard imports
try:
    from trade_system.web.state import DashboardState
    from trade_system.web.server import start_server_background
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False


logger = logging.getLogger(__name__)


class LiveOrchestrator:
    """
    Main orchestrator for paper/live trading.

    Coordinates the trading loop:
    1. Receive candles from Binance WebSocket
    2. Update state builder with new data
    3. Every sample_interval bars, generate trading decision
    4. Execute paper trades based on signals
    5. Monitor and manage open positions

    All settings are read from config/config.yaml unless overridden.

    Usage:
        orchestrator = LiveOrchestrator()
        await orchestrator.start()
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        capital: Optional[float] = None,
        horizon: Optional[int] = None,
        sample_interval: Optional[int] = None,
        min_expectancy: Optional[float] = None,
        max_distance: Optional[float] = None,
        blocked_regimes: Optional[list] = None,
        web_port: int = 8080,
    ):
        """
        Initialize the orchestrator.

        All parameters are optional - defaults are read from config/config.yaml.
        Command-line arguments can override config values.

        Args:
            config_path: Path to config file (uses default if None)
            capital: Starting capital in USD (default: from config)
            horizon: Outcome horizon in minutes (default: from config)
            sample_interval: Check for signals every N bars (default: from config)
            min_expectancy: Minimum expectancy to take trade (default: from config)
            max_distance: Maximum distance for similarity matching (default: from config)
            blocked_regimes: List of regimes to avoid (default: from config)
            web_port: Port for web dashboard (default: 8080)
        """
        self.config = get_config(config_path)

        # Trading parameters - read from config, allow override
        self.capital = capital if capital is not None else self.config.get("decision.capital", 200.0)
        self.horizon = horizon if horizon is not None else self.config.get("similarity.default_horizon", 5)
        self.sample_interval = sample_interval if sample_interval is not None else self.config.get("backtest.sample_interval", 15)
        self.min_expectancy = min_expectancy if min_expectancy is not None else self.config.get("decision.min_expectancy", 0.001)
        self.max_distance = max_distance if max_distance is not None else self.config.get("decision.max_distance", 3.0)
        self.blocked_regimes = blocked_regimes if blocked_regimes is not None else self.config.get("decision.blocked_regimes", [])

        # Components (initialized in start())
        self.connector: Optional[BinanceConnector] = None
        self.state_builder: Optional[RealtimeStateBuilder] = None
        self.similarity_engine: Optional[SimilarityEngine] = None
        self.decision_engine: Optional[DecisionEngine] = None
        self.executor: Optional[PaperExecutor] = None
        self.position_manager: Optional[PositionManager] = None

        # Web dashboard
        self.web_port = web_port
        self.web_state: Optional[DashboardState] = None
        self._web_server_task: Optional[asyncio.Task] = None

        # State
        self._running = False
        self._bar_count = 0
        self._last_signal_time: Optional[datetime] = None
        self._start_time: Optional[datetime] = None
        self._current_regime: Optional[str] = None

        # Data paths
        self.data_dir = Path(self.config.get("paths.data_dir", "data"))
        self.session_file = self.data_dir / "paper_trading" / "session_state.json"

        # Trading pair for dashboard
        self._pair = self.config.get("data.pair", "BTCUSDT")

    async def start(self) -> None:
        """Start the paper trading system."""
        logger.info("=" * 60)
        logger.info("STARTING PAPER TRADING SYSTEM")
        logger.info("=" * 60)

        self._start_time = datetime.now(timezone.utc)
        self._running = True

        try:
            # Initialize components
            await self._initialize_components()

            # Initialize web dashboard
            if self.web_port and WEB_AVAILABLE:
                await self._start_web_dashboard()

            # Set up signal handlers
            self._setup_signal_handlers()

            # Start main loop
            await self._run_main_loop()

        except Exception as e:
            logger.error(f"Error in paper trading: {e}")
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the paper trading system gracefully."""
        logger.info("Stopping paper trading system...")
        self._running = False

        # Stop web dashboard
        if self.web_state:
            self.web_state.stop()
        if self._web_server_task:
            self._web_server_task.cancel()
            try:
                await self._web_server_task
            except asyncio.CancelledError:
                pass

        # Save session state
        if self.position_manager:
            self.position_manager.save_session()

        # Close WebSocket
        if self.connector:
            await self.connector.stop()

        # Print final summary
        self._print_final_summary()

        logger.info("Paper trading stopped")

    async def _initialize_components(self) -> None:
        """Initialize all trading components."""
        logger.info("Initializing components...")

        # Get trading pair and timeframe from config
        pair = self.config.get("data.pair", "BTCUSDT")
        timeframe = self.config.get("data.timeframe", "1m")

        # 1. Load similarity engine from saved data
        logger.info("  Loading similarity engine...")
        outcome_path = self.data_dir / "outcomes" / f"{pair}_{timeframe}_outcomes.parquet"
        regime_path = self.data_dir / "regimes" / f"{pair}_{timeframe}_regimes.parquet"

        if not outcome_path.exists():
            raise FileNotFoundError(f"Outcome data not found: {outcome_path}")
        if not regime_path.exists():
            raise FileNotFoundError(f"Regime data not found: {regime_path}")

        outcome_df = pd.read_parquet(outcome_path)
        regime_df = pd.read_parquet(regime_path)

        self.similarity_engine = SimilarityEngine(
            outcome_df=outcome_df,
            regime_df=regime_df,
            k=self.config.get("similarity.k", 200),
            backend=self.config.get("similarity.backend", "bruteforce"),
        )
        logger.info(f"    Loaded {len(outcome_df):,} training samples")

        # 2. Initialize decision engine
        logger.info("  Initializing decision engine...")
        self.decision_engine = DecisionEngine(
            capital=self.capital,
            risk_per_trade=self.config.get("decision.risk_per_trade", 0.005),
            min_expectancy=self.min_expectancy,
            max_distance=self.max_distance,
            blocked_regimes=self.blocked_regimes,
            min_mfe=self.config.get("decision.min_mfe", 0.0),
            max_leverage=self.config.get("decision.max_leverage", 1.0),
        )

        # 3. Initialize paper executor
        logger.info("  Initializing paper executor...")
        self.executor = PaperExecutor(
            capital=self.capital,
            slippage_pct=self.config.get("backtest.slippage_pct", 0.0005),
            commission_pct=self.config.get("backtest.commission_pct", 0.0004),
        )

        # 4. Initialize position manager
        self.position_manager = PositionManager(
            executor=self.executor,
            max_bars_in_trade=self.config.get("backtest.max_bars_in_trade", 120),
            session_file=self.session_file,
        )

        # Try to load previous session
        if self.position_manager.load_session():
            logger.info("    Resumed from previous session")

        # 5. Initialize Binance connector
        logger.info("  Connecting to Binance...")
        buffer_size = self.config.get("normalization.window", 2000) + 500
        self.connector = BinanceConnector(
            symbol=pair,
            interval=timeframe,
            buffer_size=buffer_size,
            on_candle_close=self._on_candle_close,
            on_candle_update=self._on_candle_update,
        )

        # 6. Initialize state builder (will be done after connector gets historical data)
        logger.info("  State builder will initialize after historical data fetch...")

        logger.info(f"Components initialized for {pair} {timeframe}")

    def _setup_signal_handlers(self) -> None:
        """Set up graceful shutdown handlers."""
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._running = False
            # Also stop the connector to break the WebSocket loop
            if self.connector:
                self.connector._running = False
                if self.connector._ws:
                    # Schedule the WebSocket close in the event loop
                    asyncio.get_event_loop().call_soon_threadsafe(
                        lambda: asyncio.create_task(self.connector._ws.close())
                    )

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    async def _run_main_loop(self) -> None:
        """Run the main trading loop."""
        logger.info("Starting WebSocket connection...")

        # Start connector (this will fetch historical data and begin streaming)
        await self.connector.start()

    async def _on_candle_update(self, candle: Dict) -> None:
        """Called on every tick (real-time price update)."""
        if not self._running:
            return

        current_price = candle["close"]

        # Update web dashboard with real-time price (throttled)
        if self.web_state:
            self.web_state.update_status(price=current_price)

    async def _on_candle_close(self, candle: Dict) -> None:
        """Called when a candle closes."""
        if not self._running:
            return

        self._bar_count += 1
        current_price = candle["close"]

        # Initialize state builder on first candle (after historical data is loaded)
        if self.state_builder is None:
            await self._initialize_state_builder()
            if self.state_builder is None:
                return

        # Update state builder
        try:
            self.state_builder.update(candle)
            self._current_regime = self.state_builder.get_current_regime()
        except Exception as e:
            logger.error(f"Error updating state: {e}")
            return

        # Check position exit conditions
        if self.position_manager.has_position:
            trade = self.position_manager.on_new_bar(current_price)
            if trade:
                self._log_trade_closed(trade)

        # Check for new signals at sample interval
        if self._bar_count % self.sample_interval == 0:
            await self._check_for_signal(candle)

        # Update dashboard
        self._update_dashboard(current_price)

        # Log status periodically
        if self._bar_count % 60 == 0:  # Every hour
            self._log_status(current_price)

    async def _start_web_dashboard(self) -> None:
        """Start the web dashboard server."""
        if not WEB_AVAILABLE:
            logger.warning("Web dashboard not available (install fastapi uvicorn)")
            return

        self.web_state = DashboardState()
        self.web_state.set_config(
            pair=self._pair,
            horizon=self.horizon,
            sample_interval=self.sample_interval,
            min_expectancy=self.min_expectancy,
            initial_capital=self.capital,
        )
        self.web_state.start()

        # Start server as background task
        self._web_server_task = start_server_background(
            host="0.0.0.0",
            port=self.web_port,
            state=self.web_state,
        )

        logger.info(f"Web dashboard available at http://0.0.0.0:{self.web_port}")

    async def _initialize_state_builder(self) -> None:
        """Initialize state builder with historical data from connector."""
        historical_df = self.connector.get_candle_buffer()

        if len(historical_df) < 250:
            logger.warning(f"Not enough historical data: {len(historical_df)} bars")
            return

        logger.info(f"Initializing state builder with {len(historical_df)} historical bars...")

        self.state_builder = RealtimeStateBuilder(
            normalization_window=self.config.get("normalization.window", 2000),
            regime_smoothing_window=self.config.get("regime.smoothing_window", 30),
        )

        self.state_builder.initialize(historical_df)
        logger.info("State builder ready")

    async def _check_for_signal(self, candle: Dict) -> None:
        """Check for trading signals."""
        if not self.state_builder or not self.state_builder.is_ready:
            return

        # Skip if already have position
        if self.position_manager.has_position:
            return

        current_price = candle["close"]
        current_state = self.state_builder.get_current_state()
        current_regime = self.state_builder.get_current_regime()
        timestamp = candle["open_time"]

        # Query similarity engine
        try:
            sim_result = self.similarity_engine.query(
                current_state=current_state,
                regime=current_regime,
                horizon=self.horizon,
                max_timestamp=timestamp,
            )
        except Exception as e:
            logger.error(f"Similarity query error: {e}")
            return

        # Get decision
        decision = self.decision_engine.decide(sim_result, current_regime)

        if decision["action"] == "TRADE":
            self._last_signal_time = datetime.now(timezone.utc)

            logger.info(
                f"[SIGNAL] {decision['direction']} | "
                f"exp={decision['expectancy']*100:.3f}% | "
                f"regime={current_regime} | "
                f"price={current_price:.2f}"
            )

            # Add signal to web dashboard
            if self.web_state:
                self.web_state.add_signal({
                    "action": "TRADE",
                    "side": decision['direction'],
                    "expectancy": decision['expectancy'],
                    "distance": sim_result.get('distance_mean'),
                    "regime": current_regime,
                    "time": datetime.now(timezone.utc).isoformat(),
                })

            # Execute paper trade
            opened = self.position_manager.open_trade(
                decision=decision,
                current_price=current_price,
                regime=current_regime,
            )

            if opened:
                logger.info(
                    f"[TRADE] Opened {decision['direction']} | "
                    f"size=${decision['position_size']:.2f} | "
                    f"TP={decision['take_profit_pct']*100:.2f}% | "
                    f"SL={decision['stop_loss_pct']*100:.2f}%"
                )
        else:
            # Add skipped signal to web dashboard
            if self.web_state:
                self.web_state.add_signal({
                    "action": "SKIP",
                    "expectancy": decision.get('expectancy', 0),
                    "distance": sim_result.get('distance_mean') if sim_result.get('status') == 'OK' else None,
                    "reason": decision.get('reason', 'UNKNOWN'),
                    "regime": current_regime,
                    "time": datetime.now(timezone.utc).isoformat(),
                })

            # Log no-trade reason occasionally
            if self._bar_count % (self.sample_interval * 4) == 0:
                logger.debug(f"No trade: {decision.get('reason', 'UNKNOWN')}")

    def _log_trade_closed(self, trade) -> None:
        """Log when a trade is closed."""
        pnl_symbol = "+" if trade.pnl > 0 else ""
        logger.info(
            f"[CLOSED] {trade.side} | "
            f"PnL: {pnl_symbol}${trade.pnl:.2f} ({pnl_symbol}{trade.pnl_pct*100:.2f}%) | "
            f"reason={trade.exit_reason}"
        )

        # Save session after each trade
        self.position_manager.save_session()

        # Add trade to web dashboard
        if self.web_state:
            self.web_state.add_trade({
                "trade_id": trade.trade_id,
                "side": trade.side,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "entry_time": trade.entry_time.isoformat() if trade.entry_time else "",
                "exit_time": trade.exit_time.isoformat() if trade.exit_time else "",
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "exit_reason": trade.exit_reason,
                "status": "CLOSED",
            })

    def _update_dashboard(self, current_price: float) -> None:
        """Update the dashboard display."""
        if not self.position_manager:
            return

        # Get session stats
        stats = self.position_manager.get_session_summary(current_price)

        # Get position info using the existing method
        position_info = self.position_manager.get_position_info(current_price)
        if position_info:
            # Add size_btc for dashboard
            position_info["has_position"] = True
            position_info["size_btc"] = position_info["size_usd"] / position_info["entry_price"]
        else:
            position_info = {"has_position": False}

        # Calculate next check time
        bars_until_check = self.sample_interval - (self._bar_count % self.sample_interval)
        next_check_time = datetime.now(timezone.utc) + timedelta(minutes=bars_until_check)

        # Update web dashboard
        if self.web_state:
            self.web_state.update_status(
                price=current_price,
                regime=self._current_regime,
                bar_count=self._bar_count,
                next_check_in=bars_until_check,
            )
            self.web_state.update_position(
                has_position=position_info.get("has_position", False),
                side=position_info.get("side"),
                size_btc=position_info.get("size_btc", 0),
                entry_price=position_info.get("entry_price"),
                tp_price=position_info.get("tp_price"),
                sl_price=position_info.get("sl_price"),
                unrealized_pnl=position_info.get("unrealized_pnl", 0),
                unrealized_pnl_pct=position_info.get("unrealized_pnl_pct", 0),
            )
            self.web_state.update_stats(
                capital=stats.get("capital"),
                total_pnl=stats.get("total_pnl"),
                total_pnl_pct=stats.get("total_pnl_pct"),
                total_trades=stats.get("total_trades"),
                wins=stats.get("wins"),
                losses=stats.get("losses"),
                win_rate=stats.get("win_rate"),
                total_commission=stats.get("total_commission"),
            )

    def _log_status(self, current_price: float) -> None:
        """Log current status."""
        if not self.position_manager:
            return

        summary = self.position_manager.get_session_summary(current_price)
        uptime = datetime.now(timezone.utc) - self._start_time

        logger.info(
            f"[STATUS] Uptime: {uptime} | "
            f"Capital: ${summary['capital']:.2f} | "
            f"Trades: {summary['total_trades']} | "
            f"Win Rate: {summary['win_rate']*100:.0f}% | "
            f"PnL: ${summary['total_pnl']:.2f} ({summary['total_pnl_pct']*100:.2f}%)"
        )

    def _print_final_summary(self) -> None:
        """Print final trading summary."""
        if not self.position_manager or not self.connector:
            return

        current_price = self.connector.get_current_price() or 0
        summary = self.position_manager.get_session_summary(current_price)

        print("\n" + "=" * 60)
        print("PAPER TRADING SESSION SUMMARY")
        print("=" * 60)
        print(f"  Duration:        {datetime.now(timezone.utc) - self._start_time}")
        print(f"  Bars Processed:  {self._bar_count}")
        print(f"  Initial Capital: ${summary['initial_capital']:.2f}")
        print(f"  Final Capital:   ${summary['capital']:.2f}")
        print(f"  Total P&L:       ${summary['total_pnl']:.2f} ({summary['total_pnl_pct']*100:.2f}%)")
        print(f"  Total Trades:    {summary['total_trades']}")
        print(f"  Win Rate:        {summary['win_rate']*100:.0f}%")
        print(f"  Commission Paid: ${summary['total_commission']:.2f}")
        print("=" * 60)


async def main():
    """Entry point for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    orchestrator = LiveOrchestrator(
        capital=200.0,
        horizon=5,
        sample_interval=15,
        min_expectancy=0.001,
        max_distance=3.0,
        blocked_regimes=[],
    )

    await orchestrator.start()


if __name__ == "__main__":
    asyncio.run(main())
