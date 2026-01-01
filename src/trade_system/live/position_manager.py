"""
Position Manager for Paper Trading.

Provides higher-level position management and trade tracking.
Works with PaperExecutor to manage the lifecycle of positions.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict
from pathlib import Path
import json

from .paper_executor import PaperExecutor, PaperTrade, ExitReason


logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages positions and provides trade lifecycle control.

    Features:
    - Single position at a time (no pyramiding)
    - Timeout-based exit (max bars in trade)
    - Session state persistence
    - Trade history export
    """

    def __init__(
        self,
        executor: PaperExecutor,
        max_bars_in_trade: int = 120,
        session_file: Optional[Path] = None,
    ):
        """
        Initialize the position manager.

        Args:
            executor: PaperExecutor instance
            max_bars_in_trade: Force exit after N bars (0 = no timeout)
            session_file: Path to save/load session state
        """
        self.executor = executor
        self.max_bars_in_trade = max_bars_in_trade
        self.session_file = session_file

        # Track bars since entry
        self._bars_in_trade = 0
        self._entry_bar_time: Optional[datetime] = None

    def on_new_bar(self, current_price: float) -> Optional[PaperTrade]:
        """
        Called on each new bar. Checks exit conditions.

        Args:
            current_price: Current bar close price

        Returns:
            PaperTrade if position was closed, None otherwise
        """
        if not self.executor.has_position:
            return None

        # Increment bar counter
        self._bars_in_trade += 1

        # Check timeout
        if self.max_bars_in_trade > 0 and self._bars_in_trade >= self.max_bars_in_trade:
            logger.info(f"Position timeout after {self._bars_in_trade} bars")
            return self.executor.close_position(current_price, ExitReason.TIMEOUT)

        # Check TP/SL
        return self.executor.check_exit_conditions(current_price)

    def open_trade(
        self,
        decision: Dict,
        current_price: float,
        regime: str = "",
    ) -> bool:
        """
        Open a trade based on decision engine output.

        Args:
            decision: Decision dict from DecisionEngine
            current_price: Current market price
            regime: Current market regime

        Returns:
            True if trade was opened, False otherwise
        """
        if decision.get("action") != "TRADE":
            return False

        if self.executor.has_position:
            logger.debug("Cannot open trade: already have position")
            return False

        order = self.executor.open_position(
            side=decision["direction"],
            size_usd=decision["position_size"],
            current_price=current_price,
            tp_pct=decision["take_profit_pct"],
            sl_pct=decision["stop_loss_pct"],
            regime=regime,
        )

        if order:
            self._bars_in_trade = 0
            self._entry_bar_time = datetime.now(timezone.utc)
            return True

        return False

    def force_close(self, current_price: float, reason: str = "MANUAL") -> Optional[PaperTrade]:
        """Force close current position."""
        if not self.executor.has_position:
            return None

        exit_reason = ExitReason.MANUAL
        if reason == "SIGNAL":
            exit_reason = ExitReason.SIGNAL

        trade = self.executor.close_position(current_price, exit_reason)
        self._bars_in_trade = 0
        return trade

    def get_position_info(self, current_price: float) -> Optional[Dict]:
        """Get current position information."""
        if not self.executor.has_position:
            return None

        pos = self.executor.current_position
        unrealized_pnl = self.executor.get_unrealized_pnl(current_price)

        return {
            "side": pos.side.value,
            "size_usd": pos.size_usd,
            "entry_price": pos.entry_price,
            "current_price": current_price,
            "tp_price": pos.tp_price,
            "sl_price": pos.sl_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl / pos.size_usd,
            "bars_in_trade": self._bars_in_trade,
            "entry_time": pos.timestamp.isoformat(),
        }

    def get_session_summary(self, current_price: float) -> Dict:
        """Get full session summary."""
        stats = self.executor.get_stats()

        return {
            "capital": self.executor.capital,
            "initial_capital": self.executor.initial_capital,
            "realized_pnl": self.executor.get_realized_pnl(),
            "unrealized_pnl": self.executor.get_unrealized_pnl(current_price),
            "total_pnl": self.executor.get_total_pnl(current_price),
            "total_pnl_pct": (self.executor.capital - self.executor.initial_capital) / self.executor.initial_capital,
            "has_position": self.executor.has_position,
            "position_side": self.executor.position_side,
            **stats,
        }

    def save_session(self) -> None:
        """Save session state to file."""
        if self.session_file is None:
            return

        session_data = {
            "capital": self.executor.capital,
            "initial_capital": self.executor.initial_capital,
            "total_commission": self.executor.total_commission,
            "trade_history": [
                {
                    "trade_id": t.trade_id,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "side": t.side,
                    "size_usd": t.size_usd,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason,
                    "regime": t.regime,
                }
                for t in self.executor.trade_history
            ],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.session_file, "w") as f:
            json.dump(session_data, f, indent=2)

        logger.info(f"Session saved to {self.session_file}")

    def load_session(self) -> bool:
        """Load session state from file."""
        if self.session_file is None or not self.session_file.exists():
            return False

        try:
            with open(self.session_file) as f:
                session_data = json.load(f)

            self.executor.capital = session_data["capital"]
            self.executor.initial_capital = session_data["initial_capital"]
            self.executor.total_commission = session_data["total_commission"]

            # Restore trade history
            self.executor.trade_history = []
            for t in session_data.get("trade_history", []):
                trade = PaperTrade(
                    trade_id=t["trade_id"],
                    entry_time=datetime.fromisoformat(t["entry_time"]),
                    exit_time=datetime.fromisoformat(t["exit_time"]),
                    side=t["side"],
                    size_usd=t["size_usd"],
                    entry_price=t["entry_price"],
                    exit_price=t["exit_price"],
                    tp_price=t.get("tp_price", 0),
                    sl_price=t.get("sl_price", 0),
                    pnl=t["pnl"],
                    pnl_pct=t["pnl_pct"],
                    commission=t.get("commission", 0),
                    exit_reason=t["exit_reason"],
                    regime=t.get("regime", ""),
                )
                self.executor.trade_history.append(trade)

            logger.info(f"Session loaded from {self.session_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False

    @property
    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self.executor.has_position
