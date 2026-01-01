"""
Paper Trade Executor.

Simulates order execution without placing real orders.
Tracks positions and P&L for paper trading validation.
"""

import csv
import logging
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict
from enum import Enum


logger = logging.getLogger(__name__)


class OrderSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class ExitReason(Enum):
    TAKE_PROFIT = "TP"
    STOP_LOSS = "SL"
    TIMEOUT = "TIMEOUT"
    SIGNAL = "SIGNAL"
    MANUAL = "MANUAL"


@dataclass
class PaperOrder:
    """Represents a paper trade order."""
    order_id: str
    timestamp: datetime
    side: OrderSide
    size_usd: float           # Position size in USD
    entry_price: float        # Entry price after slippage
    tp_price: float           # Take profit price
    sl_price: float           # Stop loss price
    commission: float         # Commission paid
    status: OrderStatus = OrderStatus.OPEN

    # Exit fields (filled when closed)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[ExitReason] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class PaperTrade:
    """Completed trade record."""
    trade_id: str
    entry_time: datetime
    exit_time: datetime
    side: str
    size_usd: float
    entry_price: float
    exit_price: float
    tp_price: float
    sl_price: float
    pnl: float
    pnl_pct: float
    commission: float
    exit_reason: str
    regime: str = ""


class PaperExecutor:
    """
    Simulates order execution for paper trading.

    Features:
    - Market order simulation with slippage
    - Commission calculation
    - TP/SL monitoring
    - Trade history logging
    """

    def __init__(
        self,
        capital: float = 200.0,
        slippage_pct: float = 0.0005,      # 0.05%
        commission_pct: float = 0.0004,    # 0.04%
        trades_csv_path: Optional[str] = None,
    ):
        """
        Initialize the paper executor.

        Args:
            capital: Starting capital in USD
            slippage_pct: Slippage percentage (applied to entry/exit)
            commission_pct: Commission percentage per trade
            trades_csv_path: Path to CSV file for trade logging (auto-created if not exists)
        """
        self.initial_capital = capital
        self.capital = capital
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct

        # State
        self.current_position: Optional[PaperOrder] = None
        self.trade_history: List[PaperTrade] = []
        self.total_commission = 0.0

        # CSV trade logging
        if trades_csv_path is None:
            data_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "paper_trades"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.trades_csv_path = data_dir / "paper_trades.csv"
        else:
            self.trades_csv_path = Path(trades_csv_path)
            self.trades_csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_csv()

    def _init_csv(self) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.trades_csv_path.exists():
            headers = [
                "trade_id", "entry_time", "exit_time", "side", "size_usd",
                "entry_price", "exit_price", "tp_price", "sl_price",
                "pnl", "pnl_pct", "commission", "exit_reason", "regime"
            ]
            with open(self.trades_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            logger.info(f"Created trades CSV: {self.trades_csv_path}")

    def _append_trade_to_csv(self, trade: PaperTrade) -> None:
        """Append a completed trade to the CSV file."""
        row = [
            trade.trade_id,
            trade.entry_time.isoformat(),
            trade.exit_time.isoformat(),
            trade.side,
            f"{trade.size_usd:.2f}",
            f"{trade.entry_price:.2f}",
            f"{trade.exit_price:.2f}",
            f"{trade.tp_price:.2f}",
            f"{trade.sl_price:.2f}" if trade.sl_price > 0 else "None",
            f"{trade.pnl:.4f}",
            f"{trade.pnl_pct:.6f}",
            f"{trade.commission:.4f}",
            trade.exit_reason,
            trade.regime,
        ]
        with open(self.trades_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        logger.debug(f"Trade {trade.trade_id} saved to CSV")

    def open_position(
        self,
        side: str,
        size_usd: float,
        current_price: float,
        tp_pct: float,
        sl_pct: float,
        regime: str = "",
    ) -> Optional[PaperOrder]:
        """
        Open a new paper position.

        Args:
            side: "LONG" or "SHORT"
            size_usd: Position size in USD
            current_price: Current market price
            tp_pct: Take profit percentage (e.g., 0.005 for 0.5%)
            sl_pct: Stop loss percentage (e.g., 0.003 for 0.3%)
            regime: Current market regime (for logging)

        Returns:
            PaperOrder if opened, None if rejected
        """
        if self.current_position is not None:
            logger.warning("Cannot open position: already have an open position")
            return None

        if size_usd > self.capital:
            logger.warning(f"Position size ${size_usd:.2f} exceeds capital ${self.capital:.2f}")
            size_usd = self.capital

        # Apply slippage to entry
        if side == "LONG":
            entry_price = current_price * (1 + self.slippage_pct)
            tp_price = entry_price * (1 + tp_pct)
            sl_price = 0.0 if sl_pct <= 0 else entry_price * (1 - sl_pct)
        else:  # SHORT
            entry_price = current_price * (1 - self.slippage_pct)
            tp_price = entry_price * (1 - tp_pct)
            sl_price = 0.0 if sl_pct <= 0 else entry_price * (1 + sl_pct)

        # Calculate commission
        commission = size_usd * self.commission_pct
        self.total_commission += commission

        # Create order
        order = PaperOrder(
            order_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc),
            side=OrderSide.LONG if side == "LONG" else OrderSide.SHORT,
            size_usd=size_usd,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            commission=commission,
        )

        self.current_position = order
        self._regime = regime

        logger.info(
            f"[PAPER] Opened {side} position: "
            f"size=${size_usd:.2f}, entry={entry_price:.2f}, "
            f"TP={tp_price:.2f}, SL={sl_price:.2f}"
        )

        return order

    def check_exit_conditions(self, current_price: float) -> Optional[PaperTrade]:
        """
        Check if current position should be closed based on TP/SL.

        Args:
            current_price: Current market price

        Returns:
            PaperTrade if position was closed, None otherwise
        """
        if self.current_position is None:
            return None

        pos = self.current_position

        if pos.side == OrderSide.LONG:
            # Long: TP if price >= tp_price, SL if price <= sl_price (only if SL is set)
            if current_price >= pos.tp_price:
                return self.close_position(current_price, ExitReason.TAKE_PROFIT)
            elif pos.sl_price > 0 and current_price <= pos.sl_price:
                return self.close_position(current_price, ExitReason.STOP_LOSS)
        else:
            # Short: TP if price <= tp_price, SL if price >= sl_price (only if SL is set)
            if current_price <= pos.tp_price:
                return self.close_position(current_price, ExitReason.TAKE_PROFIT)
            elif pos.sl_price > 0 and current_price >= pos.sl_price:
                return self.close_position(current_price, ExitReason.STOP_LOSS)

        return None

    def close_position(
        self,
        exit_price: float,
        reason: ExitReason = ExitReason.MANUAL,
    ) -> Optional[PaperTrade]:
        """
        Close the current position.

        Args:
            exit_price: Price at which to close
            reason: Reason for closing

        Returns:
            PaperTrade record
        """
        if self.current_position is None:
            logger.warning("No position to close")
            return None

        pos = self.current_position

        # Apply slippage to exit
        if pos.side == OrderSide.LONG:
            actual_exit = exit_price * (1 - self.slippage_pct)
            pnl = (actual_exit - pos.entry_price) / pos.entry_price * pos.size_usd
        else:
            actual_exit = exit_price * (1 + self.slippage_pct)
            pnl = (pos.entry_price - actual_exit) / pos.entry_price * pos.size_usd

        # Subtract exit commission
        exit_commission = pos.size_usd * self.commission_pct
        self.total_commission += exit_commission
        pnl -= exit_commission

        pnl_pct = pnl / pos.size_usd

        # Update capital
        self.capital += pnl

        # Create trade record
        trade = PaperTrade(
            trade_id=pos.order_id,
            entry_time=pos.timestamp,
            exit_time=datetime.now(timezone.utc),
            side=pos.side.value,
            size_usd=pos.size_usd,
            entry_price=pos.entry_price,
            exit_price=actual_exit,
            tp_price=pos.tp_price,
            sl_price=pos.sl_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=pos.commission + exit_commission,
            exit_reason=reason.value,
            regime=getattr(self, '_regime', ''),
        )

        self.trade_history.append(trade)
        self.current_position = None

        # Save to CSV for persistence across restarts
        self._append_trade_to_csv(trade)

        logger.info(
            f"[PAPER] Closed {trade.side} position: "
            f"exit={actual_exit:.2f}, PnL=${pnl:.2f} ({pnl_pct*100:.2f}%), "
            f"reason={reason.value}"
        )

        return trade

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for open position."""
        if self.current_position is None:
            return 0.0

        pos = self.current_position

        if pos.side == OrderSide.LONG:
            return (current_price - pos.entry_price) / pos.entry_price * pos.size_usd
        else:
            return (pos.entry_price - current_price) / pos.entry_price * pos.size_usd

    def get_realized_pnl(self) -> float:
        """Get total realized P&L from closed trades."""
        return sum(t.pnl for t in self.trade_history)

    def get_total_pnl(self, current_price: float) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.get_realized_pnl() + self.get_unrealized_pnl(current_price)

    def get_stats(self) -> Dict:
        """Get trading statistics."""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0,
                "avg_pnl": 0.0,
                "total_commission": self.total_commission,
            }

        wins = sum(1 for t in self.trade_history if t.pnl > 0)
        losses = len(self.trade_history) - wins
        total_pnl = self.get_realized_pnl()

        return {
            "total_trades": len(self.trade_history),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(self.trade_history) if self.trade_history else 0.0,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl / self.initial_capital,
            "avg_pnl": total_pnl / len(self.trade_history),
            "total_commission": self.total_commission,
        }

    @property
    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self.current_position is not None

    @property
    def position_side(self) -> Optional[str]:
        """Get current position side."""
        if self.current_position:
            return self.current_position.side.value
        return None
