import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Trade:
    """Represents a single trade with entry/exit details."""

    # Entry info
    signal_time: pd.Timestamp      # When signal was generated
    entry_time: pd.Timestamp       # When trade was opened (next bar)
    entry_price: float
    direction: str                 # LONG or SHORT
    position_size: float           # In USD

    # Risk levels (as prices)
    stop_loss_price: float
    take_profit_price: float

    # Risk levels (as percentages, for reference)
    stop_loss_pct: float
    take_profit_pct: float

    # Metadata
    regime: str
    expectancy: float

    # Exit info (filled when trade closes)
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # TP_HIT, SL_HIT, TIMEOUT

    # P&L (filled when trade closes)
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    bars_held: int = 0


class TradeSimulator:
    """Simulates trade execution with slippage and commissions."""

    def __init__(
        self,
        slippage_pct: float = 0.0005,
        commission_pct: float = 0.0004,
        max_bars_in_trade: int = 120
    ):
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        self.max_bars_in_trade = max_bars_in_trade

    def open_trade(
        self,
        decision: dict,
        signal_time: pd.Timestamp,
        entry_price: float,
        regime: str
    ) -> Trade:
        """
        Open a new trade based on decision engine output.

        Args:
            decision: Output from DecisionEngine.decide()
            signal_time: Time when signal was generated
            entry_price: Price at which to enter (next bar open)
            regime: Current market regime
        """
        direction = decision["direction"]

        # Apply slippage (adverse to our trade)
        if direction == "LONG":
            fill_price = entry_price * (1 + self.slippage_pct)
        else:
            fill_price = entry_price * (1 - self.slippage_pct)

        # Calculate stop loss and take profit prices
        stop_pct = decision["stop_loss_pct"]
        tp_pct = decision["take_profit_pct"]

        if direction == "LONG":
            stop_loss_price = fill_price * (1 - stop_pct)
            take_profit_price = fill_price * (1 + tp_pct)
        else:
            stop_loss_price = fill_price * (1 + stop_pct)
            take_profit_price = fill_price * (1 - tp_pct)

        return Trade(
            signal_time=signal_time,
            entry_time=signal_time,  # Will be updated to actual entry bar
            entry_price=fill_price,
            direction=direction,
            position_size=decision["position_size"],
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            stop_loss_pct=stop_pct,
            take_profit_pct=tp_pct,
            regime=regime,
            expectancy=decision["expectancy"],
        )

    def update_trade(
        self,
        trade: Trade,
        bar: pd.Series,
        current_time: pd.Timestamp
    ) -> Trade:
        """
        Update trade status based on current bar.

        Checks if stop loss or take profit was hit during this bar.
        Returns the same trade object (possibly with exit info filled).
        """
        trade.bars_held += 1

        high = bar["high"]
        low = bar["low"]
        close = bar["close"]

        # Check for stop loss / take profit hits
        if trade.direction == "LONG":
            # For LONG: SL hit if low <= stop, TP hit if high >= target
            sl_hit = low <= trade.stop_loss_price
            tp_hit = high >= trade.take_profit_price
        else:
            # For SHORT: SL hit if high >= stop, TP hit if low <= target
            sl_hit = high >= trade.stop_loss_price
            tp_hit = low <= trade.take_profit_price

        # Determine exit (if both hit in same bar, assume SL hit first - conservative)
        if sl_hit:
            trade = self._close_trade(trade, trade.stop_loss_price, current_time, "SL_HIT")
        elif tp_hit:
            trade = self._close_trade(trade, trade.take_profit_price, current_time, "TP_HIT")
        elif trade.bars_held >= self.max_bars_in_trade:
            trade = self._close_trade(trade, close, current_time, "TIMEOUT")

        return trade

    def _close_trade(
        self,
        trade: Trade,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str
    ) -> Trade:
        """Close a trade and calculate P&L."""

        # Apply slippage on exit (adverse to our trade)
        if trade.direction == "LONG":
            fill_price = exit_price * (1 - self.slippage_pct)
        else:
            fill_price = exit_price * (1 + self.slippage_pct)

        # Calculate raw P&L
        if trade.direction == "LONG":
            pnl_pct = (fill_price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - fill_price) / trade.entry_price

        # Deduct commission (both entry and exit)
        pnl_pct -= 2 * self.commission_pct

        # Calculate dollar P&L
        pnl = trade.position_size * pnl_pct

        trade.exit_time = exit_time
        trade.exit_price = fill_price
        trade.exit_reason = exit_reason
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct

        return trade

    def force_exit(
        self,
        trade: Trade,
        exit_price: float,
        exit_time: pd.Timestamp
    ) -> Trade:
        """Force exit a trade at market price."""
        return self._close_trade(trade, exit_price, exit_time, "FORCED")
