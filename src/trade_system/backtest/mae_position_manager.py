"""
MAE Position Manager: Hold/Cut decisions based on live MAE tracking.

Key insight from analysis:
- If MAE < 50bp, likely Case 2/3 (will recover) → HOLD
- If MAE > 50bp, likely Case 1 (wrong direction) → CUT

This replaces fixed stop-loss with data-driven position management.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class PositionAction(Enum):
    """Actions for position management."""
    HOLD = "HOLD"
    CUT = "CUT"
    WIN = "WIN"
    TIMEOUT = "TIMEOUT"


@dataclass
class PositionState:
    """Current state of an open position."""
    entry_price: float
    direction: str  # "LONG" or "SHORT"
    target_bps: float
    bars_in_trade: int = 0
    current_mae_bps: float = 0.0
    current_mfe_bps: float = 0.0
    lowest_price: float = 0.0
    highest_price: float = 0.0


class MAEPositionManager:
    """
    Position manager that uses MAE threshold for hold/cut decisions.

    The key insight: 84-95% of trades eventually recover if drawdown < 50bp.
    Only cut when MAE exceeds the threshold (likely wrong direction).
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with configurable parameters.

        Args:
            config: Dictionary with parameter overrides, or None for defaults
        """
        config = config or {}

        # MAE threshold for cutting (from analysis: 50bp separates timing vs direction)
        self.mae_cut_threshold = config.get("mae_cut_threshold", 50.0)

        # Adaptive horizon settings
        self.adaptive_horizon = config.get("adaptive_horizon", True)
        self.max_bars_in_trade = config.get("max_bars_in_trade", 200)

        # Target reached settings
        self.target_buffer_bps = config.get("target_buffer_bps", 0.0)  # No buffer needed with limit orders

    def update(
        self,
        position: PositionState,
        current_price: float,
        risk_profile: Optional[Dict] = None
    ) -> Tuple[PositionAction, str]:
        """
        Update position and decide whether to HOLD, CUT, or WIN.

        Args:
            position: Current position state
            current_price: Current market price
            risk_profile: Optional risk profile for dynamic thresholds

        Returns:
            Tuple of (action: PositionAction, reason: str)
        """
        # Update position metrics
        position.bars_in_trade += 1
        self._update_metrics(position, current_price)

        # Check 1: Target hit (WIN)
        if self._check_target_hit(position, current_price):
            return PositionAction.WIN, f"Target hit at {position.current_mfe_bps:.1f}bp"

        # Check 2: MAE threshold exceeded (CUT)
        if position.current_mae_bps > self.mae_cut_threshold:
            return PositionAction.CUT, f"MAE {position.current_mae_bps:.1f}bp > {self.mae_cut_threshold}bp threshold"

        # Check 3: Time limit exceeded (TIMEOUT)
        max_bars = self._get_max_bars(risk_profile)
        if position.bars_in_trade > max_bars:
            return PositionAction.TIMEOUT, f"Exceeded {max_bars} bars in trade"

        # Otherwise, HOLD and wait for recovery
        return PositionAction.HOLD, f"MAE {position.current_mae_bps:.1f}bp within threshold, holding"

    def _update_metrics(self, position: PositionState, current_price: float):
        """Update position MAE/MFE metrics."""
        if position.direction == "LONG":
            # For long: MAE is how far below entry, MFE is how far above
            if position.lowest_price == 0 or current_price < position.lowest_price:
                position.lowest_price = current_price
            if position.highest_price == 0 or current_price > position.highest_price:
                position.highest_price = current_price

            position.current_mae_bps = (position.entry_price - position.lowest_price) / position.entry_price * 10000
            position.current_mfe_bps = (position.highest_price - position.entry_price) / position.entry_price * 10000

        else:  # SHORT
            # For short: MAE is how far above entry, MFE is how far below
            if position.highest_price == 0 or current_price > position.highest_price:
                position.highest_price = current_price
            if position.lowest_price == 0 or current_price < position.lowest_price:
                position.lowest_price = current_price

            position.current_mae_bps = (position.highest_price - position.entry_price) / position.entry_price * 10000
            position.current_mfe_bps = (position.entry_price - position.lowest_price) / position.entry_price * 10000

    def _check_target_hit(self, position: PositionState, current_price: float) -> bool:
        """Check if target has been hit."""
        target_pct = (position.target_bps - self.target_buffer_bps) / 10000

        if position.direction == "LONG":
            target_price = position.entry_price * (1 + target_pct)
            return current_price >= target_price
        else:  # SHORT
            target_price = position.entry_price * (1 - target_pct)
            return current_price <= target_price

    def _get_max_bars(self, risk_profile: Optional[Dict]) -> int:
        """Get max bars in trade, optionally from risk profile."""
        if self.adaptive_horizon and risk_profile is not None:
            # Use 90th percentile recovery time from risk profile
            recovery_90pct = risk_profile.get("recovery_90pct", self.max_bars_in_trade)
            if not np.isnan(recovery_90pct):
                return int(min(recovery_90pct, self.max_bars_in_trade))
        return self.max_bars_in_trade

    def open_position(
        self,
        entry_price: float,
        direction: str,
        target_bps: float
    ) -> PositionState:
        """
        Create a new position state.

        Args:
            entry_price: Entry price
            direction: "LONG" or "SHORT"
            target_bps: Target in basis points

        Returns:
            New PositionState object
        """
        return PositionState(
            entry_price=entry_price,
            direction=direction,
            target_bps=target_bps,
            bars_in_trade=0,
            current_mae_bps=0.0,
            current_mfe_bps=0.0,
            lowest_price=entry_price,
            highest_price=entry_price,
        )

    def get_exit_price(
        self,
        position: PositionState,
        action: PositionAction,
        current_price: float
    ) -> float:
        """
        Calculate exit price based on action.

        Args:
            position: Position state
            action: Exit action
            current_price: Current market price

        Returns:
            Exit price
        """
        if action == PositionAction.WIN:
            # Exit at target
            target_pct = position.target_bps / 10000
            if position.direction == "LONG":
                return position.entry_price * (1 + target_pct)
            else:
                return position.entry_price * (1 - target_pct)
        else:
            # Exit at current price
            return current_price

    def calculate_pnl_bps(
        self,
        position: PositionState,
        exit_price: float,
        fee_bps: float = 8.0
    ) -> float:
        """
        Calculate P&L in basis points.

        Args:
            position: Position state
            exit_price: Exit price
            fee_bps: Round-trip fees in bps

        Returns:
            P&L in basis points (after fees)
        """
        if position.direction == "LONG":
            raw_pnl_bps = (exit_price - position.entry_price) / position.entry_price * 10000
        else:
            raw_pnl_bps = (position.entry_price - exit_price) / position.entry_price * 10000

        return raw_pnl_bps - fee_bps


if __name__ == "__main__":
    # Test the position manager
    print("="*60)
    print("MAE POSITION MANAGER TEST")
    print("="*60)

    manager = MAEPositionManager({
        "mae_cut_threshold": 50.0,
        "max_bars_in_trade": 200,
    })

    # Simulate a trade that recovers (Case 2)
    print("\n--- Test 1: Recovery scenario (Case 2) ---")
    position = manager.open_position(
        entry_price=50000.0,
        direction="LONG",
        target_bps=15.0
    )

    # Simulate price path: goes down 30bp, then recovers
    prices = [
        50000.0,  # Entry
        49985.0,  # -3bp
        49950.0,  # -10bp
        49900.0,  # -20bp
        49850.0,  # -30bp (MAE)
        49900.0,  # -20bp (recovering)
        49950.0,  # -10bp
        50000.0,  # 0bp
        50050.0,  # +10bp
        50075.0,  # +15bp (TARGET HIT!)
    ]

    for i, price in enumerate(prices[1:], 1):
        action, reason = manager.update(position, price)
        print(f"  Bar {i}: Price={price:.0f}, MAE={position.current_mae_bps:.1f}bp, Action={action.value}")
        if action != PositionAction.HOLD:
            print(f"         Reason: {reason}")
            break

    # Simulate a trade that fails (Case 1)
    print("\n--- Test 2: Wrong direction scenario (Case 1) ---")
    position = manager.open_position(
        entry_price=50000.0,
        direction="LONG",
        target_bps=15.0
    )

    # Simulate price path: goes down past 50bp threshold
    prices = [
        50000.0,  # Entry
        49950.0,  # -10bp
        49900.0,  # -20bp
        49850.0,  # -30bp
        49800.0,  # -40bp
        49750.0,  # -50bp (threshold!)
        49700.0,  # -60bp (CUT!)
    ]

    for i, price in enumerate(prices[1:], 1):
        action, reason = manager.update(position, price)
        print(f"  Bar {i}: Price={price:.0f}, MAE={position.current_mae_bps:.1f}bp, Action={action.value}")
        if action != PositionAction.HOLD:
            print(f"         Reason: {reason}")
            break
