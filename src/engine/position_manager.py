"""Position Manager — tracks open positions, handles exits and re-entry.

Exit plans (priority order on every tick):
  V1 (default): PT_TARGET / PT_LOCK / MID_TRAIL / LOCKED_PROFIT / STOP_LOSS
  V2         : V1 minus LOCKED_PROFIT (no static 15 bps lock)

Bar-level: time exit on bar v1_max_bars (NO_ZONE if pnl>=0, TIME_EXIT otherwise).
Re-entry: disabled — hook left in place but never fires under V1/V2 (it keys off
the obsolete TRAILING_STOP reason from the removed V2-old plan).
"""

from dataclasses import dataclass, field
from typing import Optional

from .config.constants import FEES_BPS
from .config.schema import AppConfig
from .strategy import Direction, SignalType


@dataclass
class TradeRecord:
    """Completed trade record."""
    signal_time: object
    entry_time: object
    exit_time: object
    direction: str
    signal_type: str
    entry_price: float
    exit_price: float
    gross_profit_bps: float
    net_profit_bps: float
    mfe_bps: float
    mae_bps: float
    exit_bar: int
    exit_reason: str
    is_reentry: bool = False


@dataclass
class OpenPosition:
    """State of an open position being tracked bar-by-bar."""
    direction: Direction
    signal_type: SignalType
    entry_price: float
    entry_time: object
    signal_time: object
    max_bars: int
    is_reentry: bool = False

    # Updated each bar
    bars_held: int = 0
    highest_profit_bps: float = 0.0
    mfe_bps: float = 0.0
    mae_bps: float = 0.0
    pt_armed: bool = False


class V12PositionManager:
    """Manages position lifecycle: entry, bar updates, exits, re-entry.

    exit_version:
      "v1" — full V1 rules (includes LOCKED_PROFIT static exit at 15 bps peak)
      "v2" — V1 minus LOCKED_PROFIT
    """

    def __init__(self, config: AppConfig, exit_version: str = "v1"):
        if exit_version not in ("v1", "v2"):
            raise ValueError(f"exit_version must be 'v1' or 'v2', got {exit_version!r}")
        self.cfg = config
        self.exit_version = exit_version
        self.position: Optional[OpenPosition] = None
        self.trades: list[TradeRecord] = []

        # Re-entry state
        self._last_exit_direction: Optional[Direction] = None
        self._last_exit_signal_type: Optional[SignalType] = None
        self._last_exit_reason: Optional[str] = None
        self._last_exit_bar_index: int = -999
        self._reentry_count: int = 0

    @property
    def is_in_position(self) -> bool:
        return self.position is not None

    def open_position(
        self,
        direction: Direction,
        signal_type: SignalType,
        entry_price: float,
        entry_time: object,
        signal_time: object,
        is_reentry: bool = False,
    ) -> None:
        """Open a new position."""
        self.position = OpenPosition(
            direction=direction,
            signal_type=signal_type,
            entry_price=entry_price,
            entry_time=entry_time,
            signal_time=signal_time,
            max_bars=self.cfg.exit.v1_max_bars,
            is_reentry=is_reentry,
        )

    def on_tick(self, price: float, tick_time: object) -> Optional[TradeRecord]:
        """Check price-based exits on every tick (PT/MID_TRAIL/LOCKED_PROFIT/STOP_LOSS)."""
        pos = self.position
        if pos is None:
            return None

        # Current PnL from live price
        if pos.direction == Direction.LONG:
            tick_pnl = (price - pos.entry_price) / pos.entry_price * 10000
        else:
            tick_pnl = (pos.entry_price - price) / pos.entry_price * 10000

        # Update peak MFE in real-time
        if tick_pnl > pos.mfe_bps:
            pos.mfe_bps = tick_pnl
        if tick_pnl > pos.highest_profit_bps:
            pos.highest_profit_bps = tick_pnl
        if tick_pnl < pos.mae_bps:
            pos.mae_bps = tick_pnl

        e = self.cfg.exit

        # Arm PT once peak touches pt_arm within max bar
        if pos.highest_profit_bps >= e.v1_pt_arm_bps and pos.bars_held <= e.v1_pt_max_bar:
            pos.pt_armed = True

        # 1a. PT_TARGET: take profit at 80 (tick price, can gap above)
        if pos.pt_armed and tick_pnl >= e.v1_pt_target_bps:
            return self._close_position(tick_pnl, tick_time, pos.bars_held, "PT_TARGET")

        # 1b. PT_LOCK: stop order at 60 — exit at IDEALIZED 60 bps
        if pos.pt_armed and tick_pnl <= e.v1_pt_lock_bps:
            return self._close_position(e.v1_pt_lock_bps, tick_time, pos.bars_held, "PT_LOCK")

        # 2. MID_TRAIL: arm at 25, trail 10 (not if already pt_armed)
        if pos.highest_profit_bps >= e.v1_mid_trail_arm_bps and not pos.pt_armed:
            drawdown = pos.highest_profit_bps - tick_pnl
            if drawdown >= e.v1_mid_trail_width_bps:
                exit_bps = pos.highest_profit_bps - e.v1_mid_trail_width_bps
                return self._close_position(exit_bps, tick_time, pos.bars_held, "MID_TRAIL")

        # 3. LOCKED_PROFIT (static) — V1 only; V2 skips this rule
        if self.exit_version == "v1":
            if pos.highest_profit_bps >= e.v1_lock_arm_bps and tick_pnl <= e.v1_lock_trigger_bps:
                return self._close_position(tick_pnl, tick_time, pos.bars_held, "LOCKED_PROFIT")

        # 4. STOP_LOSS: hard cap — exit at IDEALIZED -10 bps (stop-market simulation)
        if tick_pnl <= e.v1_stop_loss_bps:
            return self._close_position(e.v1_stop_loss_bps, tick_time, pos.bars_held, "STOP_LOSS")

        return None

    def on_bar(self, high: float, low: float, close: float, bar_time: object, bar_index: int) -> Optional[TradeRecord]:
        """Process a new bar close. Handles time exit (price exits via on_tick)."""
        pos = self.position
        if pos is None:
            return None

        pos.bars_held += 1

        # Calculate P&L in bps
        if pos.direction == Direction.LONG:
            bar_mfe = (high - pos.entry_price) / pos.entry_price * 10000
            bar_mae = (low - pos.entry_price) / pos.entry_price * 10000
            bar_close_pnl = (close - pos.entry_price) / pos.entry_price * 10000
        else:
            bar_mfe = (pos.entry_price - low) / pos.entry_price * 10000
            bar_mae = (pos.entry_price - high) / pos.entry_price * 10000
            bar_close_pnl = (pos.entry_price - close) / pos.entry_price * 10000

        # Update MFE/MAE from bar extremes
        if bar_mfe > pos.mfe_bps:
            pos.mfe_bps = bar_mfe
        if bar_mae < pos.mae_bps:
            pos.mae_bps = bar_mae
        if bar_mfe > pos.highest_profit_bps:
            pos.highest_profit_bps = bar_mfe

        # V1 only: time exit on max bar (price exits handled by on_tick)
        if pos.bars_held >= pos.max_bars:
            if bar_close_pnl >= 0:
                return self._close_position(bar_close_pnl, bar_time, bar_index, "NO_ZONE")
            return self._close_position(bar_close_pnl, bar_time, bar_index, "TIME_EXIT")
        return None

    def _close_position(
        self, gross_profit_bps: float, exit_time: object, bar_index: int, reason: str
    ) -> TradeRecord:
        """Close position, record trade, update re-entry state."""
        pos = self.position

        if pos.direction == Direction.LONG:
            exit_price = pos.entry_price * (1 + gross_profit_bps / 10000)
        else:
            exit_price = pos.entry_price * (1 - gross_profit_bps / 10000)

        trade = TradeRecord(
            signal_time=pos.signal_time,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            direction=pos.direction.value,
            signal_type=pos.signal_type.value,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            gross_profit_bps=gross_profit_bps,
            net_profit_bps=gross_profit_bps - FEES_BPS,
            mfe_bps=pos.mfe_bps,
            mae_bps=pos.mae_bps,
            exit_bar=pos.bars_held,
            exit_reason=reason,
            is_reentry=pos.is_reentry,
        )
        self.trades.append(trade)

        # Track re-entry state
        self._last_exit_direction = pos.direction
        self._last_exit_signal_type = pos.signal_type
        self._last_exit_reason = reason
        self._last_exit_bar_index = bar_index

        if pos.is_reentry:
            self._reentry_count += 1
        elif reason == "TRAILING_STOP":
            # Original trade hit TS — enable re-entry
            self._reentry_count = 0

        self.position = None
        return trade

    def can_reenter(self, current_bar_index: int, regime_valid: bool) -> bool:
        """Check if re-entry conditions are met.

        Matches EXP-012 behavior: re-entry is checked ONCE at exit + cooldown.
        If regime fails at that exact bar, re-entry is abandoned (not retried later).

        Conditions:
          1. Re-entry is enabled in config
          2. Last exit was TRAILING_STOP (not TIME_EXIT)
          3. Haven't exceeded max re-entries for this signal chain
          4. Current bar is EXACTLY at cooldown point (one-shot check)
          5. Market regime still valid (bull for LONG, bear for SHORT)
        """
        rc = self.cfg.reentry
        if not rc.enabled:
            return False
        if self._last_exit_reason != "TRAILING_STOP":
            return False
        if self._reentry_count >= rc.max_reentries:
            return False
        # One-shot: only check at the exact cooldown bar
        if current_bar_index != self._last_exit_bar_index + rc.cooldown_bars:
            return False
        if not regime_valid:
            # Regime failed at cooldown bar — abandon re-entry entirely
            self._last_exit_reason = None
            return False
        return True

    @property
    def reentry_direction(self) -> Optional[Direction]:
        """Direction for re-entry (same as last closed trade)."""
        return self._last_exit_direction

    @property
    def reentry_signal_type(self) -> Optional[SignalType]:
        """Signal type for re-entry (same as last closed trade)."""
        return self._last_exit_signal_type

    def reset_reentry(self) -> None:
        """Reset re-entry state (called when a new original signal fires)."""
        self._reentry_count = 0
        self._last_exit_reason = None
        self._last_exit_direction = None
        self._last_exit_signal_type = None
        self._last_exit_bar_index = -999
