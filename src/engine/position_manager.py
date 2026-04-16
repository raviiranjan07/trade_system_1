"""V2 Position Manager — Tracks open positions, handles exits and re-entry.

Exit mechanics (EXP-5 V2, frozen 2026-03-29):
  - Trailing stop: 20 bps LONG, 30 bps SHORT
  - Tighten: bar 4+ to 6 bps
  - Early cut: bar 3 MFE<3, bar 4 MFE<5
  - Breakeven lock: MFE>=15, floor 9 bps gross
  - Time exit: bar 10
  - Re-entry: 1 max, 2 bar cooldown after trailing stop
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
    trailing_stop_bps: float
    max_bars: int
    is_reentry: bool = False

    # Updated each bar
    bars_held: int = 0
    highest_profit_bps: float = 0.0
    mfe_bps: float = 0.0
    mae_bps: float = 0.0
    # V3 state
    pt_armed: bool = False


class V12PositionManager:
    """Manages position lifecycle: entry, bar updates, exits, re-entry."""

    def __init__(self, config: AppConfig, exit_version: str = "v2"):
        self.cfg = config
        self.exit_version = exit_version  # "v2" or "v1"
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
        if direction == Direction.LONG:
            ts = self.cfg.exit.long_trailing_stop_bps
        else:
            ts = self.cfg.exit.short_trailing_stop_bps

        max_bars = self.cfg.exit.v1_max_bars if self.exit_version == "v1" else self.cfg.exit.max_bars
        self.position = OpenPosition(
            direction=direction,
            signal_type=signal_type,
            entry_price=entry_price,
            entry_time=entry_time,
            signal_time=signal_time,
            trailing_stop_bps=ts,
            max_bars=max_bars,
            is_reentry=is_reentry,
        )

    def on_tick(self, price: float, tick_time: object) -> Optional[TradeRecord]:
        """Check price-based exits on every tick (real-time monitoring).

        Branches based on exit_version:
          v2: trailing stop + BE lock (existing V2 logic)
          v1: PT_TARGET, PT_LOCK, MID_TRAIL, LOCKED_PROFIT (new V3 logic)

        Returns TradeRecord if position was closed, None if still open.
        """
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

        if self.exit_version == "v1":
            return self._on_tick_v1(pos, price, tick_pnl, tick_time)
        return self._on_tick_v2(pos, price, tick_pnl, tick_time)

    def _on_tick_v2(self, pos, price, tick_pnl, tick_time) -> Optional[TradeRecord]:
        """V2 tick logic: trailing stop + BE lock."""
        tighten_bar = self.cfg.exit.tighten_after_bar
        if pos.bars_held > tighten_bar:
            active_trailing = self.cfg.exit.tight_trailing_stop_bps
        else:
            active_trailing = pos.trailing_stop_bps

        be_activation = self.cfg.exit.breakeven_activation_mfe
        be_floor = self.cfg.exit.breakeven_floor_gross_bps
        be_active = pos.highest_profit_bps >= be_activation

        drawdown = pos.highest_profit_bps - tick_pnl
        if drawdown >= active_trailing and pos.highest_profit_bps > 0:
            reason = "TIGHT_TS" if pos.bars_held > tighten_bar else "TRAILING_STOP"
            return self._close_position(tick_pnl, tick_time, pos.bars_held, reason)

        if be_active and tick_pnl <= be_floor:
            return self._close_position(tick_pnl, tick_time, pos.bars_held, "BE_LOCK")

        return None

    def _on_tick_v1(self, pos, price, tick_pnl, tick_time) -> Optional[TradeRecord]:
        """V3 tick logic: PT (60/80) + MID_TRAIL (25/10) + LOCKED_PROFIT (15) + STOP_LOSS (-10)."""
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

        # 3. LOCKED_PROFIT (static)
        if pos.highest_profit_bps >= e.v1_lock_arm_bps and tick_pnl <= e.v1_lock_trigger_bps:
            return self._close_position(tick_pnl, tick_time, pos.bars_held, "LOCKED_PROFIT")

        # 4. STOP_LOSS: hard cap — exit at IDEALIZED -10 bps (stop-market simulation)
        if tick_pnl <= e.v1_stop_loss_bps:
            return self._close_position(e.v1_stop_loss_bps, tick_time, pos.bars_held, "STOP_LOSS")

        return None

    def on_bar(self, high: float, low: float, close: float, bar_time: object, bar_index: int) -> Optional[TradeRecord]:
        """Process a new bar close. Handles time-based exits (EARLY_CUT, TIME_EXIT).

        Price-based exits (trailing stop, BE lock) are handled by on_tick.
        This method is a safety net: if on_tick wasn't called, it also checks
        price exits using bar_close_pnl.

        Returns TradeRecord if position was closed, None if still open.
        """
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

        # V1 branch
        if self.exit_version == "v1":
            # V1: only time exit here — price exits handled by on_tick
            if pos.bars_held >= pos.max_bars:
                if bar_close_pnl >= 0:
                    return self._close_position(bar_close_pnl, bar_time, bar_index, "NO_ZONE")
                return self._close_position(bar_close_pnl, bar_time, bar_index, "TIME_EXIT")
            return None

        # V2 branch (existing logic)
        tighten_bar = self.cfg.exit.tighten_after_bar
        if pos.bars_held > tighten_bar:
            active_trailing = self.cfg.exit.tight_trailing_stop_bps
        else:
            active_trailing = pos.trailing_stop_bps

        be_activation = self.cfg.exit.breakeven_activation_mfe
        be_floor = self.cfg.exit.breakeven_floor_gross_bps
        be_active = pos.highest_profit_bps >= be_activation

        drawdown = pos.highest_profit_bps - bar_close_pnl
        if drawdown >= active_trailing and pos.highest_profit_bps > 0:
            reason = "TIGHT_TS" if pos.bars_held > tighten_bar else "TRAILING_STOP"
            return self._close_position(bar_close_pnl, bar_time, bar_index, reason)

        if be_active and bar_close_pnl <= be_floor:
            return self._close_position(bar_close_pnl, bar_time, bar_index, "BE_LOCK")

        early_cut_bar3 = self.cfg.exit.early_cut_bar3_mfe
        early_cut_bar4 = self.cfg.exit.early_cut_bar4_mfe
        if pos.bars_held == 3 and pos.highest_profit_bps < early_cut_bar3:
            return self._close_position(bar_close_pnl, bar_time, bar_index, "EARLY_CUT")
        if pos.bars_held == 4 and pos.highest_profit_bps < early_cut_bar4:
            return self._close_position(bar_close_pnl, bar_time, bar_index, "EARLY_CUT")

        if pos.bars_held >= pos.max_bars:
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
