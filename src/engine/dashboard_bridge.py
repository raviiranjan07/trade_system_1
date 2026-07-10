"""Dashboard bridge — translates TrackEvents into dashboard state updates.

The ONLY place trading objects meet the dashboard. Tracks return TrackEvent
notes; the orchestrator hands each note to this bridge; the bridge turns it
into update_model()/add_model_trade() calls on the (duck-typed) dashboard
state. No per-model code: one handle() serves every track, so model #5
gets a dashboard card by registration alone.

Presentation-only math (liq price, live PnL for display) lives here —
never in track.py.
"""

import logging
from dataclasses import asdict

from .risk.exchange_math import calc_liq_distance_bps
from .strategy import Direction
from .track import (
    StrategyTrack,
    TrackEvent,
    KIND_TRADE_CLOSED,
    KIND_POSITION_OPENED,
    KIND_POSITION_UPDATE,
    KIND_ENTRY_SKIPPED,
    KIND_PREDICTION,
)

logger = logging.getLogger(__name__)


def calc_liq_price(entry_price: float, direction: str, wallet: float, qty: float) -> float:
    """Liquidation price for display (moved out of old bot.py)."""
    if qty <= 0:
        return 0.0
    liq_dist_bps = calc_liq_distance_bps(wallet, qty, entry_price)
    if direction == "LONG":
        return round(entry_price * (1 - liq_dist_bps / 10000), 2)
    return round(entry_price * (1 + liq_dist_bps / 10000), 2)


class DashboardBridge:
    """Feeds one dashboard state from any number of tracks."""

    def __init__(self, state):
        """state: web.state.DashboardState (duck-typed — no web import here)."""
        self.state = state

    # ------------------------------------------------------------ setup

    def register(self, track: StrategyTrack, title: str, accent_color: str,
                 daily_expected_bps: float = 0.0, sort_order: int = 99) -> None:
        """Create the track's dashboard card and push its initial state."""
        self.state.register_model(
            track.name,
            title=title,
            accent_color=accent_color,
            exit_version=track.exit_version.upper(),
            daily_expected_bps=daily_expected_bps,
            sort_order=sort_order,
        )
        self.state.update_model(track.name, model_loaded=track.source.loaded)
        self._push_account(track)
        self._push_stats(track)

    # ------------------------------------------------------------ events

    def handle(self, track: StrategyTrack, event: TrackEvent, price: float = 0.0) -> None:
        """Turn one TrackEvent into dashboard updates."""
        if event.kind == KIND_TRADE_CLOSED:
            self._on_trade_closed(track, event)
        elif event.kind == KIND_POSITION_OPENED:
            self._on_position_opened(track, event)
        elif event.kind == KIND_POSITION_UPDATE:
            self._on_position_update(track, price)
        elif event.kind == KIND_PREDICTION:
            self.state.update_model(
                track.name,
                long_pct=event.prediction.long_pct,
                short_pct=event.prediction.short_pct,
                signal_status=event.prediction.status,
                last_prob=event.prediction.prob,
            )
        elif event.kind == KIND_ENTRY_SKIPPED:
            self.state.update_model(track.name, total_skips=track.total_skips)

    def handle_all(self, track: StrategyTrack, events: list, price: float = 0.0) -> None:
        for event in events:
            self.handle(track, event, price=price)

    # ------------------------------------------------------------ internals

    def _on_trade_closed(self, track: StrategyTrack, event: TrackEvent) -> None:
        trade = event.trade
        liq = calc_liq_price(trade.entry_price, trade.direction, track.wallet, event.qty)
        payload = asdict(trade)
        payload["exit_time"] = str(trade.exit_time)
        payload["entry_time"] = str(trade.entry_time)
        payload["signal_time"] = str(trade.signal_time)
        payload["qty"] = event.qty
        payload["pnl_usd"] = event.pnl_usd
        payload["liq_price"] = liq
        self.state.add_model_trade(track.name, payload)

        # Clear the position card, refresh stats + account
        self.state.update_model(
            track.name,
            has_position=False,
            position_side=None,
            position_pnl_bps=0.0,
            position_entry_price=0.0,
            position_highest_profit_bps=0.0,
            position_bars_held=0,
            position_mfe_bps=0.0,
            position_mae_bps=0.0,
            position_liq_price=0.0,
        )
        self._push_stats(track)
        if not event.restored:
            self._push_account(track)

    def _on_position_opened(self, track: StrategyTrack, event: TrackEvent) -> None:
        pos = track.pm.position
        if pos is None:
            return
        liq = calc_liq_price(pos.entry_price, pos.direction.value, track.wallet, track.qty)
        self.state.update_model(
            track.name,
            has_position=True,
            position_side=pos.direction.value,
            position_pnl_bps=0.0,
            position_entry_price=pos.entry_price,
            position_highest_profit_bps=0.0,
            position_bars_held=0,
            position_max_bars=pos.max_bars,
            position_mfe_bps=0.0,
            position_mae_bps=0.0,
            position_liq_price=liq,
            is_reentry=event.is_reentry,
            last_qty=track.qty,
        )

    def _on_position_update(self, track: StrategyTrack, price: float) -> None:
        pos = track.pm.position
        if pos is None or price <= 0:
            return
        if pos.direction == Direction.LONG:
            pnl = (price - pos.entry_price) / pos.entry_price * 10000
        else:
            pnl = (pos.entry_price - price) / pos.entry_price * 10000
        liq = calc_liq_price(pos.entry_price, pos.direction.value, track.wallet, track.qty)
        self.state.update_model(
            track.name,
            has_position=True,
            position_side=pos.direction.value,
            position_pnl_bps=round(pnl, 1),
            position_entry_price=pos.entry_price,
            position_highest_profit_bps=round(pos.highest_profit_bps, 1),
            position_bars_held=pos.bars_held,
            position_max_bars=pos.max_bars,
            position_mfe_bps=round(pos.mfe_bps, 1),
            position_mae_bps=round(pos.mae_bps, 1),
            position_liq_price=liq,
            is_reentry=pos.is_reentry,
        )

    def _push_stats(self, track: StrategyTrack) -> None:
        self.state.update_model(track.name, **asdict(track.stats()))

    def _push_account(self, track: StrategyTrack) -> None:
        self.state.update_model(
            track.name,
            wallet_usd=round(track.wallet, 2),
            drawdown_pct=round(track.health.get_drawdown_pct(track.wallet), 4),
            health_multiplier=round(track.health.get_risk_multiplier(track.wallet), 2),
            consecutive_losses=track.health.consecutive_losses,
            recent_winrate=round(track.health.get_recent_winrate(), 2),
            peak_usd=round(track.health.peak, 2),
            last_qty=track.qty,
            last_pnl_usd=round(track.last_pnl_usd, 4),
            total_skips=track.total_skips,
            model_loaded=track.source.loaded,
        )
