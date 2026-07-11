"""StrategyTrack — the live trading life of ONE strategy (one lane per model).

Contents:
  1. DATA MODELS   — the shapes data travels in
  2. SignalSource  — the socket (one interface)
  3. Plugs         — MLSource (+ model-specific subclasses when needed)
  4. StrategyTrack — the lane: state + lifecycle

Existing data models reused from elsewhere (NOT redefined here):
  TradeRecord, OpenPosition  — engine/position_manager.py
  RiskDecision               — engine/risk/risk_calculator.py
  Signal                     — engine/strategy.py
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .config.schema import AppConfig
from .persistence import StateStore, TradeLog, TrackState  # noqa: F401 (TrackState re-exported)
from .position_manager import V12PositionManager, TradeRecord
from .risk.account_health import AccountHealthMonitor
from .risk.exchange_constants import DEFAULT_CAPITAL
from .risk.risk_calculator import RiskCalculator, RiskDecision
from .risk.decision_logger import DecisionLogger
from .strategy import Signal

logger = logging.getLogger(__name__)


# =====================================================================
# 1. DATA MODELS
# =====================================================================

@dataclass
class Prediction:
    """Model confidence for one bar — display only, not a trade decision."""
    long_pct: float          # P(LONG) as percentage, e.g. 63.2
    short_pct: float         # P(SHORT) as percentage
    status: str              # ">>> ML_LONG SIGNAL <<<" / "NO TRADE (not confident)"
    prob: float              # raw P(LONG) 0..1 (V3: P(LONG) of 3-class softmax)


@dataclass
class EntryProposal:
    """A signal source's answer to 'should we enter on this bar?'."""
    signal: Signal


# --- TrackEvent: the note a track hands back to the orchestrator -----
# `kind` labels which of the five possible happenings this note reports;
# it tells the reader which payload fields below are meaningful.

KIND_TRADE_CLOSED = "trade_closed"        # a trade finished (live or restored)
KIND_POSITION_OPENED = "position_opened"  # entry accepted and opened
KIND_POSITION_UPDATE = "position_update"  # still in position (live PnL refresh)
KIND_ENTRY_SKIPPED = "entry_skipped"      # signal fired but risk said SKIP
KIND_PREDICTION = "prediction"            # model confidence for display


@dataclass
class TrackEvent:
    """One thing that happened inside a track during a lifecycle call.

    Which payload fields are set depends on `kind`:
      trade_closed    → trade, qty, pnl_usd, restored
      position_opened → signal, qty, decision, is_reentry
      entry_skipped   → signal, decision, skip_reason, is_reentry
      prediction      → prediction
      position_update → (no payload — reader pulls live state off the track)
    """
    kind: str
    track: str                                   # track name, e.g. "ML_V1"
    trade: Optional[TradeRecord] = None
    qty: float = 0.0
    pnl_usd: float = 0.0
    signal: Optional[Signal] = None
    is_reentry: bool = False
    decision: Optional[RiskDecision] = None
    skip_reason: str = ""
    prediction: Optional[Prediction] = None
    restored: bool = False                       # True when replayed from CSV


@dataclass
class TrackStats:
    """Aggregate performance over all closed trades of one track."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_bps: float = 0.0
    avg_bps: float = 0.0
    profit_factor: float = 0.0

    @classmethod
    def from_trades(cls, trades: list) -> "TrackStats":
        """The ONE place the stats formulas live (was inlined 6x in old bot.py)."""
        if not trades:
            return cls()
        wins = [t for t in trades if t.net_profit_bps > 0]
        losses = [t for t in trades if t.net_profit_bps <= 0]
        total_bps = sum(t.net_profit_bps for t in trades)
        gross_win = sum(t.net_profit_bps for t in wins)
        gross_loss = abs(sum(t.net_profit_bps for t in losses))
        return cls(
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(trades),
            total_bps=round(total_bps, 1),
            avg_bps=round(total_bps / len(trades), 1),
            profit_factor=round(gross_win / gross_loss, 2) if gross_loss > 0 else 0.0,
        )


# TrackState (the risk-state JSON schema) lives in persistence.py — it is
# the disk schema, so it belongs with the disk code. Re-exported above.


# =====================================================================
# 2. THE SOCKET — SignalSource
# =====================================================================

class SignalSource(ABC):
    """The single socket a StrategyTrack plugs its signal supplier into.

    The track only ever presses these four pins; it never knows what
    sits behind them. Every plug (MLSource and any subclass) must
    implement all four.

    Contract:
      - prepare() is called ONCE per bar close, on the shared indicator
        DataFrame; prediction() and propose_entry() receive its output.
      - prediction() is display-only and must have NO side effects.
      - propose_entry() answers for exactly one bar (bar_idx); returning
        None means "no entry on this bar".
    """

    @property
    @abstractmethod
    def loaded(self) -> bool:
        """Is this source able to produce signals? (model file found, etc.)
        When False the track skips this source entirely."""

    @abstractmethod
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add this source's features to (a copy of) the indicator df.

        Called once per bar close while flat. Must not mutate the shared
        df in place — copy if columns are added.
        """

    @abstractmethod
    def prediction(self, df: pd.DataFrame, bar_idx: int) -> Optional[Prediction]:
        """Model confidence for `bar_idx`, for dashboard display only.

        Returns None when unavailable (features NaN, rule-based source).
        Thresholds used for the status text MUST be the same ones
        propose_entry() trades on — never a separate hard-coded copy.
        """

    @abstractmethod
    def propose_entry(self, df: pd.DataFrame, bar_idx: int,
                      bar_time) -> Optional[EntryProposal]:
        """Should we enter on this bar? None = no. The track still runs
        the risk check on any proposal — a proposal is not yet a trade."""


# =====================================================================
# 3. THE PLUGS — one per signal MECHANISM (not one per model)
# =====================================================================

class MLSource(SignalSource):
    """Plug for binary P(LONG) generators — serves BOTH ML V1 and ML Attn.

    Wraps any BaseSignalGenerator subclass whose predict() returns a
    single LONG probability. Display thresholds come from the generator
    itself (loaded from params.yaml), so the status text can never
    disagree with the signal the generator actually trades.
    """

    def __init__(self, generator):
        self.gen = generator

    @property
    def loaded(self) -> bool:
        return self.gen.loaded

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            return self.gen.compute_features(df.copy())
        except Exception as e:
            logger.warning("compute_features failed: %s", e)
            return df.copy()

    def prediction(self, df: pd.DataFrame, bar_idx: int) -> Optional[Prediction]:
        features = self.gen.get_features_for_bar(df, bar_idx)
        if features is None:
            return None
        prob = self.gen.predict(features)
        if prob > self.gen.conf_long:
            status = f">>> {self.gen.signal_type_long.value} SIGNAL <<<"
        elif prob < (1 - self.gen.conf_short):
            status = f">>> {self.gen.signal_type_short.value} SIGNAL <<<"
        else:
            status = "NO TRADE (not confident)"
        return Prediction(
            long_pct=round(prob * 100, 1),
            short_pct=round((1 - prob) * 100, 1),
            status=status,
            prob=float(prob),
        )

    def propose_entry(self, df, bar_idx, bar_time) -> Optional[EntryProposal]:
        signal = self.gen.predict_bar(df, bar_idx)
        if signal is None:
            return None
        return EntryProposal(signal=signal)


# A model whose display prediction differs from binary P(LONG) (e.g. a
# 3-class LONG/SHORT/SKIP head) gets its own plug: subclass MLSource and
# override prediction() only — prepare/propose_entry stay inherited as
# long as the generator's predict_bar() handles the signal logic.
# (Reference: the retired MLV3Source in git history.)


# =====================================================================
# 4. THE TRACK — one class, one instance per model
# =====================================================================

class StrategyTrack:
    """One strategy's isolated state + lifecycle (see module docstring).

    Everything ISOLATED per model lives here: position manager, wallet,
    health, risk calculator, decision logger, trades CSV, state file.
    Everything SHARED (market feed, indicators, dashboard) stays outside;
    data comes in through method arguments and leaves as TrackEvents.
    """

    def __init__(
        self,
        name: str,
        source: SignalSource,
        config: AppConfig,
        exit_version: str,
        trade_log: TradeLog,
        state_store: StateStore,
        decision_logger: DecisionLogger,
        csv_tag: str,
        worst_loss_bps: float = 865,
        allow_same_bar_entry: bool = True,
    ):
        """
        Args:
            name: track label, e.g. "ML_V1" — used in logs and events.
            source: the plug in this track's signal socket.
            config: shared AppConfig (exit rules, re-entry settings).
            exit_version: "v1" or "v2" — this model's exit rule set.
            trade_log: this track's trades CSV (persistence.TradeLog).
            state_store: this track's wallet/health JSON (persistence.StateStore).
            decision_logger: this track's risk decision audit log.
            csv_tag: value written to the CSV config_hash column.
            worst_loss_bps: worst historical loss for position sizing.
                            TODO: read per-model from its training manifest.
            allow_same_bar_entry: may a new entry open on the same bar a
                trade closed? Default True (all current tracks).
        """
        self.name = name
        self.source = source
        self.cfg = config
        self.pm = V12PositionManager(config, exit_version=exit_version)
        self.exit_version = exit_version

        self.wallet: float = DEFAULT_CAPITAL
        self.qty: float = 0.0
        self.total_skips: int = 0
        self.last_pnl_usd: float = 0.0

        self._worst_loss_bps = worst_loss_bps
        self.health = AccountHealthMonitor()
        self.risk_calc = RiskCalculator(worst_loss_bps=worst_loss_bps, health=self.health)
        self.decision_logger = decision_logger

        self.trade_log = trade_log
        self.state_store = state_store
        self.csv_tag = csv_tag
        self.allow_same_bar_entry = allow_same_bar_entry

        self._load_state()

        if self.source.loaded:
            logger.info("[%s] track ready — signals enabled", self.name)
        else:
            logger.warning("[%s] model not found — signals disabled", self.name)

    # ---------------------------------------------------------- persistence
    # All file mechanics live in persistence.py; the track only says
    # "save my state" / "give me my history".

    def _load_state(self) -> None:
        """Restore wallet/health/skips from the state store (if present)."""
        state = self.state_store.load()
        if state is None:
            self.health.update(self.wallet)
            logger.info("[%s] no risk state — starting fresh: wallet=$%.2f",
                        self.name, self.wallet)
            return
        self.wallet = state.wallet
        self.total_skips = state.total_skips
        if state.health:
            self.health = AccountHealthMonitor.from_dict(state.health)
            self.risk_calc = RiskCalculator(
                worst_loss_bps=self._worst_loss_bps, health=self.health,
            )
        self.health.update(self.wallet)
        logger.info(
            "[%s] loaded risk state: wallet=$%.2f | peak=$%.2f | streak=%d",
            self.name, self.wallet, self.health.peak, self.health.consecutive_losses,
        )

    def save_state(self) -> None:
        self.state_store.save(TrackState(
            wallet=self.wallet,
            total_skips=self.total_skips,
            health=self.health.to_dict(),
        ))

    def restore(self) -> list:
        """Reload past trades into the position manager.

        Returns one restored trade_closed event per row so the orchestrator
        can replay them into the dashboard. Wallet/health were already
        restored by _load_state() in __init__.
        """
        events = []
        for trade, qty in self.trade_log.read_all():
            self.pm.trades.append(trade)
            pnl_usd = qty * trade.entry_price * (trade.net_profit_bps / 10000) if qty > 0 else 0.0
            events.append(TrackEvent(
                kind=KIND_TRADE_CLOSED, track=self.name, trade=trade,
                qty=qty, pnl_usd=round(pnl_usd, 4), restored=True,
            ))
        if events:
            logger.info("[%s] restored %d trades from %s",
                        self.name, len(events), self.trade_log.path.name)
        return events

    # ---------------------------------------------------------- lifecycle

    def on_tick(self, price: float, tick_time) -> list:
        """Price-based exit check on every WebSocket tick."""
        if not self.source.loaded or not self.pm.is_in_position:
            return []
        trade = self.pm.on_tick(price, tick_time)
        if trade is not None:
            logger.info(
                "[%s] TICK-EXIT %s @ %.2f | %s | %+.1f bps",
                self.name, trade.exit_reason, trade.exit_price,
                trade.direction, trade.net_profit_bps,
            )
            return [self._close_trade(trade)]
        return [TrackEvent(kind=KIND_POSITION_UPDATE, track=self.name)]

    def on_bar_close(self, candle: dict, df: pd.DataFrame, bar_index: int) -> list:
        """Full bar-close lifecycle: bar exits, then prediction + entry.

        Args:
            candle: closed candle (open/high/low/close/volume/open_time).
            df: SHARED indicator DataFrame (compute_indicators output).
            bar_index: global bar counter from the orchestrator.
        """
        if not self.source.loaded:
            return []

        events = []
        bar_time = candle["open_time"]
        latest_idx = len(df) - 1

        # In position: bar-level exits (time exit; price exits ran on ticks)
        if self.pm.is_in_position:
            trade = self.pm.on_bar(
                high=candle["high"], low=candle["low"], close=candle["close"],
                bar_time=bar_time, bar_index=bar_index,
            )
            if trade:
                logger.info(
                    "[%s] CLOSED %s (%s) | %s | %+.1f bps | bar %d | %s",
                    self.name, trade.direction, trade.signal_type,
                    trade.exit_reason, trade.net_profit_bps, trade.exit_bar, bar_time,
                )
                events.append(self._close_trade(trade))
                if not self.allow_same_bar_entry:
                    return events
            else:
                events.append(TrackEvent(kind=KIND_POSITION_UPDATE, track=self.name))
                return events

        # Flat: prediction display, then entry attempt
        df_feat = self.source.prepare(df)

        pred = self.source.prediction(df_feat, latest_idx)
        if pred is not None:
            events.append(TrackEvent(kind=KIND_PREDICTION, track=self.name, prediction=pred))
            logger.info("[%s] LONG=%.1f%% SHORT=%.1f%% | %s",
                        self.name, pred.long_pct, pred.short_pct, pred.status)

        proposal = self.source.propose_entry(df_feat, latest_idx, bar_time)
        if proposal is not None:
            events.append(self._attempt_entry(proposal, candle["close"], bar_time))
        return events

    # ---------------------------------------------------------- internals

    def _attempt_entry(self, proposal: EntryProposal, price: float, bar_time) -> TrackEvent:
        """Risk-check a proposal; open the position or record the skip."""
        decision = self.risk_calc.calculate(self.wallet, price)
        self.decision_logger.log_decision(
            decision, self.wallet, price, proposal.signal.signal_type.value,
        )
        if decision.action == "SKIP":
            self.total_skips += 1
            logger.info("[%s] SKIP %s (%s): %s",
                        self.name, proposal.signal.direction.value,
                        proposal.signal.signal_type.value, decision.skip_reason)
            return TrackEvent(
                kind=KIND_ENTRY_SKIPPED, track=self.name, signal=proposal.signal,
                decision=decision, skip_reason=decision.skip_reason,
            )

        self.qty = decision.qty
        self.pm.open_position(
            direction=proposal.signal.direction,
            signal_type=proposal.signal.signal_type,
            entry_price=price,
            entry_time=bar_time,
            signal_time=proposal.signal.timestamp,
        )
        logger.info("[%s] ENTRY %s (%s) @ %.2f | %s",
                    self.name, proposal.signal.direction.value,
                    proposal.signal.signal_type.value, price, bar_time)
        return TrackEvent(
            kind=KIND_POSITION_OPENED, track=self.name, signal=proposal.signal,
            qty=self.qty, decision=decision,
        )

    def _close_trade(self, trade: TradeRecord) -> TrackEvent:
        """THE close chain — the only place a closed trade is processed.

        Fixed order: CSV -> wallet -> health -> decision log -> save -> qty=0.
        """
        closed_qty = self.qty
        self.trade_log.append(trade, self.csv_tag, closed_qty)

        pnl_usd = 0.0
        if closed_qty > 0:
            pnl_usd = closed_qty * trade.entry_price * (trade.net_profit_bps / 10000)
            self.wallet = max(self.wallet + pnl_usd, 0.01)
            self.health.update(self.wallet, trade.net_profit_bps)
            self.decision_logger.log_outcome(trade.net_profit_bps, pnl_usd)
            self.save_state()
            logger.info(
                "[%s] risk: wallet=$%.2f | pnl=%+.2f | qty=%.3f | health=%.2f",
                self.name, self.wallet, pnl_usd, closed_qty,
                self.health.get_risk_multiplier(self.wallet),
            )

        self.qty = 0.0
        self.last_pnl_usd = pnl_usd
        return TrackEvent(
            kind=KIND_TRADE_CLOSED, track=self.name, trade=trade,
            qty=closed_qty, pnl_usd=round(pnl_usd, 4),
        )

    # ---------------------------------------------------------- views

    def stats(self) -> TrackStats:
        """Aggregate performance over all closed trades."""
        return TrackStats.from_trades(self.pm.trades)
