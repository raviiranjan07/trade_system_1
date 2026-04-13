"""V1.3 Trading Bot — Live/Paper trading using Binance WebSocket.

Connects to Binance Futures 15m kline stream and runs V1.3 strategy.
Uses the SAME strategy + position_manager as the backtest (verified identical).

Run: PYTHONPATH=src python -m v12.bot
"""

import asyncio
import csv
import logging
import signal
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from .config.constants import SYMBOL, TIMEFRAME
from .config.loader import load_config
from .config.schema import AppConfig
from .position_manager import V12PositionManager, TradeRecord
from .signals.direction_v15 import DirectionV15 as MLSignalGenerator
from .signals.direction_attention import DirectionAttention as MLAttnSignalGenerator
from .strategy import V12Strategy, Direction, Signal, SignalType
from .risk.risk_calculator import RiskCalculator, RiskConfig, RiskDecision
from .risk.account_health import AccountHealthMonitor, HealthConfig
from .risk.exchange_constants import DEFAULT_CAPITAL
from .risk.exchange_math import calc_liq_distance_bps
from .risk.tests.decision_logger import DecisionLogger
from .risk.tests.preflight import run_preflight

logger = logging.getLogger(__name__)

# Need 200+ bars for SMA200 warm-up + margin
MIN_BUFFER_SIZE = 300
BUFFER_SIZE = 500


def _calc_liq_price(entry_price: float, direction: str, wallet: float, qty: float) -> float:
    """Calculate liquidation price from entry, direction, wallet, qty."""
    if qty <= 0:
        return 0.0
    liq_dist_bps = calc_liq_distance_bps(wallet, qty, entry_price)
    if direction == "LONG":
        return round(entry_price * (1 - liq_dist_bps / 10000), 2)
    else:
        return round(entry_price * (1 + liq_dist_bps / 10000), 2)



class V12Bot:
    """Live/Paper trading bot for V1.3 strategy."""

    def __init__(self, config: AppConfig):
        self.cfg = config
        self.strategy = V12Strategy(config)
        self.pm = V12PositionManager(config)

        # Track bar index for position manager
        self._bar_count = 0
        self._last_signal: Optional[Signal] = None

        # Trade logging
        self._trades_dir = Path("data/trades")
        self._trades_dir.mkdir(parents=True, exist_ok=True)
        self._trades_csv = self._trades_dir / f"trades_{config.execution.mode}.csv"
        self._init_csv()

        # Risk management
        self._wallet = DEFAULT_CAPITAL
        self._health = AccountHealthMonitor()
        self._risk_calc = RiskCalculator(worst_loss_bps=865, health=self._health)
        self._decision_logger = DecisionLogger()
        self._current_qty = 0.0
        self._current_decision: Optional[RiskDecision] = None
        self._total_skips = 0
        self._risk_state_path = Path("data/trades/risk_state.json")
        self._load_risk_state()

        # ML signal generator — separate position manager, wallet, and CSV
        self._ml_gen = MLSignalGenerator(
            model_path=Path("models/direction_v15/direction_model.onnx"),
            scaler_path=Path("models/direction_v15/scaler.npz"),
        )
        self._ml_pm = V12PositionManager(config)
        self._ml_wallet = DEFAULT_CAPITAL
        self._ml_health = AccountHealthMonitor()
        self._ml_risk_calc = RiskCalculator(worst_loss_bps=865, health=self._ml_health)
        self._ml_decision_logger = DecisionLogger(
            log_dir="data/trades/risk_logs/ml"
        )
        self._ml_qty = 0.0
        self._ml_trades_csv = self._trades_dir / f"trades_ml_{config.execution.mode}.csv"
        self._init_ml_csv()
        self._ml_risk_state_path = Path("data/trades/risk_state_ml.json")
        self._load_ml_risk_state()
        if self._ml_gen.loaded:
            logger.info("ML V1.5 model loaded — ML_LONG/ML_SHORT signals enabled")
        else:
            logger.warning("ML V1.5 model not found — ML signals disabled")

        # ML Attention signal generator — A/B test alongside V1.5
        self._ml_attn_gen = MLAttnSignalGenerator(
            model_path=Path("models/direction_attention/attention_model.onnx"),
            scaler_path=Path("models/direction_attention/scaler.npz"),
        )
        self._ml_attn_pm = V12PositionManager(config)
        self._ml_attn_wallet = DEFAULT_CAPITAL
        self._ml_attn_health = AccountHealthMonitor()
        self._ml_attn_risk_calc = RiskCalculator(worst_loss_bps=865, health=self._ml_attn_health)
        self._ml_attn_decision_logger = DecisionLogger(
            log_dir="data/trades/risk_logs/ml_attn"
        )
        self._ml_attn_qty = 0.0
        self._ml_attn_trades_csv = self._trades_dir / f"trades_ml_attn_{config.execution.mode}.csv"
        self._init_ml_attn_csv()
        self._ml_attn_risk_state_path = Path("data/trades/risk_state_ml_attn.json")
        self._load_ml_attn_risk_state()
        if self._ml_attn_gen.loaded:
            logger.info("ML Attention model loaded — ML_ATTN_LONG/ML_ATTN_SHORT signals enabled")
        else:
            logger.warning("ML Attention model not found — ML_ATTN signals disabled")

        # Session state
        self._running = False
        self._connector = None

        # Dashboard state (optional — set via set_dashboard())
        self._dashboard = None
        self._chart_history_loaded = False

    def _load_risk_state(self) -> None:
        """Load wallet + health state from disk (survives bot restarts)."""
        if not self._risk_state_path.exists():
            self._health.update(self._wallet)
            logger.info("No risk state found — starting fresh: wallet=$%.2f", self._wallet)
            return
        import json
        try:
            with open(self._risk_state_path) as f:
                state = json.load(f)
            self._wallet = state.get('wallet', DEFAULT_CAPITAL)
            health_data = state.get('health', {})
            if health_data:
                self._health = AccountHealthMonitor.from_dict(health_data)
                self._risk_calc = RiskCalculator(worst_loss_bps=865, health=self._health)
            self._health.update(self._wallet)
            logger.info(
                "Loaded risk state: wallet=$%.2f | peak=$%.2f | streak=%d",
                self._wallet, self._health.peak, self._health.consecutive_losses,
            )
        except Exception as e:
            logger.warning("Failed to load risk state: %s — starting fresh", e)
            self._health.update(self._wallet)

    def _save_risk_state(self) -> None:
        """Save wallet + health state to disk."""
        import json
        state = {
            'wallet': self._wallet,
            'health': self._health.to_dict(),
        }
        with open(self._risk_state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def _update_risk_after_trade(self, trade: TradeRecord) -> None:
        """Update wallet, health monitor, and decision logger after trade close."""
        if self._current_qty > 0:
            pnl_usd = self._current_qty * trade.entry_price * (trade.net_profit_bps / 10000)
            self._wallet += pnl_usd
            self._wallet = max(self._wallet, 0.01)

            self._health.update(self._wallet, trade.net_profit_bps)
            self._decision_logger.log_outcome(trade.net_profit_bps, pnl_usd)
            self._save_risk_state()

            logger.info(
                "Risk: wallet=$%.2f | pnl=%+.2f | qty=%.3f | health=%.2f",
                self._wallet, pnl_usd, self._current_qty,
                self._health.get_risk_multiplier(self._wallet),
            )
            self._push_risk_to_dashboard(last_pnl_usd=pnl_usd)

        self._current_qty = 0.0
        self._current_decision = None

    def _push_risk_to_dashboard(self, last_pnl_usd: float = 0.0) -> None:
        """Push current risk state to dashboard."""
        if not self._dashboard:
            return
        self._dashboard.update_risk(
            wallet_usd=self._wallet,
            drawdown_pct=self._health.get_drawdown_pct(self._wallet),
            health_multiplier=self._health.get_risk_multiplier(self._wallet),
            consecutive_losses=self._health.consecutive_losses,
            recent_winrate=self._health.get_recent_winrate(),
            peak_usd=self._health.peak,
            last_qty=self._current_qty,
            last_pnl_usd=last_pnl_usd,
            total_skips=self._total_skips,
        )

    def set_dashboard(self, dashboard_state) -> None:
        """Attach dashboard state for real-time UI updates."""
        self._dashboard = dashboard_state
        # Push config to dashboard
        dashboard_state.set_config(
            pair=SYMBOL,
            timeframe=TIMEFRAME,
            mode=self.cfg.execution.mode,
            trailing_stop_long=self.cfg.exit.long_trailing_stop_bps,
            trailing_stop_short=self.cfg.exit.short_trailing_stop_bps,
            max_bars=self.cfg.exit.max_bars,
            reentry_enabled=self.cfg.reentry.enabled,
            config_hash=self.cfg.config_hash(),
        )
        # Push initial risk state
        try:
            self._push_risk_to_dashboard()
        except Exception as e:
            logger.warning("Failed to push initial risk state: %s", e)
        # Push initial ML Attention state
        dashboard_state.update_ml_attn(
            ml_attn_model_loaded=self._ml_attn_gen.loaded,
            ml_attn_wallet_usd=round(self._ml_attn_wallet, 2),
        )

        # Push initial ML state (trades loaded later by _load_ml_trades_from_csv)
        try:
            self._push_ml_to_dashboard()
        except Exception as e:
            logger.warning("Failed to push initial ML state: %s", e)

        # Reload previous trades so dashboard survives restarts
        try:
            self._load_trades_from_csv()
        except Exception as e:
            logger.error("Failed to load V1.4 trades from CSV: %s", e)

        # Load ML trades independently (not nested in V1.4 loading)
        try:
            self._load_ml_trades_from_csv()
        except Exception as e:
            logger.error("Failed to load ML trades from CSV: %s", e)

        # Load ML Attention trades
        try:
            self._load_ml_attn_trades_from_csv()
        except Exception as e:
            logger.error("Failed to load ML Attention trades from CSV: %s", e)

    def _load_trades_from_csv(self) -> None:
        """Load previous trades from CSV into position manager + dashboard."""
        if not self._trades_csv.exists():
            logger.warning("Trades CSV not found: %s", self._trades_csv)
            return
        try:
            df = pd.read_csv(self._trades_csv)
        except Exception as e:
            logger.error("Failed to read trades CSV: %s", e)
            return
        if df.empty:
            return

        logger.info("Reloading %d previous trades from %s", len(df), self._trades_csv.name)

        for _, row in df.iterrows():
            try:
                direction = str(row.get("direction", "LONG"))
                trade = TradeRecord(
                    signal_time=row.get("signal_time", ""),
                    entry_time=row.get("entry_time", ""),
                    exit_time=row.get("exit_time", ""),
                    direction=direction,
                    signal_type=str(row.get("signal_type", "V12_LONG" if direction == "LONG" else "V12_SHORT")),
                    entry_price=float(row.get("entry_price", 0)),
                    exit_price=float(row.get("exit_price", 0)),
                    gross_profit_bps=float(row.get("gross_profit_bps", 0)),
                    net_profit_bps=float(row.get("net_profit_bps", 0)),
                    mfe_bps=float(row.get("mfe_bps", 0)),
                    mae_bps=float(row.get("mae_bps", 0)),
                    exit_bar=int(row.get("exit_bar", 0)),
                    exit_reason=str(row.get("exit_reason", "")),
                    is_reentry=str(row.get("is_reentry", "False")).strip().lower() == "true",
                )
                self.pm.trades.append(trade)
            except Exception as e:
                logger.warning("Failed to load V1.4 trade row: %s", e)
                continue

        # Push trades + stats to dashboard
        if not self._dashboard:
            return

        for i, trade in enumerate(self.pm.trades):
            try:
                row = df.iloc[i] if i < len(df) else None
                qty = float(row.get("qty", 0)) if row is not None and "qty" in row and pd.notna(row.get("qty")) else 0.0
                pnl_usd = qty * trade.entry_price * (trade.net_profit_bps / 10000) if qty > 0 else 0
                self._dashboard.add_trade({
                    "trade_id": f"R{i+1}",
                    "direction": trade.direction,
                    "signal_type": trade.signal_type,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "net_profit_bps": trade.net_profit_bps,
                    "mfe_bps": trade.mfe_bps,
                    "mae_bps": trade.mae_bps,
                    "exit_bar": trade.exit_bar,
                    "exit_reason": trade.exit_reason,
                    "is_reentry": trade.is_reentry,
                    "exit_time": str(trade.exit_time),
                    "entry_time": str(trade.entry_time),
                    "qty": qty,
                    "pnl_usd": round(pnl_usd, 4),
                })
            except Exception as e:
                logger.warning("Failed to push V1.4 trade to dashboard: %s", e)
                continue

        trades = self.pm.trades
        wins = [t for t in trades if t.net_profit_bps > 0]
        losses = [t for t in trades if t.net_profit_bps <= 0]
        total_bps = sum(t.net_profit_bps for t in trades)
        gross_win = sum(t.net_profit_bps for t in wins) if wins else 0
        gross_loss = abs(sum(t.net_profit_bps for t in losses)) if losses else 0

        self._dashboard.update_stats(
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(trades) if trades else 0,
            total_bps=round(total_bps, 1),
            avg_bps=round(total_bps / len(trades), 1) if trades else 0,
            profit_factor=round(gross_win / gross_loss, 2) if gross_loss > 0 else 0,
        )
        logger.info(
            "Restored dashboard: %d trades, %+.0f bps, PF %.2f",
            len(trades), total_bps,
            round(gross_win / gross_loss, 2) if gross_loss > 0 else 0,
        )

    def _load_ml_trades_from_csv(self) -> None:
        """Load previous ML trades from CSV into dashboard."""
        if not self._ml_trades_csv.exists():
            logger.warning("ML trades CSV not found: %s", self._ml_trades_csv)
            return
        try:
            df = pd.read_csv(self._ml_trades_csv)
        except Exception as e:
            logger.error("Failed to read ML trades CSV: %s", e)
            return
        if df.empty:
            return

        logger.info("Reloading %d ML trades from %s", len(df), self._ml_trades_csv.name)

        for _, row in df.iterrows():
            try:
                trade = TradeRecord(
                    signal_time=row.get("signal_time", ""),
                    entry_time=row.get("entry_time", ""),
                    exit_time=row.get("exit_time", ""),
                    direction=str(row.get("direction", "LONG")),
                    signal_type=str(row.get("signal_type", "ML_LONG")),
                    entry_price=float(row.get("entry_price", 0)),
                    exit_price=float(row.get("exit_price", 0)),
                    gross_profit_bps=float(row.get("gross_profit_bps", 0)),
                    net_profit_bps=float(row.get("net_profit_bps", 0)),
                    mfe_bps=float(row.get("mfe_bps", 0)),
                    mae_bps=float(row.get("mae_bps", 0)),
                    exit_bar=int(row.get("exit_bar", 0)),
                    exit_reason=str(row.get("exit_reason", "")),
                    is_reentry=str(row.get("is_reentry", "False")).strip().lower() == "true",
                )
                self._ml_pm.trades.append(trade)

                qty = float(row.get("qty", 0)) if "qty" in row and pd.notna(row.get("qty")) else 0.0
                self._dashboard.add_ml_trade({
                    "direction": str(row.get("direction", "LONG")),
                    "signal_type": str(row.get("signal_type", "ML_LONG")),
                    "entry_price": float(row.get("entry_price", 0)),
                    "exit_price": float(row.get("exit_price", 0)),
                    "net_profit_bps": float(row.get("net_profit_bps", 0)),
                    "mfe_bps": float(row.get("mfe_bps", 0)),
                    "mae_bps": float(row.get("mae_bps", 0)),
                    "exit_bar": int(row.get("exit_bar", 0)),
                    "exit_reason": str(row.get("exit_reason", "")),
                    "exit_time": str(row.get("exit_time", "")),
                    "entry_time": str(row.get("entry_time", "")),
                    "qty": float(row.get("qty", 0)),
                })
            except Exception as e:
                logger.warning("Failed to load ML trade row: %s", e)
                continue

        ml_trades = df
        ml_wins = len(ml_trades[ml_trades["net_profit_bps"] > 0])
        ml_losses = len(ml_trades[ml_trades["net_profit_bps"] <= 0])
        ml_total_bps = ml_trades["net_profit_bps"].sum()

        self._dashboard.update_ml(
            ml_wallet_usd=round(self._ml_wallet, 2),
            ml_model_loaded=self._ml_gen.loaded,
            ml_total_trades=len(ml_trades),
            ml_wins=ml_wins,
            ml_losses=ml_losses,
            ml_total_bps=round(ml_total_bps, 1),
            ml_avg_bps=round(ml_total_bps / len(ml_trades), 1) if len(ml_trades) > 0 else 0,
            ml_win_rate=round(ml_wins / len(ml_trades), 3) if len(ml_trades) > 0 else 0,
        )
        logger.info(
            "Restored ML dashboard: %d trades, %+.1f bps, wallet=$%.2f",
            len(ml_trades), ml_total_bps, self._ml_wallet,
        )

    def _load_ml_attn_trades_from_csv(self) -> None:
        """Load previous ML Attention trades from CSV into dashboard."""
        if not self._ml_attn_trades_csv.exists():
            return
        try:
            df = pd.read_csv(self._ml_attn_trades_csv)
        except Exception as e:
            logger.error("Failed to read ML Attention trades CSV: %s", e)
            return
        if df.empty:
            return

        logger.info("Reloading %d ML Attention trades from %s", len(df), self._ml_attn_trades_csv.name)

        for _, row in df.iterrows():
            try:
                trade = TradeRecord(
                    signal_time=row.get("signal_time", ""),
                    entry_time=row.get("entry_time", ""),
                    exit_time=row.get("exit_time", ""),
                    direction=str(row.get("direction", "LONG")),
                    signal_type=str(row.get("signal_type", "ML_ATTN_LONG")),
                    entry_price=float(row.get("entry_price", 0)),
                    exit_price=float(row.get("exit_price", 0)),
                    gross_profit_bps=float(row.get("gross_profit_bps", 0)),
                    net_profit_bps=float(row.get("net_profit_bps", 0)),
                    mfe_bps=float(row.get("mfe_bps", 0)),
                    mae_bps=float(row.get("mae_bps", 0)),
                    exit_bar=int(row.get("exit_bar", 0)),
                    exit_reason=str(row.get("exit_reason", "")),
                    is_reentry=str(row.get("is_reentry", "False")).strip().lower() == "true",
                )
                self._ml_attn_pm.trades.append(trade)

                if self._dashboard:
                    ep = float(row.get("entry_price", 0))
                    d = str(row.get("direction", "LONG"))
                    q = float(row.get("qty", 0)) if "qty" in row and pd.notna(row.get("qty")) else 0.0
                    liq = _calc_liq_price(ep, d, self._ml_attn_wallet, q) if q > 0 else 0.0
                    self._dashboard.add_ml_attn_trade({
                        "direction": d,
                        "signal_type": str(row.get("signal_type", "ML_ATTN_LONG")),
                        "entry_price": ep,
                        "exit_price": float(row.get("exit_price", 0)),
                        "net_profit_bps": float(row.get("net_profit_bps", 0)),
                        "mfe_bps": float(row.get("mfe_bps", 0)),
                        "mae_bps": float(row.get("mae_bps", 0)),
                        "exit_bar": int(row.get("exit_bar", 0)),
                        "exit_reason": str(row.get("exit_reason", "")),
                        "exit_time": str(row.get("exit_time", "")),
                        "entry_time": str(row.get("entry_time", "")),
                        "qty": q,
                        "liq_price": liq,
                    })
            except Exception as e:
                logger.warning("Failed to load ML Attention trade row: %s", e)
                continue

        wins = len(df[df["net_profit_bps"] > 0])
        total_bps = df["net_profit_bps"].sum()

        self._push_ml_attn_to_dashboard()
        logger.info(
            "Restored ML Attention dashboard: %d trades, %+.1f bps, wallet=$%.2f",
            len(df), total_bps, self._ml_attn_wallet,
        )

    def _init_csv(self) -> None:
        """Initialize CSV for trade logging."""
        if not self._trades_csv.exists():
            headers = [
                "signal_time", "entry_time", "exit_time", "direction", "signal_type",
                "entry_price", "exit_price", "gross_profit_bps", "net_profit_bps",
                "mfe_bps", "mae_bps", "exit_bar", "exit_reason", "is_reentry",
                "config_hash", "qty",
            ]
            with open(self._trades_csv, "w", newline="") as f:
                csv.writer(f).writerow(headers)
            logger.info("Created trades CSV: %s", self._trades_csv)

    def _init_ml_csv(self) -> None:
        """Initialize CSV for ML trade logging."""
        if not self._ml_trades_csv.exists():
            headers = [
                "signal_time", "entry_time", "exit_time", "direction", "signal_type",
                "entry_price", "exit_price", "gross_profit_bps", "net_profit_bps",
                "mfe_bps", "mae_bps", "exit_bar", "exit_reason", "is_reentry",
                "config_hash", "qty",
            ]
            with open(self._ml_trades_csv, "w", newline="") as f:
                csv.writer(f).writerow(headers)
            logger.info("Created ML trades CSV: %s", self._ml_trades_csv)

    def _load_ml_risk_state(self) -> None:
        """Load ML wallet + health state from disk."""
        if not self._ml_risk_state_path.exists():
            self._ml_health.update(self._ml_wallet)
            logger.info("No ML risk state — starting fresh: wallet=$%.2f", self._ml_wallet)
            return
        import json
        try:
            with open(self._ml_risk_state_path) as f:
                state = json.load(f)
            self._ml_wallet = state.get('wallet', DEFAULT_CAPITAL)
            health_data = state.get('health', {})
            if health_data:
                self._ml_health = AccountHealthMonitor.from_dict(health_data)
                self._ml_risk_calc = RiskCalculator(worst_loss_bps=865, health=self._ml_health)
            self._ml_health.update(self._ml_wallet)
            logger.info("Loaded ML risk state: wallet=$%.2f", self._ml_wallet)
        except Exception as e:
            logger.warning("Failed to load ML risk state: %s — starting fresh", e)
            self._ml_health.update(self._ml_wallet)

    def _save_ml_risk_state(self) -> None:
        """Save ML wallet + health state to disk."""
        import json
        state = {
            'wallet': self._ml_wallet,
            'health': self._ml_health.to_dict(),
        }
        with open(self._ml_risk_state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def _log_ml_trade(self, trade: TradeRecord) -> None:
        """Append ML trade to separate CSV."""
        row = [
            trade.signal_time, trade.entry_time, trade.exit_time,
            trade.direction, trade.signal_type,
            f"{trade.entry_price:.2f}", f"{trade.exit_price:.2f}",
            f"{trade.gross_profit_bps:.1f}", f"{trade.net_profit_bps:.1f}",
            f"{trade.mfe_bps:.1f}", f"{trade.mae_bps:.1f}",
            trade.exit_bar, trade.exit_reason, trade.is_reentry,
            "ml_v1", f"{self._ml_qty:.4f}",
        ]
        with open(self._ml_trades_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _update_ml_risk_after_trade(self, trade: TradeRecord) -> None:
        """Update ML wallet and health after trade close."""
        if self._ml_qty > 0:
            pnl_usd = self._ml_qty * trade.entry_price * (trade.net_profit_bps / 10000)
            self._ml_wallet += pnl_usd
            self._ml_wallet = max(self._ml_wallet, 0.01)
            self._ml_health.update(self._ml_wallet, trade.net_profit_bps)
            self._ml_decision_logger.log_outcome(trade.net_profit_bps, pnl_usd)
            self._save_ml_risk_state()
            logger.info(
                "ML Risk: wallet=$%.2f | pnl=%+.2f | qty=%.3f",
                self._ml_wallet, pnl_usd, self._ml_qty,
            )
            self._push_ml_to_dashboard(last_pnl_usd=pnl_usd)
        self._ml_qty = 0.0

    def _push_ml_to_dashboard(self, last_pnl_usd: float = 0.0) -> None:
        """Push ML state to dashboard."""
        if not self._dashboard:
            return
        self._dashboard.update_ml(
            ml_wallet_usd=round(self._ml_wallet, 2),
            ml_total_trades=len([t for t in self._ml_pm.trades]),
            ml_has_position=self._ml_pm.is_in_position,
            ml_model_loaded=self._ml_gen.loaded,
            ml_drawdown_pct=self._ml_health.get_drawdown_pct(self._ml_wallet),
            ml_health_multiplier=self._ml_health.get_risk_multiplier(self._ml_wallet),
            ml_consecutive_losses=self._ml_health.consecutive_losses,
            ml_recent_winrate=self._ml_health.get_recent_winrate(),
            ml_peak_usd=self._ml_health.peak,
            ml_last_pnl_usd=last_pnl_usd,
            ml_last_qty=self._ml_qty,
            ml_total_skips=0,
        )

    def _push_ml_trade_to_dashboard(self, trade: TradeRecord) -> None:
        """Push completed ML trade to dashboard."""
        if not self._dashboard:
            return
        liq_price = _calc_liq_price(trade.entry_price, trade.direction, self._ml_wallet, self._ml_qty)
        self._dashboard.add_ml_trade({
            "direction": trade.direction,
            "signal_type": trade.signal_type,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "net_profit_bps": trade.net_profit_bps,
            "mfe_bps": trade.mfe_bps,
            "mae_bps": trade.mae_bps,
            "exit_bar": trade.exit_bar,
            "exit_reason": trade.exit_reason,
            "is_reentry": trade.is_reentry,
            "exit_time": str(trade.exit_time),
            "entry_time": str(trade.entry_time),
            "qty": self._ml_qty,
            "liq_price": liq_price,
        })
        # Update ML stats
        ml_trades = self._ml_pm.trades
        if ml_trades:
            wins = sum(1 for t in ml_trades if t.net_profit_bps > 0)
            total = len(ml_trades)
            total_bps = sum(t.net_profit_bps for t in ml_trades)
            self._dashboard.update_ml(
                ml_wallet_usd=round(self._ml_wallet, 2),
                ml_total_trades=total,
                ml_wins=wins,
                ml_losses=total - wins,
                ml_win_rate=round(wins / total, 3) if total > 0 else 0,
                ml_total_bps=round(total_bps, 1),
                ml_avg_bps=round(total_bps / total, 1) if total > 0 else 0,
                ml_has_position=False,
                ml_model_loaded=True,
            )

    # =================================================================
    # ML ATTENTION (A/B test) — mirrors ML V1.5 methods above
    # =================================================================

    def _init_ml_attn_csv(self) -> None:
        """Initialize CSV for ML Attention trade logging."""
        if not self._ml_attn_trades_csv.exists():
            headers = [
                "signal_time", "entry_time", "exit_time", "direction", "signal_type",
                "entry_price", "exit_price", "gross_profit_bps", "net_profit_bps",
                "mfe_bps", "mae_bps", "exit_bar", "exit_reason", "is_reentry",
                "config_hash", "qty",
            ]
            with open(self._ml_attn_trades_csv, "w", newline="") as f:
                csv.writer(f).writerow(headers)
            logger.info("Created ML Attention trades CSV: %s", self._ml_attn_trades_csv)

    def _load_ml_attn_risk_state(self) -> None:
        """Load ML Attention wallet + health state from disk."""
        if not self._ml_attn_risk_state_path.exists():
            self._ml_attn_health.update(self._ml_attn_wallet)
            logger.info("No ML Attention risk state — starting fresh: wallet=$%.2f", self._ml_attn_wallet)
            return
        import json
        try:
            with open(self._ml_attn_risk_state_path) as f:
                state = json.load(f)
            self._ml_attn_wallet = state.get('wallet', DEFAULT_CAPITAL)
            health_data = state.get('health', {})
            if health_data:
                self._ml_attn_health = AccountHealthMonitor.from_dict(health_data)
                self._ml_attn_risk_calc = RiskCalculator(worst_loss_bps=865, health=self._ml_attn_health)
            self._ml_attn_health.update(self._ml_attn_wallet)
            logger.info("Loaded ML Attention risk state: wallet=$%.2f", self._ml_attn_wallet)
        except Exception as e:
            logger.warning("Failed to load ML Attention risk state: %s — starting fresh", e)
            self._ml_attn_health.update(self._ml_attn_wallet)

    def _save_ml_attn_risk_state(self) -> None:
        """Save ML Attention wallet + health state to disk."""
        import json
        state = {
            'wallet': self._ml_attn_wallet,
            'health': self._ml_attn_health.to_dict(),
        }
        with open(self._ml_attn_risk_state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def _log_ml_attn_trade(self, trade: TradeRecord) -> None:
        """Append ML Attention trade to separate CSV."""
        row = [
            trade.signal_time, trade.entry_time, trade.exit_time,
            trade.direction, trade.signal_type,
            f"{trade.entry_price:.2f}", f"{trade.exit_price:.2f}",
            f"{trade.gross_profit_bps:.1f}", f"{trade.net_profit_bps:.1f}",
            f"{trade.mfe_bps:.1f}", f"{trade.mae_bps:.1f}",
            trade.exit_bar, trade.exit_reason, trade.is_reentry,
            "ml_attn_v1", f"{self._ml_attn_qty:.4f}",
        ]
        with open(self._ml_attn_trades_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _update_ml_attn_risk_after_trade(self, trade: TradeRecord) -> None:
        """Update ML Attention wallet and health after trade close."""
        if self._ml_attn_qty > 0:
            pnl_usd = self._ml_attn_qty * trade.entry_price * (trade.net_profit_bps / 10000)
            self._ml_attn_wallet += pnl_usd
            self._ml_attn_wallet = max(self._ml_attn_wallet, 0.01)
            self._ml_attn_health.update(self._ml_attn_wallet, trade.net_profit_bps)
            self._ml_attn_decision_logger.log_outcome(trade.net_profit_bps, pnl_usd)
            self._save_ml_attn_risk_state()
            logger.info(
                "ML Attn Risk: wallet=$%.2f | pnl=%+.2f | qty=%.3f",
                self._ml_attn_wallet, pnl_usd, self._ml_attn_qty,
            )
            self._push_ml_attn_to_dashboard()
        self._ml_attn_qty = 0.0

    def _push_ml_attn_to_dashboard(self) -> None:
        """Push ML Attention state to dashboard."""
        if not self._dashboard:
            return
        attn_trades = self._ml_attn_pm.trades
        wins = sum(1 for t in attn_trades if t.net_profit_bps > 0)
        total = len(attn_trades)
        total_bps = sum(t.net_profit_bps for t in attn_trades)
        self._dashboard.update_ml_attn(
            ml_attn_wallet_usd=round(self._ml_attn_wallet, 2),
            ml_attn_total_trades=total,
            ml_attn_wins=wins,
            ml_attn_losses=total - wins,
            ml_attn_win_rate=round(wins / total, 3) if total > 0 else 0,
            ml_attn_total_bps=round(total_bps, 1),
            ml_attn_avg_bps=round(total_bps / total, 1) if total > 0 else 0,
            ml_attn_has_position=self._ml_attn_pm.is_in_position,
            ml_attn_model_loaded=self._ml_attn_gen.loaded,
        )

    def _log_trade(self, trade: TradeRecord) -> None:
        """Append trade to CSV."""
        row = [
            trade.signal_time, trade.entry_time, trade.exit_time,
            trade.direction, trade.signal_type,
            f"{trade.entry_price:.2f}", f"{trade.exit_price:.2f}",
            f"{trade.gross_profit_bps:.1f}", f"{trade.net_profit_bps:.1f}",
            f"{trade.mfe_bps:.1f}", f"{trade.mae_bps:.1f}",
            trade.exit_bar, trade.exit_reason, trade.is_reentry,
            self.cfg.config_hash(), f"{self._current_qty:.4f}",
        ]
        with open(self._trades_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _push_trade_to_dashboard(self, trade: TradeRecord) -> None:
        """Push completed trade + updated stats to dashboard."""
        if not self._dashboard:
            return

        pnl_usd = self._current_qty * trade.entry_price * (trade.net_profit_bps / 10000) if self._current_qty > 0 else 0
        liq_price = _calc_liq_price(trade.entry_price, trade.direction, self._wallet, self._current_qty)
        self._dashboard.add_trade({
            "trade_id": f"{trade.direction[0]}{self._bar_count}",
            "direction": trade.direction,
            "signal_type": trade.signal_type,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "net_profit_bps": trade.net_profit_bps,
            "mfe_bps": trade.mfe_bps,
            "mae_bps": trade.mae_bps,
            "exit_bar": trade.exit_bar,
            "exit_reason": trade.exit_reason,
            "is_reentry": trade.is_reentry,
            "exit_time": str(trade.exit_time),
            "qty": self._current_qty,
            "pnl_usd": round(pnl_usd, 4),
            "liq_price": liq_price,
        })

        # Clear position
        self._dashboard.clear_position()

        # Update stats
        trades = self.pm.trades
        wins = [t for t in trades if t.net_profit_bps > 0]
        losses = [t for t in trades if t.net_profit_bps <= 0]
        total_bps = sum(t.net_profit_bps for t in trades)
        gross_win = sum(t.net_profit_bps for t in wins) if wins else 0
        gross_loss = abs(sum(t.net_profit_bps for t in losses)) if losses else 0

        self._dashboard.update_stats(
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(trades) if trades else 0,
            total_bps=round(total_bps, 1),
            avg_bps=round(total_bps / len(trades), 1) if trades else 0,
            profit_factor=round(gross_win / gross_loss, 2) if gross_loss > 0 else 0,
        )

    def _load_chart_history(self) -> None:
        """Load 15m candle history from Binance REST API and push to dashboard.

        Uses native Binance 15m candles (not aggregated from 1-min DB data)
        so SMA200 and other indicators match Binance exactly.
        Falls back to TimescaleDB if the Binance API call fails.
        """
        if not self._dashboard:
            return

        df = self._fetch_chart_from_binance()
        if df is None:
            df = self._fetch_chart_from_db()
        if df is None:
            return

        # Compute indicators on full dataset (SMA200 needs warmup)
        df = self.strategy.compute_indicators(df)
        df = df.dropna(subset=["open", "high", "low", "close"])
        # Trim warmup: keep only last 11 days for display
        cutoff = df.index.max() - pd.Timedelta(days=11)
        df = df[df.index >= cutoff]

        # Convert to chart format
        candles = []
        for idx, row in df.iterrows():
            candles.append({
                "time": int(idx.timestamp()),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": round(float(row["volume"]), 2),
                "sma": round(float(row["sma"]), 2) if pd.notna(row.get("sma")) else None,
                "rsi": round(float(row["rsi"]), 1) if pd.notna(row.get("rsi")) else None,
            })

        self._dashboard.set_candle_history(candles)

        # Also set status indicators so top cards match chart from startup
        last_row = df.iloc[-1]
        regime = "BULL" if last_row.get("bull_market", False) else "BEAR"
        self._dashboard.update_status(
            price=round(float(last_row["close"]), 2),
            regime=regime,
            rsi=round(float(last_row["rsi"]), 1) if pd.notna(last_row.get("rsi")) else 0,
            atr_percentile=round(float(last_row["atr_percentile"]), 1) if pd.notna(last_row.get("atr_percentile")) else 0,
            ema_separation=round(float(last_row["ema_separation"]), 4) if pd.notna(last_row.get("ema_separation")) else 0,
        )

        # Push signal proximity at startup so tracker shows immediately
        self._dashboard.update_proximity(self._compute_proximity(last_row))

        self._chart_history_loaded = True
        logger.info("Loaded %d 15m candles for chart display", len(candles))

    def _fetch_chart_from_binance(self) -> Optional[pd.DataFrame]:
        """Fetch native 15m candles from Binance REST API.

        Returns ~1500 candles (~15.6 days) which provides 11 days display
        plus warmup for SMA200. Returns None on failure.
        """
        import requests as req

        logger.info("Loading chart history from Binance REST API...")
        try:
            resp = req.get(
                "https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": SYMBOL, "interval": TIMEFRAME, "limit": 1500},
                timeout=15,
            )
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:
            logger.warning("Binance API chart fetch failed: %s", e)
            return None

        if not raw:
            logger.warning("No candles returned from Binance API")
            return None

        # Binance kline: [open_time, open, high, low, close, volume, close_time, ...]
        rows = []
        for k in raw:
            rows.append({
                "time": pd.Timestamp(k[0], unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        df = pd.DataFrame(rows)
        df.set_index("time", inplace=True)
        # Drop last candle — it's the currently forming (incomplete) candle.
        # Using it would give stale RSI/SMA that won't match the final close.
        # The live candle mechanism handles the current candle separately.
        if len(df) > 1:
            df = df.iloc[:-1]
        logger.info("Fetched %d native 15m candles from Binance", len(df))
        return df

    def _fetch_chart_from_db(self) -> Optional[pd.DataFrame]:
        """Fallback: fetch 15m candles from TimescaleDB (aggregated from 1-min).

        Returns None on failure.
        """
        import os
        from dotenv import load_dotenv
        load_dotenv()
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            logger.warning("DATABASE_URL not set, skipping DB chart fallback")
            return None

        logger.info("Falling back to TimescaleDB for chart history...")
        try:
            engine = create_engine(db_url, connect_args={"connect_timeout": 10})
            query = text("""
                SELECT
                    time_bucket('15 minutes', time) AS time,
                    first(open, time) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, time) AS close,
                    sum(volume) AS volume
                FROM ohlcv_data
                WHERE pair = :pair
                  AND time >= NOW() - INTERVAL '14 days'
                GROUP BY 1
                ORDER BY 1
            """)
            df = pd.read_sql(query, engine, params={"pair": SYMBOL})
            engine.dispose()
        except Exception as e:
            logger.error("DB chart fetch failed: %s", e)
            return None

        if df.empty:
            logger.warning("No chart history returned from DB")
            return None

        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df.index = df.index.tz_localize(None)
        logger.info("Fetched %d aggregated 15m candles from DB", len(df))
        return df

    def _push_position_to_dashboard(self, close_price: float) -> None:
        """Push current position state to dashboard."""
        if not self._dashboard or not self.pm.position:
            return

        pos = self.pm.position
        if pos.direction == Direction.LONG:
            current_pnl = (close_price - pos.entry_price) / pos.entry_price * 10000
        else:
            current_pnl = (pos.entry_price - close_price) / pos.entry_price * 10000

        # Show active trailing stop (tightened after configured bar)
        tighten_bar = self.cfg.exit.tighten_after_bar
        if pos.bars_held > tighten_bar:
            active_ts = self.cfg.exit.tight_trailing_stop_bps
        else:
            active_ts = pos.trailing_stop_bps

        liq_price = _calc_liq_price(pos.entry_price, pos.direction.value, self._wallet, self._current_qty)

        self._dashboard.update_position(
            has_position=True,
            side=pos.direction.value,
            entry_price=pos.entry_price,
            trailing_stop_bps=active_ts,
            highest_profit_bps=round(pos.highest_profit_bps, 1),
            current_pnl_bps=round(current_pnl, 1),
            bars_held=pos.bars_held,
            max_bars=pos.max_bars,
            mfe_bps=round(pos.mfe_bps, 1),
            mae_bps=round(pos.mae_bps, 1),
            is_reentry=pos.is_reentry,
            liq_price=liq_price,
        )

    def _push_ml_position_to_dashboard(self, close_price: float) -> None:
        """Push ML position state to dashboard."""
        if not self._dashboard or not self._ml_pm.position:
            return

        pos = self._ml_pm.position
        if pos.direction == Direction.LONG:
            current_pnl = (close_price - pos.entry_price) / pos.entry_price * 10000
        else:
            current_pnl = (pos.entry_price - close_price) / pos.entry_price * 10000

        tighten_bar = self.cfg.exit.tighten_after_bar
        if pos.bars_held > tighten_bar:
            active_ts = self.cfg.exit.tight_trailing_stop_bps
        else:
            active_ts = pos.trailing_stop_bps

        liq_price = _calc_liq_price(pos.entry_price, pos.direction.value, self._ml_wallet, self._ml_qty)

        self._dashboard.update_ml(
            ml_has_position=True,
            ml_position_side=pos.direction.value,
            ml_position_pnl_bps=round(current_pnl, 1),
            ml_position_entry_price=pos.entry_price,
            ml_position_trailing_stop_bps=active_ts,
            ml_position_highest_profit_bps=round(pos.highest_profit_bps, 1),
            ml_position_bars_held=pos.bars_held,
            ml_position_max_bars=pos.max_bars,
            ml_position_mfe_bps=round(pos.mfe_bps, 1),
            ml_position_mae_bps=round(pos.mae_bps, 1),
            ml_position_liq_price=liq_price,
        )

    async def start(self) -> None:
        """Start the bot with Binance WebSocket."""
        # Load connector directly — bypass live/__init__.py (old orchestrator imports)
        import importlib.util, pathlib
        _spec = importlib.util.spec_from_file_location(
            "binance_connector",
            pathlib.Path(__file__).parents[1] / "trade_system" / "live" / "binance_connector.py",
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        BinanceConnector = _mod.BinanceConnector

        logger.info(
            "Starting ALPHA | mode=%s | hash=%s | leverage=%dx | size=$%.0f",
            self.cfg.execution.mode,
            self.cfg.config_hash(),
            self.cfg.execution.leverage,
            self.cfg.execution.position_size_usd,
        )

        self._running = True
        self._initial_push_done = False
        self._connector = BinanceConnector(
            symbol=SYMBOL,
            interval=TIMEFRAME,
            buffer_size=BUFFER_SIZE,
            on_candle_close=self._on_candle_close,
            on_candle_update=self._on_candle_update,
        )

        # Start dashboard — mark as RUNNING so web server can show loading UI
        if self._dashboard:
            self._dashboard.start()

        # Yield to event loop so the web server task can bind its port
        await asyncio.sleep(0.1)

        # Load chart history in a thread (blocking I/O: requests.get + pd.read_sql)
        # so the web server stays responsive and UI renders immediately
        if self._dashboard:
            await asyncio.get_event_loop().run_in_executor(
                None, self._load_chart_history
            )

        # Handle graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except NotImplementedError:
                pass  # Windows doesn't support add_signal_handler

        try:
            await self._connector.start()
        except KeyboardInterrupt:
            await self.stop()

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        logger.info("Stopping ALPHA...")
        self._running = False
        if self._connector:
            await self._connector.stop()

        if self._dashboard:
            self._dashboard.stop()

        # Log final stats
        trades = self.pm.trades
        if trades:
            total = sum(t.net_profit_bps for t in trades)
            wins = sum(1 for t in trades if t.net_profit_bps > 0)
            logger.info(
                "Session complete: %d trades, %d wins (%.1f%%), %+.0f bps total",
                len(trades), wins, wins / len(trades) * 100, total,
            )

    async def _on_candle_update(self, candle: dict) -> None:
        """Called on every WebSocket tick. Pushes initial state + live price."""
        if not self._dashboard:
            return

        # First tick after buffer is ready: mark as initialized
        if not self._initial_push_done:
            df = self._connector.get_candle_buffer()
            if len(df) < MIN_BUFFER_SIZE:
                return

            if self._chart_history_loaded:
                # DB loaded OK — indicators already set from DB, just update price
                self._dashboard.update_status(
                    price=candle["close"],
                    bar_count=self._bar_count,
                )
            else:
                # DB was unreachable — set indicators from live buffer as fallback
                df = self.strategy.compute_indicators(df)
                latest = df.iloc[-1]
                regime = "BULL" if latest.get("bull_market", False) else "BEAR"
                self._dashboard.update_status(
                    price=candle["close"],
                    regime=regime,
                    bar_count=self._bar_count,
                    rsi=round(float(latest.get("rsi", 0)), 1),
                    atr_percentile=round(float(latest.get("atr_percentile", 0)), 1),
                    ema_separation=round(float(latest.get("ema_separation", 0)), 4),
                )

            self._initial_push_done = True
            logger.info("Dashboard initialized from historical buffer (%d bars)", len(df))
            return

        # Subsequent ticks: update price + live candle
        self._dashboard.update_status(price=candle["close"])
        self._dashboard.update_current_candle({
            "time": int(candle["open_time"].timestamp()),
            "open": round(candle["open"], 2),
            "high": round(candle["high"], 2),
            "low": round(candle["low"], 2),
            "close": round(candle["close"], 2),
            "volume": round(candle.get("volume", 0), 2),
        })

    async def _on_candle_close(self, candle: dict) -> None:
        """Called on each 15m candle close. Core bot logic."""
        self._bar_count += 1

        # Build indicator DataFrame from buffer
        df = self._connector.get_candle_buffer()
        if len(df) < MIN_BUFFER_SIZE:
            logger.debug("Warming up: %d/%d bars", len(df), MIN_BUFFER_SIZE)
            return

        # Compute indicators
        df = self.strategy.compute_indicators(df)
        latest_idx = len(df) - 1
        latest = df.iloc[latest_idx]

        current_price = candle["close"]
        bar_time = candle["open_time"]

        # Determine regime
        regime = "BULL" if latest.get("bull_market", False) else "BEAR"

        # Push status + indicators + closed candle to dashboard
        if self._dashboard:
            self._dashboard.update_status(
                price=current_price,
                regime=regime,
                bar_count=self._bar_count,
                rsi=round(float(latest.get("rsi", 0)), 1),
                atr_percentile=round(float(latest.get("atr_percentile", 0)), 1),
                ema_separation=round(float(latest.get("ema_separation", 0)), 4),
            )
            self._dashboard.close_candle({
                "time": int(df.index[-1].timestamp()),
                "open": round(float(latest["open"]), 2),
                "high": round(float(latest["high"]), 2),
                "low": round(float(latest["low"]), 2),
                "close": round(float(latest["close"]), 2),
                "volume": round(float(latest.get("volume", 0)), 2),
                "sma": round(float(latest["sma"]), 2) if pd.notna(latest.get("sma")) else None,
                "rsi": round(float(latest["rsi"]), 1) if pd.notna(latest.get("rsi")) else None,
            })
            self._dashboard.update_proximity(self._compute_proximity(latest))

        # === ML: If in ML position, update with new bar ===
        if self._ml_gen.loaded and self._ml_pm.is_in_position:
            ml_trade = self._ml_pm.on_bar(
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                bar_time=bar_time,
                bar_index=self._bar_count,
            )
            if ml_trade:
                self._log_ml_trade(ml_trade)
                self._update_ml_risk_after_trade(ml_trade)
                self._push_ml_trade_to_dashboard(ml_trade)
                # Clear ALL ML position fields from dashboard
                self._dashboard.update_ml(
                    ml_has_position=False,
                    ml_position_side=None,
                    ml_position_pnl_bps=0.0,
                    ml_position_entry_price=0.0,
                    ml_position_trailing_stop_bps=0.0,
                    ml_position_highest_profit_bps=0.0,
                    ml_position_bars_held=0,
                    ml_position_max_bars=10,
                    ml_position_mfe_bps=0.0,
                    ml_position_mae_bps=0.0,
                )
                logger.info(
                    "[ML] CLOSED %s (%s) | %s | %+.1f bps | bar %d | %s",
                    ml_trade.direction,
                    ml_trade.signal_type,
                    ml_trade.exit_reason,
                    ml_trade.net_profit_bps,
                    ml_trade.exit_bar,
                    bar_time,
                )
            else:
                # Still in position — push live ML position data
                self._push_ml_position_to_dashboard(candle["close"])

        # === ML: If not in ML position, check for ML signal ===
        if self._ml_gen.loaded and not self._ml_pm.is_in_position:
            df_ml = self._ml_gen.compute_features_from_df(df.copy())
            ml_features = self._ml_gen.get_features_for_bar(df_ml, latest_idx)
            if ml_features is not None:
                ml_prob = self._ml_gen.predict(ml_features)
                long_pct = ml_prob * 100
                short_pct = (1 - ml_prob) * 100
                if ml_prob > 0.60:
                    status = ">>> ML_LONG SIGNAL <<<"
                elif ml_prob < 0.35:
                    status = ">>> ML_SHORT SIGNAL <<<"
                else:
                    status = "NO TRADE (not confident)"
                # Push prediction to dashboard
                self._dashboard.update_ml(
                    ml_last_prob=ml_prob,
                    ml_long_pct=round(long_pct, 1),
                    ml_short_pct=round(short_pct, 1),
                    ml_signal_status=status,
                )
                logger.info(
                    "[ML] LONG=%.1f%% SHORT=%.1f%% | %s",
                    long_pct, short_pct, status,
                )
            ml_signal = self._ml_gen.predict_bar(df_ml, latest_idx)
            if ml_signal is not None:
                ml_decision = self._ml_risk_calc.calculate(self._ml_wallet, current_price)
                self._ml_decision_logger.log_decision(
                    ml_decision, self._ml_wallet, current_price,
                    ml_signal.signal_type.value,
                )
                if ml_decision.action != "SKIP":
                    self._ml_qty = ml_decision.qty
                    self._ml_pm.reset_reentry()
                    self._ml_pm.open_position(
                        direction=ml_signal.direction,
                        signal_type=ml_signal.signal_type,
                        entry_price=current_price,
                        entry_time=bar_time,
                        signal_time=ml_signal.timestamp,
                    )
                    self._push_ml_to_dashboard()
                    # Push fresh position data immediately (not stale from previous trade)
                    ml_liq = _calc_liq_price(current_price, ml_signal.direction.value, self._ml_wallet, self._ml_qty)
                    self._dashboard.update_ml(
                        ml_has_position=True,
                        ml_position_side=ml_signal.direction.value,
                        ml_position_pnl_bps=0.0,
                        ml_position_entry_price=current_price,
                        ml_position_trailing_stop_bps=20 if ml_signal.direction == Direction.LONG else 30,
                        ml_position_highest_profit_bps=0.0,
                        ml_position_bars_held=0,
                        ml_position_max_bars=self.cfg.exit.max_bars,
                        ml_position_mfe_bps=0.0,
                        ml_position_mae_bps=0.0,
                        ml_position_liq_price=ml_liq,
                        ml_last_qty=self._ml_qty,
                    )
                    logger.info(
                        "[ML] ENTRY %s (%s) @ %.2f | prob=%.3f | %s",
                        ml_signal.direction.value,
                        ml_signal.signal_type.value,
                        current_price,
                        ml_signal.rsi,  # prob stored in rsi field
                        bar_time,
                    )

        # === ML ATTENTION: If in position, update with new bar ===
        if self._ml_attn_gen.loaded and self._ml_attn_pm.is_in_position:
            ml_attn_trade = self._ml_attn_pm.on_bar(
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                bar_time=bar_time,
                bar_index=self._bar_count,
            )
            if ml_attn_trade:
                self._log_ml_attn_trade(ml_attn_trade)
                attn_trade_qty = self._ml_attn_qty  # save before reset
                attn_liq = _calc_liq_price(ml_attn_trade.entry_price, ml_attn_trade.direction, self._ml_attn_wallet, attn_trade_qty)
                self._update_ml_attn_risk_after_trade(ml_attn_trade)
                if self._dashboard:
                    self._dashboard.add_ml_attn_trade({
                        "direction": ml_attn_trade.direction,
                        "signal_type": ml_attn_trade.signal_type,
                        "entry_price": ml_attn_trade.entry_price,
                        "exit_price": ml_attn_trade.exit_price,
                        "net_profit_bps": ml_attn_trade.net_profit_bps,
                        "mfe_bps": ml_attn_trade.mfe_bps,
                        "mae_bps": ml_attn_trade.mae_bps,
                        "exit_bar": ml_attn_trade.exit_bar,
                        "exit_reason": ml_attn_trade.exit_reason,
                        "is_reentry": ml_attn_trade.is_reentry,
                        "exit_time": str(ml_attn_trade.exit_time),
                        "entry_time": str(ml_attn_trade.entry_time),
                        "qty": attn_trade_qty,
                        "liq_price": attn_liq,
                    })
                    self._dashboard.update_ml_attn(
                        ml_attn_has_position=False,
                        ml_attn_position_side=None,
                        ml_attn_position_pnl_bps=0.0,
                    )
                logger.info(
                    "[ML_ATTN] CLOSED %s (%s) | %s | %+.1f bps | bar %d | %s",
                    ml_attn_trade.direction,
                    ml_attn_trade.signal_type,
                    ml_attn_trade.exit_reason,
                    ml_attn_trade.net_profit_bps,
                    ml_attn_trade.exit_bar,
                    bar_time,
                )
            else:
                # Still in position — push live PnL
                if self._dashboard and self._ml_attn_pm.position:
                    pos = self._ml_attn_pm.position
                    close_price = candle["close"]
                    if pos.direction == Direction.LONG:
                        pnl = (close_price - pos.entry_price) / pos.entry_price * 10000
                    else:
                        pnl = (pos.entry_price - close_price) / pos.entry_price * 10000
                    self._dashboard.update_ml_attn(
                        ml_attn_position_pnl_bps=round(pnl, 1),
                        ml_attn_position_bars_held=pos.bars_held,
                        ml_attn_position_mfe_bps=round(pos.mfe_bps, 1),
                        ml_attn_position_mae_bps=round(pos.mae_bps, 1),
                    )

        # === ML ATTENTION: If not in position, check for signal ===
        if self._ml_attn_gen.loaded and not self._ml_attn_pm.is_in_position:
            try:
                df_attn = self._ml_attn_gen.compute_features(df.copy())
            except Exception as e:
                logger.warning("[ML_ATTN] compute_features failed: %s", e)
                df_attn = df.copy()
            # Push prediction to dashboard
            attn_features = self._ml_attn_gen.get_features_for_bar(df_attn, latest_idx)
            if attn_features is None:
                logger.debug("[ML_ATTN] Features unavailable for bar %d (NaN or missing)", latest_idx)
            if attn_features is not None and self._dashboard:
                attn_prob = self._ml_attn_gen.predict(attn_features)
                attn_long_pct = attn_prob * 100
                attn_short_pct = (1 - attn_prob) * 100
                if attn_prob > 0.60:
                    attn_status = ">>> ML_ATTN_LONG SIGNAL <<<"
                elif attn_prob < 0.40:
                    attn_status = ">>> ML_ATTN_SHORT SIGNAL <<<"
                else:
                    attn_status = "NO TRADE (not confident)"
                self._dashboard.update_ml_attn(
                    ml_attn_last_prob=attn_prob,
                    ml_attn_long_pct=round(attn_long_pct, 1),
                    ml_attn_short_pct=round(attn_short_pct, 1),
                    ml_attn_signal_status=attn_status,
                )
                logger.info(
                    "[ML_ATTN] LONG=%.1f%% SHORT=%.1f%% | %s",
                    attn_long_pct, attn_short_pct, attn_status,
                )
            attn_signal = self._ml_attn_gen.predict_bar(df_attn, latest_idx)
            if attn_signal is not None:
                attn_decision = self._ml_attn_risk_calc.calculate(self._ml_attn_wallet, current_price)
                self._ml_attn_decision_logger.log_decision(
                    attn_decision, self._ml_attn_wallet, current_price,
                    attn_signal.signal_type.value,
                )
                if attn_decision.action != "SKIP":
                    self._ml_attn_qty = attn_decision.qty
                    self._ml_attn_pm.reset_reentry()
                    self._ml_attn_pm.open_position(
                        direction=attn_signal.direction,
                        signal_type=attn_signal.signal_type,
                        entry_price=current_price,
                        entry_time=bar_time,
                        signal_time=attn_signal.timestamp,
                    )
                    self._push_ml_attn_to_dashboard()
                    if self._dashboard:
                        self._dashboard.update_ml_attn(
                            ml_attn_has_position=True,
                            ml_attn_position_side=attn_signal.direction.value,
                            ml_attn_position_pnl_bps=0.0,
                            ml_attn_position_entry_price=current_price,
                            ml_attn_position_trailing_stop_bps=20 if attn_signal.direction == Direction.LONG else 30,
                            ml_attn_position_highest_profit_bps=0.0,
                            ml_attn_position_bars_held=0,
                            ml_attn_position_max_bars=self.cfg.exit.max_bars,
                            ml_attn_position_mfe_bps=0.0,
                            ml_attn_position_mae_bps=0.0,
                            ml_attn_position_liq_price=_calc_liq_price(current_price, attn_signal.direction.value, self._ml_attn_wallet, self._ml_attn_qty),
                            ml_attn_last_qty=self._ml_attn_qty,
                        )
                    logger.info(
                        "[ML_ATTN] ENTRY %s (%s) @ %.2f | prob=%.3f | %s",
                        attn_signal.direction.value,
                        attn_signal.signal_type.value,
                        current_price,
                        attn_signal.rsi,
                        bar_time,
                    )

        # === V1.4: If in position, update with new bar ===
        if self.pm.is_in_position:
            trade = self.pm.on_bar(
                high=candle["high"],
                low=candle["low"],
                close=candle["close"],
                bar_time=bar_time,
                bar_index=self._bar_count,
            )
            if trade:
                self._log_trade(trade)
                self._push_trade_to_dashboard(trade)
                self._update_risk_after_trade(trade)
                logger.info(
                    "[%s] CLOSED %s (%s) | %s | %+.1f bps | bar %d | %s",
                    self.cfg.execution.mode.upper(),
                    trade.direction,
                    trade.signal_type,
                    trade.exit_reason,
                    trade.net_profit_bps,
                    trade.exit_bar,
                    bar_time,
                )
            else:
                # Still in position — update dashboard with current state
                self._push_position_to_dashboard(current_price)
            return

        # === Not in position: check re-entry ===
        if self.pm.reentry_signal_type is not None:
            regime_ok = self._check_regime(self.pm.reentry_signal_type, latest)
            if self.pm.can_reenter(self._bar_count, regime_ok):
                # Risk check for re-entry
                decision = self._risk_calc.calculate(self._wallet, current_price)
                self._decision_logger.log_decision(
                    decision, self._wallet, current_price,
                    self.pm.reentry_signal_type.value,
                )
                if decision.action == "SKIP":
                    self._total_skips += 1
                    self._push_risk_to_dashboard()
                    logger.info(
                        "[%s] SKIP re-entry %s: %s",
                        self.cfg.execution.mode.upper(),
                        self.pm.reentry_signal_type.value,
                        decision.skip_reason,
                    )
                    return
                self._current_qty = decision.qty
                self._current_decision = decision

                entry_price = current_price
                self.pm.open_position(
                    direction=self.pm.reentry_direction,
                    signal_type=self.pm.reentry_signal_type,
                    entry_price=entry_price,
                    entry_time=bar_time,
                    signal_time=bar_time,
                    is_reentry=True,
                )
                if self._dashboard:
                    self._push_position_to_dashboard(current_price)
                    self._dashboard.add_signal({
                        "action": "REENTRY",
                        "direction": self.pm.reentry_direction.value,
                        "signal_type": self.pm.reentry_signal_type.value,
                        "rsi": float(latest.get("rsi", 0)),
                        "time": str(bar_time),
                    })
                logger.info(
                    "[%s] RE-ENTRY %s (%s) @ %.2f | %s",
                    self.cfg.execution.mode.upper(),
                    self.pm.reentry_direction.value,
                    self.pm.reentry_signal_type.value,
                    entry_price,
                    bar_time,
                )
                return

        # === Check for new signal ===
        signals = self.strategy.generate_signals(df)
        if not signals:
            return

        # Only care about signal on the latest bar
        latest_signal = signals[-1]
        if latest_signal.bar_index != latest_idx:
            return

        # New signal fires — clear old re-entry state regardless of risk outcome
        self.pm.reset_reentry()

        # Risk check before opening
        decision = self._risk_calc.calculate(self._wallet, current_price)
        self._decision_logger.log_decision(
            decision, self._wallet, current_price,
            latest_signal.signal_type.value,
        )
        if decision.action == "SKIP":
            self._total_skips += 1
            self._push_risk_to_dashboard()
            logger.info(
                "[%s] SKIP %s (%s): %s",
                self.cfg.execution.mode.upper(),
                latest_signal.direction.value,
                latest_signal.signal_type.value,
                decision.skip_reason,
            )
            return
        self._current_qty = decision.qty
        self._current_decision = decision
        entry_price = current_price

        self.pm.open_position(
            direction=latest_signal.direction,
            signal_type=latest_signal.signal_type,
            entry_price=entry_price,
            entry_time=bar_time,
            signal_time=latest_signal.timestamp,
        )

        if self._dashboard:
            self._push_position_to_dashboard(current_price)
            self._dashboard.add_signal({
                "action": "ENTRY",
                "direction": latest_signal.direction.value,
                "signal_type": latest_signal.signal_type.value,
                "rsi": float(latest_signal.rsi),
                "time": str(bar_time),
            })

        logger.info(
            "[%s] ENTRY %s (%s) @ %.2f | RSI=%.1f | %s",
            self.cfg.execution.mode.upper(),
            latest_signal.direction.value,
            latest_signal.signal_type.value,
            entry_price,
            latest_signal.rsi,
            bar_time,
        )

    def _compute_proximity(self, latest: pd.Series) -> dict:
        """Compute signal proximity for all 4 signal types."""
        c = self.cfg
        rsi = float(latest.get("rsi", 50))
        atr_pctl = float(latest.get("atr_percentile", 0))
        ema_sep = float(latest.get("ema_separation", 0))
        is_bull = bool(latest.get("bull_market", False))

        return {
            "V12_LONG": {
                "direction": "LONG",
                "conditions": [
                    {"name": "RSI", "op": "<", "threshold": c.strategy.rsi_oversold,
                     "current": round(rsi, 1), "met": rsi < c.strategy.rsi_oversold},
                    {"name": "Regime", "needed": "BULL", "met": is_bull},
                    {"name": "ATR", "op": ">=", "threshold": c.long_filters.atr_percentile_min,
                     "current": round(atr_pctl, 1), "met": atr_pctl >= c.long_filters.atr_percentile_min},
                    {"name": "EMA", "op": ">=", "threshold": c.long_filters.ema_separation_min,
                     "current": round(ema_sep, 2), "met": ema_sep >= c.long_filters.ema_separation_min},
                ],
            },
            "V12_SHORT": {
                "direction": "SHORT",
                "conditions": [
                    {"name": "RSI", "op": ">", "threshold": c.strategy.rsi_overbought,
                     "current": round(rsi, 1), "met": rsi > c.strategy.rsi_overbought},
                    {"name": "Regime", "needed": "BEAR", "met": not is_bull},
                ],
            },
            "BEAR_LONG": {
                "direction": "LONG",
                "conditions": [
                    {"name": "RSI", "op": "<", "threshold": c.bear_long_filters.rsi_threshold,
                     "current": round(rsi, 1), "met": rsi < c.bear_long_filters.rsi_threshold},
                    {"name": "Regime", "needed": "BEAR", "met": not is_bull},
                    {"name": "EMA", "op": ">=", "threshold": c.bear_long_filters.ema_separation_min,
                     "current": round(ema_sep, 2), "met": ema_sep >= c.bear_long_filters.ema_separation_min},
                ],
            },
            "BULL_SHORT": {
                "direction": "SHORT",
                "conditions": [
                    {"name": "RSI", "op": ">", "threshold": c.bull_short_filters.rsi_threshold,
                     "current": round(rsi, 1), "met": rsi > c.bull_short_filters.rsi_threshold},
                    {"name": "Regime", "needed": "BULL", "met": is_bull},
                    {"name": "ATR", "op": ">=", "threshold": c.bull_short_filters.atr_percentile_min,
                     "current": round(atr_pctl, 1), "met": atr_pctl >= c.bull_short_filters.atr_percentile_min},
                    {"name": "EMA", "op": ">=", "threshold": c.bull_short_filters.ema_separation_min,
                     "current": round(ema_sep, 2), "met": ema_sep >= c.bull_short_filters.ema_separation_min},
                ],
            },
        }

    def _check_regime(self, signal_type: SignalType, latest_bar: pd.Series) -> bool:
        """Check if market regime is still valid for re-entry signal type.

        V1.3: regime depends on SIGNAL TYPE, not just direction:
          V12_LONG / BULL_SHORT -> needs bull (price > SMA200)
          V12_SHORT / BEAR_LONG -> needs bear (price < SMA200)
        """
        if signal_type in (SignalType.V12_LONG, SignalType.BULL_SHORT):
            return bool(latest_bar["bull_market"])
        else:  # V12_SHORT, BEAR_LONG
            return bool(latest_bar["bear_market"])


async def main():
    """Run the V1.3 bot with dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config()
    bot = V12Bot(config)

    # Run risk calculator preflight check
    logger.info("Running risk calculator preflight...")
    if not run_preflight(wallet=bot._wallet, btc_price=100000):
        logger.error("Risk calculator preflight FAILED — refusing to start")
        return
    logger.info(
        "Preflight PASSED | wallet=$%.2f | decision_log=%s",
        bot._wallet, bot._decision_logger.log_path,
    )

    # Start dashboard web server
    from src.web.server import run_server
    from src.web.state import dashboard_state

    port = 8080
    bot.set_dashboard(dashboard_state)

    # Run web server and bot concurrently
    web_task = asyncio.create_task(run_server(port=port, state=dashboard_state))
    logger.info("Dashboard running at http://localhost:%d", port)

    try:
        await bot.start()
    finally:
        web_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
