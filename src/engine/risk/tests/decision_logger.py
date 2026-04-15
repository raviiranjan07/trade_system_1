"""Decision logger — records every risk decision during live/paper trading.

Logs to a single CSV file that persists across bot restarts.

Usage in bot:
    from engine.risk.tests.decision_logger import DecisionLogger

    logger = DecisionLogger("data/trades/risk_logs")
    # Before trade:
    decision = risk_calculator.calculate(wallet, btc_price)
    logger.log_decision(decision, wallet, btc_price, signal_type)
    # After trade:
    logger.log_outcome(pnl_bps, pnl_usd)
"""
import csv
import os
from datetime import datetime
from pathlib import Path


HEADERS = [
    'timestamp',
    'wallet_usd',
    'btc_price',
    'signal_type',
    'action',
    'qty',
    'position_usd',
    'margin_usd',
    'risk_pct',
    'safety_stop_bps',
    'health_multiplier',
    'skip_reason',
    'pnl_bps',
    'pnl_usd',
    'reasoning',
]


class DecisionLogger:
    """Logs every risk decision to CSV for post-session analysis."""

    def __init__(self, log_dir: str = "data/trades/risk_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Single file — append across sessions
        self.filepath = self.log_dir / "decisions.csv"

        # Track current decision for pairing with outcome
        self._pending_decision = None
        self._row_count = 0

        # Write header only if file doesn't exist
        if not self.filepath.exists():
            self._write_header()

    def _write_header(self):
        """Write CSV header."""
        with open(self.filepath, 'w', newline='') as f:
            csv.writer(f).writerow(HEADERS)

    def log_decision(self, decision, wallet: float, btc_price: float,
                     signal_type: str = ""):
        """Log a risk decision before the trade executes.

        Call log_outcome() after the trade closes to complete the record.
        """
        self._pending_decision = {
            'timestamp': datetime.utcnow().isoformat(),
            'wallet_usd': wallet,
            'btc_price': btc_price,
            'signal_type': signal_type,
            'action': decision.action,
            'qty': decision.qty,
            'position_usd': decision.position_usd,
            'margin_usd': decision.margin_usd,
            'risk_pct': decision.risk_pct,
            'safety_stop_bps': decision.safety_stop_bps,
            'health_multiplier': decision.health_multiplier,
            'skip_reason': decision.skip_reason,
            'reasoning': ' | '.join(decision.reasoning),
        }

        # If SKIP, write immediately (no outcome to wait for)
        if decision.action == "SKIP":
            self._write_row(self._pending_decision, pnl_bps=0, pnl_usd=0)
            self._pending_decision = None

    def log_outcome(self, pnl_bps: float, pnl_usd: float):
        """Log the trade outcome after it closes.

        Must be called after log_decision() for TRADE actions.
        """
        if self._pending_decision is None:
            return

        self._write_row(self._pending_decision, pnl_bps, pnl_usd)
        self._pending_decision = None

    def _write_row(self, decision: dict, pnl_bps: float, pnl_usd: float):
        """Write one row to CSV."""
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                decision['timestamp'],
                f"{decision['wallet_usd']:.2f}",
                f"{decision['btc_price']:.2f}",
                decision['signal_type'],
                decision['action'],
                f"{decision['qty']:.4f}",
                f"{decision['position_usd']:.2f}",
                f"{decision['margin_usd']:.2f}",
                f"{decision['risk_pct']:.4f}",
                f"{decision['safety_stop_bps']:.1f}",
                f"{decision['health_multiplier']:.2f}",
                decision['skip_reason'],
                f"{pnl_bps:.2f}",
                f"{pnl_usd:.4f}",
                decision['reasoning'],
            ])
        self._row_count += 1

    @property
    def total_logged(self) -> int:
        return self._row_count

    @property
    def log_path(self) -> str:
        return str(self.filepath)
