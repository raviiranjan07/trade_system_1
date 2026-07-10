"""Persistence — everything that touches the disk for a track, and nothing else.

If a track's data must survive a restart, it goes through this file:
  TrackPaths / paths_for()  — the ONE naming rule (track name -> its files)
  migrate_legacy()          — one-time rename of pre-refactor files (run at cutover!)
  TrackState                — schema of the risk-state JSON
  TradeLog                  — the trades CSV (append + read back)
  StateStore                — the risk-state JSON (save + load)

Trading logic never opens a file; storage code never makes a trade decision.
Swapping CSV/JSON for SQLite later means rewriting THIS file only.
"""

import csv
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

from .position_manager import TradeRecord
from .risk.exchange_constants import DEFAULT_CAPITAL

logger = logging.getLogger(__name__)

TRADES_CSV_HEADERS = [
    "signal_time", "entry_time", "exit_time", "direction", "signal_type",
    "entry_price", "exit_price", "gross_profit_bps", "net_profit_bps",
    "mfe_bps", "mae_bps", "exit_bar", "exit_reason", "is_reentry",
    "config_hash", "qty",
]


# =====================================================================
# Naming rule — track name in, paths out. No mapping table, no exceptions.
# =====================================================================

@dataclass(frozen=True)
class TrackPaths:
    """The three disk locations belonging to one track."""
    trades_csv: Path
    state_json: Path
    decision_log_dir: Path


def paths_for(name: str, mode: str, trades_dir: Path) -> TrackPaths:
    """Derive a track's file paths from its name.

    E.g. name="ML_V1", mode="paper", trades_dir=data/trades ->
      data/trades/trades_ml_v1_paper.csv
      data/trades/risk_state_ml_v1.json
      data/trades/risk_logs/ml_v1/
    """
    key = name.lower()
    trades_dir = Path(trades_dir)
    return TrackPaths(
        trades_csv=trades_dir / f"trades_{key}_{mode}.csv",
        state_json=trades_dir / f"risk_state_{key}.json",
        decision_log_dir=trades_dir / "risk_logs" / key,
    )


def migrate_legacy(trades_dir: Path, mode: str) -> list:
    """One-time rename of pre-refactor files to the uniform naming scheme.

    ONLY renames when the old file exists AND the new one doesn't, so it is
    safe to run repeatedly. MUST NOT run while the old bot.py is still in
    use — old bot writes to the legacy names and history would split.
    Call it once, at orchestrator cutover.

    Returns list of (old, new) paths that were moved.
    """
    trades_dir = Path(trades_dir)
    renames = [
        # V1.4 files had no prefix -> V_RULE_BASED
        (trades_dir / f"trades_{mode}.csv",
         trades_dir / f"trades_v_rule_based_{mode}.csv"),
        (trades_dir / "risk_state.json",
         trades_dir / "risk_state_v_rule_based.json"),
        (trades_dir / "risk_logs" / "decisions.csv",
         trades_dir / "risk_logs" / "v_rule_based" / "decisions.csv"),
        # "ml" -> "ml_v1"
        (trades_dir / f"trades_ml_{mode}.csv",
         trades_dir / f"trades_ml_v1_{mode}.csv"),
        (trades_dir / "risk_state_ml.json",
         trades_dir / "risk_state_ml_v1.json"),
        (trades_dir / "risk_logs" / "ml" / "decisions.csv",
         trades_dir / "risk_logs" / "ml_v1" / "decisions.csv"),
        # ml_attn / ml_v3 already match the scheme — nothing to do
    ]
    def _header_only(p: Path) -> bool:
        """True when a file holds no data rows (just a header, or empty).

        Constructing tracks creates empty new-scheme files; those must not
        block migrating the real history into their place.
        """
        try:
            return len(p.read_text().strip().splitlines()) <= 1
        except Exception:
            return False

    moved = []
    for old, new in renames:
        if not old.exists():
            continue
        if new.exists():
            if not _header_only(new):
                continue  # real data already at destination — never overwrite
            new.unlink()
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        moved.append((old, new))
        logger.info("migrated: %s -> %s", old, new)
    return moved


# =====================================================================
# TrackState — schema of the risk-state JSON
# =====================================================================

@dataclass
class TrackState:
    """What survives a restart — the shape of the risk-state JSON file.

    `health` is the AccountHealthMonitor.to_dict() payload
    (peak, consecutive_losses, recent_trades).
    """
    wallet: float = DEFAULT_CAPITAL
    total_skips: int = 0
    health: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrackState":
        """Tolerant loader — missing keys fall back to fresh defaults
        (keeps compatibility with the old {'wallet', 'health'} files)."""
        return cls(
            wallet=data.get("wallet", DEFAULT_CAPITAL),
            total_skips=data.get("total_skips", 0),
            health=data.get("health", {}) or {},
        )


# =====================================================================
# TradeLog — the trades CSV
# =====================================================================

class TradeLog:
    """Owns one track's trades CSV: schema, formatting, append, read-back."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(TRADES_CSV_HEADERS)
            logger.info("created trades CSV: %s", self.path)

    def append(self, trade: TradeRecord, tag: str, qty: float) -> None:
        """Write one closed trade. Formats match the pre-refactor files."""
        row = [
            trade.signal_time, trade.entry_time, trade.exit_time,
            trade.direction, trade.signal_type,
            f"{trade.entry_price:.2f}", f"{trade.exit_price:.2f}",
            f"{trade.gross_profit_bps:.1f}", f"{trade.net_profit_bps:.1f}",
            f"{trade.mfe_bps:.1f}", f"{trade.mae_bps:.1f}",
            trade.exit_bar, trade.exit_reason, trade.is_reentry,
            tag, f"{qty:.4f}",
        ]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def read_all(self) -> list:
        """Read back all trades as (TradeRecord, qty) pairs.

        Bad rows are logged and skipped; missing file -> empty list.
        """
        if not self.path.exists():
            return []
        try:
            df = pd.read_csv(self.path)
        except Exception as e:
            logger.error("failed to read trades CSV %s: %s", self.path, e)
            return []
        if df.empty:
            return []

        out = []
        for _, row in df.iterrows():
            try:
                trade = TradeRecord(
                    signal_time=row.get("signal_time", ""),
                    entry_time=row.get("entry_time", ""),
                    exit_time=row.get("exit_time", ""),
                    direction=str(row.get("direction", "LONG")),
                    signal_type=str(row.get("signal_type", "")),
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
                qty = float(row.get("qty", 0)) if "qty" in row and pd.notna(row.get("qty")) else 0.0
                out.append((trade, qty))
            except Exception as e:
                logger.warning("skipping bad trade row in %s: %s", self.path.name, e)
                continue
        return out


# =====================================================================
# StateStore — the risk-state JSON
# =====================================================================

class StateStore:
    """Owns one track's risk-state JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> Optional[TrackState]:
        """Load saved state. Returns None when there is nothing usable
        (no file, or corrupt) — the caller starts fresh."""
        if not self.path.exists():
            return None
        try:
            with open(self.path) as f:
                return TrackState.from_dict(json.load(f))
        except Exception as e:
            logger.warning("failed to load state %s: %s — starting fresh", self.path, e)
            return None

    def save(self, state: TrackState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
