"""Account health monitor — tracks wallet state and outputs risk multiplier.

Tracks three metrics:
  1. Drawdown from peak wallet balance
  2. Consecutive losing streak
  3. Recent win rate (rolling window)

Each metric produces a factor (0.0–1.0). Combined:
  risk_multiplier = max(floor, dd_factor × streak_factor × recent_factor)

When wallet makes a new peak → drawdown resets → multiplier recovers to 1.0.

ALL threshold parameters are PLACEHOLDERS — determined through backtesting.
"""
from collections import deque
from dataclasses import dataclass, field


@dataclass
class HealthConfig:
    """Configuration for account health thresholds.

    All values are PLACEHOLDERS — to be determined through testing.
    """
    # Drawdown thresholds
    drawdown_threshold_1: float = 0.15      # 15% DD → start reducing
    drawdown_threshold_2: float = 0.30      # 30% DD → aggressive reduce
    drawdown_reduction_1: float = 0.70      # multiply risk by 0.70
    drawdown_reduction_2: float = 0.40      # multiply risk by 0.40

    # Consecutive loss streak
    consec_loss_threshold: int = 5          # 5 losses → reduce
    consec_loss_reduction: float = 0.50     # multiply risk by 0.50

    # Recent performance window
    recent_lookback: int = 20               # rolling window size
    recent_winrate_floor: float = 0.35      # below 35% → reduce
    recent_reduction: float = 0.60          # multiply risk by 0.60

    # Floor — never reduce below this
    risk_floor: float = 0.20


class AccountHealthMonitor:
    """Tracks account health and outputs a risk multiplier."""

    def __init__(self, config: HealthConfig = None):
        self.config = config or HealthConfig()
        self.peak: float = 0.0
        self.consecutive_losses: int = 0
        self.recent_trades: deque = deque(maxlen=self.config.recent_lookback)

    def update(self, wallet_balance: float, trade_pnl_bps: float = None):
        """Call after each trade with current wallet balance.

        Args:
            wallet_balance: Current wallet balance in USD.
            trade_pnl_bps: P&L of the trade in bps (positive=win, negative=loss).
                           Pass None on first call (initialization).
        """
        # Update peak
        if wallet_balance > self.peak:
            self.peak = wallet_balance

        # Record trade result
        if trade_pnl_bps is not None:
            self.recent_trades.append(trade_pnl_bps)
            if trade_pnl_bps <= 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

    def get_drawdown_pct(self, wallet_balance: float) -> float:
        """Current drawdown as fraction (0.0 = at peak, 0.3 = 30% below peak)."""
        if self.peak <= 0:
            return 0.0
        return max(0.0, (self.peak - wallet_balance) / self.peak)

    def get_recent_winrate(self) -> float:
        """Win rate of recent trades. Returns 1.0 if no history yet."""
        if not self.recent_trades:
            return 1.0
        wins = sum(1 for t in self.recent_trades if t > 0)
        return wins / len(self.recent_trades)

    def get_risk_multiplier(self, wallet_balance: float) -> float:
        """Combined risk multiplier from all three metrics.

        Returns:
            Float between config.risk_floor and 1.0.
            1.0 = full size, lower = reduce position.
        """
        cfg = self.config

        # Factor 1: Drawdown
        dd = self.get_drawdown_pct(wallet_balance)
        if dd >= cfg.drawdown_threshold_2:
            dd_factor = cfg.drawdown_reduction_2
        elif dd >= cfg.drawdown_threshold_1:
            dd_factor = cfg.drawdown_reduction_1
        else:
            dd_factor = 1.0

        # Factor 2: Consecutive losses
        if self.consecutive_losses >= cfg.consec_loss_threshold:
            streak_factor = cfg.consec_loss_reduction
        else:
            streak_factor = 1.0

        # Factor 3: Recent win rate
        winrate = self.get_recent_winrate()
        if winrate < cfg.recent_winrate_floor:
            recent_factor = cfg.recent_reduction
        else:
            recent_factor = 1.0

        combined = dd_factor * streak_factor * recent_factor
        return max(cfg.risk_floor, min(1.0, combined))

    def get_snapshot(self) -> dict:
        """Current state for logging/dashboard."""
        return {
            'peak': self.peak,
            'consecutive_losses': self.consecutive_losses,
            'recent_trades_count': len(self.recent_trades),
            'recent_winrate': self.get_recent_winrate(),
        }

    def to_dict(self) -> dict:
        """Serialize state for persistence (bot restart)."""
        return {
            'peak': self.peak,
            'consecutive_losses': self.consecutive_losses,
            'recent_trades': list(self.recent_trades),
        }

    @classmethod
    def from_dict(cls, data: dict, config: HealthConfig = None) -> 'AccountHealthMonitor':
        """Restore state from persisted dict."""
        monitor = cls(config=config)
        monitor.peak = data.get('peak', 0.0)
        monitor.consecutive_losses = data.get('consecutive_losses', 0)
        for t in data.get('recent_trades', []):
            monitor.recent_trades.append(t)
        return monitor
