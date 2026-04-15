"""Adaptive risk calculator — decides position size for each trade.

Inputs:  wallet balance + BTC price + account health
Output:  position size (qty) + details

Logic:
  1. Can wallet afford minimum trade? No → SKIP
  2. Calculate base position from wallet × risk %
  3. Adjust for account health (drawdown/streak → reduce)
  4. Ensure position >= exchange minimum
  5. Calculate safety stop from liquidation distance
  6. Return position size + reasoning

ALL parameters are PLACEHOLDERS — determined through backtesting.
"""
import math
from dataclasses import dataclass, field

from .exchange_constants import LEVERAGE, STEP_SIZE, MIN_QTY
from .exchange_math import (
    calc_min_qty,
    calc_margin,
    calc_liq_distance_bps,
    calc_risk_pct,
)
from .account_health import AccountHealthMonitor


@dataclass
class RiskConfig:
    """Risk calculator configuration. All values are PLACEHOLDERS."""
    base_risk_pct: float = 0.10     # % of wallet willing to lose per trade
    safety_pct: float = 0.60        # safety stop as % of liquidation distance


@dataclass
class RiskDecision:
    """Output of every calculate() call."""
    action: str                     # "TRADE" or "SKIP"
    qty: float = 0.0                # BTC quantity to trade
    position_usd: float = 0.0      # notional position value
    margin_usd: float = 0.0        # margin required
    risk_pct: float = 0.0          # worst-case loss as % of wallet
    safety_stop_bps: float = 0.0   # safety stop in bps
    health_multiplier: float = 1.0  # from account health monitor
    skip_reason: str = ""           # why skipped (empty if TRADE)
    reasoning: list = field(default_factory=list)


class RiskCalculator:
    """Adaptive position sizer. Uses wallet + health to decide how much to bet.

    Args:
        worst_loss_bps: Worst historical loss in bps (positive number, e.g. 865).
                        Source: experiments/layer1/L1R-001/train_stats.json
        config: RiskConfig with sizing parameters.
        health: AccountHealthMonitor instance.
    """

    def __init__(
        self,
        worst_loss_bps: float,
        config: RiskConfig = None,
        health: AccountHealthMonitor = None,
    ):
        self.worst_loss_bps = abs(worst_loss_bps)
        self.config = config or RiskConfig()
        self.health = health or AccountHealthMonitor()

    def calculate(self, wallet: float, btc_price: float) -> RiskDecision:
        """Calculate position size for a trade.

        Args:
            wallet: Current wallet balance in USD.
            btc_price: Current BTC price in USD.

        Returns:
            RiskDecision with action, qty, and full reasoning.
        """
        reasoning = []

        # Step 1: Exchange minimums
        min_qty = calc_min_qty(btc_price)
        min_margin = calc_margin(min_qty, btc_price)
        reasoning.append(f"Exchange min: {min_qty} BTC, margin ${min_margin:.2f}")

        # Step 2: Can wallet afford minimum?
        if wallet < min_margin:
            reasoning.append(f"Wallet ${wallet:.2f} < min margin ${min_margin:.2f}")
            return RiskDecision(
                action="SKIP",
                skip_reason="insufficient_funds",
                reasoning=reasoning,
            )

        # Step 3: Account health multiplier
        health_mult = self.health.get_risk_multiplier(wallet)
        reasoning.append(f"Health multiplier: {health_mult:.2f}")

        # Step 4: Calculate risk budget
        base_risk = wallet * self.config.base_risk_pct
        adjusted_risk = base_risk * health_mult
        reasoning.append(
            f"Risk budget: ${base_risk:.2f} base × {health_mult:.2f} = ${adjusted_risk:.2f}"
        )

        # Step 5: Convert risk budget to position size
        # adjusted_risk = max $ we're willing to lose
        # position × (worst_loss_bps / 10000) = adjusted_risk
        # position = adjusted_risk / (worst_loss_bps / 10000)
        position = adjusted_risk / (self.worst_loss_bps / 10000)
        qty = position / btc_price
        qty = math.floor(qty / STEP_SIZE) * STEP_SIZE
        reasoning.append(f"Raw qty: {qty:.4f} BTC (position ${position:.2f})")

        # Step 6: Ensure >= exchange minimum
        if qty < min_qty:
            qty = min_qty
            reasoning.append(f"Rounded up to exchange min: {qty} BTC")

        # Step 7: Verify wallet can afford the margin
        margin = calc_margin(qty, btc_price)
        if wallet < margin:
            qty = min_qty
            margin = calc_margin(qty, btc_price)
            reasoning.append(f"Margin check: reduced to min qty {qty} BTC")

        # Step 8: Calculate actual risk and safety stop
        position_usd = qty * btc_price
        risk_pct = calc_risk_pct(wallet, qty, btc_price, self.worst_loss_bps)
        liq_dist = calc_liq_distance_bps(wallet, qty, btc_price)
        safety_stop = liq_dist * self.config.safety_pct

        reasoning.append(f"Position: {qty} BTC (${position_usd:.2f})")
        reasoning.append(f"Margin: ${margin:.2f}")
        reasoning.append(f"Risk: {risk_pct:.1%} of wallet (worst case)")
        reasoning.append(f"Liq distance: {liq_dist:.0f} bps")
        reasoning.append(f"Safety stop: {safety_stop:.0f} bps ({self.config.safety_pct:.0%} of liq)")

        return RiskDecision(
            action="TRADE",
            qty=qty,
            position_usd=position_usd,
            margin_usd=margin,
            risk_pct=risk_pct,
            safety_stop_bps=safety_stop,
            health_multiplier=health_mult,
            reasoning=reasoning,
        )
