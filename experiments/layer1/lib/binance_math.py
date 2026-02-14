"""Binance position calculations — single source of truth."""
import math
from .constants import LEVERAGE, MAINT_MARGIN_RATE, MIN_QTY, STEP_SIZE, MIN_NOTIONAL


def calc_min_qty(btc_price: float) -> float:
    """Minimum BTC qty for a given price (from min notional $100)."""
    raw = MIN_NOTIONAL / btc_price
    return max(MIN_QTY, math.ceil(raw / STEP_SIZE) * STEP_SIZE)


def calc_margin(qty: float, btc_price: float) -> float:
    """Margin required at 125x leverage."""
    return qty * btc_price / LEVERAGE


def calc_liq_distance_bps(wallet: float, qty: float, btc_price: float) -> float:
    """How many bps from entry until liquidation (cross margin).

    Liquidation occurs when:
      wallet + unrealized_pnl = maintenance_margin
      wallet + qty * btc_price * (move/10000) = qty * btc_price * MAINT_MARGIN_RATE

    Solving for move (in bps):
      move = (MAINT_MARGIN_RATE * qty * btc_price - wallet) / (qty * btc_price) * 10000
      move = (MAINT_MARGIN_RATE - wallet / position) * 10000
    """
    position = qty * btc_price
    if position == 0:
        return float('inf')
    maint = position * MAINT_MARGIN_RATE
    buffer = wallet - maint
    return buffer / position * 10000


def calc_max_qty(wallet: float, btc_price: float, max_risk_frac: float,
                 worst_loss_bps: float) -> float:
    """Max qty that limits worst-case loss to max_risk_frac of wallet.

    worst_loss_bps should be positive (e.g., 865 for -865 bps worst trade).

    max_risk_dollar = wallet * max_risk_frac
    position * worst_loss_bps / 10000 = max_risk_dollar
    qty = max_risk_dollar / (worst_loss_bps / 10000) / btc_price
    """
    max_risk_dollar = wallet * max_risk_frac
    position = max_risk_dollar / (worst_loss_bps / 10000)
    qty = position / btc_price
    # Round down to step size
    return math.floor(qty / STEP_SIZE) * STEP_SIZE


def calc_risk_pct(wallet: float, qty: float, btc_price: float,
                  loss_bps: float) -> float:
    """What % of wallet is lost on a given loss (in bps).

    loss_bps should be positive.
    """
    position = qty * btc_price
    loss_dollar = position * loss_bps / 10000
    if wallet <= 0:
        return float('inf')
    return loss_dollar / wallet
