"""Test the risk calculator on historical trades.

Compares:
  A) Adaptive risk calculator (sizing + health monitor)
  B) Fixed 0.001 BTC (current bot behavior, no risk management)

Runs trades sequentially (not shuffled) to see real equity curves.
"""
import sys
sys.path.insert(0, "src")

from v12.risk.trade_loader import load_enriched_trades
from v12.risk.risk_calculator import RiskCalculator, RiskConfig
from v12.risk.account_health import AccountHealthMonitor, HealthConfig
from v12.risk.exchange_constants import DEFAULT_CAPITAL, LEVERAGE, MAINT_MARGIN_RATE


def simulate_adaptive(trades, capital, worst_loss_bps):
    """Run trades through adaptive risk calculator."""
    health = AccountHealthMonitor()
    health.update(capital)
    calc = RiskCalculator(worst_loss_bps=worst_loss_bps, health=health)

    wallet = capital
    peak = capital
    max_dd = 0.0
    wins = 0
    losses = 0
    skips = 0
    equity = [capital]

    for t in trades:
        decision = calc.calculate(wallet, t['btc_price'])

        if decision.action == "SKIP":
            skips += 1
            equity.append(wallet)
            continue

        # Apply P&L
        pnl = decision.qty * t['btc_price'] * (t['bps'] / 10000)

        # Check liquidation
        position = decision.qty * t['btc_price']
        maint = position * MAINT_MARGIN_RATE
        max_loss = wallet - maint
        if pnl < -max_loss:
            wallet = 0.01  # liquidated
        else:
            wallet = max(wallet + pnl, 0.01)

        # Track stats
        if t['bps'] > 0:
            wins += 1
        else:
            losses += 1

        if wallet > peak:
            peak = wallet
        dd = (peak - wallet) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

        # Update health monitor
        health.update(wallet, t['bps'])
        equity.append(wallet)

    return {
        'final': wallet,
        'peak': peak,
        'max_dd': max_dd,
        'wins': wins,
        'losses': losses,
        'skips': skips,
        'total_trades': len(trades),
        'equity': equity,
    }


def simulate_fixed(trades, capital, fixed_qty=0.001):
    """Run trades with fixed 0.001 BTC (no risk management)."""
    wallet = capital
    peak = capital
    max_dd = 0.0
    wins = 0
    losses = 0
    skips = 0
    equity = [capital]

    for t in trades:
        qty = fixed_qty
        margin = qty * t['btc_price'] / LEVERAGE

        # Skip if can't afford
        if wallet < margin:
            skips += 1
            equity.append(wallet)
            continue

        # Apply P&L
        pnl = qty * t['btc_price'] * (t['bps'] / 10000)

        position = qty * t['btc_price']
        maint = position * MAINT_MARGIN_RATE
        max_loss = wallet - maint
        if pnl < -max_loss:
            wallet = 0.01
        else:
            wallet = max(wallet + pnl, 0.01)

        if t['bps'] > 0:
            wins += 1
        else:
            losses += 1

        if wallet > peak:
            peak = wallet
        dd = (peak - wallet) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

        equity.append(wallet)

    return {
        'final': wallet,
        'peak': peak,
        'max_dd': max_dd,
        'wins': wins,
        'losses': losses,
        'skips': skips,
        'total_trades': len(trades),
        'equity': equity,
    }


def print_result(label, r):
    """Print one simulation result."""
    total = r['wins'] + r['losses']
    wr = r['wins'] / total * 100 if total > 0 else 0
    print(f"  {label}")
    print(f"    Start: ${DEFAULT_CAPITAL:.2f} -> Final: ${r['final']:.2f}")
    print(f"    Peak: ${r['peak']:.2f} | Max DD: {r['max_dd']:.1%}")
    print(f"    Trades: {total} ({r['wins']}W / {r['losses']}L) | Win rate: {wr:.1f}%")
    print(f"    Skips: {r['skips']}")
    print()


if __name__ == "__main__":
    capital = DEFAULT_CAPITAL

    # Get worst loss from TRAIN for risk calculator
    print("Loading TRAIN trades...")
    train_trades = load_enriched_trades("train")
    worst_loss = abs(min(t['bps'] for t in train_trades))
    print(f"  {len(train_trades)} trades, worst loss: {worst_loss:.1f} bps")
    print()

    print("Loading OOS trades...")
    oos_trades = load_enriched_trades("oos")
    print(f"  {len(oos_trades)} trades")
    print()

    # === TRAIN ===
    print("=" * 60)
    print("TRAIN (2020-2023)")
    print("=" * 60)
    train_adaptive = simulate_adaptive(train_trades, capital, worst_loss)
    train_fixed = simulate_fixed(train_trades, capital)
    print_result("A) Adaptive Risk Calculator", train_adaptive)
    print_result("B) Fixed 0.001 BTC (no risk mgmt)", train_fixed)

    # === OOS ===
    print("=" * 60)
    print("OOS (2024-2025)")
    print("=" * 60)
    oos_adaptive = simulate_adaptive(oos_trades, capital, worst_loss)
    oos_fixed = simulate_fixed(oos_trades, capital)
    print_result("A) Adaptive Risk Calculator", oos_adaptive)
    print_result("B) Fixed 0.001 BTC (no risk mgmt)", oos_fixed)

    # === Summary ===
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'':>30s} | {'TRAIN':>12s} | {'OOS':>12s}")
    print(f"  {'-'*60}")
    print(f"  {'A) Adaptive final $':>30s} | ${train_adaptive['final']:>10.2f} | ${oos_adaptive['final']:>10.2f}")
    print(f"  {'B) Fixed final $':>30s} | ${train_fixed['final']:>10.2f} | ${oos_fixed['final']:>10.2f}")
    print(f"  {'A) Max drawdown':>30s} | {train_adaptive['max_dd']:>11.1%} | {oos_adaptive['max_dd']:>11.1%}")
    print(f"  {'B) Max drawdown':>30s} | {train_fixed['max_dd']:>11.1%} | {oos_fixed['max_dd']:>11.1%}")
    print(f"  {'A) Skips':>30s} | {train_adaptive['skips']:>12d} | {oos_adaptive['skips']:>12d}")
    print(f"  {'B) Skips':>30s} | {train_fixed['skips']:>12d} | {oos_fixed['skips']:>12d}")
