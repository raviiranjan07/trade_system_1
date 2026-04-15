"""Stress test — worst-case trade ordering.

Tests:
  1. All biggest losses first (worst possible start)
  2. All losses first, then all wins
  3. Alternating loss-loss-win pattern (grinding)

Shows: does the wallet survive? How low does it go?

Run: python src/engine/risk/tests/test_stress.py
"""
import sys
sys.path.insert(0, "src")

from engine.risk.risk_calculator import RiskCalculator, RiskConfig
from engine.risk.account_health import AccountHealthMonitor, HealthConfig
from engine.risk.trade_loader import load_enriched_trades
from engine.risk.exchange_constants import DEFAULT_CAPITAL, MAINT_MARGIN_RATE


def simulate_order(trades, capital, worst_loss_bps):
    """Run trades in given order through risk calculator."""
    health = AccountHealthMonitor()
    health.update(capital)
    calc = RiskCalculator(worst_loss_bps=worst_loss_bps, health=health)

    wallet = capital
    peak = capital
    min_wallet = capital
    max_dd = 0.0
    skips = 0

    for t in trades:
        d = calc.calculate(wallet, t['btc_price'])
        if d.action == "SKIP":
            skips += 1
            continue

        pnl = d.qty * t['btc_price'] * (t['bps'] / 10000)
        position = d.qty * t['btc_price']
        maint = position * MAINT_MARGIN_RATE
        max_loss = wallet - maint

        if pnl < -max_loss:
            wallet = 0.01
        else:
            wallet = max(wallet + pnl, 0.01)

        if wallet > peak:
            peak = wallet
        if wallet < min_wallet:
            min_wallet = wallet
        dd = (peak - wallet) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

        health.update(wallet, t['bps'])

    return {
        'final': wallet,
        'min_wallet': min_wallet,
        'max_dd': max_dd,
        'skips': skips,
    }


def print_result(label, r, capital):
    """Print one stress test result."""
    survived = r['final'] > 1.0
    status = "SURVIVED" if survived else "RUINED"
    print(f"  {label}")
    print(f"    {status}: ${capital} -> ${r['final']:.2f} (low: ${r['min_wallet']:.2f})")
    print(f"    Max DD: {r['max_dd']:.1%} | Skips: {r['skips']}")
    print()


if __name__ == "__main__":
    print("Loading trades...")
    train_trades = load_enriched_trades("train")
    oos_trades = load_enriched_trades("oos")
    worst_loss = abs(min(t['bps'] for t in train_trades))

    for period_name, trades in [("TRAIN", train_trades), ("OOS", oos_trades)]:
        for capital in [5, 100, 500]:
            print(f"\n{'='*50}")
            print(f"  {period_name} — Starting capital: ${capital}")
            print(f"{'='*50}")

            # Order 1: Biggest losses first
            sorted_worst = sorted(trades, key=lambda t: t['bps'])
            r = simulate_order(sorted_worst, capital, worst_loss)
            print_result("Worst losses first", r, capital)

            # Order 2: All losses first, then all wins
            losses = [t for t in trades if t['bps'] <= 0]
            wins = [t for t in trades if t['bps'] > 0]
            losses_first = losses + wins
            r = simulate_order(losses_first, capital, worst_loss)
            print_result("All losses first, then wins", r, capital)

            # Order 3: Grinding — loss, loss, win, loss, loss, win...
            losses_copy = list(losses)
            wins_copy = list(wins)
            grinding = []
            li, wi = 0, 0
            while li < len(losses_copy) or wi < len(wins_copy):
                if li < len(losses_copy):
                    grinding.append(losses_copy[li]); li += 1
                if li < len(losses_copy):
                    grinding.append(losses_copy[li]); li += 1
                if wi < len(wins_copy):
                    grinding.append(wins_copy[wi]); wi += 1
            r = simulate_order(grinding, capital, worst_loss)
            print_result("Grinding (loss-loss-win pattern)", r, capital)

            # Order 4: Normal order (baseline)
            r = simulate_order(trades, capital, worst_loss)
            print_result("Normal order (baseline)", r, capital)
