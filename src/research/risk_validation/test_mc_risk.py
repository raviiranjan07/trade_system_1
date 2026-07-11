"""Monte Carlo with risk calculator — ruin probability test.

Shuffles trades 1000 times, runs each path through the risk calculator.
Shows: how many paths go to ruin, median/worst final wallet.
Compares: with risk calculator vs fixed 0.001 BTC.

Run: python src/research/risk_validation/test_mc_risk.py
"""
import sys
sys.path.insert(0, "src")

import numpy as np

from research.trade_loader import load_enriched_trades
from engine.risk.risk_calculator import RiskCalculator, RiskConfig
from engine.risk.account_health import AccountHealthMonitor, HealthConfig
from engine.risk.exchange_constants import DEFAULT_CAPITAL, LEVERAGE, MAINT_MARGIN_RATE


def mc_adaptive(trades, capital, worst_loss_bps, n_sims=1000, seed=42):
    """Monte Carlo with adaptive risk calculator."""
    rng = np.random.default_rng(seed)
    finals = []
    max_dds = []
    ruin_count = 0

    for _ in range(n_sims):
        shuffled = list(trades)
        rng.shuffle(shuffled)

        health = AccountHealthMonitor()
        health.update(capital)
        calc = RiskCalculator(worst_loss_bps=worst_loss_bps, health=health)

        wallet = capital
        peak = capital
        max_dd = 0.0

        for t in shuffled:
            d = calc.calculate(wallet, t['btc_price'])
            if d.action == "SKIP":
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
            dd = (peak - wallet) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

            health.update(wallet, t['bps'])

        finals.append(wallet)
        max_dds.append(max_dd)
        if wallet < 1.0:
            ruin_count += 1

    return np.array(finals), np.array(max_dds), ruin_count


def mc_fixed(trades, capital, n_sims=1000, seed=42, fixed_qty=0.001):
    """Monte Carlo with fixed 0.001 BTC."""
    rng = np.random.default_rng(seed)
    finals = []
    max_dds = []
    ruin_count = 0

    for _ in range(n_sims):
        shuffled = list(trades)
        rng.shuffle(shuffled)

        wallet = capital
        peak = capital
        max_dd = 0.0

        for t in shuffled:
            margin = fixed_qty * t['btc_price'] / LEVERAGE
            if wallet < margin:
                continue

            pnl = fixed_qty * t['btc_price'] * (t['bps'] / 10000)
            position = fixed_qty * t['btc_price']
            maint = position * MAINT_MARGIN_RATE
            max_loss = wallet - maint

            if pnl < -max_loss:
                wallet = 0.01
            else:
                wallet = max(wallet + pnl, 0.01)

            if wallet > peak:
                peak = wallet
            dd = (peak - wallet) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        finals.append(wallet)
        max_dds.append(max_dd)
        if wallet < 1.0:
            ruin_count += 1

    return np.array(finals), np.array(max_dds), ruin_count


def print_mc(label, finals, max_dds, ruin_count, n_sims):
    """Print MC results."""
    print(f"  {label}")
    print(f"    Median final:  ${np.median(finals):>10,.2f}")
    print(f"    P5 final:      ${np.percentile(finals, 5):>10,.2f}")
    print(f"    P95 final:     ${np.percentile(finals, 95):>10,.2f}")
    print(f"    Worst path:    ${np.min(finals):>10,.2f}")
    print(f"    Avg max DD:    {np.mean(max_dds):>10.1%}")
    print(f"    P95 max DD:    {np.percentile(max_dds, 95):>10.1%}")
    print(f"    Ruin (<$1):    {ruin_count}/{n_sims} ({ruin_count/n_sims*100:.1f}%)")
    print()


if __name__ == "__main__":
    n_sims = 1000

    print("Loading trades...")
    train_trades = load_enriched_trades("train")
    oos_trades = load_enriched_trades("oos")
    worst_loss = abs(min(t['bps'] for t in train_trades))
    print(f"  TRAIN: {len(train_trades)} | OOS: {len(oos_trades)} | Sims: {n_sims}")

    for capital in [5, 100, 500]:
        print(f"\n{'='*50}")
        print(f"  Starting capital: ${capital} | {n_sims} simulations")
        print(f"{'='*50}")

        for period_name, trades in [("TRAIN", train_trades), ("OOS", oos_trades)]:
            print(f"\n  --- {period_name} ---")

            f_a, d_a, r_a = mc_adaptive(trades, capital, worst_loss, n_sims)
            print_mc("A) Adaptive Risk Calculator", f_a, d_a, r_a, n_sims)

            f_b, d_b, r_b = mc_fixed(trades, capital, n_sims)
            print_mc("B) Fixed 0.001 BTC", f_b, d_b, r_b, n_sims)
