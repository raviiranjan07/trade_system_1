"""L1-EXP-006c: Kelly Optimal Margin % at 125x Leverage

SETUP: Binance 125x leverage, Cross margin
QUESTION: What % of wallet should be used as margin per trade?

margin = pct × wallet
position = margin × 125
effective_leverage = pct × 125

Tests: 1% to 100% margin allocation
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import math
import numpy as np
from engine.backtest import run_backtest
from engine.config.loader import load_config

STARTING_CAPITAL = 10.0
LEVERAGE = 125
N_SIMS = 1000
MAINT_MARGIN_RATE = 0.004
BINANCE_MIN_NOTIONAL = 100
BINANCE_MIN_QTY = 0.001
BINANCE_STEP_SIZE = 0.001

config = load_config()
trades = run_backtest(config)

trade_data = []
for t in trades:
    trade_data.append({
        'bps': t.net_profit_bps,
        'btc_price': t.entry_price,
        'direction': t.direction,
        'signal_type': t.signal_type if hasattr(t, 'signal_type') else 'UNKNOWN',
    })

returns = [td['bps'] for td in trade_data]

print("=" * 110)
print("L1-EXP-006c: KELLY OPTIMAL MARGIN % AT 125x LEVERAGE")
print("=" * 110)
print(f"  V1.3.2: {len(trades)} trades, {sum(1 for r in returns if r > 0)/len(returns)*100:.1f}% win")
print(f"  Leverage: {LEVERAGE}x (Binance setting)")
print(f"  Return range: {min(returns):.1f} to {max(returns):.1f} bps")
print()


def calc_max_dd(equity):
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def simulate(trade_list, margin_pct, capital=STARTING_CAPITAL):
    """Simulate with fixed margin % at 125x leverage.

    margin = margin_pct × wallet
    position = margin × 125
    Cross margin: full wallet backs position for liquidation.
    """
    equity = [capital]
    liquidated = 0

    for td in trade_list:
        eq = equity[-1]

        # Calculate margin and position
        margin = eq * margin_pct
        position = margin * LEVERAGE

        # Binance minimum check
        if position < BINANCE_MIN_NOTIONAL:
            # Use minimum position instead
            qty = max(BINANCE_MIN_QTY, math.ceil(BINANCE_MIN_NOTIONAL / td['btc_price'] / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE)
            position = qty * td['btc_price']

        # PnL
        pnl = position * (td['bps'] / 10000)

        # Cross margin liquidation: full equity backs position
        maint = position * MAINT_MARGIN_RATE
        max_loss = eq - maint
        if pnl < -max_loss:
            equity.append(0.01)
            liquidated += 1
        else:
            equity.append(max(eq + pnl, 0.01))

    return equity, liquidated


def run_mc(margin_pct):
    """Run MC simulation for a given margin %."""
    np.random.seed(42)
    n = len(trade_data)

    finals = []
    max_dds = []
    ruin_count = 0
    liq_total = 0

    for _ in range(N_SIMS):
        indices = np.random.choice(n, n, replace=True)
        sample = [trade_data[i] for i in indices]

        equity, liquidated = simulate(sample, margin_pct)
        final_eq = equity[-1]
        finals.append(final_eq)
        max_dds.append(calc_max_dd(equity))
        if final_eq < 1.0:
            ruin_count += 1
        liq_total += liquidated

    finals = np.array(finals)
    max_dds = np.array(max_dds)

    return {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'p95': np.percentile(finals, 95),
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'ruin_pct': ruin_count / N_SIMS * 100,
        'avg_liqs': liq_total / N_SIMS,
    }


# ============================================================
# PART 1: BROAD SWEEP (1% to 100%)
# ============================================================
print("=" * 110)
print("  PART 1: MARGIN % SWEEP AT 125x LEVERAGE")
print("=" * 110)
print()

margin_pcts = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]

print(f"  {'Margin%':>8s} | {'Eff Lev':>8s} | {'Position':>10s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'AvgDD':>7s} | {'Ruin':>6s} | {'Liqs':>5s}")
print(f"  {'-'*110}")

results = []
for pct in margin_pcts:
    eff_lev = pct * LEVERAGE
    position_10 = 10 * pct * LEVERAGE  # position with $10 wallet
    r = run_mc(pct)
    r['margin_pct'] = pct
    r['eff_lev'] = eff_lev
    results.append(r)
    print(f"  {pct*100:7.1f}% | {eff_lev:6.1f}x | ${position_10:>8,.0f} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:4.1f}% | {r['avg_liqs']:4.1f}")

# Find optimal
best = max(results, key=lambda x: x['median'])
best_safe = max([r for r in results if r['ruin_pct'] <= 1.0], key=lambda x: x['median'])

print()
print(f"  BEST MEDIAN: {best['margin_pct']*100:.0f}% margin = {best['eff_lev']:.0f}x effective | Median ${best['median']:,.0f} | Ruin {best['ruin_pct']:.1f}%")
print(f"  BEST SAFE (ruin<=1%): {best_safe['margin_pct']*100:.0f}% margin = {best_safe['eff_lev']:.0f}x effective | Median ${best_safe['median']:,.0f} | Ruin {best_safe['ruin_pct']:.1f}%")

# ============================================================
# PART 2: FINE-GRAINED AROUND OPTIMAL
# ============================================================
print()
print("=" * 110)
print("  PART 2: FINE-GRAINED SWEEP AROUND OPTIMAL")
print("=" * 110)
print()

# Find the peak region and sweep finely
peak_pct = best_safe['margin_pct']
fine_pcts = sorted(set([
    max(0.01, peak_pct - 0.10),
    max(0.01, peak_pct - 0.08),
    max(0.01, peak_pct - 0.06),
    max(0.01, peak_pct - 0.04),
    max(0.01, peak_pct - 0.02),
    peak_pct,
    peak_pct + 0.02,
    peak_pct + 0.04,
    peak_pct + 0.06,
    peak_pct + 0.08,
    peak_pct + 0.10,
]))

print(f"  {'Margin%':>8s} | {'Eff Lev':>8s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'AvgDD':>7s} | {'Ruin':>6s}")
print(f"  {'-'*85}")

fine_results = []
for pct in fine_pcts:
    if pct <= 0 or pct > 1.0:
        continue
    eff_lev = pct * LEVERAGE
    r = run_mc(pct)
    r['margin_pct'] = pct
    r['eff_lev'] = eff_lev
    fine_results.append(r)
    marker = " <-- PEAK" if pct == peak_pct else ""
    print(f"  {pct*100:7.1f}% | {eff_lev:6.1f}x | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:4.1f}%{marker}")


# ============================================================
# PART 3: KELLY THEORETICAL CALCULATION
# ============================================================
print()
print("=" * 110)
print("  PART 3: KELLY THEORETICAL vs MC OPTIMAL")
print("=" * 110)
print()

wins = [r for r in returns if r > 0]
losses = [r for r in returns if r <= 0]
W = len(wins) / len(returns)
avg_win = np.mean(wins)
avg_loss = abs(np.mean(losses))
R = avg_win / avg_loss

# Classic Kelly fraction
kelly_f = W - (1 - W) / R
half_kelly_f = kelly_f / 2

# Convert to margin % at 125x
kelly_margin_pct = kelly_f / LEVERAGE * 100  # because eff_lev = margin_pct * 125, and kelly gives eff_lev
# Actually: Kelly gives optimal fraction of bankroll to risk
# At 125x leverage: fraction of bankroll as margin = kelly_f * (avg payoff) / leverage
# Simpler: kelly_optimal_leverage = kelly_f * ... let me use the standard formula

# Kelly optimal leverage (from EXP-004)
# f = W - (1-W)/R = fraction of wealth to bet
# For leveraged trading: optimal_leverage = f / (avg_loss_per_unit)
# avg_loss_per_unit = avg_loss_bps / 10000
kelly_lev = kelly_f / (avg_loss / 10000)
half_kelly_lev = kelly_lev / 2
kelly_margin = half_kelly_lev / LEVERAGE  # margin % at 125x

print(f"  V1.3.2 stats:")
print(f"    Win rate: {W*100:.1f}%")
print(f"    Avg win: +{avg_win:.1f} bps")
print(f"    Avg loss: -{avg_loss:.1f} bps")
print(f"    Payoff ratio: {R:.2f}")
print(f"    Kelly fraction: {kelly_f:.4f}")
print()
print(f"  Kelly optimal leverage: {kelly_lev:.1f}x")
print(f"  Half-Kelly leverage: {half_kelly_lev:.1f}x")
print(f"  Half-Kelly as margin % at 125x: {kelly_margin*100:.1f}%")
print()

# MC optimal
best_fine = max([r for r in fine_results if r['ruin_pct'] <= 1.0], key=lambda x: x['median']) if fine_results else best_safe
print(f"  MC optimal (ruin<=1%): {best_fine['margin_pct']*100:.1f}% margin = {best_fine['eff_lev']:.1f}x effective")
print(f"  Kelly says: {kelly_margin*100:.1f}% margin = {half_kelly_lev:.1f}x effective")
print()
print(f"  CONCLUSION: Optimal margin per trade = ~{best_fine['margin_pct']*100:.0f}% of wallet at 125x leverage")
