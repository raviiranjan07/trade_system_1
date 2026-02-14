"""L1-EXP-002c: Position Sizing in BTC Qty

QUESTION: How many 0.001 BTC units to trade per signal?

SETUP (FIXED, NEVER CHANGES):
  - Binance leverage: 125x
  - Margin mode: Cross
  - Starting wallet: $10
  - Min qty: 0.001 BTC, step: 0.001 BTC

VARIABLE:
  - Position size in BTC qty (multiples of 0.001)
  - As wallet grows, qty scales up

KEY INSIGHT:
  - More qty = more profit per trade but closer liquidation
  - Liquidation depends on: wallet vs (position * maint_rate)
  - Liq buffer = (wallet - maint) / position

TESTS:
  Part 1: Fixed qty (1x-10x) at $10 — no scaling, shows liquidation
  Part 2: Scaling rules — when to increase qty as wallet grows
  Part 3: Kelly — what qty maximizes geometric growth
  Part 4: MC validation — 1000 paths for best configs
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import math
import numpy as np
from v12.backtest import run_backtest
from v12.config.loader import load_config

# ============================================================
# CONSTANTS
# ============================================================
STARTING_CAPITAL = 10.0
LEVERAGE = 125
MAINT_MARGIN_RATE = 0.004
BINANCE_MIN_QTY = 0.001
BINANCE_STEP_SIZE = 0.001
BINANCE_MIN_NOTIONAL = 100
N_SIMS = 1000

# ============================================================
# LOAD TRADES
# ============================================================
config = load_config()
trades = run_backtest(config)

trade_data = []
for t in trades:
    btc_price = t.entry_price
    qty_min = max(BINANCE_MIN_QTY,
                  math.ceil(BINANCE_MIN_NOTIONAL / btc_price / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE)
    trade_data.append({
        'bps': t.net_profit_bps,
        'btc_price': btc_price,
        'qty_min': qty_min,
        'pos_min': qty_min * btc_price,
    })

returns = [td['bps'] for td in trade_data]
wins = [r for r in returns if r > 0]
losses = [r for r in returns if r <= 0]

print("=" * 100)
print("L1-EXP-002c: POSITION SIZING IN BTC QTY")
print("=" * 100)
print(f"  V1.3.2: {len(returns)} trades, {len(wins)/len(returns)*100:.1f}% win")
print(f"  Avg win: +{np.mean(wins):.1f} bps | Avg loss: {np.mean(losses):.1f} bps")
print(f"  Best: +{max(returns):.1f} bps | Worst: {min(returns):.1f} bps")
print(f"  Wallet: $10 | Leverage: 125x (fixed) | Cross margin")
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


# ============================================================
# PART 1: FIXED QTY — Liquidation analysis
# ============================================================
print("=" * 100)
print("PART 1: FIXED QTY AT $10 WALLET — liquidation for each size")
print("=" * 100)
print()

# Use average BTC price from our trades
avg_btc = np.mean([td['btc_price'] for td in trade_data])
print(f"  Average BTC price in OOS trades: ${avg_btc:,.0f}")
print()

print(f"  {'Qty':>8s} | {'Position':>10s} | {'Margin':>8s} | {'Maint':>8s} | {'Liq Buffer':>10s} | {'Liq LONG':>12s} | {'Liq SHORT':>12s} | {'Worst OK?':>10s}")
print(f"  {'-'*100}")

worst_trade_bps = min(returns)

for mult in range(1, 11):
    qty = BINANCE_MIN_QTY * mult
    position = qty * avg_btc
    margin = position / LEVERAGE
    maint = position * MAINT_MARGIN_RATE
    liq_buffer_dollar = STARTING_CAPITAL - maint
    liq_buffer_bps = liq_buffer_dollar / position * 10000
    liq_long = avg_btc * (1 - liq_buffer_dollar / position)
    liq_short = avg_btc * (1 + liq_buffer_dollar / position)
    worst_ok = "YES" if abs(worst_trade_bps) < liq_buffer_bps else "NO"

    print(f"  {qty:.3f} | ${position:>8,.0f} | ${margin:>6.2f} | ${maint:>6.2f} | {liq_buffer_bps:>8.0f} bps | ${liq_long:>10,.0f} | ${liq_short:>10,.0f} | {worst_ok:>10s}")

print()
print(f"  Worst trade: {worst_trade_bps:.1f} bps")
print(f"  Any qty where worst trade causes liquidation = NOT SAFE")


# ============================================================
# PART 2: FIXED QTY — Performance (no scaling)
# ============================================================
print()
print("=" * 100)
print("PART 2: FIXED QTY — performance over 220 trades (no scaling)")
print("=" * 100)
print()

def simulate_fixed_qty(trade_list, qty_mult, capital=STARTING_CAPITAL):
    """Trade fixed BTC qty regardless of wallet size."""
    equity = [capital]
    skipped = 0
    liquidated = 0

    for td in trade_list:
        eq = equity[-1]
        qty = td['qty_min'] * qty_mult
        position = qty * td['btc_price']
        margin = position / LEVERAGE
        maint = position * MAINT_MARGIN_RATE

        # Can afford margin?
        if eq < margin:
            equity.append(eq)
            skipped += 1
            continue

        pnl = position * (td['bps'] / 10000)

        # Cross margin liq check
        max_loss = eq - maint
        if pnl < -max_loss:
            equity.append(0.01)
            liquidated += 1
        else:
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated

print(f"  {'Qty Mult':>8s} | {'Final':>12s} | {'Return':>10s} | {'MaxDD':>7s} | {'MinEq':>8s} | {'Skip':>6s} | {'Liq':>4s}")
print(f"  {'-'*75}")

for mult in range(1, 11):
    eq, skip, liq = simulate_fixed_qty(trade_data, mult)
    final = eq[-1]
    ret = (final - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    dd = calc_max_dd(eq)
    min_eq = min(eq)
    print(f"  {mult:>7d}x | ${final:>10,.2f} | {ret:>+8.1f}% | {dd*100:5.1f}% | ${min_eq:>6.2f} | {skip:>6d} | {liq:>4d}")

print()
print("  NOTE: Fixed qty = LINEAR growth (no compounding)")


# ============================================================
# PART 3: SCALING QTY — increase qty as wallet grows
# ============================================================
print()
print("=" * 100)
print("PART 3: SCALING QTY — position grows with wallet")
print("=" * 100)
print()
print("  Rule: qty = floor(wallet / step_size) * 0.001 BTC")
print("  step_size = how many $ per 0.001 BTC step")
print("  Example: step_size=$3 means at $10 wallet -> 3x qty, at $30 -> 10x qty")
print()

def simulate_scaling_qty(trade_list, dollars_per_step, capital=STARTING_CAPITAL):
    """Scale BTC qty with wallet size.

    qty = floor(wallet / dollars_per_step) * 0.001
    Enforces minimum 0.001 BTC and Binance constraints.
    """
    equity = [capital]
    skipped = 0
    liquidated = 0

    for td in trade_list:
        eq = equity[-1]

        # Calculate qty based on wallet
        steps = max(1, int(eq / dollars_per_step))
        qty = steps * BINANCE_STEP_SIZE

        # Enforce minimum
        qty = max(qty, td['qty_min'])

        position = qty * td['btc_price']
        margin = position / LEVERAGE
        maint = position * MAINT_MARGIN_RATE

        # Can afford?
        if eq < margin:
            equity.append(eq)
            skipped += 1
            continue

        pnl = position * (td['bps'] / 10000)

        # Liq check
        max_loss = eq - maint
        if pnl < -max_loss:
            equity.append(0.01)
            liquidated += 1
        else:
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated

# Test different step sizes
step_sizes = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]

print(f"  {'$/step':>8s} | {'Qty@$10':>8s} | {'Qty@$100':>9s} | {'Qty@$1K':>9s} | {'Final':>14s} | {'MaxDD':>7s} | {'MinEq':>8s} | {'Liq':>4s}")
print(f"  {'-'*95}")

for step in step_sizes:
    eq, skip, liq = simulate_scaling_qty(trade_data, step)
    final = eq[-1]
    dd = calc_max_dd(eq)
    min_eq = min(eq)

    qty_10 = max(1, int(10 / step)) * 0.001
    qty_100 = max(1, int(100 / step)) * 0.001
    qty_1000 = max(1, int(1000 / step)) * 0.001

    print(f"  ${step:>5.1f} | {qty_10:.3f} | {qty_100:.3f}  | {qty_1000:.3f}  | ${final:>12,.0f} | {dd*100:5.1f}% | ${min_eq:>6.2f} | {liq:>4d}")


# ============================================================
# PART 4: MC — 1000 paths for each scaling rule
# ============================================================
print()
print("=" * 100)
print("PART 4: MONTE CARLO — 1000 paths per scaling rule")
print("=" * 100)
print()

print(f"  {'$/step':>8s} | {'Median':>14s} | {'P5':>14s} | {'P25':>14s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*90}")

mc_results = {}
for step in step_sizes:
    np.random.seed(42)
    finals = []
    max_dds = []
    ruin_count = 0

    for _ in range(N_SIMS):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        eq, _, liq = simulate_scaling_qty(shuffled, step)
        finals.append(eq[-1])
        max_dds.append(calc_max_dd(eq))
        if eq[-1] < 1.0:
            ruin_count += 1

    mc_results[step] = {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'ruin_pct': ruin_count / N_SIMS * 100,
    }

    r = mc_results[step]
    print(f"  ${step:>5.1f} | ${r['median']:>12,.0f} | ${r['p5']:>12,.0f} | ${r['p25']:>12,.0f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%")


# ============================================================
# PART 5: FINE SWEEP around best
# ============================================================
print()
print("=" * 100)
print("PART 5: FINE SWEEP around optimal $/step")
print("=" * 100)
print()

# Find best safe from broad sweep
safe = {k: v for k, v in mc_results.items() if v['ruin_pct'] <= 1.0}
if safe:
    best_step = max(safe, key=lambda k: safe[k]['median'])
    # Fine sweep around it
    fine_steps = sorted(set([round(best_step + d * 0.25, 2) for d in range(-6, 7) if best_step + d * 0.25 > 0]))

    print(f"  Best from broad: ${best_step}/step | Sweeping nearby values")
    print()
    print(f"  {'$/step':>8s} | {'Median':>14s} | {'P5':>14s} | {'AvgDD':>7s} | {'Ruin%':>6s}")
    print(f"  {'-'*65}")

    fine_results = {}
    for step in fine_steps:
        np.random.seed(42)
        finals = []
        max_dds = []
        ruin_count = 0

        for _ in range(N_SIMS):
            shuffled = list(trade_data)
            np.random.shuffle(shuffled)
            eq, _, _ = simulate_scaling_qty(shuffled, step)
            finals.append(eq[-1])
            max_dds.append(calc_max_dd(eq))
            if eq[-1] < 1.0:
                ruin_count += 1

        fine_results[step] = {
            'median': np.median(finals),
            'p5': np.percentile(finals, 5),
            'avg_dd': np.mean(max_dds),
            'ruin_pct': ruin_count / N_SIMS * 100,
        }

        r = fine_results[step]
        marker = " <--" if step == best_step else ""
        print(f"  ${step:>5.2f} | ${r['median']:>12,.0f} | ${r['p5']:>12,.0f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%{marker}")

    # Find optimal in fine
    fine_safe = {k: v for k, v in fine_results.items() if v['ruin_pct'] <= 1.0}
    if fine_safe:
        optimal_step = max(fine_safe, key=lambda k: fine_safe[k]['median'])
        print()
        print(f"  OPTIMAL: ${optimal_step}/step")
        print(f"    At $10 wallet: {max(1, int(10 / optimal_step))}x qty = {max(1, int(10 / optimal_step)) * 0.001:.3f} BTC")
        print(f"    At $100 wallet: {max(1, int(100 / optimal_step))}x qty = {max(1, int(100 / optimal_step)) * 0.001:.3f} BTC")
        print(f"    At $1000 wallet: {max(1, int(1000 / optimal_step))}x qty = {max(1, int(1000 / optimal_step)) * 0.001:.3f} BTC")


# ============================================================
# PART 6: VERDICT
# ============================================================
print()
print("=" * 100)
print("PART 6: VERDICT")
print("=" * 100)
print()

# Collect all safe results
all_mc = {}
all_mc.update(mc_results)
if 'fine_results' in dir():
    all_mc.update(fine_results)

all_safe = {k: v for k, v in all_mc.items() if v['ruin_pct'] <= 1.0}
zero_ruin = {k: v for k, v in all_mc.items() if v['ruin_pct'] == 0}

if all_safe:
    opt = max(all_safe, key=lambda k: all_safe[k]['median'])
    r = all_safe[opt]
    print(f"  OPTIMAL (ruin <= 1%): ${opt}/step")
    print(f"    Median: ${r['median']:,.0f} | P5: ${r['p5']:,.0f} | AvgDD: {r['avg_dd']*100:.1f}% | Ruin: {r['ruin_pct']:.1f}%")
    print(f"    At $10: {max(1,int(10/opt))}x qty ({max(1,int(10/opt))*0.001:.3f} BTC)")
    print(f"    At $100: {max(1,int(100/opt))}x qty ({max(1,int(100/opt))*0.001:.3f} BTC)")
    print()

if zero_ruin:
    con = max(zero_ruin, key=lambda k: zero_ruin[k]['median'])
    r = zero_ruin[con]
    print(f"  CONSERVATIVE (0% ruin): ${con}/step")
    print(f"    Median: ${r['median']:,.0f} | P5: ${r['p5']:,.0f} | AvgDD: {r['avg_dd']*100:.1f}% | Ruin: {r['ruin_pct']:.1f}%")
    print(f"    At $10: {max(1,int(10/con))}x qty ({max(1,int(10/con))*0.001:.3f} BTC)")
    print(f"    At $100: {max(1,int(100/con))}x qty ({max(1,int(100/con))*0.001:.3f} BTC)")
    print()

print("  HOW IT WORKS:")
print("    1. Check wallet balance")
print("    2. qty = floor(wallet / $/step) * 0.001 BTC")
print("    3. Enforce minimum 0.001 BTC")
print("    4. Place order with qty at 125x leverage")
print("    5. Margin = position / 125 (auto-calculated by Binance)")
print("    6. Full wallet backs trade (cross margin)")
