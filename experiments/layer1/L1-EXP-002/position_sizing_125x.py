"""L1-EXP-002b: Position Sizing at 125x

QUESTION: What margin% per trade gives optimal growth with acceptable risk?

SETUP (FIXED):
  - Binance leverage setting: 125x (never changes)
  - Margin mode: Cross (full wallet backs trade)
  - Starting capital: $10
  - Position = margin% * wallet * 125
  - Min position enforced: dynamic from BTC price (0.001 BTC min qty)

VARIABLE:
  - margin% = fraction of wallet used as margin per trade
  - This controls position size and risk

TESTS:
  Part 1: Broad sweep (2% to 50% margin) - find the shape
  Part 2: Kelly theoretical - what math says is optimal
  Part 3: Fine sweep around optimal - find exact cliff
  Part 4: MC validation at chosen margin% - 1000 paths
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
# LOAD V1.3.2 TRADES
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

print("=" * 110)
print("L1-EXP-002b: POSITION SIZING AT 125x")
print("=" * 110)
print(f"  V1.3.2: {len(returns)} trades, {len(wins)/len(returns)*100:.1f}% win")
print(f"  Mean: {np.mean(returns):+.1f} bps | Std: {np.std(returns):.1f} bps")
print(f"  Avg win: {np.mean(wins):+.1f} bps | Avg loss: {np.mean(losses):.1f} bps")
print(f"  Best: {max(returns):+.1f} bps | Worst: {min(returns):.1f} bps")
print(f"  Setup: $10 start, 125x leverage, cross margin")
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
    """Simulate with fixed margin% at 125x.

    position = margin_pct * wallet * 125
    If position < min Binance position, use min position instead.
    Cross margin: full wallet backs trade for liquidation.
    """
    equity = [capital]
    liquidated = 0
    skipped = 0

    for td in trade_list:
        eq = equity[-1]

        # Calculate position
        margin = eq * margin_pct
        position = margin * LEVERAGE

        # Enforce Binance minimum
        if position < td['pos_min']:
            position = td['pos_min']

        # Can we afford the margin?
        margin_req = position / LEVERAGE
        if eq < margin_req:
            equity.append(eq)
            skipped += 1
            continue

        # PnL
        pnl = position * (td['bps'] / 10000)

        # Cross margin liquidation check
        maint = position * MAINT_MARGIN_RATE
        max_loss = eq - maint
        if pnl < -max_loss:
            equity.append(0.01)
            liquidated += 1
        else:
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


def run_mc(margin_pct):
    """Run 1000 MC paths for a given margin%."""
    np.random.seed(42)
    n = len(trade_data)

    finals = []
    max_dds = []
    ruin_count = 0
    liq_total = 0

    for _ in range(N_SIMS):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        eq, skipped, liquidated = simulate(shuffled, margin_pct)
        finals.append(eq[-1])
        max_dds.append(calc_max_dd(eq))
        if eq[-1] < 1.0:
            ruin_count += 1
        liq_total += liquidated

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
# PART 1: BROAD SWEEP (2% to 50% margin)
# ============================================================
print("=" * 110)
print("PART 1: BROAD SWEEP -- margin% from 2% to 50%")
print("=" * 110)
print()

broad_pcts = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,
              0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36,
              0.38, 0.40, 0.45, 0.50]

print(f"  {'Margin%':>8s} | {'Eff Lev':>8s} | {'Orig Final':>14s} | {'MC Median':>14s} | {'MC P5':>14s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*110}")

broad_results = []
for pct in broad_pcts:
    # Original order
    eq_orig, _, _ = simulate(trade_data, pct)
    orig_final = eq_orig[-1]

    # MC
    r = run_mc(pct)
    r['margin_pct'] = pct
    r['eff_lev'] = pct * LEVERAGE
    r['orig_final'] = orig_final
    broad_results.append(r)

    print(f"  {pct*100:6.0f}% | {r['eff_lev']:6.1f}x | ${orig_final:>12,.0f} | ${r['median']:>12,.0f} | ${r['p5']:>12,.0f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%")

# Find best safe config
safe_results = [r for r in broad_results if r['ruin_pct'] <= 1.0]
if safe_results:
    best_safe = max(safe_results, key=lambda x: x['median'])
    print()
    print(f"  BEST (ruin <= 1%): {best_safe['margin_pct']*100:.0f}% margin = {best_safe['eff_lev']:.0f}x eff | Median ${best_safe['median']:,.0f} | Ruin {best_safe['ruin_pct']:.1f}%")

best_overall = max(broad_results, key=lambda x: x['median'])
print(f"  BEST (any ruin): {best_overall['margin_pct']*100:.0f}% margin = {best_overall['eff_lev']:.0f}x eff | Median ${best_overall['median']:,.0f} | Ruin {best_overall['ruin_pct']:.1f}%")


# ============================================================
# PART 2: KELLY THEORETICAL
# ============================================================
print()
print("=" * 110)
print("PART 2: KELLY THEORETICAL")
print("=" * 110)
print()

W = len(wins) / len(returns)
avg_win = np.mean(wins)
avg_loss_abs = abs(np.mean(losses))
R = avg_win / avg_loss_abs

# Classic Kelly
kelly_f = W - (1 - W) / R
kelly_lev = kelly_f / (avg_loss_abs / 10000)
half_kelly_lev = kelly_lev / 2
kelly_margin = kelly_lev / LEVERAGE  # full Kelly as margin%
half_kelly_margin = half_kelly_lev / LEVERAGE  # half Kelly as margin%

# Mean-Variance Kelly
mu = np.mean(returns) / 10000
var = np.var(returns) / (10000**2)
mv_kelly_lev = mu / var if var > 0 else 0
mv_half_kelly_lev = mv_kelly_lev / 2
mv_kelly_margin = mv_kelly_lev / LEVERAGE
mv_half_kelly_margin = mv_half_kelly_lev / LEVERAGE

# Geometric Kelly (maximize log growth)
# Optimal f maximizes E[ln(1 + f*r)] where r = return per unit leverage
# Numerically find it
best_geo_f = 0
best_geo_growth = -999
for f_test in np.arange(0.01, 1.0, 0.005):
    log_growth = 0
    for r in returns:
        pnl_pct = f_test * LEVERAGE * r / 10000
        if 1 + pnl_pct > 0:
            log_growth += np.log(1 + pnl_pct)
        else:
            log_growth = -999
            break
    if log_growth > best_geo_growth:
        best_geo_growth = log_growth
        best_geo_f = f_test

print(f"  V1.3.2 stats:")
print(f"    Win rate: {W*100:.1f}%")
print(f"    Avg win: +{avg_win:.1f} bps | Avg loss: -{avg_loss_abs:.1f} bps")
print(f"    Payoff ratio: {R:.2f}")
print()
print(f"  Classic Kelly:")
print(f"    Kelly fraction: {kelly_f:.4f}")
print(f"    Full Kelly leverage: {kelly_lev:.1f}x -> margin% = {kelly_margin*100:.1f}%")
print(f"    Half Kelly leverage: {half_kelly_lev:.1f}x -> margin% = {half_kelly_margin*100:.1f}%")
print()
print(f"  Mean-Variance Kelly:")
print(f"    Full Kelly leverage: {mv_kelly_lev:.1f}x -> margin% = {mv_kelly_margin*100:.1f}%")
print(f"    Half Kelly leverage: {mv_half_kelly_lev:.1f}x -> margin% = {mv_half_kelly_margin*100:.1f}%")
print()
print(f"  Geometric Kelly (maximize log growth):")
print(f"    Optimal margin%: {best_geo_f*100:.1f}% -> eff leverage: {best_geo_f*LEVERAGE:.1f}x")
print(f"    Log growth per trade: {best_geo_growth/len(returns):.6f}")


# ============================================================
# PART 3: FINE SWEEP AROUND OPTIMAL
# ============================================================
print()
print("=" * 110)
print("PART 3: FINE SWEEP -- finding the exact cliff")
print("=" * 110)
print()

# Find the region where ruin transitions from 0% to >0%
# From broad sweep, find where ruin first appears
cliff_region_start = 0.10
for r in broad_results:
    if r['ruin_pct'] > 0:
        cliff_region_start = r['margin_pct'] - 0.06
        break

cliff_region_end = min(cliff_region_start + 0.16, 0.50)

fine_pcts = sorted(set(np.arange(max(0.02, cliff_region_start),
                                  cliff_region_end + 0.01, 0.01).round(2)))

print(f"  Sweeping {cliff_region_start*100:.0f}% to {cliff_region_end*100:.0f}% in 1% steps")
print()
print(f"  {'Margin%':>8s} | {'Eff Lev':>8s} | {'MC Median':>14s} | {'MC P5':>14s} | {'MC P25':>14s} | {'AvgDD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*90}")

fine_results = []
for pct in fine_pcts:
    r = run_mc(pct)
    r['margin_pct'] = pct
    r['eff_lev'] = pct * LEVERAGE
    fine_results.append(r)

    marker = ""
    if abs(pct - half_kelly_margin) < 0.005:
        marker = " <-- HALF KELLY"
    elif abs(pct - best_geo_f) < 0.005:
        marker = " <-- GEO KELLY"

    print(f"  {pct*100:6.0f}% | {r['eff_lev']:6.1f}x | ${r['median']:>12,.0f} | ${r['p5']:>12,.0f} | ${r['p25']:>12,.0f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%{marker}")

# Find optimal in fine sweep
fine_safe = [r for r in fine_results if r['ruin_pct'] <= 1.0]
if fine_safe:
    best_fine = max(fine_safe, key=lambda x: x['median'])
    print()
    print(f"  OPTIMAL (ruin <= 1%): {best_fine['margin_pct']*100:.0f}% margin = {best_fine['eff_lev']:.0f}x eff")
    print(f"    Median: ${best_fine['median']:,.0f} | P5: ${best_fine['p5']:,.0f} | AvgDD: {best_fine['avg_dd']*100:.1f}% | Ruin: {best_fine['ruin_pct']:.1f}%")


# ============================================================
# PART 4: DEEP MC AT KEY MARGIN% VALUES
# ============================================================
print()
print("=" * 110)
print("PART 4: DEEP MC (5000 paths) AT KEY VALUES")
print("=" * 110)
print()

# Pick key values to test with more MC paths
N_DEEP = 5000
key_pcts = sorted(set([
    half_kelly_margin,
    best_geo_f,
    0.16,  # 20x effective
    0.20,  # 25x effective
    0.24,  # 30x effective
]))
# Round to nearest 1%
key_pcts = sorted(set([round(p, 2) for p in key_pcts]))

print(f"  Running {N_DEEP} MC paths per config...")
print()
print(f"  {'Margin%':>8s} | {'Eff Lev':>8s} | {'Median':>14s} | {'P5':>14s} | {'P25':>14s} | {'P75':>14s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*120}")

for pct in key_pcts:
    np.random.seed(42)
    n = len(trade_data)
    finals = []
    max_dds = []
    ruin_count = 0

    for _ in range(N_DEEP):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        eq, _, _ = simulate(shuffled, pct)
        finals.append(eq[-1])
        max_dds.append(calc_max_dd(eq))
        if eq[-1] < 1.0:
            ruin_count += 1

    median = np.median(finals)
    p5 = np.percentile(finals, 5)
    p25 = np.percentile(finals, 25)
    p75 = np.percentile(finals, 75)
    avg_dd = np.mean(max_dds)
    p95_dd = np.percentile(max_dds, 95)
    ruin_pct = ruin_count / N_DEEP * 100

    label = ""
    if abs(pct - half_kelly_margin) < 0.005:
        label = " (Half Kelly)"
    elif abs(pct - best_geo_f) < 0.005:
        label = " (Geo Kelly)"

    print(f"  {pct*100:6.0f}% | {pct*LEVERAGE:6.1f}x | ${median:>12,.0f} | ${p5:>12,.0f} | ${p25:>12,.0f} | ${p75:>12,.0f} | {avg_dd*100:5.1f}% | {p95_dd*100:5.1f}% | {ruin_pct:5.2f}%{label}")


# ============================================================
# PART 5: VERDICT
# ============================================================
print()
print("=" * 110)
print("PART 5: VERDICT")
print("=" * 110)
print()

# Collect all results for summary
all_safe = [r for r in broad_results + fine_results if r['ruin_pct'] <= 1.0]
if all_safe:
    optimal = max(all_safe, key=lambda x: x['median'])
    conservative = max([r for r in all_safe if r['ruin_pct'] == 0], key=lambda x: x['median']) if [r for r in all_safe if r['ruin_pct'] == 0] else None

    print(f"  OPTIMAL (max median, ruin <= 1%):")
    print(f"    Margin: {optimal['margin_pct']*100:.0f}% | Eff leverage: {optimal['eff_lev']:.0f}x")
    print(f"    Position at $10 wallet: ${10 * optimal['margin_pct'] * LEVERAGE:,.0f}")
    print(f"    Position at $100 wallet: ${100 * optimal['margin_pct'] * LEVERAGE:,.0f}")
    print(f"    MC Median: ${optimal['median']:,.0f} | P5: ${optimal['p5']:,.0f} | Ruin: {optimal['ruin_pct']:.1f}%")
    print()

    if conservative:
        print(f"  CONSERVATIVE (max median, ruin = 0%):")
        print(f"    Margin: {conservative['margin_pct']*100:.0f}% | Eff leverage: {conservative['eff_lev']:.0f}x")
        print(f"    Position at $10 wallet: ${10 * conservative['margin_pct'] * LEVERAGE:,.0f}")
        print(f"    Position at $100 wallet: ${100 * conservative['margin_pct'] * LEVERAGE:,.0f}")
        print(f"    MC Median: ${conservative['median']:,.0f} | P5: ${conservative['p5']:,.0f} | Ruin: {conservative['ruin_pct']:.1f}%")
        print()

    print(f"  Kelly says:")
    print(f"    Classic half-Kelly: {half_kelly_margin*100:.1f}% margin = {half_kelly_lev:.1f}x eff")
    print(f"    Mean-Var half-Kelly: {mv_half_kelly_margin*100:.1f}% margin = {mv_half_kelly_lev:.1f}x eff")
    print(f"    Geometric optimal: {best_geo_f*100:.1f}% margin = {best_geo_f*LEVERAGE:.1f}x eff")
    print()
    print(f"  ANSWER: Use {optimal['margin_pct']*100:.0f}% margin per trade at 125x leverage")
    print(f"          Position = {optimal['margin_pct']*100:.0f}% * wallet * 125")
