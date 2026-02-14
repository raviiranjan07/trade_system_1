"""L1-EXP-003: Binance Reality Check - Funding Rates + Exact Liquidation

QUESTION: Do our EXP-002 results hold up when we add real Binance costs?

WHAT WE TEST:
  1. Trade duration analysis (how many bars, do trades cross funding times?)
  2. Funding rate impact at different rate levels (0.005%, 0.01%, 0.03%, 0.05%)
  3. Exact Binance liquidation formula (maintenance margin = 0.4%)
  4. Combined impact on Iso 5% hybrid (our EXP-002 winner)

FUNDING RATE FACTS:
  - Binance charges/pays every 8 hours (00:00, 08:00, 16:00 UTC)
  - Rate varies: typically 0.005-0.03%, can spike to 0.1%+
  - If LONG and rate > 0: you PAY | If SHORT and rate > 0: you RECEIVE
  - If LONG and rate < 0: you RECEIVE | If SHORT and rate < 0: you PAY
  - Net effect over time is roughly balanced, but we model worst case

LIQUIDATION FORMULA (Binance Isolated):
  - LONG: Liq Price = Entry * (1 - (margin / position) + maintenance_rate)
  - SHORT: Liq Price = Entry * (1 + (margin / position) - maintenance_rate)
  - Maintenance margin rate for BTCUSDT: 0.4% (Tier 1, <$50K)
"""
import sys
sys.path.insert(0, "src")

import numpy as np
from v12.backtest import run_backtest
from v12.config.loader import load_config

STARTING_CAPITAL = 10.0
MIN_NOTIONAL = 170.0
MAX_LEVERAGE_SETTING = 125
N_SIMS = 1000
MAINT_MARGIN_RATE = 0.004  # 0.4% for BTCUSDT Tier 1

config = load_config()
trades = run_backtest(config)

# Extract all trade data
returns = [t.net_profit_bps for t in trades]
bars_held = [t.exit_bar for t in trades]
directions = [t.direction for t in trades]
exit_reasons = [t.exit_reason for t in trades]
entry_times = [t.entry_time for t in trades]
exit_times = [t.exit_time for t in trades]
signal_types = [t.signal_type for t in trades]

print("=" * 120)
print("L1-EXP-003: BINANCE REALITY CHECK - Funding Rates + Exact Liquidation")
print("=" * 120)
print(f"V1.3.2: {len(trades)} trades, {sum(1 for r in returns if r > 0)/len(returns)*100:.1f}% win")
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
# PART 1: TRADE DURATION ANALYSIS
# ============================================================
print("=" * 120)
print("PART 1: TRADE DURATION ANALYSIS")
print("=" * 120)
print()

print(f"  Bars held distribution (each bar = 15 minutes):")
print(f"    Mean:   {np.mean(bars_held):.1f} bars ({np.mean(bars_held)*15:.0f} min)")
print(f"    Median: {np.median(bars_held):.0f} bars ({np.median(bars_held)*15:.0f} min)")
print(f"    Min:    {min(bars_held)} bars ({min(bars_held)*15} min)")
print(f"    Max:    {max(bars_held)} bars ({max(bars_held)*15} min)")
print()

# Distribution of bars held
from collections import Counter
bar_counts = Counter(bars_held)
print(f"  {'Bars':>6s} | {'Minutes':>8s} | {'Count':>6s} | {'%':>6s} | {'Cum%':>6s}")
print(f"  {'-'*45}")
cum = 0
for b in sorted(bar_counts.keys()):
    cum += bar_counts[b]
    print(f"  {b:>6d} | {b*15:>7d}m | {bar_counts[b]:>6d} | {bar_counts[b]/len(trades)*100:5.1f}% | {cum/len(trades)*100:5.1f}%")

print()

# By exit reason
print(f"  By exit reason:")
reason_counts = Counter(exit_reasons)
for reason, count in reason_counts.most_common():
    avg_bars = np.mean([b for b, r in zip(bars_held, exit_reasons) if r == reason])
    print(f"    {reason:>15s}: {count:>4d} trades, avg {avg_bars:.1f} bars ({avg_bars*15:.0f} min)")

print()

# Funding time crossings
# Funding at 00:00, 08:00, 16:00 UTC - every 480 minutes (32 bars)
# A trade crosses funding if it spans one of these boundaries
FUNDING_INTERVAL_BARS = 32  # 480 min / 15 min per bar

# Check how many trades could cross a funding time
# Max hold = 10 bars = 150 min. Funding every 480 min.
# So a trade can cross AT MOST one funding time
# Probability: trade_duration / funding_interval

trades_crossing_funding = 0
for i, t in enumerate(trades):
    try:
        entry_h = t.entry_time.hour
        entry_m = t.entry_time.minute
        exit_h = t.exit_time.hour
        exit_m = t.exit_time.minute

        entry_min = entry_h * 60 + entry_m
        exit_min = exit_h * 60 + exit_m

        # Handle day boundary
        if exit_min < entry_min:
            exit_min += 1440

        # Funding times in minutes: 0, 480, 960 (00:00, 08:00, 16:00)
        for ft in [0, 480, 960, 1440, 1920]:
            if entry_min < ft <= exit_min:
                trades_crossing_funding += 1
                break
    except:
        pass

print(f"  Funding time analysis:")
print(f"    Funding charged at: 00:00, 08:00, 16:00 UTC (every 8 hours)")
print(f"    Max trade duration: {max(bars_held)} bars = {max(bars_held)*15} min")
print(f"    Funding interval: {FUNDING_INTERVAL_BARS} bars = 480 min")
print(f"    Trades crossing funding time: {trades_crossing_funding}/{len(trades)} ({trades_crossing_funding/len(trades)*100:.1f}%)")
print(f"    Avg trade duration: {np.mean(bars_held)*15:.0f} min vs 480 min funding interval")
print(f"    Expected crossings (uniform): {np.mean(bars_held)*15/480*100:.1f}% of trades")
print()

# LONG vs SHORT breakdown (matters for funding direction)
long_count = sum(1 for d in directions if d == "LONG")
short_count = sum(1 for d in directions if d == "SHORT")
print(f"  Direction breakdown:")
print(f"    LONG:  {long_count} trades ({long_count/len(trades)*100:.1f}%)")
print(f"    SHORT: {short_count} trades ({short_count/len(trades)*100:.1f}%)")
print(f"    (When funding > 0: LONG pays, SHORT receives)")
print(f"    (BTC funding is mostly positive -> LONG pays more often)")


# ============================================================
# PART 2: FUNDING RATE IMPACT
# ============================================================
print()
print("=" * 120)
print("PART 2: FUNDING RATE IMPACT - How much does funding cost?")
print("=" * 120)
print()

# Model: for each trade that crosses a funding time, add/subtract funding cost
# funding_cost = position_size * funding_rate
# LONG pays when rate > 0, SHORT receives when rate > 0
# Worst case: assume ALL funding payments are costs (conservative)

funding_rates = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]  # in percent

print("  WORST CASE: Assume every funding payment is a COST (never receive)")
print()

# Calculate funding cost per trade
# Only trades crossing funding time pay
# Cost = position * rate

# How many trades cross funding? Use our actual count
pct_crossing = trades_crossing_funding / len(trades)

print(f"  Position size: $170 (Phase 1)")
print(f"  Trades crossing funding time: {trades_crossing_funding}/{len(trades)} ({pct_crossing*100:.1f}%)")
print()

print(f"  {'Funding Rate':>14s} | {'Cost/Trade':>11s} | {'Total Cost':>11s} | {'Bps/Trade':>10s} | {'Total Bps':>10s} | {'% of Profit':>12s}")
print(f"  {'-'*85}")

total_profit_bps = sum(returns)
total_profit_dollars = sum(r * 170 / 10000 for r in returns)  # approximate

for rate in funding_rates:
    cost_per_trade = 170 * (rate / 100)  # $ per funding event
    total_cost = cost_per_trade * trades_crossing_funding
    bps_per_trade = cost_per_trade / 170 * 10000  # convert to bps
    total_bps = bps_per_trade * trades_crossing_funding
    pct_of_profit = total_cost / total_profit_dollars * 100 if total_profit_dollars > 0 else 0

    print(f"  {rate:>13.3f}% | ${cost_per_trade:10.4f} | ${total_cost:10.2f} | {bps_per_trade:>9.1f} | {total_bps:>9.1f} | {pct_of_profit:>10.1f}%")

print()

# More realistic: LONG pays, SHORT receives (when funding > 0)
print("  REALISTIC: LONG pays, SHORT receives (typical positive funding)")
print()

for rate in [0.01, 0.03, 0.05]:
    long_cost = 0
    short_income = 0

    for i, t in enumerate(trades):
        try:
            entry_h = t.entry_time.hour
            entry_m = t.entry_time.minute
            exit_h = t.exit_time.hour
            exit_m = t.exit_time.minute
            entry_min = entry_h * 60 + entry_m
            exit_min = exit_h * 60 + exit_m
            if exit_min < entry_min:
                exit_min += 1440

            crosses = False
            for ft in [0, 480, 960, 1440, 1920]:
                if entry_min < ft <= exit_min:
                    crosses = True
                    break

            if crosses:
                cost = 170 * (rate / 100)
                if t.direction == "LONG":
                    long_cost += cost
                else:
                    short_income += cost
        except:
            pass

    net = long_cost - short_income
    print(f"  Rate {rate:.3f}%: LONG pays ${long_cost:.2f}, SHORT receives ${short_income:.2f}, NET cost = ${net:.2f} ({net/total_profit_dollars*100:.1f}% of profit)")


# ============================================================
# PART 3: EXACT BINANCE LIQUIDATION
# ============================================================
print()
print("=" * 120)
print("PART 3: EXACT BINANCE LIQUIDATION (Isolated Margin)")
print("=" * 120)
print()

print("  Binance Isolated Liquidation Formula:")
print("    LONG:  Liquidated when price drops to: Entry * (1 - margin/position + maint_rate)")
print("    SHORT: Liquidated when price rises to: Entry * (1 + margin/position - maint_rate)")
print(f"    Maintenance margin rate: {MAINT_MARGIN_RATE*100:.1f}% (BTCUSDT Tier 1)")
print()

# Compare our model vs exact Binance liquidation
# Our model: liquidated when loss > margin
# Binance: liquidated slightly EARLIER because of maintenance margin

# For $170 position with different margins:
print(f"  {'Margin':>10s} | {'Our Model':>20s} | {'Binance Exact':>20s} | {'Difference':>12s}")
print(f"  {'-'*70}")

for margin in [0.50, 1.00, 1.50, 2.00, 3.00]:
    # Our model: liquidated when loss = margin
    our_liq_bps = margin / 170 * 10000  # loss in bps that triggers liquidation

    # Binance: you get liquidated when unrealized loss = margin - maintenance_margin
    # maintenance_margin = position * maint_rate = 170 * 0.004 = $0.68
    maint_margin = 170 * MAINT_MARGIN_RATE
    binance_liq_loss = margin - maint_margin  # you lose MORE of your margin to maint
    binance_liq_bps = binance_liq_loss / 170 * 10000

    diff_bps = our_liq_bps - binance_liq_bps

    print(f"  ${margin:>8.2f} | {our_liq_bps:>8.1f} bps loss | {binance_liq_bps:>8.1f} bps loss | {diff_bps:>+8.1f} bps")

print()
print(f"  Maintenance margin = $170 * {MAINT_MARGIN_RATE*100:.1f}% = ${170 * MAINT_MARGIN_RATE:.2f}")
print(f"  Binance liquidates {170 * MAINT_MARGIN_RATE:.2f} bps EARLIER than our model")
print(f"  At 5% margin ($0.50): our model says liq at 29.4 bps loss, Binance at -10.6 bps")
print(f"  WAIT - this means at $0.50 margin, Binance would liquidate BEFORE any loss!")
print()

# Recalculate: what's the MINIMUM margin to not get instantly liquidated?
min_margin = 170 * MAINT_MARGIN_RATE
print(f"  CRITICAL: Minimum margin to not get instantly liquidated = ${min_margin:.2f}")
print(f"  Our 5% of $10 = $0.50 < ${min_margin:.2f} -> WOULD GET INSTANTLY LIQUIDATED!")
print()

# What equity do we need for 5% to be safe?
safe_equity = min_margin / 0.05
print(f"  5% margin >= ${min_margin:.2f} requires equity >= ${safe_equity:.2f}")
print(f"  10% margin >= ${min_margin:.2f} requires equity >= ${min_margin/0.10:.2f}")
print(f"  15% margin >= ${min_margin:.2f} requires equity >= ${min_margin/0.15:.2f}")
print(f"  20% margin >= ${min_margin:.2f} requires equity >= ${min_margin/0.20:.2f}")
print()

# What does this mean for our configs?
print("  IMPACT ON OUR CONFIGS:")
print(f"  At $10 equity, 5% margin = $0.50:")
print(f"    Margin ($0.50) - Maintenance (${min_margin:.2f}) = ${0.50 - min_margin:.2f}")
if 0.50 < min_margin:
    print(f"    NEGATIVE! Trade would be instantly liquidated on Binance!")
    print(f"    Need margin > ${min_margin:.2f} to open position")
else:
    print(f"    Max loss before liquidation: ${0.50 - min_margin:.2f} = {(0.50-min_margin)/170*10000:.1f} bps")

print()
print(f"  At $10 equity, 10% margin = $1.00:")
usable = 1.00 - min_margin
print(f"    Margin ($1.00) - Maintenance (${min_margin:.2f}) = ${usable:.2f}")
print(f"    Max loss before liquidation: ${usable:.2f} = {usable/170*10000:.1f} bps")

print()
print(f"  At $10 equity, 20% margin = $2.00:")
usable = 2.00 - min_margin
print(f"    Margin ($2.00) - Maintenance (${min_margin:.2f}) = ${usable:.2f}")
print(f"    Max loss before liquidation: ${usable:.2f} = {usable/170*10000:.1f} bps")


# ============================================================
# PART 4: CORRECTED SIMULATION - Exact Binance liquidation
# ============================================================
print()
print("=" * 120)
print("PART 4: CORRECTED SIMULATION - With maintenance margin")
print("=" * 120)
print()


def simulate_binance_pct_hybrid(rets, position_size, pct, threshold, scale_lev,
                                 maint_rate=MAINT_MARGIN_RATE, funding_rate=0.0,
                                 funding_cross_pct=0.0, capital=STARTING_CAPITAL):
    """Phase 1: Isolated with exact Binance liquidation. Phase 2: Cross scaling.

    Binance liquidation: when loss >= margin - (position * maint_rate)
    """
    equity = [capital]
    skipped = 0
    liquidated = 0
    funding_paid = 0.0

    maint_margin = position_size * maint_rate

    for i, r in enumerate(rets):
        eq = equity[-1]

        if eq < threshold:
            # Phase 1: Isolated margin with Binance mechanics
            margin = eq * pct

            # Can't open if margin < maintenance margin
            if margin <= maint_margin:
                equity.append(eq)
                skipped += 1
                continue

            pnl = position_size * (r / 10000)

            # Funding cost (probabilistic)
            if funding_rate > 0 and np.random.random() < funding_cross_pct:
                fund_cost = position_size * (funding_rate / 100)
                pnl -= fund_cost
                funding_paid += fund_cost

            # Binance liquidation: lose margin when loss >= margin - maint_margin
            max_loss = margin - maint_margin
            if pnl < -max_loss:
                # LIQUIDATED: lose entire allocated margin
                equity.append(eq - margin)
                liquidated += 1
            else:
                equity.append(max(eq + pnl, 0.01))
        else:
            # Phase 2: Cross margin scaling
            pos = eq * scale_lev
            pnl = pos * (r / 10000)

            # Funding on Phase 2 too
            if funding_rate > 0 and np.random.random() < funding_cross_pct:
                fund_cost = pos * (funding_rate / 100)
                pnl -= fund_cost
                funding_paid += fund_cost

            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated, funding_paid


# Original order comparison
print("--- ORIGINAL ORDER ---")
print()
print(f"  {'Config':>45s} | {'Final':>10s} | {'MaxDD':>7s} | {'Skip':>5s} | {'Liq':>4s} | {'Fund$':>7s}")
print(f"  {'-'*95}")

configs = [
    ("EXP-002 winner (no Binance costs)", 0.05, 0.0, 0.0),
    ("+ Maintenance margin only", 0.05, 0.0, 0.0),  # handled by function
    ("+ Funding 0.01% (typical)", 0.05, 0.01, pct_crossing),
    ("+ Funding 0.03% (high)", 0.05, 0.03, pct_crossing),
    ("+ Funding 0.05% (stress)", 0.05, 0.05, pct_crossing),
    ("10% margin + maint + fund 0.01%", 0.10, 0.01, pct_crossing),
    ("15% margin + maint + fund 0.01%", 0.15, 0.01, pct_crossing),
    ("20% margin + maint + fund 0.01%", 0.20, 0.01, pct_crossing),
]

np.random.seed(42)
for name, pct, frate, fcross in configs:
    eq, sk, liq, fpaid = simulate_binance_pct_hybrid(returns, 170, pct, 25, 20,
                                                      funding_rate=frate,
                                                      funding_cross_pct=fcross)
    dd = calc_max_dd(eq)
    print(f"  {name:>45s} | ${eq[-1]:9.2f} | {dd*100:5.1f}% | {sk:>5d} | {liq:>4d} | ${fpaid:6.2f}")


# ============================================================
# PART 5: MONTE CARLO with Binance costs
# ============================================================
print()
print("=" * 120)
print("PART 5: MONTE CARLO - With Binance reality costs")
print("=" * 120)
print()

np.random.seed(42)

mc_configs = {
    'EXP-002: 5% no costs': (0.05, 0.0, 0.0),
    '5% + maint + fund 0.01%': (0.05, 0.01, pct_crossing),
    '5% + maint + fund 0.03%': (0.05, 0.03, pct_crossing),
    '10% + maint + fund 0.01%': (0.10, 0.01, pct_crossing),
    '15% + maint + fund 0.01%': (0.15, 0.01, pct_crossing),
    '20% + maint + fund 0.01%': (0.20, 0.01, pct_crossing),
}

print(f"  {'Config':>30s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'P75':>12s} | {'AvgDD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*105}")

mc_results = {}
for name, (pct, frate, fcross) in mc_configs.items():
    finals = []
    max_dds = []
    ruined = 0

    for _ in range(N_SIMS):
        shuffled = list(returns)
        np.random.shuffle(shuffled)
        eq, sk, liq, fp = simulate_binance_pct_hybrid(shuffled, 170, pct, 25, 20,
                                                       funding_rate=frate,
                                                       funding_cross_pct=fcross)
        finals.append(eq[-1])
        dd = calc_max_dd(eq)
        max_dds.append(dd)
        if eq[-1] < 1.36:
            ruined += 1

    mc_results[name] = {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'avg_dd': np.mean(max_dds),
        'ruin_pct': ruined / N_SIMS * 100,
    }

    r = mc_results[name]
    print(f"  {name:>30s} | ${r['median']:11.2f} | ${r['p5']:11.2f} | ${r['p25']:11.2f} | ${r['p75']:11.2f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%")


# ============================================================
# VERDICT
# ============================================================
print()
print("=" * 120)
print("VERDICT")
print("=" * 120)
print()

base = mc_results.get('EXP-002: 5% no costs', {})
real = mc_results.get('5% + maint + fund 0.01%', {})

if base and real:
    print(f"  EXP-002 (no costs):     Median ${base['median']:>10.2f} | P5 ${base['p5']:>10.2f} | Ruin {base['ruin_pct']:.1f}%")
    print(f"  With Binance reality:   Median ${real['median']:>10.2f} | P5 ${real['p5']:>10.2f} | Ruin {real['ruin_pct']:.1f}%")
    print()

    median_diff = (real['median'] - base['median']) / base['median'] * 100
    p5_diff = (real['p5'] - base['p5']) / base['p5'] * 100

    print(f"  Median change: {median_diff:+.1f}%")
    print(f"  P5 change:     {p5_diff:+.1f}%")
    print(f"  Ruin change:   {real['ruin_pct'] - base['ruin_pct']:+.1f}%")

# Check all configs
print()
print("  RANKING by P5:")
sorted_results = sorted(mc_results.items(), key=lambda x: x[1]['p5'], reverse=True)
for i, (name, r) in enumerate(sorted_results):
    print(f"    #{i+1}: {name:>30s} - P5 ${r['p5']:>10.2f}, Median ${r['median']:>10.2f}, Ruin {r['ruin_pct']:.1f}%")


# ============================================================
# PART 6: MINIMUM POSITION SIZE - Binance constraints
# ============================================================
print()
print("=" * 120)
print("PART 6: MINIMUM POSITION SIZE - How Binance constraints change with BTC price")
print("=" * 120)
print()

# Real Binance specs (from API query):
# LOT_SIZE: minQty=0.001, stepSize=0.001
# MIN_NOTIONAL: 100 USDT
BINANCE_MIN_QTY = 0.001   # BTC
BINANCE_STEP_SIZE = 0.001  # BTC
BINANCE_MIN_NOTIONAL = 100  # USDT

print(f"  Binance BTCUSDT Perpetual constraints:")
print(f"    Min quantity: {BINANCE_MIN_QTY} BTC (step: {BINANCE_STEP_SIZE})")
print(f"    Min notional: ${BINANCE_MIN_NOTIONAL}")
print()

# Show how minimum position changes with BTC price
import math

print(f"  {'BTC Price':>12s} | {'0.001 BTC':>10s} | {'Min Qty':>10s} | {'Min Position':>13s} | {'Maint Margin':>13s} | {'5% of $10':>10s} | {'5% Works?':>10s}")
print(f"  {'-'*100}")

btc_prices = [40000, 50000, 60000, 70000, 80000, 90000, 95000, 100000, 110000, 120000]

for price in btc_prices:
    qty_001 = BINANCE_MIN_QTY * price
    # Min qty to meet notional: ceil to step size
    min_qty_for_notional = math.ceil(BINANCE_MIN_NOTIONAL / price / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE
    min_qty = max(BINANCE_MIN_QTY, min_qty_for_notional)
    min_pos = min_qty * price
    maint = min_pos * MAINT_MARGIN_RATE
    margin_5pct = 10 * 0.05  # $0.50
    works = "YES" if margin_5pct > maint else "NO"

    print(f"  ${price:>10,d} | ${qty_001:>8.0f} | {min_qty:.3f} BTC | ${min_pos:>11.0f} | ${maint:>11.2f} | ${margin_5pct:>8.2f} | {works:>10s}")

print()
print("  KEY INSIGHT: Minimum position size CHANGES with BTC price!")
print(f"  At BTC=$95K: min = 0.002 BTC = $190 (our $170 would be REJECTED!)")
print(f"  At BTC=$100K+: min = 0.001 BTC = $100+")
print()

# What position sizes are valid across the full price range?
print("  Valid fixed position sizes across BTC price range:")
print()
for target_pos in [100, 120, 140, 160, 170, 190, 200]:
    valid_prices = []
    invalid_prices = []
    for price in btc_prices:
        min_qty_for_notional = math.ceil(BINANCE_MIN_NOTIONAL / price / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE
        min_qty = max(BINANCE_MIN_QTY, min_qty_for_notional)
        min_pos = min_qty * price
        if target_pos >= min_pos:
            valid_prices.append(price)
        else:
            invalid_prices.append(price)

    maint = target_pos * MAINT_MARGIN_RATE
    margin_5pct = 10 * 0.05
    can_5pct = margin_5pct > maint

    status = "ALL" if len(invalid_prices) == 0 else f"Fails at: {', '.join(f'${p//1000}K' for p in invalid_prices)}"
    print(f"    ${target_pos:>3d}: maint=${maint:.2f}, 5% works={can_5pct} | {status}")

print()

# ============================================================
# PART 7: MC with different position sizes (Binance-corrected)
# ============================================================
print("=" * 120)
print("PART 7: MC - Position size comparison with Binance maintenance margin")
print("=" * 120)
print()

# Test: what if we use smaller position to make 5% margin work?
# Position $100: maint = $0.40, 5% of $10 = $0.50 > $0.40 -> WORKS!
# Position $120: maint = $0.48, 5% of $10 = $0.50 > $0.48 -> barely works
# Position $170: maint = $0.68, 5% of $10 = $0.50 < $0.68 -> FAILS

print("  Logic: smaller position -> lower maintenance margin -> 5% might work")
print("  But: smaller position -> less profit per trade")
print()

np.random.seed(42)

pos_configs = {}
for pos_size in [100, 120, 140, 170]:
    for pct in [0.05, 0.10, 0.15, 0.20]:
        name = f"${pos_size} pos, {int(pct*100)}% margin"
        pos_configs[name] = (pos_size, pct)

print(f"  {'Config':>30s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'P75':>12s} | {'Ruin%':>6s} | {'Skip':>5s} | {'Liq':>4s}")
print(f"  {'-'*110}")

pos_mc_results = {}
for name, (pos_size, pct) in pos_configs.items():
    finals = []
    max_dds = []
    ruined = 0
    total_skip = 0
    total_liq = 0

    for _ in range(N_SIMS):
        shuffled = list(returns)
        np.random.shuffle(shuffled)
        eq, sk, liq, fp = simulate_binance_pct_hybrid(shuffled, pos_size, pct, 25, 20,
                                                        funding_rate=0.01,
                                                        funding_cross_pct=pct_crossing)
        finals.append(eq[-1])
        dd = calc_max_dd(eq)
        max_dds.append(dd)
        total_skip += sk
        total_liq += liq
        if eq[-1] < 1.0:
            ruined += 1

    avg_skip = total_skip / N_SIMS
    avg_liq = total_liq / N_SIMS

    pos_mc_results[name] = {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'ruin_pct': ruined / N_SIMS * 100,
        'avg_skip': avg_skip,
        'avg_liq': avg_liq,
    }

    r = pos_mc_results[name]
    print(f"  {name:>30s} | ${r['median']:11.2f} | ${r['p5']:11.2f} | ${r['p25']:11.2f} | ${r['p75']:11.2f} | {r['ruin_pct']:5.1f}% | {r['avg_skip']:5.1f} | {r['avg_liq']:4.1f}")

print()
print("  COMPARISON: Position size vs safety")
print()
for pos_size in [100, 120, 140, 170]:
    name_5 = f"${pos_size} pos, 5% margin"
    name_15 = f"${pos_size} pos, 15% margin"
    r5 = pos_mc_results.get(name_5, {})
    r15 = pos_mc_results.get(name_15, {})
    if r5 and r15:
        maint = pos_size * MAINT_MARGIN_RATE
        print(f"  ${pos_size} position (maint=${maint:.2f}):")
        print(f"     5% margin: Median ${r5['median']:>10.2f}, P5 ${r5['p5']:>10.2f}, Ruin {r5['ruin_pct']:.1f}%, Skip {r5['avg_skip']:.0f}")
        print(f"    15% margin: Median ${r15['median']:>10.2f}, P5 ${r15['p5']:>10.2f}, Ruin {r15['ruin_pct']:.1f}%, Skip {r15['avg_skip']:.0f}")
        print()
