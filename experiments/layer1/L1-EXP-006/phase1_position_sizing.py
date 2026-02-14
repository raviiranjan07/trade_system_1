"""L1-EXP-006b: Phase 1 Position Sizing Test

QUESTION: In Phase 1 (wallet < $15), should we vary position size?

CONFIGS TESTED:
  A: Always 1x minimum ($130 = 0.002 BTC) [CURRENT BASELINE]
  B: 2x minimum ($260 = 0.004 BTC) on strong signals (V12_SHORT, BEAR_LONG), 1x on others
  C: Always 2x minimum ($260 = 0.004 BTC)
  D: 3x minimum ($390 = 0.006 BTC) - aggressive
  E: Signal-quality based: V12_SHORT=2x, BEAR_LONG=2x, BULL_SHORT=1.5x, V12_LONG=1x

Phase 2 is identical across all configs: equity * 20x (or 25x)
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import math
import numpy as np
from v12.backtest import run_backtest
from v12.config.loader import load_config

STARTING_CAPITAL = 10.0
MAX_LEVERAGE_SETTING = 125
N_SIMS = 1000
MAINT_MARGIN_RATE = 0.004
BINANCE_MIN_QTY = 0.001
BINANCE_STEP_SIZE = 0.001
BINANCE_MIN_NOTIONAL = 100
PHASE1_THRESHOLD = 15

# Load V1.3.2 trades
config = load_config()
trades = run_backtest(config)

trade_data = []
for t in trades:
    btc_price = t.entry_price
    qty = max(BINANCE_MIN_QTY, math.ceil(BINANCE_MIN_NOTIONAL / btc_price / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE)
    position = qty * btc_price
    maint = position * MAINT_MARGIN_RATE
    trade_data.append({
        'bps': t.net_profit_bps,
        'btc_price': btc_price,
        'qty_1x': qty,
        'position_1x': position,
        'maint_1x': maint,
        'direction': t.direction,
        'signal_type': t.signal_type if hasattr(t, 'signal_type') else 'UNKNOWN',
    })

returns = [td['bps'] for td in trade_data]

# Count signal types
from collections import Counter
sig_counts = Counter(td['signal_type'] for td in trade_data)

print("=" * 100)
print("L1-EXP-006b: PHASE 1 POSITION SIZING TEST")
print("=" * 100)
print(f"  V1.3.2: {len(trades)} trades, {sum(1 for r in returns if r > 0)/len(returns)*100:.1f}% win")
print(f"  Signal types: {dict(sig_counts)}")
print(f"  1x position range: ${min(td['position_1x'] for td in trade_data):,.0f} - ${max(td['position_1x'] for td in trade_data):,.0f}")
print(f"  2x position range: ${min(td['position_1x'] for td in trade_data)*2:,.0f} - ${max(td['position_1x'] for td in trade_data)*2:,.0f}")
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


# Position size multiplier functions
def multiplier_1x(signal_type):
    """Always 1x (baseline)."""
    return 1.0

def multiplier_2x_strong(signal_type):
    """2x on strong signals (V12_SHORT, BEAR_LONG), 1x on others."""
    if signal_type in ('V12_SHORT', 'BEAR_LONG'):
        return 2.0
    return 1.0

def multiplier_2x_all(signal_type):
    """Always 2x."""
    return 2.0

def multiplier_3x_all(signal_type):
    """Always 3x."""
    return 3.0

def multiplier_signal_quality(signal_type):
    """Signal-quality based sizing."""
    if signal_type == 'V12_SHORT':
        return 2.0
    elif signal_type == 'BEAR_LONG':
        return 2.0
    elif signal_type == 'BULL_SHORT':
        return 1.5
    else:  # V12_LONG (weakest)
        return 1.0


def simulate_phase1_sizing(trade_list, mult_fn, scale_lev, capital=STARTING_CAPITAL):
    """Simulate with variable Phase 1 position sizing.

    Phase 1: position = mult_fn(signal_type) * base_position (cross margin)
    Phase 2: position = equity * scale_lev (fixed leverage)
    """
    equity = [capital]
    skipped = 0
    liquidated = 0

    for td in trade_list:
        eq = equity[-1]

        if eq < PHASE1_THRESHOLD:
            # Phase 1: variable position based on signal
            mult = mult_fn(td['signal_type'])
            qty = td['qty_1x'] * mult
            pos = qty * td['btc_price']
            maint = pos * MAINT_MARGIN_RATE

            margin_req = pos / MAX_LEVERAGE_SETTING
            if eq < margin_req:
                equity.append(eq)
                skipped += 1
                continue

            pnl = pos * (td['bps'] / 10000)

            # Cross liq: when equity drops to maintenance level
            max_loss = eq - maint
            if pnl < -max_loss:
                equity.append(0.01)
                liquidated += 1
            else:
                equity.append(max(eq + pnl, 0.01))
        else:
            # Phase 2: fixed scaling (identical for all configs)
            pos = eq * scale_lev
            pnl = pos * (td['bps'] / 10000)
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


def run_mc(mult_fn, scale_lev, label):
    """Run MC simulation for a given Phase 1 sizing config."""
    np.random.seed(42)
    n = len(trade_data)

    finals = []
    max_dds = []
    ruin_count = 0
    liq_count = 0
    skip_count = 0

    for _ in range(N_SIMS):
        indices = np.random.choice(n, n, replace=True)
        sample = [trade_data[i] for i in indices]

        equity, skipped, liquidated = simulate_phase1_sizing(sample, mult_fn, scale_lev)
        final_eq = equity[-1]
        finals.append(final_eq)
        max_dds.append(calc_max_dd(equity))
        if final_eq < 1.0:
            ruin_count += 1
        liq_count += liquidated
        skip_count += skipped

    finals = np.array(finals)
    max_dds = np.array(max_dds)

    return {
        'label': label,
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'ruin_pct': ruin_count / N_SIMS * 100,
        'avg_liqs': liq_count / N_SIMS,
        'avg_skips': skip_count / N_SIMS,
    }


# ============================================================
# PART 1: PHASE 1 POSITION SIZING COMPARISON
# ============================================================
configs = [
    (multiplier_1x, "A: Always 1x (baseline)"),
    (multiplier_2x_strong, "B: 2x on strong signals"),
    (multiplier_2x_all, "C: Always 2x"),
    (multiplier_3x_all, "D: Always 3x"),
    (multiplier_signal_quality, "E: Signal-quality based"),
]

for phase2_lev in [20, 25]:
    print("=" * 100)
    print(f"  PHASE 1 SIZING WITH PHASE 2 = {phase2_lev}x")
    print("=" * 100)
    print()
    print(f"  {'Config':>30s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'Ruin':>6s} | {'Liqs':>5s} | {'Skips':>5s}")
    print(f"  {'-'*110}")

    results = []
    for mult_fn, label in configs:
        r = run_mc(mult_fn, phase2_lev, label)
        results.append(r)
        print(f"  {label:>30s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['ruin_pct']:4.1f}% | {r['avg_liqs']:4.1f} | {r['avg_skips']:4.1f}")

    print()

    # Compare each config to baseline
    baseline = results[0]
    print(f"  COMPARISON TO BASELINE ({baseline['label']}):")
    for r in results[1:]:
        median_diff = (r['median'] / baseline['median'] - 1) * 100
        p5_diff = (r['p5'] / baseline['p5'] - 1) * 100
        ruin_diff = r['ruin_pct'] - baseline['ruin_pct']
        print(f"    {r['label']:>30s}: Median {median_diff:+.1f}% | P5 {p5_diff:+.1f}% | Ruin {ruin_diff:+.1f}pp")
    print()


# ============================================================
# PART 2: WHY DOES LARGER POSITION HELP/HURT?
# ============================================================
print("=" * 100)
print("  PART 2: PHASE 1 TRADE ANALYSIS")
print("=" * 100)
print()

# Analyze Phase 1 trades specifically
# Phase 1 trades are early in the sequence - how many trades to reach $15?
# Simulate one path with actual trade order to see Phase 1 duration
equity = STARTING_CAPITAL
phase1_trades_1x = 0
phase1_trades_2x = 0
for td in trade_data:
    if equity >= PHASE1_THRESHOLD:
        break
    # 1x
    pos = td['position_1x']
    pnl = pos * (td['bps'] / 10000)
    equity_1x = equity + pnl
    phase1_trades_1x += 1

# Reset for 2x
equity_1x_path = STARTING_CAPITAL
equity_2x_path = STARTING_CAPITAL
for i, td in enumerate(trade_data):
    if equity_1x_path >= PHASE1_THRESHOLD and equity_2x_path >= PHASE1_THRESHOLD:
        break

    pos_1x = td['position_1x']
    pos_2x = td['position_1x'] * 2
    pnl_1x = pos_1x * (td['bps'] / 10000)
    pnl_2x = pos_2x * (td['bps'] / 10000)

    if equity_1x_path < PHASE1_THRESHOLD:
        equity_1x_path = max(equity_1x_path + pnl_1x, 0.01)
    if equity_2x_path < PHASE1_THRESHOLD:
        equity_2x_path = max(equity_2x_path + pnl_2x, 0.01)

    if i < 30:  # Show first 30 trades
        in_p1_1x = "P1" if equity_1x_path < PHASE1_THRESHOLD else "P2"
        in_p1_2x = "P1" if equity_2x_path < PHASE1_THRESHOLD else "P2"
        print(f"  Trade {i+1:3d} | {td['signal_type']:>12s} | {td['bps']:+7.1f} bps | 1x: ${equity_1x_path:8.2f} ({in_p1_1x}) | 2x: ${equity_2x_path:8.2f} ({in_p1_2x})")


# ============================================================
# PART 3: LIQUIDATION RISK AT DIFFERENT POSITION SIZES
# ============================================================
print()
print("=" * 100)
print("  PART 3: LIQUIDATION RISK IN PHASE 1")
print("=" * 100)
print()

# For each position multiplier, what's the liquidation buffer?
for mult_label, mult in [("1x", 1), ("2x", 2), ("3x", 3)]:
    avg_pos = np.mean([td['position_1x'] for td in trade_data]) * mult
    eff_lev = avg_pos / STARTING_CAPITAL
    if eff_lev > 0:
        liq_buffer_pct = (1 / eff_lev - MAINT_MARGIN_RATE) * 100
    else:
        liq_buffer_pct = 999
    liq_buffer_bps = liq_buffer_pct * 100

    worst_trade = min(returns)
    survives_worst = abs(worst_trade) < liq_buffer_bps
    survives_2x = abs(worst_trade * 2) < liq_buffer_bps

    print(f"  {mult_label}: Avg position ${avg_pos:,.0f} | Eff leverage {eff_lev:.1f}x | Liq buffer {liq_buffer_bps:.0f} bps")
    print(f"       Survives worst ({worst_trade:.0f} bps): {'YES' if survives_worst else 'NO'}")
    print(f"       Survives 2x worst ({worst_trade*2:.0f} bps): {'YES' if survives_2x else 'NO'}")
    print()

print()
print("=" * 100)
print("  CONCLUSION")
print("=" * 100)
