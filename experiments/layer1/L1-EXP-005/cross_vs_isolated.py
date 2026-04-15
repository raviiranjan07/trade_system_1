"""L1-EXP-005: Cross vs Isolated - Head-to-Head with Binance Reality

QUESTION: Is cross margin actually better than isolated in Phase 1?

Uses REAL position sizes from actual BTC entry prices (EXP-003 formula):
  qty = max(0.001, ceil(100 / btc_price / 0.001) * 0.001)
  position = qty * btc_price  (varies $100-$190 per trade)

Each trade has a DIFFERENT position size and maintenance margin.
"""
import sys
sys.path.insert(0, "src")

import math
import numpy as np
from engine.backtest import run_backtest
from engine.config.loader import load_config

STARTING_CAPITAL = 10.0
MAX_LEVERAGE_SETTING = 125
N_SIMS = 1000
MAINT_MARGIN_RATE = 0.004
BINANCE_MIN_QTY = 0.001
BINANCE_STEP_SIZE = 0.001
BINANCE_MIN_NOTIONAL = 100

config = load_config()
trades = run_backtest(config)

# Extract per-trade data with REAL position sizes
trade_data = []
for t in trades:
    btc_price = t.entry_price
    qty = max(BINANCE_MIN_QTY, math.ceil(BINANCE_MIN_NOTIONAL / btc_price / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE)
    position = qty * btc_price
    maint = position * MAINT_MARGIN_RATE
    trade_data.append({
        'bps': t.net_profit_bps,
        'btc_price': btc_price,
        'qty': qty,
        'position': position,
        'maint_margin': maint,
    })

returns = [td['bps'] for td in trade_data]
positions = [td['position'] for td in trade_data]
maint_margins = [td['maint_margin'] for td in trade_data]
losses = [r for r in returns if r <= 0]

print("=" * 100)
print("L1-EXP-005: CROSS vs ISOLATED - Dynamic Position Sizing")
print("=" * 100)
print(f"V1.3.2: {len(returns)} trades, {sum(1 for r in returns if r > 0)/len(returns)*100:.1f}% win")
print(f"Avg loss: {np.mean(losses):.1f} bps | Worst: {min(returns):.1f} bps")
print()


# ============================================================
# PART 1: Position size distribution from actual trades
# ============================================================
print("=" * 100)
print("PART 1: ACTUAL POSITION SIZES (from V1.3.2 entry prices)")
print("=" * 100)
print()

print(f"  BTC price range: ${min(td['btc_price'] for td in trade_data):,.0f} - ${max(td['btc_price'] for td in trade_data):,.0f}")
print(f"  Position range:  ${min(positions):,.0f} - ${max(positions):,.0f}")
print(f"  Maint margin:    ${min(maint_margins):.2f} - ${max(maint_margins):.2f}")
print()

# Show distribution
from collections import Counter
pos_rounded = [round(p / 10) * 10 for p in positions]
pos_counts = Counter(pos_rounded)
print(f"  {'Position $':>12s} | {'Count':>6s} | {'Maint $':>8s} | {'Iso 15% usable':>15s} | {'Liq threshold':>14s}")
print(f"  {'-'*70}")
for pos in sorted(pos_counts.keys()):
    maint = pos * MAINT_MARGIN_RATE
    usable = 1.50 - maint  # at $10 equity
    liq_bps = usable / pos * 10000 if usable > 0 else 0
    status = f"{liq_bps:.1f} bps" if usable > 0 else "CANT OPEN"
    print(f"  ${pos:>10.0f} | {pos_counts[pos]:>6d} | ${maint:>6.2f} | ${usable:>13.2f} | {status:>14s}")

print()

# Per-trade position sizes for first 10 trades
print("  First 10 trades (actual values):")
print(f"  {'#':>4s} | {'BTC Price':>10s} | {'Qty':>8s} | {'Position':>10s} | {'Maint':>8s} | {'Bps':>8s} | {'PnL $':>8s}")
print(f"  {'-'*70}")
for i, td in enumerate(trade_data[:10]):
    pnl = td['position'] * td['bps'] / 10000
    print(f"  {i+1:>4d} | ${td['btc_price']:>9,.0f} | {td['qty']:.3f} | ${td['position']:>8.0f} | ${td['maint_margin']:>.2f} | {td['bps']:>+7.1f} | ${pnl:>+7.2f}")


# ============================================================
# PART 2: Loss zone analysis with REAL positions
# ============================================================
print()
print("=" * 100)
print("PART 2: LOSS ZONE ANALYSIS (per-trade, real positions)")
print("=" * 100)
print()

# For each losing trade, calculate which zone it falls in at $10 equity
zone_same = 0     # loss < iso liquidation threshold
zone_cross = 0    # iso liquidated but cross loss < margin (cross wins)
zone_iso = 0      # iso liquidated but cross loss > margin (iso wins)
cross_savings = 0.0
iso_savings = 0.0

print(f"  Losing trades where methods DIFFER (at $10 equity, 15% iso margin):")
print(f"  {'#':>4s} | {'Bps':>8s} | {'Position':>10s} | {'Cross loss':>11s} | {'Iso loss':>11s} | {'Winner':>8s} | {'Savings':>8s}")
print(f"  {'-'*75}")

for i, td in enumerate(trade_data):
    if td['bps'] >= 0:
        continue

    pos = td['position']
    maint = td['maint_margin']
    iso_margin = STARTING_CAPITAL * 0.15  # $1.50
    usable = iso_margin - maint

    if usable <= 0:
        # Can't even open in isolated mode
        continue

    loss_bps = abs(td['bps'])
    actual_loss = pos * loss_bps / 10000

    if actual_loss <= usable:
        # Both methods: same loss
        zone_same += 1
    elif actual_loss <= iso_margin:
        # Iso: liquidated (lose full margin $1.50), but actual loss < $1.50
        # Cross: just loses actual_loss
        zone_cross += 1
        saving = iso_margin - actual_loss
        cross_savings += saving
        print(f"  {i+1:>4d} | {td['bps']:>+7.1f} | ${pos:>8.0f} | ${actual_loss:>9.2f} | ${iso_margin:>9.2f} | {'CROSS':>8s} | ${saving:>6.2f}")
    else:
        # Iso: liquidated (lose $1.50), cross: loses > $1.50
        zone_iso += 1
        saving = actual_loss - iso_margin
        iso_savings += saving
        print(f"  {i+1:>4d} | {td['bps']:>+7.1f} | ${pos:>8.0f} | ${actual_loss:>9.2f} | ${iso_margin:>9.2f} | {'ISO':>8s} | ${saving:>6.2f}")

print()
print(f"  SUMMARY (at $10 equity):")
print(f"    Same result:      {zone_same} trades")
print(f"    Cross wins:       {zone_cross} trades (saves ${cross_savings:.2f} total)")
print(f"    Isolated wins:    {zone_iso} trades (saves ${iso_savings:.2f} total)")
net = iso_savings - cross_savings
print(f"    Net advantage:    {'ISOLATED' if net > 0 else 'CROSS'} saves ${abs(net):.2f}")


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
# PART 3: SIMULATION with per-trade position sizes
# ============================================================
print()
print("=" * 100)
print("PART 3: SIMULATION - Cross vs Isolated (Original Order, Real Positions)")
print("=" * 100)
print()


def simulate_cross_dynamic(trade_list, threshold, scale_lev, capital=STARTING_CAPITAL):
    """Phase 1: Cross margin, dynamic position per trade. Phase 2: Cross scaling."""
    equity = [capital]
    skipped = 0
    liquidated = 0

    for td in trade_list:
        eq = equity[-1]

        if eq < threshold:
            pos = td['position']
            maint = td['maint_margin']
            margin_req = pos / MAX_LEVERAGE_SETTING

            if eq < margin_req:
                equity.append(eq)
                skipped += 1
                continue

            pnl = pos * (td['bps'] / 10000)
            max_loss = eq - maint
            if pnl < -max_loss:
                equity.append(0.01)
                liquidated += 1
            else:
                equity.append(max(eq + pnl, 0.01))
        else:
            pos = eq * scale_lev
            pnl = pos * (td['bps'] / 10000)
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


def simulate_iso_dynamic(trade_list, pct, threshold, scale_lev, capital=STARTING_CAPITAL):
    """Phase 1: Isolated margin (pct% of equity), dynamic position. Phase 2: Cross scaling."""
    equity = [capital]
    skipped = 0
    liquidated = 0

    for td in trade_list:
        eq = equity[-1]

        if eq < threshold:
            pos = td['position']
            maint = td['maint_margin']
            margin = eq * pct

            if margin <= maint:
                equity.append(eq)
                skipped += 1
                continue

            pnl = pos * (td['bps'] / 10000)
            max_loss = margin - maint
            if pnl < -max_loss:
                equity.append(eq - margin)
                liquidated += 1
            else:
                equity.append(max(eq + pnl, 0.01))
        else:
            pos = eq * scale_lev
            pnl = pos * (td['bps'] / 10000)
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


# Original order
configs_orig = [
    ("Cross", lambda td: simulate_cross_dynamic(td, 25, 20)),
    ("Iso 15%", lambda td: simulate_iso_dynamic(td, 0.15, 25, 20)),
    ("Iso 20%", lambda td: simulate_iso_dynamic(td, 0.20, 25, 20)),
    ("Iso 25%", lambda td: simulate_iso_dynamic(td, 0.25, 25, 20)),
]

print(f"  {'Config':>20s} | {'Final':>12s} | {'MaxDD':>8s} | {'Skip':>5s} | {'Liq':>4s} | {'Phase1 exit':>12s}")
print(f"  {'-'*75}")

for name, sim_fn in configs_orig:
    eq, sk, liq = sim_fn(trade_data)
    dd = calc_max_dd(eq)
    phase1_exit = "N/A"
    for i, e in enumerate(eq):
        if e >= 25:
            phase1_exit = f"Trade {i}"
            break
    print(f"  {name:>20s} | ${eq[-1]:11.2f} | {dd*100:6.1f}% | {sk:>5d} | {liq:>4d} | {phase1_exit:>12s}")


# ============================================================
# PART 4: MONTE CARLO with per-trade positions
# ============================================================
print()
print("=" * 100)
print("PART 4: MONTE CARLO - 1000 shuffled paths (real positions)")
print("=" * 100)
print()

np.random.seed(42)

mc_configs = {
    'Cross -> 20x @$25': ('cross', None),
    'Iso 10% -> 20x @$25': ('iso', 0.10),
    'Iso 15% -> 20x @$25': ('iso', 0.15),
    'Iso 20% -> 20x @$25': ('iso', 0.20),
    'Iso 25% -> 20x @$25': ('iso', 0.25),
    'Iso 30% -> 20x @$25': ('iso', 0.30),
}

print(f"  {'Config':>25s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'P75':>12s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'Ruin%':>6s} | {'AvgLiq':>7s}")
print(f"  {'-'*120}")

mc_results = {}
for name, (mode, pct) in mc_configs.items():
    finals = []
    max_dds = []
    ruined = 0
    total_liq = 0

    for _ in range(N_SIMS):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)

        if mode == 'cross':
            eq, sk, liq = simulate_cross_dynamic(shuffled, 25, 20)
        else:
            eq, sk, liq = simulate_iso_dynamic(shuffled, pct, 25, 20)

        finals.append(eq[-1])
        dd = calc_max_dd(eq)
        max_dds.append(dd)
        total_liq += liq
        if eq[-1] < 1.0:
            ruined += 1

    mc_results[name] = {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'ruin_pct': ruined / N_SIMS * 100,
        'avg_liq': total_liq / N_SIMS,
    }

    r = mc_results[name]
    print(f"  {name:>25s} | ${r['median']:11.2f} | ${r['p5']:11.2f} | ${r['p25']:11.2f} | ${r['p75']:11.2f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}% | {r['avg_liq']:6.1f}")


# ============================================================
# PART 5: PATH-BY-PATH comparison
# ============================================================
print()
print("=" * 100)
print("PART 5: PATH-BY-PATH - Cross vs Iso 15% on same 1000 shuffles")
print("=" * 100)
print()

np.random.seed(42)

cross_finals = []
iso15_finals = []

for _ in range(N_SIMS):
    shuffled = list(trade_data)
    np.random.shuffle(shuffled)

    eq_c, _, _ = simulate_cross_dynamic(shuffled, 25, 20)
    eq_i, _, _ = simulate_iso_dynamic(shuffled, 0.15, 25, 20)

    cross_finals.append(eq_c[-1])
    iso15_finals.append(eq_i[-1])

cross_wins = sum(1 for c, i in zip(cross_finals, iso15_finals) if c > i)
iso_wins = sum(1 for c, i in zip(cross_finals, iso15_finals) if i > c)
ties = sum(1 for c, i in zip(cross_finals, iso15_finals) if abs(c - i) < 0.01)

print(f"  Cross wins:    {cross_wins}/1000 ({cross_wins/10:.1f}%)")
print(f"  Iso 15% wins:  {iso_wins}/1000 ({iso_wins/10:.1f}%)")
print(f"  Ties:          {ties}/1000 ({ties/10:.1f}%)")
print()

diffs = [c - i for c, i in zip(cross_finals, iso15_finals)]
print(f"  Avg difference:    ${np.mean(diffs):+.2f} ({'CROSS' if np.mean(diffs) > 0 else 'ISO'} better)")
print(f"  Median difference: ${np.median(diffs):+.2f}")
print()

print(f"  PERCENTILE COMPARISON:")
print(f"  {'Percentile':>12s} | {'Cross':>12s} | {'Iso 15%':>12s} | {'Diff':>12s} | {'Winner':>8s}")
print(f"  {'-'*65}")
for pctl in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    c = np.percentile(cross_finals, pctl)
    i = np.percentile(iso15_finals, pctl)
    diff = c - i
    winner = "CROSS" if diff > 0 else "ISO"
    print(f"  {'P'+str(pctl):>12s} | ${c:11.2f} | ${i:11.2f} | ${diff:+11.2f} | {winner:>8s}")


# ============================================================
# VERDICT
# ============================================================
print()
print("=" * 100)
print("VERDICT: CROSS vs ISOLATED (with real Binance position sizes)")
print("=" * 100)
print()

rc = mc_results['Cross -> 20x @$25']
ri = mc_results['Iso 15% -> 20x @$25']

print(f"  {'Metric':>20s} | {'Cross':>15s} | {'Iso 15%':>15s} | {'Winner':>8s}")
print(f"  {'-'*65}")
print(f"  {'MC Median':>20s} | ${rc['median']:14.2f} | ${ri['median']:14.2f} | {'CROSS' if rc['median'] > ri['median'] else 'ISO':>8s}")
print(f"  {'MC P5':>20s} | ${rc['p5']:14.2f} | ${ri['p5']:14.2f} | {'CROSS' if rc['p5'] > ri['p5'] else 'ISO':>8s}")
print(f"  {'MC P25':>20s} | ${rc['p25']:14.2f} | ${ri['p25']:14.2f} | {'CROSS' if rc['p25'] > ri['p25'] else 'ISO':>8s}")
print(f"  {'Avg DD':>20s} | {rc['avg_dd']*100:13.1f}% | {ri['avg_dd']*100:13.1f}% | {'CROSS' if rc['avg_dd'] < ri['avg_dd'] else 'ISO':>8s}")
print(f"  {'P95 DD':>20s} | {rc['p95_dd']*100:13.1f}% | {ri['p95_dd']*100:13.1f}% | {'CROSS' if rc['p95_dd'] < ri['p95_dd'] else 'ISO':>8s}")
print(f"  {'Ruin %':>20s} | {rc['ruin_pct']:13.1f}% | {ri['ruin_pct']:13.1f}% | {'CROSS' if rc['ruin_pct'] < ri['ruin_pct'] else 'ISO':>8s}")
print(f"  {'Avg Liquidations':>20s} | {rc['avg_liq']:14.1f} | {ri['avg_liq']:14.1f} | {'CROSS' if rc['avg_liq'] < ri['avg_liq'] else 'ISO':>8s}")
print()
print(f"  Path-by-path: Cross wins {cross_wins}/1000, Iso wins {iso_wins}/1000, Ties {ties}/1000")
