"""L1-EXP-002 v2: Complete Position Sizing Grid

ONE QUESTION: What is the best way to grow $10 to max on Binance with V1.3.2 trades?

VARIABLES:
  1. Phase 1 margin mode: Cross, Iso 15%, Iso 20%, Iso 25%, Iso 30%
  2. Phase 2 leverage: 20x, 25x, 30x
  3. Transition threshold: $15, $20, $25, $30

FIXED:
  - Starting capital: $10
  - Position size: DYNAMIC per BTC entry price (Binance formula)
  - Maintenance margin: 0.4% of position
  - Phase 1: fixed position (min Binance qty). Phase 2: scaling (equity * leverage)

GRID: 5 modes x 3 leverages x 4 thresholds = 60 configs, 1000 MC paths each
"""
import sys
sys.path.insert(0, "src")

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

config = load_config()
trades = run_backtest(config)

# Build per-trade data with REAL position sizes
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
losses = [r for r in returns if r <= 0]

print("=" * 110)
print("L1-EXP-002 v2: COMPLETE POSITION SIZING GRID")
print("=" * 110)
print(f"V1.3.2: {len(returns)} trades, {sum(1 for r in returns if r > 0)/len(returns)*100:.1f}% win")
print(f"Mean: {np.mean(returns):+.1f} bps | Worst: {min(returns):.1f} bps | Best: {max(returns):+.1f} bps")
print(f"Position range: ${min(positions):,.0f} - ${max(positions):,.0f} (dynamic per BTC price)")
print(f"Starting capital: ${STARTING_CAPITAL} | MC paths: {N_SIMS}")
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
# SIMULATION FUNCTIONS
# ============================================================
def simulate_cross(trade_list, threshold, scale_lev, capital=STARTING_CAPITAL):
    """Phase 1: Cross margin (full equity backs position).
       Phase 2: Cross scaling (position = equity * leverage)."""
    equity = [capital]
    skipped = 0
    liquidated = 0

    for td in trade_list:
        eq = equity[-1]

        if eq < threshold:
            # Phase 1: cross margin, fixed position per trade
            pos = td['position']
            maint = td['maint_margin']
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
            # Phase 2: scaling
            pos = eq * scale_lev
            pnl = pos * (td['bps'] / 10000)
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


def simulate_isolated(trade_list, pct, threshold, scale_lev, capital=STARTING_CAPITAL):
    """Phase 1: Isolated margin (pct% of equity per trade).
       Phase 2: Cross scaling (position = equity * leverage)."""
    equity = [capital]
    skipped = 0
    liquidated = 0

    for td in trade_list:
        eq = equity[-1]

        if eq < threshold:
            # Phase 1: isolated margin
            pos = td['position']
            maint = td['maint_margin']
            margin = eq * pct

            # Can't open if margin doesn't cover maintenance
            if margin <= maint:
                equity.append(eq)
                skipped += 1
                continue

            pnl = pos * (td['bps'] / 10000)

            # Isolated liq: when loss exceeds usable margin
            max_loss = margin - maint
            if pnl < -max_loss:
                equity.append(eq - margin)  # lose allocated margin only
                liquidated += 1
            else:
                equity.append(max(eq + pnl, 0.01))
        else:
            # Phase 2: scaling (same as cross)
            pos = eq * scale_lev
            pnl = pos * (td['bps'] / 10000)
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


# ============================================================
# PART 1: ORIGINAL ORDER - All configs
# ============================================================
print("=" * 110)
print("PART 1: ORIGINAL TRADE ORDER - All configs")
print("=" * 110)
print()

phase1_modes = {
    'Cross': ('cross', None),
    'Iso 15%': ('iso', 0.15),
    'Iso 20%': ('iso', 0.20),
    'Iso 25%': ('iso', 0.25),
    'Iso 30%': ('iso', 0.30),
}
phase2_levs = [20, 25, 30]
thresholds = [15, 20, 25, 30]

print(f"  {'Mode':>10s} | {'P2 Lev':>7s} | {'Thresh':>7s} | {'Final':>12s} | {'MaxDD':>8s} | {'Skip':>5s} | {'Liq':>4s} | {'P1 trades':>10s}")
print(f"  {'-'*85}")

for mode_name, (mode, pct) in phase1_modes.items():
    for lev in phase2_levs:
        for thresh in thresholds:
            if mode == 'cross':
                eq, sk, liq = simulate_cross(trade_data, thresh, lev)
            else:
                eq, sk, liq = simulate_isolated(trade_data, pct, thresh, lev)

            dd = calc_max_dd(eq)
            # Count Phase 1 trades
            p1_count = 0
            sim_eq = STARTING_CAPITAL
            for td in trade_data:
                if sim_eq >= thresh:
                    break
                p1_count += 1
                sim_eq += td['position'] * (td['bps'] / 10000)

            print(f"  {mode_name:>10s} | {lev:>5d}x | ${thresh:>5d} | ${eq[-1]:11.2f} | {dd*100:6.1f}% | {sk:>5d} | {liq:>4d} | {p1_count:>10d}")
    print()


# ============================================================
# PART 2: MONTE CARLO GRID - The main event
# ============================================================
print()
print("=" * 110)
print("PART 2: MONTE CARLO GRID - 1000 paths per config")
print("=" * 110)
print()

np.random.seed(42)

# Store all results
grid_results = {}

total_configs = len(phase1_modes) * len(phase2_levs) * len(thresholds)
config_num = 0

for mode_name, (mode, pct) in phase1_modes.items():
    for lev in phase2_levs:
        for thresh in thresholds:
            config_num += 1
            key = (mode_name, lev, thresh)

            finals = []
            max_dds = []
            ruined = 0
            total_liq = 0
            total_skip = 0

            for _ in range(N_SIMS):
                shuffled = list(trade_data)
                np.random.shuffle(shuffled)

                if mode == 'cross':
                    eq, sk, liq = simulate_cross(shuffled, thresh, lev)
                else:
                    eq, sk, liq = simulate_isolated(shuffled, pct, thresh, lev)

                finals.append(eq[-1])
                dd = calc_max_dd(eq)
                max_dds.append(dd)
                total_liq += liq
                total_skip += sk
                if eq[-1] < 1.0:
                    ruined += 1

            grid_results[key] = {
                'median': np.median(finals),
                'p5': np.percentile(finals, 5),
                'p25': np.percentile(finals, 25),
                'p75': np.percentile(finals, 75),
                'p95': np.percentile(finals, 95),
                'avg_dd': np.mean(max_dds),
                'p95_dd': np.percentile(max_dds, 95),
                'ruin_pct': ruined / N_SIMS * 100,
                'avg_liq': total_liq / N_SIMS,
                'avg_skip': total_skip / N_SIMS,
            }

            if config_num % 15 == 0:
                print(f"  Progress: {config_num}/{total_configs} configs done...")

print(f"  Done! {total_configs} configs x {N_SIMS} paths = {total_configs * N_SIMS:,} simulations")
print()


# ============================================================
# PART 3: RESULTS TABLE - Sorted by P5
# ============================================================
print("=" * 110)
print("PART 3: FULL GRID RESULTS - Sorted by MC P5 (worst 5% luck)")
print("=" * 110)
print()

print(f"  {'#':>3s} | {'Mode':>8s} | {'Lev':>4s} | {'Thr':>5s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'Ruin':>5s} | {'Liq':>5s}")
print(f"  {'-'*110}")

sorted_results = sorted(grid_results.items(), key=lambda x: x[1]['p5'], reverse=True)

for rank, (key, r) in enumerate(sorted_results):
    mode_name, lev, thresh = key
    marker = " <--" if rank < 5 else ""
    print(f"  {rank+1:>3d} | {mode_name:>8s} | {lev:>3d}x | ${thresh:>3d} | ${r['median']:11.2f} | ${r['p5']:11.2f} | ${r['p25']:11.2f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['ruin_pct']:4.1f}% | {r['avg_liq']:4.1f}{marker}")


# ============================================================
# PART 4: BEST BY CATEGORY
# ============================================================
print()
print("=" * 110)
print("PART 4: BEST CONFIG BY CATEGORY")
print("=" * 110)
print()

# Best P5 (safest)
print("  TOP 5 by P5 (best worst-case):")
for i, (key, r) in enumerate(sorted(grid_results.items(), key=lambda x: x[1]['p5'], reverse=True)[:5]):
    mode_name, lev, thresh = key
    print(f"    #{i+1}: {mode_name} / {lev}x / ${thresh} -> Median ${r['median']:,.0f}, P5 ${r['p5']:,.0f}, DD {r['avg_dd']*100:.1f}%")

print()

# Best Median (highest typical return)
print("  TOP 5 by Median (best typical outcome):")
for i, (key, r) in enumerate(sorted(grid_results.items(), key=lambda x: x[1]['median'], reverse=True)[:5]):
    mode_name, lev, thresh = key
    print(f"    #{i+1}: {mode_name} / {lev}x / ${thresh} -> Median ${r['median']:,.0f}, P5 ${r['p5']:,.0f}, DD {r['avg_dd']*100:.1f}%")

print()

# Best P5/DD ratio (risk-adjusted)
print("  TOP 5 by P5/AvgDD ratio (risk-adjusted):")
ratios = {k: v['p5'] / v['avg_dd'] if v['avg_dd'] > 0 else 0 for k, v in grid_results.items()}
for i, (key, ratio) in enumerate(sorted(ratios.items(), key=lambda x: x[1], reverse=True)[:5]):
    r = grid_results[key]
    mode_name, lev, thresh = key
    print(f"    #{i+1}: {mode_name} / {lev}x / ${thresh} -> P5 ${r['p5']:,.0f}, DD {r['avg_dd']*100:.1f}%, Ratio {ratio:,.0f}")

print()

# Lowest DD with good returns
print("  TOP 5 by lowest AvgDD (with Median > $10K):")
safe = {k: v for k, v in grid_results.items() if v['median'] > 10000}
for i, (key, r) in enumerate(sorted(safe.items(), key=lambda x: x[1]['avg_dd'])[:5]):
    mode_name, lev, thresh = key
    print(f"    #{i+1}: {mode_name} / {lev}x / ${thresh} -> DD {r['avg_dd']*100:.1f}%, Median ${r['median']:,.0f}, P5 ${r['p5']:,.0f}")


# ============================================================
# PART 5: CROSS vs ISO comparison at each leverage
# ============================================================
print()
print("=" * 110)
print("PART 5: CROSS vs ISOLATED - At each Phase 2 leverage")
print("=" * 110)
print()

for lev in phase2_levs:
    print(f"  --- Phase 2 = {lev}x ---")
    print(f"  {'Mode':>10s} | {'Thresh':>7s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s} | {'Liq':>5s}")
    print(f"  {'-'*70}")

    for thresh in thresholds:
        for mode_name in phase1_modes:
            key = (mode_name, lev, thresh)
            r = grid_results[key]
            print(f"  {mode_name:>10s} | ${thresh:>5d} | ${r['median']:11.2f} | ${r['p5']:11.2f} | {r['avg_dd']*100:5.1f}% | {r['avg_liq']:4.1f}")
        print()
    print()


# ============================================================
# PART 6: LEVERAGE COMPARISON (fixing mode + threshold)
# ============================================================
print("=" * 110)
print("PART 6: LEVERAGE IMPACT - How much does Phase 2 leverage matter?")
print("=" * 110)
print()

for mode_name in ['Cross', 'Iso 20%']:
    print(f"  --- {mode_name} ---")
    print(f"  {'Lev':>5s} | {'Thresh':>7s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'P75':>12s} | {'AvgDD':>7s}")
    print(f"  {'-'*75}")

    for lev in phase2_levs:
        for thresh in thresholds:
            key = (mode_name, lev, thresh)
            r = grid_results[key]
            print(f"  {lev:>4d}x | ${thresh:>5d} | ${r['median']:11.2f} | ${r['p5']:11.2f} | ${r['p25']:11.2f} | ${r['p75']:11.2f} | {r['avg_dd']*100:5.1f}%")
    print()


# ============================================================
# PART 7: THRESHOLD IMPACT
# ============================================================
print("=" * 110)
print("PART 7: THRESHOLD IMPACT - When to switch from Phase 1 to Phase 2")
print("=" * 110)
print()

print("  Lower threshold = switch to compounding earlier = more growth but more risk")
print()

for mode_name in ['Cross', 'Iso 20%']:
    for lev in [20, 25, 30]:
        print(f"  {mode_name} / {lev}x:")
        for thresh in thresholds:
            key = (mode_name, lev, thresh)
            r = grid_results[key]
            print(f"    ${thresh}: Median ${r['median']:>10,.0f} | P5 ${r['p5']:>10,.0f} | DD {r['avg_dd']*100:.1f}% | P95DD {r['p95_dd']*100:.1f}%")
        print()


# ============================================================
# VERDICT
# ============================================================
print()
print("=" * 110)
print("VERDICT")
print("=" * 110)
print()

# Overall best by P5
best_p5_key = max(grid_results, key=lambda k: grid_results[k]['p5'])
best_p5 = grid_results[best_p5_key]
print(f"  SAFEST (best P5):    {best_p5_key[0]} / {best_p5_key[1]}x / ${best_p5_key[2]}")
print(f"    Median ${best_p5['median']:,.0f} | P5 ${best_p5['p5']:,.0f} | DD {best_p5['avg_dd']*100:.1f}%")
print()

# Best median
best_med_key = max(grid_results, key=lambda k: grid_results[k]['median'])
best_med = grid_results[best_med_key]
print(f"  HIGHEST (best median): {best_med_key[0]} / {best_med_key[1]}x / ${best_med_key[2]}")
print(f"    Median ${best_med['median']:,.0f} | P5 ${best_med['p5']:,.0f} | DD {best_med['avg_dd']*100:.1f}%")
print()

# Best risk-adjusted
best_ratio_key = max(ratios, key=ratios.get)
best_ratio = grid_results[best_ratio_key]
print(f"  BEST RISK-ADJUSTED:  {best_ratio_key[0]} / {best_ratio_key[1]}x / ${best_ratio_key[2]}")
print(f"    Median ${best_ratio['median']:,.0f} | P5 ${best_ratio['p5']:,.0f} | DD {best_ratio['avg_dd']*100:.1f}%")
print()

# Show the 3 practical options
print("  THREE OPTIONS FOR USER:")
print()

for label, lev in [("Conservative (20x)", 20), ("Moderate (25x)", 25), ("Aggressive (30x)", 30)]:
    # Find best P5 at this leverage
    lev_results = {k: v for k, v in grid_results.items() if k[1] == lev}
    best_key = max(lev_results, key=lambda k: lev_results[k]['p5'])
    r = grid_results[best_key]
    print(f"  {label}:")
    print(f"    Best config: {best_key[0]} / {best_key[1]}x / ${best_key[2]}")
    print(f"    Median ${r['median']:>10,.0f} | P5 ${r['p5']:>10,.0f} | DD {r['avg_dd']*100:.1f}% | P95DD {r['p95_dd']*100:.1f}%")
    print()
