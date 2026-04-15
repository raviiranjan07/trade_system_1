"""L1-EXP-002: Fixed Position Size vs Scaling — Cross Margin

QUESTION: Is it better to use a fixed position size (with high leverage
for low margin) or scale position with equity (fixed leverage)?

KEY INSIGHT: In Binance Cross Margin mode, leverage setting only controls
margin requirement, NOT P&L. P&L depends only on POSITION SIZE.
  - 125x leverage available for small positions (<$50K)
  - Margin required = position_size / leverage
  - At 125x: $170 position needs only $1.36 margin (vs $8.50 at 20x)

METHODS TESTED:
  A) Fixed position $170 (minimum, never changes)
  B) Fixed position $200, $250, $300, $500
  C) Scaling: position = equity * 10x, 15x, 20x (EXP-001 baseline)
  D) Hybrid: fixed $170 until equity threshold, then scale at 10x/15x/20x

COMPARE AGAINST: L1-EXP-001 baseline (scaling 20x)
  - Final: $56,043 | Ruin: 18.6% | MC Median: $56,043

CROSS MARGIN MODEL:
  - Can trade if: equity >= position_size / 125
  - P&L per trade = position_size * return_bps / 10000
  - Ruin = equity drops below min_margin for position
"""
import sys
sys.path.insert(0, "src")

import numpy as np
from engine.backtest import run_backtest
from engine.config.loader import load_config

# ============================================================
# CONSTANTS
# ============================================================
STARTING_CAPITAL = 10.0
MIN_NOTIONAL = 170.0
MAX_LEVERAGE_SETTING = 125  # Binance allows 125x for small positions
N_SIMS = 1000

# ============================================================
# LOAD V1.3.2 TRADES
# ============================================================
config = load_config()
trades = run_backtest(config)
returns = [t.net_profit_bps for t in trades]

wins = [r for r in returns if r > 0]
losses = [r for r in returns if r <= 0]

print("=" * 120)
print("L1-EXP-002: FIXED POSITION SIZE vs SCALING (Cross Margin)")
print("=" * 120)
print(f"V1.3.2: {len(returns)} trades, {len(wins)/len(returns)*100:.1f}% win")
print(f"Mean: {np.mean(returns):+.1f} bps | Std: {np.std(returns):.1f} bps")
print(f"Worst: {min(returns):.1f} bps | Best: {max(returns):+.1f} bps")
print(f"Cross Margin: max leverage setting = {MAX_LEVERAGE_SETTING}x")
print(f"Min margin for $170 position = ${MIN_NOTIONAL/MAX_LEVERAGE_SETTING:.2f}")
print()


# ============================================================
# CORE FUNCTIONS
# ============================================================
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


def simulate_fixed_position(rets, position_size, capital=STARTING_CAPITAL):
    """Fixed position size with cross margin (125x available)."""
    equity = [capital]
    skipped = 0
    min_margin = position_size / MAX_LEVERAGE_SETTING

    for r in rets:
        eq = equity[-1]
        # Cross margin: need at least min_margin to open position
        if eq < min_margin:
            equity.append(eq)
            skipped += 1
            continue
        pnl = position_size * (r / 10000)
        equity.append(max(eq + pnl, 0.01))
    return equity, skipped


def simulate_scaling(rets, leverage, capital=STARTING_CAPITAL):
    """Scaling position: position = equity * leverage (EXP-001 model).
    Uses 20x max leverage, needs equity * leverage >= $170."""
    equity = [capital]
    skipped = 0
    for r in rets:
        eq = equity[-1]
        pos = eq * leverage
        if pos < MIN_NOTIONAL:
            equity.append(eq)
            skipped += 1
            continue
        pnl = pos * (r / 10000)
        equity.append(max(eq + pnl, 0.01))
    return equity, skipped


def simulate_hybrid(rets, fixed_pos, threshold, scale_lev, capital=STARTING_CAPITAL):
    """Hybrid: fixed position until equity >= threshold, then scale.
    Phase 1 (equity < threshold): position = fixed_pos, 125x margin
    Phase 2 (equity >= threshold): position = equity * scale_lev
    """
    equity = [capital]
    skipped = 0
    phases = []
    min_margin_fixed = fixed_pos / MAX_LEVERAGE_SETTING

    for r in rets:
        eq = equity[-1]

        if eq < threshold:
            # Phase 1: fixed position
            if eq < min_margin_fixed:
                equity.append(eq)
                skipped += 1
                phases.append(0)
                continue
            pos = fixed_pos
            phases.append(1)
        else:
            # Phase 2: scaling
            pos = eq * scale_lev
            phases.append(2)

        pnl = pos * (r / 10000)
        equity.append(max(eq + pnl, 0.01))

    return equity, skipped, phases


# ============================================================
# PART 1: ORIGINAL ORDER — All methods
# ============================================================
print("=" * 120)
print("PART 1: ORIGINAL TRADE ORDER")
print("=" * 120)
print()

print(f"  {'Method':>40s} | {'Final':>12s} | {'Return':>10s} | {'MaxDD':>8s} | {'MinEq':>8s} | {'Skip':>5s}")
print(f"  {'-'*100}")

results = {}

# A) Fixed position sizes
for pos in [170, 200, 250, 300, 500]:
    eq, sk = simulate_fixed_position(returns, pos)
    dd = calc_max_dd(eq)
    ret = (eq[-1] - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    name = f"A) Fixed ${pos} position"
    results[name] = {'final': eq[-1], 'dd': dd, 'skipped': sk, 'equity': eq}
    print(f"  {name:>40s} | ${eq[-1]:11.2f} | {ret:+9.1f}% | {dd*100:6.1f}% | ${min(eq):6.2f} | {sk:>5d}")

print()

# C) Scaling (EXP-001 style)
for lev in [10, 15, 20]:
    eq, sk = simulate_scaling(returns, lev)
    dd = calc_max_dd(eq)
    ret = (eq[-1] - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    name = f"C) Scaling {lev}x leverage"
    results[name] = {'final': eq[-1], 'dd': dd, 'skipped': sk, 'equity': eq}
    print(f"  {name:>40s} | ${eq[-1]:11.2f} | {ret:+9.1f}% | {dd*100:6.1f}% | ${min(eq):6.2f} | {sk:>5d}")

print()

# D) Hybrid: fixed $170 until threshold, then scale
for threshold, scale_lev in [(25, 10), (25, 15), (25, 20),
                              (50, 10), (50, 15), (50, 20),
                              (100, 10), (100, 15), (100, 20)]:
    eq, sk, phases = simulate_hybrid(returns, 170, threshold, scale_lev)
    dd = calc_max_dd(eq)
    ret = (eq[-1] - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    name = f"D) Hybrid $170->scale {scale_lev}x @${threshold}"
    results[name] = {'final': eq[-1], 'dd': dd, 'skipped': sk, 'equity': eq}
    p1_count = sum(1 for p in phases if p == 1)
    p2_count = sum(1 for p in phases if p == 2)
    print(f"  {name:>40s} | ${eq[-1]:11.2f} | {ret:+9.1f}% | {dd*100:6.1f}% | ${min(eq):6.2f} | {sk:>5d}  (P1:{p1_count} P2:{p2_count})")


# ============================================================
# PART 2: WORST TRADE IMPACT — Fixed position vs Scaling
# ============================================================
print()
print("=" * 120)
print("PART 2: WORST TRADE IMPACT — How much do you lose on the worst trade?")
print("=" * 120)
print()

worst = min(returns)
print(f"  Worst trade: {worst:.1f} bps ({worst/100:.2f}%)")
print()

print(f"  {'Method':>35s} | {'Position $':>12s} | {'Loss $':>10s} | {'Loss %':>8s} | {'After':>10s} | {'Can Trade?':>10s}")
print(f"  {'-'*100}")

# Fixed positions at $10 equity
for pos in [170, 200, 250, 300, 500]:
    loss = pos * abs(worst) / 10000
    loss_pct = loss / STARTING_CAPITAL * 100
    after = STARTING_CAPITAL - loss
    min_margin = pos / MAX_LEVERAGE_SETTING
    can = "YES" if after >= min_margin else "NO"
    print(f"  {'Fixed $' + str(pos):>35s} | ${pos:11.2f} | ${loss:9.2f} | {loss_pct:6.1f}% | ${after:9.2f} | {can:>10s}")

# Scaling at $10 equity
for lev in [10, 15, 20]:
    pos = STARTING_CAPITAL * lev
    loss = pos * abs(worst) / 10000
    loss_pct = loss / STARTING_CAPITAL * 100
    after = STARTING_CAPITAL - loss
    can_scale = "YES" if after * lev >= MIN_NOTIONAL else "NO"
    print(f"  {'Scaling ' + str(lev) + 'x ($' + str(int(pos)) + ' pos)':>35s} | ${pos:11.2f} | ${loss:9.2f} | {loss_pct:6.1f}% | ${after:9.2f} | {can_scale:>10s}")


# ============================================================
# PART 3: CONSECUTIVE LOSSES TO RUIN
# ============================================================
print()
print("=" * 120)
print("PART 3: CONSECUTIVE LOSSES TO RUIN")
print("=" * 120)
print()

avg_loss = np.mean(losses)
print(f"  Using avg loss: {avg_loss:.1f} bps")
print()

print(f"  {'Method':>35s} | {'Losses to Ruin':>15s} | {'Final Equity':>12s}")
print(f"  {'-'*70}")

# Fixed position: ruin when equity < position / 125
for pos in [170, 200, 250, 300]:
    eq = STARTING_CAPITAL
    min_margin = pos / MAX_LEVERAGE_SETTING
    for i in range(100):
        if eq < min_margin:
            print(f"  {'Fixed $' + str(pos):>35s} | {i:>15d} | ${eq:11.2f}")
            break
        loss = pos * abs(avg_loss) / 10000
        eq -= loss
    else:
        print(f"  {'Fixed $' + str(pos):>35s} | {'>100':>15s} | ${eq:11.2f}")

# Scaling: ruin when equity * lev < $170
for lev in [10, 15, 20]:
    eq = STARTING_CAPITAL
    for i in range(100):
        if eq * lev < MIN_NOTIONAL:
            print(f"  {'Scaling ' + str(lev) + 'x':>35s} | {i:>15d} | ${eq:11.2f}")
            break
        loss = eq * lev * abs(avg_loss) / 10000
        eq -= loss
    else:
        print(f"  {'Scaling ' + str(lev) + 'x':>35s} | {'>100':>15s} | ${eq:11.2f}")


# ============================================================
# PART 4: MONTE CARLO — 1000 shuffled paths
# ============================================================
print()
print("=" * 120)
print("PART 4: MONTE CARLO — 1000 shuffled paths")
print("=" * 120)
print()

np.random.seed(42)

mc_configs = {
    # Fixed position sizes
    'Fixed $170': lambda r: simulate_fixed_position(r, 170),
    'Fixed $200': lambda r: simulate_fixed_position(r, 200),
    'Fixed $250': lambda r: simulate_fixed_position(r, 250),
    'Fixed $300': lambda r: simulate_fixed_position(r, 300),
    # Scaling (EXP-001 style)
    'Scaling 10x': lambda r: simulate_scaling(r, 10),
    'Scaling 15x': lambda r: simulate_scaling(r, 15),
    'Scaling 20x': lambda r: simulate_scaling(r, 20),
    # Hybrid
    'Hybrid $170->10x @$25': lambda r: simulate_hybrid(r, 170, 25, 10)[:2],
    'Hybrid $170->15x @$25': lambda r: simulate_hybrid(r, 170, 25, 15)[:2],
    'Hybrid $170->20x @$25': lambda r: simulate_hybrid(r, 170, 25, 20)[:2],
    'Hybrid $170->10x @$50': lambda r: simulate_hybrid(r, 170, 50, 10)[:2],
    'Hybrid $170->15x @$50': lambda r: simulate_hybrid(r, 170, 50, 15)[:2],
    'Hybrid $170->20x @$50': lambda r: simulate_hybrid(r, 170, 50, 20)[:2],
}

print(f"  {'Method':>27s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'P75':>12s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*110}")

mc_results = {}
for name, fn in mc_configs.items():
    finals = []
    max_dds = []
    ruined = 0

    for _ in range(N_SIMS):
        shuffled = list(returns)
        np.random.shuffle(shuffled)
        eq, sk = fn(shuffled)
        finals.append(eq[-1])
        dd = calc_max_dd(eq)
        max_dds.append(dd)

        # Ruin = can't trade anymore
        if 'Fixed' in name:
            pos = int(name.split('$')[1])
            min_margin = pos / MAX_LEVERAGE_SETTING
            if eq[-1] < min_margin:
                ruined += 1
        elif 'Scaling' in name:
            lev = int(name.split()[1].replace('x', ''))
            if eq[-1] * lev < MIN_NOTIONAL:
                ruined += 1
        else:
            # Hybrid: check if stuck (can't do either phase)
            min_margin = 170 / MAX_LEVERAGE_SETTING
            if eq[-1] < min_margin:
                ruined += 1

    mc_results[name] = {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'ruin_pct': ruined / N_SIMS * 100,
    }

    r = mc_results[name]
    print(f"  {name:>27s} | ${r['median']:11.2f} | ${r['p5']:11.2f} | ${r['p25']:11.2f} | ${r['p75']:11.2f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%")


# ============================================================
# PART 5: SURVIVAL ANALYSIS
# ============================================================
print()
print("=" * 120)
print("PART 5: SURVIVAL — % of 1000 paths that NEVER dip below threshold")
print("=" * 120)
print()

np.random.seed(42)

thresholds = [3, 5, 7, 8, 9]
key_methods = ['Fixed $170', 'Fixed $200', 'Fixed $300',
               'Scaling 20x', 'Hybrid $170->15x @$25', 'Hybrid $170->20x @$50']

print(f"  {'Method':>27s}", end="")
for t in thresholds:
    print(f" | {'>${}'.format(t):>7s}", end="")
print(f" | {'Ruin':>6s}")
print(f"  {'-'*75}")

for name in key_methods:
    fn = mc_configs[name]
    above = {t: 0 for t in thresholds}
    ruined = 0

    for _ in range(N_SIMS):
        shuffled = list(returns)
        np.random.shuffle(shuffled)
        eq, sk = fn(shuffled)
        min_eq = min(eq)
        for t in thresholds:
            if min_eq >= t:
                above[t] += 1
        min_margin = 170 / MAX_LEVERAGE_SETTING
        if eq[-1] < min_margin:
            ruined += 1

    print(f"  {name:>27s}", end="")
    for t in thresholds:
        pct = above[t] / N_SIMS * 100
        print(f" | {pct:6.1f}%", end="")
    print(f" | {ruined/N_SIMS*100:5.1f}%")


# ============================================================
# PART 6: GROWTH TRAJECTORY — When do hybrid methods start compounding?
# ============================================================
print()
print("=" * 120)
print("PART 6: GROWTH TRAJECTORY (original order)")
print("=" * 120)
print()

checkpoints = [10, 25, 50, 75, 100, 150, 200, 220]
print(f"  {'Method':>27s}", end="")
for cp in checkpoints:
    print(f" | {'@t=' + str(cp):>10s}", end="")
print()
print(f"  {'-'*120}")

trajectory_methods = {
    'Fixed $170': lambda r: simulate_fixed_position(r, 170),
    'Fixed $250': lambda r: simulate_fixed_position(r, 250),
    'Scaling 20x': lambda r: simulate_scaling(r, 20),
    'Hybrid $170->15x @$25': lambda r: simulate_hybrid(r, 170, 25, 15)[:2],
    'Hybrid $170->20x @$25': lambda r: simulate_hybrid(r, 170, 25, 20)[:2],
    'Hybrid $170->15x @$50': lambda r: simulate_hybrid(r, 170, 50, 15)[:2],
}

for name, fn in trajectory_methods.items():
    eq, _ = fn(returns)
    print(f"  {name:>27s}", end="")
    for cp in checkpoints:
        idx = min(cp, len(eq) - 1)
        print(f" | ${eq[idx]:9.2f}", end="")
    print()


# ============================================================
# PART 7: VERDICT
# ============================================================
print()
print("=" * 120)
print("VERDICT")
print("=" * 120)
print()

# Find best methods by different criteria
best_growth = max(mc_results.items(), key=lambda x: x[1]['median'])
safest = min(mc_results.items(), key=lambda x: x[1]['ruin_pct'])
best_risk_adj = max(mc_results.items(),
    key=lambda x: x[1]['median'] / x[1]['p95_dd'] if x[1]['p95_dd'] > 0 else 0)

print(f"  BEST GROWTH (MC Median):     {best_growth[0]} = ${best_growth[1]['median']:.2f}")
print(f"  SAFEST (Lowest Ruin):        {safest[0]} = {safest[1]['ruin_pct']:.1f}% ruin")
print(f"  BEST RISK-ADJUSTED:          {best_risk_adj[0]} = median/P95DD = {best_risk_adj[1]['median']/best_risk_adj[1]['p95_dd']:.1f}")
print()

# Compare key matchups
print("  KEY COMPARISONS:")
print()
for a, b in [('Fixed $170', 'Scaling 20x'),
             ('Hybrid $170->20x @$25', 'Scaling 20x'),
             ('Hybrid $170->15x @$50', 'Scaling 20x'),
             ('Fixed $170', 'Hybrid $170->20x @$25')]:
    if a in mc_results and b in mc_results:
        ra = mc_results[a]
        rb = mc_results[b]
        print(f"  {a:>27s}: Median ${ra['median']:>10.2f}, P5 ${ra['p5']:>8.2f}, Ruin {ra['ruin_pct']:>5.1f}%")
        print(f"  {b:>27s}: Median ${rb['median']:>10.2f}, P5 ${rb['p5']:>8.2f}, Ruin {rb['ruin_pct']:>5.1f}%")
        growth_winner = a if ra['median'] > rb['median'] else b
        safety_winner = a if ra['ruin_pct'] < rb['ruin_pct'] else b
        print(f"    Growth: {growth_winner} | Safety: {safety_winner}")
        print()


# ============================================================
# PART 8: ISOLATED MARGIN — Cap loss per trade
# ============================================================
print()
print("=" * 120)
print("PART 8: ISOLATED MARGIN — Allocate fixed margin per trade, cap losses")
print("=" * 120)
print()
print("  In Isolated Margin:")
print("    - You allocate a fixed $ margin per trade")
print("    - If trade loss > allocated margin: LIQUIDATED, lose only the margin")
print("    - If trade loss < allocated margin: normal exit, lose actual P&L")
print("    - Remaining balance is PROTECTED from liquidation")
print()

# Binance maintenance margin rate for BTCUSDT = 0.4%
MAINT_MARGIN_RATE = 0.004


def simulate_isolated(rets, position_size, margin_per_trade, capital=STARTING_CAPITAL):
    """Isolated margin: allocate fixed margin per trade.

    Rules:
    1. Each trade: allocate margin_per_trade from balance
    2. P&L = position_size * return_bps / 10000
    3. If loss > margin_per_trade: liquidated, lose margin_per_trade
    4. If win or small loss: get margin back +/- P&L
    5. Can't trade if balance < margin_per_trade
    """
    equity = [capital]
    skipped = 0
    liquidated = 0

    for r in rets:
        eq = equity[-1]
        if eq < margin_per_trade:
            equity.append(eq)
            skipped += 1
            continue

        pnl = position_size * (r / 10000)

        if pnl < -margin_per_trade:
            # LIQUIDATED: lose entire allocated margin
            equity.append(eq - margin_per_trade)
            liquidated += 1
        else:
            # Normal trade: P&L applied
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


def simulate_isolated_hybrid(rets, position_size, margin_per_trade,
                              threshold, scale_lev, capital=STARTING_CAPITAL):
    """Isolated margin Phase 1, then Cross margin scaling Phase 2."""
    equity = [capital]
    skipped = 0
    liquidated = 0

    for i, r in enumerate(rets):
        eq = equity[-1]

        if eq < threshold:
            # Phase 1: Isolated margin, fixed position
            if eq < margin_per_trade:
                equity.append(eq)
                skipped += 1
                continue

            pnl = position_size * (r / 10000)
            if pnl < -margin_per_trade:
                equity.append(eq - margin_per_trade)
                liquidated += 1
            else:
                equity.append(max(eq + pnl, 0.01))
        else:
            # Phase 2: Cross margin, scaling
            pos = eq * scale_lev
            pnl = pos * (r / 10000)
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


# --- PART 8A: Original order comparison ---
print("--- 8A: ORIGINAL ORDER ---")
print()
print(f"  {'Method':>45s} | {'Final':>12s} | {'MaxDD':>7s} | {'MinEq':>8s} | {'Skip':>5s} | {'Liq':>4s}")
print(f"  {'-'*100}")

# Cross margin baseline (from earlier)
eq_cross, sk_cross = simulate_fixed_position(returns, 170)
print(f"  {'Cross: Fixed $170':>45s} | ${eq_cross[-1]:11.2f} | {calc_max_dd(eq_cross)*100:5.1f}% | ${min(eq_cross):6.2f} | {sk_cross:>5d} | {'N/A':>4s}")

# Isolated margin at different margin allocations
for margin in [1.50, 2.00, 3.00, 4.00, 5.00]:
    eq_iso, sk_iso, liq_iso = simulate_isolated(returns, 170, margin)
    dd_iso = calc_max_dd(eq_iso)
    name = f"Isolated: $170 pos, ${margin:.2f} margin"
    print(f"  {name:>45s} | ${eq_iso[-1]:11.2f} | {dd_iso*100:5.1f}% | ${min(eq_iso):6.2f} | {sk_iso:>5d} | {liq_iso:>4d}")

print()

# Show which trades get liquidated at each margin level
print("  Liquidation analysis (at what margin level does each losing trade get liquidated?):")
print()
losing_trades = [(i, r) for i, r in enumerate(returns) if r < 0]
losing_trades.sort(key=lambda x: x[1])  # worst first

print(f"  {'Trade':>6s} | {'Return':>10s} | {'Loss on $170':>13s} | {'Liq@$1.50':>10s} | {'Liq@$2.00':>10s} | {'Liq@$3.00':>10s} | {'Liq@$5.00':>10s}")
print(f"  {'-'*80}")

for idx, r in losing_trades[:15]:  # Show worst 15
    loss = 170 * abs(r) / 10000
    liq_150 = "LIQ" if loss > 1.50 else f"${loss:.2f}"
    liq_200 = "LIQ" if loss > 2.00 else f"${loss:.2f}"
    liq_300 = "LIQ" if loss > 3.00 else f"${loss:.2f}"
    liq_500 = "LIQ" if loss > 5.00 else f"${loss:.2f}"
    print(f"  {idx:>6d} | {r:>+9.1f} | ${loss:12.2f} | {liq_150:>10s} | {liq_200:>10s} | {liq_300:>10s} | {liq_500:>10s}")

total_losing = len(losing_trades)
for margin in [1.50, 2.00, 3.00, 5.00]:
    liq_count = sum(1 for _, r in losing_trades if 170 * abs(r) / 10000 > margin)
    print(f"  Margin ${margin:.2f}: {liq_count}/{total_losing} losing trades get liquidated ({liq_count/total_losing*100:.0f}%)")


# --- PART 8B: Consecutive losses to ruin ---
print()
print("--- 8B: CONSECUTIVE LOSSES TO RUIN (Isolated vs Cross) ---")
print()

print(f"  {'Method':>45s} | {'Losses to Ruin':>15s}")
print(f"  {'-'*65}")

for margin in [1.50, 2.00, 3.00, 5.00]:
    eq = STARTING_CAPITAL
    for i in range(100):
        if eq < margin:
            print(f"  {'Isolated $170, $' + f'{margin:.2f} margin':>45s} | {i:>15d}")
            break
        # Worst case: every loss = liquidation = lose full margin
        eq -= margin
    else:
        print(f"  {'Isolated $170, $' + f'{margin:.2f} margin':>45s} | {'>100':>15s}")

# Cross margin comparison
eq = STARTING_CAPITAL
for i in range(100):
    if eq < 170 / MAX_LEVERAGE_SETTING:
        print(f"  {'Cross $170 (avg loss)':>45s} | {i:>15d}")
        break
    eq -= 170 * abs(np.mean(losses)) / 10000


# --- PART 8C: Monte Carlo ---
print()
print("--- 8C: MONTE CARLO — Isolated margin methods ---")
print()

np.random.seed(42)

iso_configs = {
    'Cross $170': lambda r: simulate_fixed_position(r, 170)[:2],
    'Iso $170, $1.50 margin': lambda r: simulate_isolated(r, 170, 1.50)[:2],
    'Iso $170, $2.00 margin': lambda r: simulate_isolated(r, 170, 2.00)[:2],
    'Iso $170, $3.00 margin': lambda r: simulate_isolated(r, 170, 3.00)[:2],
    'Iso $170, $5.00 margin': lambda r: simulate_isolated(r, 170, 5.00)[:2],
    'Iso Hybrid->20x @$25,$2': lambda r: simulate_isolated_hybrid(r, 170, 2.00, 25, 20)[:2],
    'Iso Hybrid->20x @$25,$3': lambda r: simulate_isolated_hybrid(r, 170, 3.00, 25, 20)[:2],
    'Cross Hybrid->20x @$25': lambda r: simulate_hybrid(r, 170, 25, 20)[:2],
}

print(f"  {'Method':>30s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'P75':>12s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*115}")

iso_mc_results = {}
for name, fn in iso_configs.items():
    finals = []
    max_dds = []
    ruined = 0

    for _ in range(N_SIMS):
        shuffled = list(returns)
        np.random.shuffle(shuffled)
        eq, sk = fn(shuffled)
        finals.append(eq[-1])
        dd = calc_max_dd(eq)
        max_dds.append(dd)
        if eq[-1] < 1.36:  # $170/125 = can't trade
            ruined += 1

    iso_mc_results[name] = {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'ruin_pct': ruined / N_SIMS * 100,
    }

    r = iso_mc_results[name]
    print(f"  {name:>30s} | ${r['median']:11.2f} | ${r['p5']:11.2f} | ${r['p25']:11.2f} | ${r['p75']:11.2f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%")


# --- PART 8D: Cross vs Isolated Verdict ---
print()
print("--- 8D: CROSS vs ISOLATED VERDICT ---")
print()

cross_hybrid = iso_mc_results.get('Cross Hybrid->20x @$25', {})
iso_hybrid_2 = iso_mc_results.get('Iso Hybrid->20x @$25,$2', {})
iso_hybrid_3 = iso_mc_results.get('Iso Hybrid->20x @$25,$3', {})

if cross_hybrid and iso_hybrid_2:
    print(f"  Cross Hybrid $170->20x @$25:  Median ${cross_hybrid['median']:>10.2f} | P5 ${cross_hybrid['p5']:>10.2f} | Ruin {cross_hybrid['ruin_pct']:.1f}%")
    print(f"  Iso Hybrid   $2 margin:       Median ${iso_hybrid_2['median']:>10.2f} | P5 ${iso_hybrid_2['p5']:>10.2f} | Ruin {iso_hybrid_2['ruin_pct']:.1f}%")
    print(f"  Iso Hybrid   $3 margin:       Median ${iso_hybrid_3['median']:>10.2f} | P5 ${iso_hybrid_3['p5']:>10.2f} | Ruin {iso_hybrid_3['ruin_pct']:.1f}%")
    print()
    print("  Does Isolated margin improve the Hybrid approach?")
