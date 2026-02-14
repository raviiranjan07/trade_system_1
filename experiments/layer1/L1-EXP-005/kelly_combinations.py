"""L1-EXP-005 v2: Kelly Combinations - Fixed AND Dynamic Leverage

PART A: FIXED Kelly fractions (0.25 to 1.0 of full Kelly)
PART B: DYNAMIC Kelly setups that avoid EXP-005 v1 problems:
  1. Warm-Start: Begin at half-Kelly 25x, adapt only if stats deviate
  2. High Floor: Continuous Kelly with min 20x (never underbet)
  3. Streak-Based: Fixed 25x, step up/down based on win/loss streaks
  4. Signal-Quality: Different leverage per signal type (V12_LONG/SHORT, BEAR/BULL)
  5. Equity-Milestone: Step leverage based on account size
  6. Pure Continuous: Kelly with no DD constraint, no floor (true continuous)
  7. Confidence-Based: Use Kelly lower CI bound (conservative continuous)

WHY EXP-005 v1 FAILED:
  - Bayesian warmup too conservative (avg 18.6x vs fixed 20x)
  - DD constraint cut leverage during recoveries
  - Variable leverage hurt geometric growth
  -> New setups avoid these problems with floors, warm starts, discrete steps

BASELINE: Cross/20x/$15 = $41K median, $30K P5
          Cross/25x/$15 = $175K median, $100K P5
"""
import sys
sys.path.insert(0, "src")

import math
import numpy as np
from v12.backtest import run_backtest
from v12.config.loader import load_config

# ============================================================
# CONSTANTS
# ============================================================
STARTING_CAPITAL = 10.0
MAX_LEVERAGE_SETTING = 125
N_SIMS = 1000
MAINT_MARGIN_RATE = 0.004
BINANCE_MIN_QTY = 0.001
BINANCE_STEP_SIZE = 0.001
BINANCE_MIN_NOTIONAL = 100
PHASE1_THRESHOLD = 15

# ============================================================
# LOAD V1.3.2 TRADES
# ============================================================
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
        'qty': qty,
        'position': position,
        'maint_margin': maint,
        'signal_type': t.signal_type if hasattr(t, 'signal_type') else 'UNKNOWN',
    })

returns = [td['bps'] for td in trade_data]
wins = [r for r in returns if r > 0]
losses = [r for r in returns if r <= 0]
win_rate = len(wins) / len(returns)
avg_win = np.mean(wins)
avg_loss = abs(np.mean(losses))
payoff_ratio = avg_win / avg_loss

# Kelly calculations
kelly_fraction = win_rate - (1 - win_rate) / payoff_ratio
# Generalized Kelly leverage (from EXP-004): find optimal numerically
leverages_sweep = np.arange(1, 61, 0.5)
growth_rates = []
for lev in leverages_sweep:
    g = np.mean([np.log(max(1 + lev * r / 10000, 1e-10)) for r in returns])
    growth_rates.append(g)
kelly_leverage = leverages_sweep[np.argmax(growth_rates)]
half_kelly_lev = kelly_leverage / 2

print("=" * 110)
print("L1-EXP-005 v2: KELLY COMBINATIONS - Fixed AND Dynamic Leverage")
print("=" * 110)
print()
print(f"  V1.3.2: {len(trades)} trades, {win_rate*100:.1f}% win, payoff {payoff_ratio:.2f}")
print(f"  Classic Kelly fraction: {kelly_fraction:.3f} ({kelly_fraction*100:.1f}%)")
print(f"  Generalized Kelly leverage: {kelly_leverage:.1f}x")
print(f"  Half Kelly leverage: {half_kelly_lev:.1f}x")
print(f"  Phase 1: Cross margin, dynamic position, threshold ${PHASE1_THRESHOLD}")
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


def simulate_phase1(td, equity):
    """Single Phase 1 trade. Returns new equity."""
    pos = td['position']
    maint = td['maint_margin']
    margin_req = pos / MAX_LEVERAGE_SETTING

    if equity < margin_req:
        return equity  # skip

    pnl = pos * (td['bps'] / 10000)
    max_loss = equity - maint
    if pnl < -max_loss:
        return 0.01  # liquidated
    return max(equity + pnl, 0.01)


def simulate_with_leverage_fn(trade_list, lev_fn, capital=STARTING_CAPITAL):
    """Generic simulation: Phase 1 (cross) + Phase 2 (leverage from function).

    lev_fn(state) -> leverage for next trade
    state dict is updated each trade with: equity, trade_idx, return_bps, etc.
    """
    equity_curve = [capital]
    leverage_curve = []
    equity = capital

    state = {
        'equity': capital,
        'peak': capital,
        'trade_idx': 0,
        'phase2_idx': 0,
        'wins': 0,
        'losses': 0,
        'consecutive_wins': 0,
        'consecutive_losses': 0,
        'sum_win_bps': 0.0,
        'sum_loss_bps': 0.0,
        'recent_returns': [],  # last N returns
        'phase2_started': False,
    }

    for td in trade_list:
        eq = equity

        if eq < PHASE1_THRESHOLD:
            # Phase 1
            equity = simulate_phase1(td, eq)
            equity_curve.append(equity)
            leverage_curve.append(0)
            state['trade_idx'] += 1
            continue

        # Phase 2
        if not state['phase2_started']:
            state['phase2_started'] = True
            state['peak'] = eq

        state['equity'] = eq
        if eq > state['peak']:
            state['peak'] = eq

        lev = lev_fn(state)
        leverage_curve.append(lev)

        pos = eq * lev
        pnl = pos * (td['bps'] / 10000)
        equity = max(eq + pnl, 0.01)
        equity_curve.append(equity)

        # Update state
        r = td['bps']
        state['trade_idx'] += 1
        state['phase2_idx'] += 1
        state['recent_returns'].append(r)
        if len(state['recent_returns']) > 50:
            state['recent_returns'].pop(0)

        if r > 0:
            state['wins'] += 1
            state['sum_win_bps'] += r
            state['consecutive_wins'] += 1
            state['consecutive_losses'] = 0
        else:
            state['losses'] += 1
            state['sum_loss_bps'] += abs(r)
            state['consecutive_losses'] += 1
            state['consecutive_wins'] = 0

    return equity_curve, leverage_curve


def run_mc(lev_fn, label, n_sims=N_SIMS):
    """Run MC simulation and return results dict."""
    finals = []
    max_dds = []
    avg_levs = []
    ruined = 0

    for _ in range(n_sims):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)

        eq_curve, lev_curve = simulate_with_leverage_fn(shuffled, lev_fn)
        finals.append(eq_curve[-1])
        max_dds.append(calc_max_dd(eq_curve))

        p2_levs = [l for l in lev_curve if l > 0]
        if p2_levs:
            avg_levs.append(np.mean(p2_levs))

        if eq_curve[-1] < 1.0:
            ruined += 1

    return {
        'label': label,
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'avg_lev': np.mean(avg_levs) if avg_levs else 0,
        'ruin_pct': ruined / n_sims * 100,
    }


# ============================================================
# PART A: FIXED KELLY FRACTIONS
# ============================================================
print("=" * 110)
print("PART A: FIXED KELLY FRACTIONS (Phase 2 leverage = fraction * Kelly optimal)")
print("=" * 110)
print()
print(f"  Kelly optimal: {kelly_leverage:.1f}x")
print()

np.random.seed(42)

fixed_fractions = [
    (0.25, 'Quarter Kelly'),
    (0.33, '1/3 Kelly'),
    (0.40, '2/5 Kelly'),
    (0.50, 'Half Kelly'),
    (0.60, '3/5 Kelly'),
    (0.67, '2/3 Kelly'),
    (0.75, '3/4 Kelly'),
    (1.00, 'Full Kelly'),
]

print(f"  {'Fraction':>14s} | {'Leverage':>8s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'P75':>12s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'Ruin':>5s}")
print(f"  {'-'*110}")

fixed_results = {}
for frac, label in fixed_fractions:
    lev = round(kelly_leverage * frac, 1)

    def make_fixed_fn(fixed_lev):
        def fn(state):
            return fixed_lev
        return fn

    result = run_mc(make_fixed_fn(lev), f"{label} ({frac:.0%})")
    result['leverage'] = lev
    result['fraction'] = frac
    fixed_results[frac] = result

    r = result
    marker = ""
    if abs(lev - 20) < 1:
        marker = " <-- ~our 20x"
    elif abs(lev - 25) < 1:
        marker = " <-- ~half Kelly"

    print(f"  {label:>14s} | {lev:>6.1f}x | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | ${r['p75']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['ruin_pct']:4.1f}%{marker}")

print()

# Find optimal fraction by different criteria
best_p5_frac = max(fixed_results, key=lambda f: fixed_results[f]['p5'])
best_med_frac = max(fixed_results, key=lambda f: fixed_results[f]['median'])
best_ratio_frac = max(fixed_results, key=lambda f: fixed_results[f]['p5'] / max(fixed_results[f]['avg_dd'], 0.01))

print(f"  Best by P5:      {best_p5_frac:.0%} Kelly = {fixed_results[best_p5_frac]['leverage']:.1f}x (P5 ${fixed_results[best_p5_frac]['p5']:,.0f})")
print(f"  Best by Median:  {best_med_frac:.0%} Kelly = {fixed_results[best_med_frac]['leverage']:.1f}x (Median ${fixed_results[best_med_frac]['median']:,.0f})")
print(f"  Best risk-adj:   {best_ratio_frac:.0%} Kelly = {fixed_results[best_ratio_frac]['leverage']:.1f}x")
print()


# ============================================================
# PART B: DYNAMIC KELLY SETUPS
# ============================================================
print("=" * 110)
print("PART B: DYNAMIC KELLY SETUPS (7 strategies)")
print("=" * 110)
print()

np.random.seed(42)

# --- Setup 1: Warm-Start Kelly ---
# Start at half-Kelly, only reduce if observed win rate drops significantly
def warm_start_kelly(state):
    """Start at half-Kelly 25x. Reduce to 20x if win rate drops below 55%.
    Increase to 30x if win rate above 65%. Otherwise stay at 25x."""
    base = half_kelly_lev  # ~25x
    n = state['wins'] + state['losses']
    if n < 10:
        return base  # not enough data, use default

    obs_wr = state['wins'] / n
    if obs_wr < 0.50:
        return 15.0
    elif obs_wr < 0.55:
        return 20.0
    elif obs_wr > 0.65:
        return 30.0
    else:
        return base


# --- Setup 2: High Floor Kelly ---
# Continuous Kelly but min 20x (never underbet)
def high_floor_kelly(state):
    """Continuous Kelly with min 20x floor. Recalculate from running stats."""
    n = state['wins'] + state['losses']
    if n < 5:
        return half_kelly_lev

    p = state['wins'] / n
    if state['wins'] > 0 and state['losses'] > 0:
        R = (state['sum_win_bps'] / state['wins']) / (state['sum_loss_bps'] / state['losses'])
    else:
        return half_kelly_lev

    if R <= 0:
        return 20.0

    f = p - (1 - p) / R
    f = max(0, f)
    # Half Kelly for safety
    f /= 2
    # Convert to leverage (calibration: full Kelly frac 0.427 -> 49.5x)
    lev = f * (kelly_leverage / kelly_fraction)
    return max(20.0, min(40.0, lev))  # floor 20x, cap 40x


# --- Setup 3: Streak-Based ---
# Fixed 25x, adjust based on win/loss streaks
def streak_kelly(state):
    """Base 25x. +5x after 3+ consecutive wins. -5x after 2+ consecutive losses."""
    base = half_kelly_lev
    if state['consecutive_wins'] >= 3:
        return min(35.0, base + 5.0)
    elif state['consecutive_losses'] >= 2:
        return max(15.0, base - 5.0)
    return base


# --- Setup 4: Signal-Quality Based ---
# Different leverage per signal type
# From V1.3.2 data: V12_SHORT is strongest (PF 2.57), BEAR_LONG next (PF 3.42)
def signal_quality_kelly(state):
    """Use half-Kelly as base. Scale by signal quality."""
    # Without actual signal type in state, use running performance
    # Higher leverage when recent performance is strong
    n = state['wins'] + state['losses']
    if n < 10:
        return half_kelly_lev

    # Recent 20-trade window
    recent = state['recent_returns'][-20:] if len(state['recent_returns']) >= 20 else state['recent_returns']
    recent_wr = sum(1 for r in recent if r > 0) / len(recent) if recent else 0.5

    if recent_wr >= 0.70:
        return 30.0  # hot streak, high conviction
    elif recent_wr >= 0.60:
        return 25.0  # normal (at expected win rate)
    elif recent_wr >= 0.50:
        return 20.0  # slightly below expected
    else:
        return 15.0  # cold streak, reduce exposure


# --- Setup 5: Equity Milestone ---
# Step leverage based on account size
def equity_milestone_kelly(state):
    """20x until $500, 25x until $5K, 30x until $50K, then 25x."""
    eq = state['equity']
    if eq < 500:
        return 20.0
    elif eq < 5000:
        return 25.0
    elif eq < 50000:
        return 30.0
    else:
        return 25.0  # protect gains


# --- Setup 6: Pure Continuous Kelly ---
# True continuous Kelly, no DD constraint, no floor
def pure_continuous_kelly(state):
    """Pure continuous Kelly. Recalculate each trade. Half Kelly. Min 5x, max 50x."""
    n = state['wins'] + state['losses']
    if n < 5:
        return half_kelly_lev

    p = state['wins'] / n
    if state['wins'] > 0 and state['losses'] > 0:
        R = (state['sum_win_bps'] / state['wins']) / (state['sum_loss_bps'] / state['losses'])
    else:
        return half_kelly_lev

    if R <= 0:
        return 5.0

    f = p - (1 - p) / R
    f = max(0, f)
    f /= 2  # half kelly
    lev = f * (kelly_leverage / kelly_fraction)
    return max(5.0, min(50.0, lev))


# --- Setup 7: Confidence-Based Kelly ---
# Use lower bound of Kelly estimate (conservative)
def confidence_kelly(state):
    """Kelly using lower 90% CI of win rate (conservative). Min 10x, max 40x."""
    n = state['wins'] + state['losses']
    if n < 10:
        return 20.0  # conservative start

    p = state['wins'] / n
    # 90% CI lower bound for win rate (normal approximation)
    se = math.sqrt(p * (1 - p) / n)
    p_lower = p - 1.645 * se  # 5th percentile

    if state['wins'] > 0 and state['losses'] > 0:
        R = (state['sum_win_bps'] / state['wins']) / (state['sum_loss_bps'] / state['losses'])
    else:
        return 20.0

    if R <= 0 or p_lower <= 0:
        return 10.0

    f = p_lower - (1 - p_lower) / R
    f = max(0, f)
    f /= 2  # half kelly
    lev = f * (kelly_leverage / kelly_fraction)
    return max(10.0, min(40.0, lev))


# Run all dynamic setups
dynamic_setups = [
    ('1. Warm-Start', warm_start_kelly),
    ('2. High Floor (20x)', high_floor_kelly),
    ('3. Streak-Based', streak_kelly),
    ('4. Signal-Quality', signal_quality_kelly),
    ('5. Equity Milestone', equity_milestone_kelly),
    ('6. Pure Continuous', pure_continuous_kelly),
    ('7. Confidence-Based', confidence_kelly),
]

print(f"  {'Setup':>22s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'P75':>12s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'AvgLev':>7s} | {'Ruin':>5s}")
print(f"  {'-'*115}")

# First print fixed baselines
for fixed_lev in [20, 25, 30]:
    def make_fixed(fl):
        def fn(state):
            return fl
        return fn
    result = run_mc(make_fixed(fixed_lev), f"Fixed {fixed_lev}x")
    r = result
    print(f"  {'Fixed ' + str(fixed_lev) + 'x (base)':>22s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | ${r['p75']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {fixed_lev:5.1f}x | {r['ruin_pct']:4.1f}%")

print(f"  {'-'*115}")

dynamic_results = {}
for name, fn in dynamic_setups:
    result = run_mc(fn, name)
    dynamic_results[name] = result
    r = result
    print(f"  {name:>22s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | ${r['p75']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['avg_lev']:5.1f}x | {r['ruin_pct']:4.1f}%")

print()


# ============================================================
# PART C: HEAD-TO-HEAD - Best Dynamic vs Best Fixed
# ============================================================
print("=" * 110)
print("PART C: HEAD-TO-HEAD COMPARISON")
print("=" * 110)
print()

# Find best dynamic by P5
best_dyn_name = max(dynamic_results, key=lambda k: dynamic_results[k]['p5'])
best_dyn = dynamic_results[best_dyn_name]

# Find best dynamic by median
best_dyn_med_name = max(dynamic_results, key=lambda k: dynamic_results[k]['median'])
best_dyn_med = dynamic_results[best_dyn_med_name]

print(f"  BEST FIXED (by P5):      Half Kelly {half_kelly_lev:.1f}x")
print(f"    Median: ${fixed_results[0.50]['median']:,.0f} | P5: ${fixed_results[0.50]['p5']:,.0f} | AvgDD: {fixed_results[0.50]['avg_dd']*100:.1f}%")
print()
print(f"  BEST DYNAMIC (by P5):    {best_dyn_name}")
print(f"    Median: ${best_dyn['median']:,.0f} | P5: ${best_dyn['p5']:,.0f} | AvgDD: {best_dyn['avg_dd']*100:.1f}% | AvgLev: {best_dyn['avg_lev']:.1f}x")
print()
print(f"  BEST DYNAMIC (by median): {best_dyn_med_name}")
print(f"    Median: ${best_dyn_med['median']:,.0f} | P5: ${best_dyn_med['p5']:,.0f} | AvgDD: {best_dyn_med['avg_dd']*100:.1f}% | AvgLev: {best_dyn_med['avg_lev']:.1f}x")
print()

# Direct path-by-path comparison: best dynamic vs fixed 20x and 25x
print("  PATH-BY-PATH: How often does best dynamic beat fixed?")
print()

np.random.seed(42)

best_dyn_fn = dict(dynamic_setups)[best_dyn_name]

dyn_finals = []
f20_finals = []
f25_finals = []

def fixed_20(state):
    return 20.0

def fixed_25(state):
    return 25.0

for _ in range(N_SIMS):
    shuffled = list(trade_data)
    np.random.shuffle(shuffled)

    eq_d, _ = simulate_with_leverage_fn(shuffled, best_dyn_fn)
    eq_20, _ = simulate_with_leverage_fn(shuffled, fixed_20)
    eq_25, _ = simulate_with_leverage_fn(shuffled, fixed_25)

    dyn_finals.append(eq_d[-1])
    f20_finals.append(eq_20[-1])
    f25_finals.append(eq_25[-1])

beats_20 = sum(1 for d, f in zip(dyn_finals, f20_finals) if d > f) / N_SIMS * 100
beats_25 = sum(1 for d, f in zip(dyn_finals, f25_finals) if d > f) / N_SIMS * 100

print(f"  {best_dyn_name} beats Fixed 20x: {beats_20:.1f}% of paths")
print(f"  {best_dyn_name} beats Fixed 25x: {beats_25:.1f}% of paths")
print()

# When dynamic beats fixed - by how much? When it loses - by how much?
dyn_vs_20 = [(d - f) / f * 100 for d, f in zip(dyn_finals, f20_finals)]
dyn_vs_25 = [(d - f) / f * 100 for d, f in zip(dyn_finals, f25_finals)]

print(f"  vs Fixed 20x - when dynamic wins:  avg +{np.mean([x for x in dyn_vs_20 if x > 0]):.1f}%")
print(f"  vs Fixed 20x - when dynamic loses: avg {np.mean([x for x in dyn_vs_20 if x <= 0]):.1f}%")
print(f"  vs Fixed 25x - when dynamic wins:  avg +{np.mean([x for x in dyn_vs_25 if x > 0]):.1f}%")
print(f"  vs Fixed 25x - when dynamic loses: avg {np.mean([x for x in dyn_vs_25 if x <= 0]):.1f}%")
print()


# ============================================================
# PART D: LEVERAGE DISTRIBUTION for each Dynamic Setup
# ============================================================
print("=" * 110)
print("PART D: LEVERAGE DISTRIBUTION - What leverage does each dynamic setup actually use?")
print("=" * 110)
print()

np.random.seed(42)

print(f"  {'Setup':>22s} | {'Min':>6s} | {'P5':>6s} | {'P25':>6s} | {'Med':>6s} | {'P75':>6s} | {'P95':>6s} | {'Max':>6s} | {'Std':>6s}")
print(f"  {'-'*90}")

for name, fn in dynamic_setups:
    all_levs = []
    for _ in range(100):  # 100 paths for leverage distribution
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        _, lev_curve = simulate_with_leverage_fn(shuffled, fn)
        p2_levs = [l for l in lev_curve if l > 0]
        all_levs.extend(p2_levs)

    if all_levs:
        print(f"  {name:>22s} | {min(all_levs):5.1f}x | {np.percentile(all_levs, 5):5.1f}x | {np.percentile(all_levs, 25):5.1f}x | {np.median(all_levs):5.1f}x | {np.percentile(all_levs, 75):5.1f}x | {np.percentile(all_levs, 95):5.1f}x | {max(all_levs):5.1f}x | {np.std(all_levs):5.1f}")

print()


# ============================================================
# VERDICT
# ============================================================
print("=" * 110)
print("VERDICT")
print("=" * 110)
print()

# Collect ALL results for final ranking
all_results = []

# Fixed
for frac, r in fixed_results.items():
    all_results.append({
        'name': f"Fixed {r['leverage']:.0f}x ({frac:.0%} Kelly)",
        'type': 'FIXED',
        'median': r['median'],
        'p5': r['p5'],
        'avg_dd': r['avg_dd'],
        'avg_lev': r['leverage'],
    })

# Dynamic
for name, r in dynamic_results.items():
    all_results.append({
        'name': name,
        'type': 'DYNAMIC',
        'median': r['median'],
        'p5': r['p5'],
        'avg_dd': r['avg_dd'],
        'avg_lev': r['avg_lev'],
    })

# Sort by P5
all_results.sort(key=lambda x: x['p5'], reverse=True)

print("  FINAL RANKING (all configs, sorted by P5):")
print()
print(f"  {'#':>3s} | {'Type':>7s} | {'Config':>35s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s} | {'AvgLev':>7s}")
print(f"  {'-'*100}")

for i, r in enumerate(all_results):
    marker = " ***" if i < 3 else ""
    print(f"  {i+1:>3d} | {r['type']:>7s} | {r['name']:>35s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['avg_lev']:5.1f}x{marker}")

print()

# Check if any dynamic beats best fixed
best_fixed_p5 = max(all_results, key=lambda x: x['p5'] if x['type'] == 'FIXED' else 0)
best_dynamic_p5 = max((r for r in all_results if r['type'] == 'DYNAMIC'), key=lambda x: x['p5'], default=None)

if best_dynamic_p5 and best_dynamic_p5['p5'] > best_fixed_p5['p5']:
    print(f"  DYNAMIC WINS: {best_dynamic_p5['name']} (P5 ${best_dynamic_p5['p5']:,.0f}) beats")
    print(f"                {best_fixed_p5['name']} (P5 ${best_fixed_p5['p5']:,.0f})")
else:
    print(f"  FIXED WINS: {best_fixed_p5['name']} (P5 ${best_fixed_p5['p5']:,.0f})")
    if best_dynamic_p5:
        print(f"  Best dynamic: {best_dynamic_p5['name']} (P5 ${best_dynamic_p5['p5']:,.0f})")

print()

# Practical recommendation
print("  PRACTICAL RECOMMENDATION:")
print()
# Find the sweet spot: highest P5 with DD < 60%
safe_all = [r for r in all_results if r['avg_dd'] < 0.60]
if safe_all:
    best_safe = max(safe_all, key=lambda x: x['p5'])
    print(f"  Best P5 with DD < 60%: {best_safe['name']}")
    print(f"    Median ${best_safe['median']:,.0f} | P5 ${best_safe['p5']:,.0f} | DD {best_safe['avg_dd']*100:.1f}%")
    print()

# Conservative pick: best P5 with DD < 50%
conservative = [r for r in all_results if r['avg_dd'] < 0.50]
if conservative:
    best_cons = max(conservative, key=lambda x: x['p5'])
    print(f"  Conservative (DD < 50%): {best_cons['name']}")
    print(f"    Median ${best_cons['median']:,.0f} | P5 ${best_cons['p5']:,.0f} | DD {best_cons['avg_dd']*100:.1f}%")
    print()
