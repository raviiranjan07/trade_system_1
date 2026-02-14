"""L1-EXP-002: Dynamic Leverage Testing

QUESTION: Can dynamically adjusting leverage during Phase 2 beat fixed leverage?

DYNAMIC METHODS TESTED:
  1. Recalculate Kelly every N trades (N=5, 10, 20, 50)
  2. Different Kelly types for recalculation:
     a. Classic (binary): f = W - (1-W)/R  [O(1) per update]
     b. Mean-Variance: L = mu/var * 10000  [O(1) per update]
     (Generalized Kelly skipped in dynamic - same result as Classic but 100x slower)
  3. Different update rules:
     a. Immediate update (use all Phase 2 trades seen so far)
     b. Rolling window (only last N trades)
     c. Exponential decay (recent trades weighted more)
  4. Leverage bounds:
     a. No floor (min 5x)
     b. Floor 20x (never underbet)
     c. Floor at 0.33 Kelly (~16x, always aggressive enough)

FIXED BASELINES: 20x, 25x, 30x (from EXP-002 grid)
Phase 1: Cross margin, dynamic position, threshold $15
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)  # flush output in real time

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
        'position': position,
        'maint_margin': maint,
    })

returns = np.array([td['bps'] for td in trade_data])
win_rate = np.sum(returns > 0) / len(returns)
avg_win = np.mean(returns[returns > 0])
avg_loss = abs(np.mean(returns[returns <= 0]))
payoff_ratio = avg_win / avg_loss
mu = np.mean(returns)
sigma2 = np.var(returns)

# Pre-calculate Kelly benchmarks
def find_gen_kelly(rets):
    """Generalized Kelly: argmax E[log(1 + L*r/10000)]."""
    best_lev = 5.0
    best_g = -np.inf
    for lev_10 in range(10, 1101):  # 1x to 55x in 0.5 steps
        lev = lev_10 / 20
        factors = 1 + lev * rets / 10000
        if np.any(factors <= 0):
            break
        g = np.mean(np.log(factors))
        if g > best_g:
            best_g = g
            best_lev = lev
    return best_lev

FULL_KELLY_LEV = find_gen_kelly(returns)
HALF_KELLY_LEV = FULL_KELLY_LEV / 2

print("=" * 110)
print("L1-EXP-002: DYNAMIC LEVERAGE TESTING")
print("=" * 110)
print()
print(f"  V1.3.2: {len(returns)} trades, {win_rate*100:.1f}% win, payoff {payoff_ratio:.2f}")
print(f"  Mean: {mu:.2f} bps, Var: {sigma2:.1f}, Std: {np.sqrt(sigma2):.1f}")
print(f"  Generalized Kelly: {FULL_KELLY_LEV:.1f}x, Half: {HALF_KELLY_LEV:.1f}x")
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
# KELLY CALCULATION FUNCTIONS (3 types)
# ============================================================
def kelly_classic(rets):
    """Classic Kelly: f = W - (1-W)/R, converted to leverage."""
    w = np.sum(rets > 0) / len(rets) if len(rets) > 0 else 0.5
    w_ret = rets[rets > 0]
    l_ret = rets[rets <= 0]
    if len(w_ret) == 0 or len(l_ret) == 0:
        return HALF_KELLY_LEV
    R = np.mean(w_ret) / abs(np.mean(l_ret))
    f = w - (1 - w) / R
    if f <= 0:
        return 5.0
    # Calibrate: full Kelly fraction -> leverage using known mapping
    multiplier = FULL_KELLY_LEV / (win_rate - (1 - win_rate) / payoff_ratio)
    return f * multiplier / 2  # half Kelly


def kelly_mean_variance(rets):
    """Mean-Variance Kelly: L = mu / var * 10000."""
    if len(rets) < 5:
        return HALF_KELLY_LEV
    m = np.mean(rets)
    v = np.var(rets)
    if v <= 0 or m <= 0:
        return 5.0
    return (m / v * 10000) / 2  # half Kelly


# ============================================================
# DYNAMIC LEVERAGE SIMULATION
# ============================================================
def simulate_dynamic(trade_list, kelly_fn, update_freq=10, window=None,
                     decay=None, min_lev=5.0, max_lev=50.0, capital=STARTING_CAPITAL):
    """Dynamic leverage simulation.

    kelly_fn: function(returns_array) -> half-kelly leverage
    update_freq: recalculate every N Phase 2 trades
    window: if set, only use last N trades for Kelly calc (rolling window)
    decay: if set, exponential decay weight (0.95 = recent trades 5% more weight)
    min_lev/max_lev: leverage bounds
    """
    equity_curve = [capital]
    leverage_curve = []
    equity = capital
    phase2_returns = []
    current_lev = HALF_KELLY_LEV  # start at known half-Kelly
    p2_count = 0

    for td in trade_list:
        eq = equity

        if eq < PHASE1_THRESHOLD:
            # Phase 1: Cross margin
            pos = td['position']
            maint = td['maint_margin']
            margin_req = pos / MAX_LEVERAGE_SETTING
            if eq < margin_req:
                equity_curve.append(eq)
                leverage_curve.append(0)
                continue
            pnl = pos * (td['bps'] / 10000)
            max_loss = eq - maint
            if pnl < -max_loss:
                equity = 0.01
            else:
                equity = max(eq + pnl, 0.01)
            equity_curve.append(equity)
            leverage_curve.append(0)
            continue

        # Phase 2: dynamic leverage
        leverage_curve.append(current_lev)
        pos = eq * current_lev
        pnl = pos * (td['bps'] / 10000)
        equity = max(eq + pnl, 0.01)
        equity_curve.append(equity)

        # Record and update
        phase2_returns.append(td['bps'])
        p2_count += 1

        # Recalculate Kelly at specified frequency
        if p2_count % update_freq == 0 and len(phase2_returns) >= 10:
            if window and len(phase2_returns) > window:
                calc_rets = np.array(phase2_returns[-window:])
            elif decay:
                n = len(phase2_returns)
                weights = np.array([decay ** (n - 1 - i) for i in range(n)])
                # Weighted returns for mean/var
                rets_arr = np.array(phase2_returns)
                w_sum = np.sum(weights)
                w_mean = np.sum(weights * rets_arr) / w_sum
                w_var = np.sum(weights * (rets_arr - w_mean) ** 2) / w_sum
                if w_var > 0 and w_mean > 0:
                    current_lev = max(min_lev, min(max_lev, (w_mean / w_var * 10000) / 2))
                continue
            else:
                calc_rets = np.array(phase2_returns)

            new_lev = kelly_fn(calc_rets)
            current_lev = max(min_lev, min(max_lev, new_lev))

    return equity_curve, leverage_curve


def run_dynamic_mc(kelly_fn, update_freq=10, window=None, decay=None,
                   min_lev=5.0, max_lev=50.0, n_sims=N_SIMS):
    """MC simulation with dynamic leverage."""
    finals = []
    max_dds = []
    avg_levs = []
    ruined = 0

    for _ in range(n_sims):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        eq_curve, lev_curve = simulate_dynamic(
            shuffled, kelly_fn, update_freq, window, decay, min_lev, max_lev
        )
        finals.append(eq_curve[-1])
        max_dds.append(calc_max_dd(eq_curve))
        p2_levs = [l for l in lev_curve if l > 0]
        if p2_levs:
            avg_levs.append(np.mean(p2_levs))
        if eq_curve[-1] < 1.0:
            ruined += 1

    return {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'avg_lev': np.mean(avg_levs) if avg_levs else 0,
        'ruin_pct': ruined / n_sims * 100,
    }


def simulate_fixed(trade_list, fixed_lev, capital=STARTING_CAPITAL):
    equity_curve = [capital]
    equity = capital
    for td in trade_list:
        eq = equity
        if eq < PHASE1_THRESHOLD:
            pos = td['position']
            maint = td['maint_margin']
            margin_req = pos / MAX_LEVERAGE_SETTING
            if eq < margin_req:
                equity_curve.append(eq)
                continue
            pnl = pos * (td['bps'] / 10000)
            max_loss = eq - maint
            if pnl < -max_loss:
                equity = 0.01
            else:
                equity = max(eq + pnl, 0.01)
        else:
            pos = eq * fixed_lev
            pnl = pos * (td['bps'] / 10000)
            equity = max(eq + pnl, 0.01)
        equity_curve.append(equity)
    return equity_curve


def run_fixed_mc(fixed_lev, n_sims=N_SIMS):
    finals = []
    max_dds = []
    ruined = 0
    for _ in range(n_sims):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        eq = simulate_fixed(shuffled, fixed_lev)
        finals.append(eq[-1])
        max_dds.append(calc_max_dd(eq))
        if eq[-1] < 1.0:
            ruined += 1
    return {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'avg_lev': fixed_lev,
        'ruin_pct': ruined / n_sims * 100,
    }


# ============================================================
# PART 1: FIXED BASELINES
# ============================================================
print("=" * 110)
print("PART 1: FIXED LEVERAGE BASELINES")
print("=" * 110)
print()

np.random.seed(42)

print(f"  {'Config':>25s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'AvgDD':>7s} | {'AvgLev':>7s} | {'Ruin':>5s}")
print(f"  {'-'*90}")

fixed_baselines = {}
for lev in [15, 20, 25, 30, 35]:
    result = run_fixed_mc(lev)
    fixed_baselines[lev] = result
    r = result
    marker = ""
    if lev == 20:
        marker = " <- conservative"
    elif lev == 25:
        marker = " <- half Kelly"
    print(f"  {'Fixed ' + str(lev) + 'x':>25s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['avg_lev']:5.1f}x | {r['ruin_pct']:4.1f}%{marker}")

print()


# ============================================================
# PART 2: DYNAMIC - Kelly Type Comparison
# ============================================================
print("=" * 110)
print("PART 2: DYNAMIC LEVERAGE - Which Kelly type for recalculation?")
print("=" * 110)
print()
print("  Update every 10 trades, all Phase 2 history, floor 5x, cap 50x")
print()

np.random.seed(42)

print(f"  {'Kelly Type':>25s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'AvgDD':>7s} | {'AvgLev':>7s} | {'Ruin':>5s}")
print(f"  {'-'*90}")

kelly_fns = [
    ('Classic (Binary)', kelly_classic),
    ('Mean-Variance', kelly_mean_variance),
]

type_results = {}
for name, fn in kelly_fns:
    result = run_dynamic_mc(fn, update_freq=10)
    type_results[name] = result
    r = result
    print(f"  {name:>25s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['avg_lev']:5.1f}x | {r['ruin_pct']:4.1f}%")

print()


# ============================================================
# PART 3: DYNAMIC - Update Frequency
# ============================================================
print("=" * 110)
print("PART 3: DYNAMIC LEVERAGE - How often to recalculate?")
print("=" * 110)
print()

# Use best Kelly type from Part 2
best_type_name = max(type_results, key=lambda k: type_results[k]['p5'])
best_type_fn = dict(kelly_fns)[best_type_name]
print(f"  Using: {best_type_name} (best P5 from Part 2)")
print()

np.random.seed(42)

print(f"  {'Update Freq':>15s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'AvgDD':>7s} | {'AvgLev':>7s} | {'Ruin':>5s}")
print(f"  {'-'*85}")

freq_results = {}
for freq in [1, 3, 5, 10, 20, 50, 100]:
    result = run_dynamic_mc(best_type_fn, update_freq=freq)
    freq_results[freq] = result
    r = result
    print(f"  {'Every ' + str(freq) + ' trades':>15s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['avg_lev']:5.1f}x | {r['ruin_pct']:4.1f}%")

# Also test "never update" = just start at half-Kelly and keep it
print(f"  {'Never (fixed)':>15s} | ${fixed_baselines[25]['median']:>10,.0f} | ${fixed_baselines[25]['p5']:>10,.0f} | ${fixed_baselines[25]['p25']:>10,.0f} | {fixed_baselines[25]['avg_dd']*100:5.1f}% | {25:5.1f}x | {fixed_baselines[25]['ruin_pct']:4.1f}%")
print()


# ============================================================
# PART 4: DYNAMIC - Rolling Window vs Full History vs Decay
# ============================================================
print("=" * 110)
print("PART 4: DYNAMIC LEVERAGE - Data window for Kelly calculation")
print("=" * 110)
print()

# Use best frequency from Part 3
best_freq = max(freq_results, key=lambda k: freq_results[k]['p5'])
print(f"  Using: {best_type_name}, update every {best_freq} trades")
print()

np.random.seed(42)

print(f"  {'Window':>25s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'AvgDD':>7s} | {'AvgLev':>7s} | {'Ruin':>5s}")
print(f"  {'-'*95}")

window_results = {}

# Full history (no window)
result = run_dynamic_mc(best_type_fn, update_freq=best_freq, window=None)
window_results['Full history'] = result
r = result
print(f"  {'Full history':>25s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['avg_lev']:5.1f}x | {r['ruin_pct']:4.1f}%")

# Rolling windows
for win in [20, 30, 50, 80, 100]:
    result = run_dynamic_mc(best_type_fn, update_freq=best_freq, window=win)
    window_results[f'Rolling {win}'] = result
    r = result
    print(f"  {'Rolling ' + str(win) + ' trades':>25s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['avg_lev']:5.1f}x | {r['ruin_pct']:4.1f}%")

# Exponential decay
for decay in [0.98, 0.95, 0.90]:
    result = run_dynamic_mc(best_type_fn, update_freq=best_freq, decay=decay)
    window_results[f'Decay {decay}'] = result
    r = result
    print(f"  {'Decay ' + str(decay):>25s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['avg_lev']:5.1f}x | {r['ruin_pct']:4.1f}%")

print()


# ============================================================
# PART 5: DYNAMIC - Leverage Floor Impact
# ============================================================
print("=" * 110)
print("PART 5: DYNAMIC LEVERAGE - Minimum leverage floor")
print("=" * 110)
print()

# Use best configs from above
best_window_name = max(window_results, key=lambda k: window_results[k]['p5'])
best_window = None
best_decay = None
if 'Rolling' in best_window_name:
    best_window = int(best_window_name.split()[-1])
elif 'Decay' in best_window_name:
    best_decay = float(best_window_name.split()[-1])

print(f"  Using: {best_type_name}, every {best_freq} trades, {best_window_name}")
print()

np.random.seed(42)

print(f"  {'Floor':>15s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'AvgDD':>7s} | {'AvgLev':>7s} | {'Ruin':>5s}")
print(f"  {'-'*85}")

floor_results = {}
for floor in [5, 10, 15, 18, 20, 22, 25]:
    result = run_dynamic_mc(best_type_fn, update_freq=best_freq,
                            window=best_window, decay=best_decay,
                            min_lev=floor)
    floor_results[floor] = result
    r = result
    print(f"  {'Min ' + str(floor) + 'x':>15s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['avg_lev']:5.1f}x | {r['ruin_pct']:4.1f}%")

print()


# ============================================================
# PART 6: BEST DYNAMIC vs FIXED - FINAL COMPARISON
# ============================================================
print("=" * 110)
print("PART 6: FINAL COMPARISON - Best Dynamic vs Fixed")
print("=" * 110)
print()

# Find best dynamic overall
best_floor = max(floor_results, key=lambda k: floor_results[k]['p5'])
best_dynamic = floor_results[best_floor]

print(f"  BEST DYNAMIC CONFIG:")
print(f"    Kelly type: {best_type_name}")
print(f"    Update freq: every {best_freq} trades")
print(f"    Data window: {best_window_name}")
print(f"    Min leverage: {best_floor}x")
print(f"    Avg leverage: {best_dynamic['avg_lev']:.1f}x")
print()

print(f"  {'Config':>30s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s} | {'AvgLev':>7s}")
print(f"  {'-'*75}")

for lev in [20, 25, 30]:
    r = fixed_baselines[lev]
    print(f"  {'Fixed ' + str(lev) + 'x':>30s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['avg_lev']:5.1f}x")

r = best_dynamic
label = f"Dynamic (best)"
print(f"  {label:>30s} | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['avg_lev']:5.1f}x")

print()

# Path-by-path comparison
np.random.seed(42)

dyn_finals = []
f20_finals = []
f25_finals = []

for _ in range(N_SIMS):
    shuffled = list(trade_data)
    np.random.shuffle(shuffled)

    eq_d, _ = simulate_dynamic(shuffled, best_type_fn, update_freq=best_freq,
                                window=best_window, decay=best_decay,
                                min_lev=best_floor)
    eq_20 = simulate_fixed(shuffled, 20)
    eq_25 = simulate_fixed(shuffled, 25)

    dyn_finals.append(eq_d[-1])
    f20_finals.append(eq_20[-1])
    f25_finals.append(eq_25[-1])

beats_20 = sum(1 for d, f in zip(dyn_finals, f20_finals) if d > f) / N_SIMS * 100
beats_25 = sum(1 for d, f in zip(dyn_finals, f25_finals) if d > f) / N_SIMS * 100

print(f"  Dynamic beats Fixed 20x: {beats_20:.1f}% of paths")
print(f"  Dynamic beats Fixed 25x: {beats_25:.1f}% of paths")
print()

if beats_25 > 50:
    print("  VERDICT: Dynamic leverage WINS over Fixed 25x")
elif beats_20 > 50:
    print("  VERDICT: Dynamic beats 20x but not 25x - use Fixed 25x instead")
else:
    print("  VERDICT: Fixed leverage WINS - dynamic adds no value")
print()
