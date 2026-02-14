"""L1-EXP-004 Part 2: Kelly Algorithm TYPES Comparison

QUESTION: Which Kelly algorithm gives the best optimal leverage for V1.3.2?

TYPES TESTED:
  1. Classic Kelly (Binary): f = W - (1-W)/R -> leverage conversion
  2. Mean-Variance Kelly (Gaussian): L = mu / sigma^2 * 10000
  3. Generalized Kelly (Numerical): argmax E[log(1 + L*r/10000)]
  4. Empirical Kelly (Bootstrap): median of 1000 bootstrap Kelly leverages
  5. Robust Kelly (Conservative): use lower 90% CI of win rate + payoff ratio

For EACH type:
  - Calculate optimal leverage (full Kelly)
  - Calculate half-Kelly leverage
  - MC simulation at both full and half
  - Compare which type's leverage gives best MC P5
"""
import sys
sys.path.insert(0, "src")

import math
import numpy as np
from scipy import stats as scipy_stats
from v12.backtest import run_backtest
from v12.config.loader import load_config

# ============================================================
# CONSTANTS
# ============================================================
STARTING_CAPITAL = 10.0
MAX_LEVERAGE_SETTING = 125
N_SIMS = 1000
N_BOOTSTRAP = 1000
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
        'position': position,
        'maint_margin': maint,
    })

returns = np.array([td['bps'] for td in trade_data])
wins = returns[returns > 0]
losses = returns[returns <= 0]
win_rate = len(wins) / len(returns)
avg_win = np.mean(wins)
avg_loss = abs(np.mean(losses))
payoff_ratio = avg_win / avg_loss
mu = np.mean(returns)
sigma2 = np.var(returns)

print("=" * 110)
print("L1-EXP-004 Part 2: KELLY ALGORITHM TYPES COMPARISON")
print("=" * 110)
print()
print(f"  V1.3.2: {len(returns)} trades, {win_rate*100:.1f}% win, payoff {payoff_ratio:.2f}")
print(f"  Mean: {mu:.2f} bps, Variance: {sigma2:.1f} bps^2, Std: {np.sqrt(sigma2):.1f} bps")
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


def simulate_fixed(trade_list, fixed_lev, capital=STARTING_CAPITAL):
    """Phase 1 (cross) + Phase 2 (fixed leverage)."""
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


def run_mc(fixed_lev, n_sims=N_SIMS):
    """MC simulation at fixed leverage."""
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
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'ruin_pct': ruined / n_sims * 100,
    }


def find_generalized_kelly(rets, max_lev=60.0, step=0.5):
    """Numerical: argmax E[log(1 + L * r / 10000)]."""
    best_lev = 1.0
    best_g = -np.inf
    for lev_int in range(int(1 / step), int(max_lev / step) + 1):
        lev = lev_int * step
        factors = 1 + lev * rets / 10000
        if np.any(factors <= 0):
            break
        g = np.mean(np.log(factors))
        if g > best_g:
            best_g = g
            best_lev = lev
    return best_lev, best_g


# ============================================================
# PART 1: Calculate Optimal Leverage for Each Kelly Type
# ============================================================
print("=" * 110)
print("PART 1: OPTIMAL LEVERAGE BY KELLY TYPE")
print("=" * 110)
print()

kelly_types = {}

# --- Type 1: Classic Kelly (Binary) ---
f_classic = win_rate - (1 - win_rate) / payoff_ratio
# Convert fraction to leverage: calibrate against generalized
gen_lev, gen_g = find_generalized_kelly(returns)
classic_multiplier = gen_lev / f_classic if f_classic > 0 else 100
classic_lev = f_classic * classic_multiplier

kelly_types['1. Classic (Binary)'] = {
    'formula': f'f = W - (1-W)/R = {win_rate:.3f} - {1-win_rate:.3f}/{payoff_ratio:.2f} = {f_classic:.4f}',
    'full_lev': classic_lev,
    'half_lev': classic_lev / 2,
    'fraction': f_classic,
}

# --- Type 2: Mean-Variance Kelly (Gaussian) ---
mv_lev = mu / sigma2 * 10000
kelly_types['2. Mean-Variance (Gaussian)'] = {
    'formula': f'L = mu/sigma^2 * 10000 = {mu:.2f}/{sigma2:.1f} * 10000 = {mv_lev:.1f}',
    'full_lev': mv_lev,
    'half_lev': mv_lev / 2,
    'fraction': mu / np.sqrt(sigma2),  # Sharpe-like
}

# --- Type 3: Generalized Kelly (Numerical) ---
kelly_types['3. Generalized (Numerical)'] = {
    'formula': f'argmax E[log(1 + L*r/10000)] = {gen_lev:.1f}x (G={gen_g:.6f})',
    'full_lev': gen_lev,
    'half_lev': gen_lev / 2,
    'fraction': gen_g,
}

# --- Type 4: Empirical Kelly (Bootstrap) ---
np.random.seed(42)
bootstrap_levs = []
for _ in range(N_BOOTSTRAP):
    sample = np.random.choice(returns, size=len(returns), replace=True)
    blev, _ = find_generalized_kelly(sample)
    bootstrap_levs.append(blev)

bootstrap_levs = np.array(bootstrap_levs)
emp_lev = np.median(bootstrap_levs)
emp_ci_low = np.percentile(bootstrap_levs, 5)
emp_ci_high = np.percentile(bootstrap_levs, 95)

kelly_types['4. Empirical (Bootstrap)'] = {
    'formula': f'Median of 1000 bootstrap: {emp_lev:.1f}x [90% CI: {emp_ci_low:.1f}-{emp_ci_high:.1f}]',
    'full_lev': emp_lev,
    'half_lev': emp_lev / 2,
    'fraction': 0,
    'ci_low': emp_ci_low,
    'ci_high': emp_ci_high,
}

# --- Type 5: Robust Kelly (Conservative) ---
# Use lower 90% CI of win rate and lower CI of payoff ratio
n = len(returns)
se_wr = math.sqrt(win_rate * (1 - win_rate) / n)
wr_lower = win_rate - 1.645 * se_wr  # 5th percentile

# Bootstrap payoff ratio CI
np.random.seed(42)
boot_payoffs = []
for _ in range(N_BOOTSTRAP):
    sample = np.random.choice(returns, size=len(returns), replace=True)
    bwins = sample[sample > 0]
    blosses = sample[sample <= 0]
    if len(bwins) > 0 and len(blosses) > 0:
        boot_payoffs.append(np.mean(bwins) / abs(np.mean(blosses)))
boot_payoffs = np.array(boot_payoffs)
payoff_lower = np.percentile(boot_payoffs, 5)

f_robust = wr_lower - (1 - wr_lower) / payoff_lower
robust_lev = f_robust * classic_multiplier if f_robust > 0 else 5.0

kelly_types['5. Robust (Conservative)'] = {
    'formula': f'Lower CI: WR={wr_lower:.3f}, R={payoff_lower:.2f} -> f={f_robust:.4f}',
    'full_lev': robust_lev,
    'half_lev': robust_lev / 2,
    'fraction': f_robust,
    'wr_lower': wr_lower,
    'payoff_lower': payoff_lower,
}

# Print all types
print(f"  {'Kelly Type':>30s} | {'Formula/Method':>60s}")
print(f"  {'-'*95}")
for name, info in kelly_types.items():
    print(f"  {name:>30s} | {info['formula']}")

print()
print(f"  {'Kelly Type':>30s} | {'Full Optimal':>12s} | {'Half Kelly':>12s}")
print(f"  {'-'*60}")
for name, info in kelly_types.items():
    print(f"  {name:>30s} | {info['full_lev']:>10.1f}x | {info['half_lev']:>10.1f}x")

print()


# ============================================================
# PART 2: MC Simulation for Each Kelly Type's Optimal Leverage
# ============================================================
print("=" * 110)
print("PART 2: MC SIMULATION - Each Kelly Type at Full and Half Leverage")
print("=" * 110)
print()

np.random.seed(42)

print(f"  {'Kelly Type':>30s} | {'Mode':>5s} | {'Lev':>6s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'Ruin':>5s}")
print(f"  {'-'*115}")

mc_results = {}
for name, info in kelly_types.items():
    for mode, lev in [('Full', info['full_lev']), ('Half', info['half_lev'])]:
        # Clamp leverage to reasonable range
        lev_clamped = max(5, min(55, lev))
        key = (name, mode)

        result = run_mc(lev_clamped)
        result['leverage'] = lev_clamped
        result['original_lev'] = lev
        mc_results[key] = result

        r = result
        print(f"  {name:>30s} | {mode:>5s} | {lev_clamped:>5.1f}x | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['ruin_pct']:4.1f}%")
    print()

# Also add bootstrap CI leverages
print("  --- Bootstrap Kelly CI boundaries ---")
for mode, lev in [('P5 (conservative)', emp_ci_low), ('Median', emp_lev), ('P95 (aggressive)', emp_ci_high)]:
    lev_clamped = max(5, min(55, lev))
    result = run_mc(lev_clamped)
    r = result
    print(f"  {'4. Bootstrap ' + mode:>30s} |       | {lev_clamped:>5.1f}x | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | ${r['p25']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['ruin_pct']:4.1f}%")

print()


# ============================================================
# PART 3: Fine-Grained Sweep Around Each Type's Optimal
# ============================================================
print("=" * 110)
print("PART 3: FINE-GRAINED SWEEP - +/- 5x around each type's half-Kelly")
print("=" * 110)
print()

np.random.seed(42)

for name, info in kelly_types.items():
    center = info['half_lev']
    print(f"  --- {name} (half-Kelly center: {center:.1f}x) ---")
    print(f"  {'Leverage':>10s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s} | {'Ruin':>5s}")
    print(f"  {'-'*55}")

    sweep_start = max(5, center - 5)
    sweep_end = min(55, center + 5)

    best_p5 = 0
    best_lev = center

    for lev_10 in range(int(sweep_start * 2), int(sweep_end * 2) + 1, 1):
        lev = lev_10 / 2
        result = run_mc(lev)
        r = result

        marker = " <--" if abs(lev - center) < 0.3 else ""
        if r['p5'] > best_p5:
            best_p5 = r['p5']
            best_lev = lev
            if not marker:
                marker = " *"

        print(f"  {lev:>8.1f}x | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:4.1f}%{marker}")

    print(f"  Best P5 in sweep: {best_lev:.1f}x (P5 ${best_p5:,.0f})")
    print()


# ============================================================
# PART 4: GROWTH RATE COMPARISON
# ============================================================
print("=" * 110)
print("PART 4: THEORETICAL GROWTH RATE - G(L) for each Kelly type")
print("=" * 110)
print()
print("  G(L) = E[log(1 + L * r / 10000)] = per-trade geometric growth rate")
print("  Higher G(L) = faster compounding")
print()

print(f"  {'Kelly Type':>30s} | {'Leverage':>8s} | {'G(L)':>12s} | {'Geo Mean':>10s} | {'E[final]':>12s}")
print(f"  {'-'*85}")

for name, info in kelly_types.items():
    for mode, lev in [('Full', info['full_lev']), ('Half', info['half_lev'])]:
        lev_c = max(5, min(55, lev))
        factors = 1 + lev_c * returns / 10000
        valid = factors > 0
        if all(valid):
            g = np.mean(np.log(factors))
            geo_mean = np.exp(g)
            expected_final = STARTING_CAPITAL * np.exp(len(returns) * g)
        else:
            g = -99
            geo_mean = 0
            expected_final = 0

        label = f"{name} ({mode})"
        print(f"  {label:>30s} | {lev_c:>6.1f}x | {g:>11.6f} | {geo_mean:>9.6f}x | ${expected_final:>10,.0f}")

print()


# ============================================================
# VERDICT
# ============================================================
print("=" * 110)
print("VERDICT: WHICH KELLY TYPE IS BEST?")
print("=" * 110)
print()

# Rank all half-Kelly results by P5
half_results = [(name, mc_results[(name, 'Half')]) for name in kelly_types]
half_results.sort(key=lambda x: x[1]['p5'], reverse=True)

print("  RANKING BY P5 (Half Kelly):")
print(f"  {'#':>3s} | {'Kelly Type':>30s} | {'Leverage':>8s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s}")
print(f"  {'-'*80}")

for i, (name, r) in enumerate(half_results):
    print(f"  {i+1:>3d} | {name:>30s} | {r['leverage']:>6.1f}x | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | {r['avg_dd']*100:5.1f}%")

print()

# Also rank full Kelly by P5
full_results = [(name, mc_results[(name, 'Full')]) for name in kelly_types]
full_results.sort(key=lambda x: x[1]['p5'], reverse=True)

print("  RANKING BY P5 (Full Kelly):")
print(f"  {'#':>3s} | {'Kelly Type':>30s} | {'Leverage':>8s} | {'Median':>12s} | {'P5':>12s} | {'AvgDD':>7s}")
print(f"  {'-'*80}")

for i, (name, r) in enumerate(full_results):
    print(f"  {i+1:>3d} | {name:>30s} | {r['leverage']:>6.1f}x | ${r['median']:>10,.0f} | ${r['p5']:>10,.0f} | {r['avg_dd']*100:5.1f}%")

print()

# Summary
best_half = half_results[0]
best_full = full_results[0]

print(f"  BEST HALF KELLY: {best_half[0]}")
print(f"    Leverage: {best_half[1]['leverage']:.1f}x | Median: ${best_half[1]['median']:,.0f} | P5: ${best_half[1]['p5']:,.0f} | DD: {best_half[1]['avg_dd']*100:.1f}%")
print()
print(f"  BEST FULL KELLY: {best_full[0]}")
print(f"    Leverage: {best_full[1]['leverage']:.1f}x | Median: ${best_full[1]['median']:,.0f} | P5: ${best_full[1]['p5']:,.0f} | DD: {best_full[1]['avg_dd']*100:.1f}%")
print()

# Key insight: do different types give meaningfully different results?
half_levs = [kelly_types[name]['half_lev'] for name in kelly_types]
print(f"  LEVERAGE SPREAD ACROSS TYPES:")
print(f"    Range: {min(half_levs):.1f}x to {max(half_levs):.1f}x")
print(f"    Spread: {max(half_levs) - min(half_levs):.1f}x")
if max(half_levs) - min(half_levs) < 5:
    print(f"    -> SMALL spread: all Kelly types agree on ~{np.mean(half_levs):.0f}x")
else:
    print(f"    -> LARGE spread: Kelly type matters!")
print()
