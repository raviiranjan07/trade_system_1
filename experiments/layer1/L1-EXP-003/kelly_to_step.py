"""L1-EXP-003: Kelly Criterion -> $/step Conversion

QUESTION: Which Kelly algorithm gives the best $/step for position sizing?

KELLY TYPES TESTED:
  1. Classic Kelly:    f = p - q/b  (binary: win rate + payoff ratio)
  2. Mean-Variance:   f = mean / variance  (uses full return distribution)
  3. Geometric:       f = argmax(E[log(1 + f*r)])  (maximize log growth)
  4. Continuous:      f = (mean - rf) / variance  (continuous returns, rf=0)
  5. Empirical:       Brute-force: which $/step maximizes geometric mean?

PROCESS:
  1. Calculate Kelly fraction from V1.3.2 trade stats
  2. Convert fraction -> $/step: step = wallet / (fraction * wallet / (0.001 * avg_btc_price))
  3. Test full Kelly, half-Kelly, quarter-Kelly
  4. MC validate each
  5. Compare to EXP-002 brute-force results ($2.00-$2.50/step)

SETUP:
  - Binance: 125x leverage (fixed), cross margin
  - Starting wallet: $10
  - Position: multiples of 0.001 BTC
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import math
import numpy as np
from engine.backtest import run_backtest
from engine.config.loader import load_config

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

returns_bps = [td['bps'] for td in trade_data]
returns_frac = [td['bps'] / 10000 for td in trade_data]  # as fraction (0.01 = 1%)
wins_bps = [r for r in returns_bps if r > 0]
losses_bps = [r for r in returns_bps if r <= 0]
avg_btc = np.mean([td['btc_price'] for td in trade_data])

print("=" * 100)
print("L1-EXP-003: KELLY CRITERION -> $/STEP CONVERSION")
print("=" * 100)
print(f"  V1.3.2: {len(returns_bps)} trades, {len(wins_bps)/len(returns_bps)*100:.1f}% win")
print(f"  Avg win: +{np.mean(wins_bps):.1f} bps | Avg loss: {np.mean(losses_bps):.1f} bps")
print(f"  Mean return: {np.mean(returns_bps):+.2f} bps | Std: {np.std(returns_bps):.2f} bps")
print(f"  Avg BTC price: ${avg_btc:,.0f}")
print(f"  Wallet: $10 | Leverage: 125x (fixed) | Cross margin")
print()


# ============================================================
# HELPER: Convert Kelly fraction to $/step
# ============================================================
def kelly_fraction_to_step(fraction, wallet=STARTING_CAPITAL, avg_price=avg_btc):
    """Convert Kelly fraction to $/step.

    Kelly fraction = what fraction of wallet to risk per trade.
    Risk per trade = position * avg_loss_fraction.
    Position = qty * btc_price.
    qty = steps * 0.001.

    fraction * wallet = position * avg_loss_frac
    position = fraction * wallet / avg_loss_frac
    qty = position / avg_price
    steps = qty / 0.001
    $/step = wallet / steps

    But we can simplify: $/step = wallet / steps
    where steps = qty / 0.001 = (fraction * wallet / avg_loss_frac) / avg_price / 0.001
    """
    avg_loss_frac = abs(np.mean(losses_bps)) / 10000  # avg loss as fraction
    position = fraction * wallet / avg_loss_frac
    qty = position / avg_price
    steps = qty / BINANCE_STEP_SIZE
    if steps < 1:
        return wallet  # minimum 1 step
    dollar_per_step = wallet / steps
    return dollar_per_step


# ============================================================
# PART 1: KELLY TYPE CALCULATIONS
# ============================================================
print("=" * 100)
print("PART 1: KELLY FRACTION CALCULATIONS")
print("=" * 100)
print()

kelly_results = {}

# --- Type 1: Classic Kelly ---
# f = p - q/b where p=win_rate, q=1-p, b=avg_win/avg_loss
p = len(wins_bps) / len(returns_bps)
q = 1 - p
b = abs(np.mean(wins_bps)) / abs(np.mean(losses_bps))  # payoff ratio
f_classic = p - q / b
kelly_results['Classic'] = f_classic
print(f"  1. CLASSIC KELLY: f = p - q/b")
print(f"     p={p:.3f}, q={q:.3f}, b={b:.2f}")
print(f"     f = {p:.3f} - {q:.3f}/{b:.2f} = {f_classic:.4f}")
print()

# --- Type 2: Mean-Variance Kelly ---
# f = mean / variance (of returns as fractions)
mean_r = np.mean(returns_frac)
var_r = np.var(returns_frac)
f_meanvar = mean_r / var_r
kelly_results['Mean-Variance'] = f_meanvar
print(f"  2. MEAN-VARIANCE KELLY: f = mean / variance")
print(f"     mean={mean_r:.6f}, var={var_r:.6f}")
print(f"     f = {f_meanvar:.4f}")
print()

# --- Type 3: Geometric Kelly ---
# f = argmax E[log(1 + f*r)] -- numerical optimization
# Search over f from 0 to 2 in small steps
best_geo_f = 0
best_geo_growth = -999
for f_test in np.arange(0.01, 3.0, 0.01):
    log_growth = np.mean([np.log(max(1 + f_test * r, 1e-10)) for r in returns_frac])
    if log_growth > best_geo_growth:
        best_geo_growth = log_growth
        best_geo_f = f_test
f_geometric = best_geo_f
kelly_results['Geometric'] = f_geometric
print(f"  3. GEOMETRIC KELLY: f = argmax E[log(1 + f*r)]")
print(f"     Searched f in [0.01, 3.0], step 0.01")
print(f"     f = {f_geometric:.4f} (log growth = {best_geo_growth:.6f})")
print()

# --- Type 4: Continuous Kelly ---
# f = (mean - rf) / variance, where rf = risk-free rate = 0
f_continuous = (mean_r - 0) / var_r
kelly_results['Continuous'] = f_continuous
print(f"  4. CONTINUOUS KELLY: f = (mean - rf) / variance")
print(f"     Same as Mean-Variance when rf=0")
print(f"     f = {f_continuous:.4f}")
print()

# --- Type 5: Empirical Kelly ---
# Brute-force: which $/step maximizes geometric mean of MC paths?
# (This is done in Part 3 below)
print(f"  5. EMPIRICAL KELLY: Brute-force $/step that maximizes geometric mean")
print(f"     (Calculated in Part 3)")
print()


# ============================================================
# PART 2: CONVERT KELLY FRACTIONS TO $/STEP
# ============================================================
print("=" * 100)
print("PART 2: KELLY FRACTION -> $/STEP CONVERSION")
print("=" * 100)
print()

print(f"  {'Kelly Type':>15s} | {'Fraction':>10s} | {'Full $/step':>12s} | {'Half $/step':>12s} | {'Quarter $/step':>14s}")
print(f"  {'-'*75}")

conversion_results = {}
for name, frac in kelly_results.items():
    full_step = kelly_fraction_to_step(frac)
    half_step = kelly_fraction_to_step(frac / 2)
    quarter_step = kelly_fraction_to_step(frac / 4)

    conversion_results[name] = {
        'fraction': frac,
        'full': full_step,
        'half': half_step,
        'quarter': quarter_step,
    }

    print(f"  {name:>15s} | {frac:>10.4f} | ${full_step:>10.2f} | ${half_step:>10.2f} | ${quarter_step:>12.2f}")

print()
print(f"  EXP-002 brute-force optimal: $2.00/step (aggressive), $2.50/step (conservative)")
print(f"  Which Kelly type matches?")
print()


# ============================================================
# PART 3: MC VALIDATION — Test each Kelly-derived $/step
# ============================================================
print("=" * 100)
print("PART 3: MC VALIDATION (1000 paths per config)")
print("=" * 100)
print()


def simulate_scaling_qty(trade_list, dollars_per_step, capital=STARTING_CAPITAL):
    """Scale BTC qty with wallet size."""
    equity = [capital]
    skipped = 0
    liquidated = 0

    for td in trade_list:
        eq = equity[-1]
        if eq <= 0.01:
            equity.append(0.01)
            continue

        steps = max(1, int(eq / dollars_per_step))
        qty = steps * BINANCE_STEP_SIZE
        qty = max(qty, td['qty_min'])

        position = qty * td['btc_price']
        margin = position / LEVERAGE
        maint = position * MAINT_MARGIN_RATE

        if eq < margin:
            equity.append(eq)
            skipped += 1
            continue

        pnl = position * (td['bps'] / 10000)
        max_loss = eq - maint
        if pnl < -max_loss:
            equity.append(0.01)
            liquidated += 1
        else:
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


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


def run_mc(step_val, n_sims=N_SIMS, seed=42):
    np.random.seed(seed)
    finals = []
    max_dds = []
    ruin_count = 0

    for _ in range(n_sims):
        shuffled = list(trade_data)
        np.random.shuffle(shuffled)
        eq, _, liq = simulate_scaling_qty(shuffled, step_val)
        finals.append(eq[-1])
        max_dds.append(calc_max_dd(eq))
        if eq[-1] < 1.0:
            ruin_count += 1

    return {
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'geo_mean': np.exp(np.mean(np.log(np.maximum(finals, 0.01)))),
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'ruin_pct': ruin_count / n_sims * 100,
    }


# Collect all $/step values to test
test_configs = []
for name, conv in conversion_results.items():
    for frac_name, step_val in [('Full', conv['full']), ('Half', conv['half']), ('Quarter', conv['quarter'])]:
        label = f"{name} {frac_name}"
        if 0.5 <= step_val <= 20:  # only test reasonable range
            test_configs.append((label, step_val))

# Add EXP-002 references
test_configs.append(("EXP-002 Aggressive", 2.00))
test_configs.append(("EXP-002 Conservative", 2.50))
test_configs.append(("EXP-002 Safe", 3.00))

# Remove near-duplicates (within $0.05)
unique_configs = []
seen_steps = []
for label, step in sorted(test_configs, key=lambda x: x[1]):
    is_dup = False
    for s in seen_steps:
        if abs(step - s) < 0.05:
            is_dup = True
            break
    if not is_dup:
        unique_configs.append((label, step))
        seen_steps.append(step)

print(f"  Testing {len(unique_configs)} configs:")
print()
print(f"  {'Config':>30s} | {'$/step':>8s} | {'Median':>14s} | {'GeoMean':>14s} | {'P5':>14s} | {'AvgDD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*110}")

mc_all = {}
for label, step in sorted(unique_configs, key=lambda x: x[1]):
    r = run_mc(step)
    mc_all[(label, step)] = r
    print(f"  {label:>30s} | ${step:>6.2f} | ${r['median']:>12,.0f} | ${r['geo_mean']:>12,.0f} | ${r['p5']:>12,.0f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%")

print()


# ============================================================
# PART 4: EMPIRICAL KELLY — Which $/step maximizes geometric mean?
# ============================================================
print("=" * 100)
print("PART 4: EMPIRICAL KELLY — Brute-force optimal $/step")
print("=" * 100)
print()

# Fine sweep from $0.50 to $10.00
empirical_steps = np.arange(0.50, 10.25, 0.25)
print(f"  Sweeping $/step from ${empirical_steps[0]:.2f} to ${empirical_steps[-1]:.2f} in $0.25 increments")
print()
print(f"  {'$/step':>8s} | {'GeoMean':>14s} | {'Median':>14s} | {'P5':>14s} | {'Ruin%':>6s}")
print(f"  {'-'*75}")

best_geo = None
best_geo_step = None
best_safe_geo = None
best_safe_step = None

for step in empirical_steps:
    r = run_mc(step)
    marker = ""
    if best_geo is None or r['geo_mean'] > best_geo:
        best_geo = r['geo_mean']
        best_geo_step = step
    if r['ruin_pct'] <= 1.0 and (best_safe_geo is None or r['geo_mean'] > best_safe_geo):
        best_safe_geo = r['geo_mean']
        best_safe_step = step

    if step in [1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 4.00, 5.00]:
        print(f"  ${step:>6.2f} | ${r['geo_mean']:>12,.0f} | ${r['median']:>12,.0f} | ${r['p5']:>12,.0f} | {r['ruin_pct']:5.1f}%")

print(f"  {'-'*75}")

# Run fine sweep around the best
if best_safe_step:
    fine_steps = np.arange(max(0.50, best_safe_step - 0.50), best_safe_step + 0.75, 0.05)
    print()
    print(f"  FINE SWEEP around optimal (${best_safe_step:.2f}):")
    print()
    print(f"  {'$/step':>8s} | {'GeoMean':>14s} | {'Median':>14s} | {'Ruin%':>6s}")
    print(f"  {'-'*55}")

    fine_best_geo = None
    fine_best_step = None

    for step in fine_steps:
        r = run_mc(step)
        if r['ruin_pct'] <= 1.0 and (fine_best_geo is None or r['geo_mean'] > fine_best_geo):
            fine_best_geo = r['geo_mean']
            fine_best_step = step
        marker = " <-- BEST" if step == fine_best_step and r['ruin_pct'] <= 1.0 else ""
        print(f"  ${step:>6.2f} | ${r['geo_mean']:>12,.0f} | ${r['median']:>12,.0f} | {r['ruin_pct']:5.1f}%{marker}")

    if fine_best_step:
        kelly_results['Empirical'] = None  # fraction not applicable
        print()
        print(f"  EMPIRICAL KELLY OPTIMAL: ${fine_best_step:.2f}/step (geo mean: ${fine_best_geo:,.0f})")


# ============================================================
# PART 5: COMPARISON — All Kelly types vs Empirical
# ============================================================
print()
print("=" * 100)
print("PART 5: WHICH KELLY TYPE IS BEST?")
print("=" * 100)
print()

print(f"  {'Kelly Type':>15s} | {'Fraction':>10s} | {'Best $/step':>12s} | {'Match Empirical?':>18s}")
print(f"  {'-'*70}")

empirical_opt = fine_best_step if 'fine_best_step' in dir() and fine_best_step else best_safe_step

for name, frac in kelly_results.items():
    if name == 'Empirical':
        print(f"  {'Empirical':>15s} | {'N/A':>10s} | ${empirical_opt:>10.2f} | {'REFERENCE':>18s}")
        continue

    conv = conversion_results[name]
    # Find which fraction (full/half/quarter) is closest to empirical
    best_match = None
    best_dist = 999
    for frac_name, step_val in [('Full', conv['full']), ('Half', conv['half']), ('Quarter', conv['quarter'])]:
        dist = abs(step_val - empirical_opt)
        if dist < best_dist:
            best_dist = dist
            best_match = (frac_name, step_val)

    match_label = f"{best_match[0]} (${best_match[1]:.2f})"
    close = "YES" if best_dist < 0.5 else "NO"
    print(f"  {name:>15s} | {frac:>10.4f} | {match_label:>12s} | {close:>18s}")

print()


# ============================================================
# PART 6: VERDICT
# ============================================================
print("=" * 100)
print("PART 6: VERDICT")
print("=" * 100)
print()

print(f"  EMPIRICAL OPTIMAL: ${empirical_opt:.2f}/step")
print(f"    = Maximizes geometric mean across 1000 MC paths")
print(f"    = At $10 wallet: {max(1,int(10/empirical_opt))}x qty ({max(1,int(10/empirical_opt))*0.001:.3f} BTC)")
print()

print("  KELLY TYPE SUMMARY:")
for name in ['Classic', 'Mean-Variance', 'Geometric', 'Continuous']:
    frac = kelly_results[name]
    conv = conversion_results[name]
    print(f"    {name}: fraction={frac:.4f}")
    print(f"      Full=${conv['full']:.2f}, Half=${conv['half']:.2f}, Quarter=${conv['quarter']:.2f}")
print()

print(f"  RECOMMENDATION:")
print(f"    Use ${empirical_opt:.2f}/step as the BASE position size")
print(f"    This is validated by MC (not just theory)")
print(f"    Kelly types that agree with this are more trustworthy")
