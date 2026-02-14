"""L1-EXP-007: Validated $/step and Kelly — Proper Train/Test Split

QUESTION: Do the $/step values and Kelly findings hold with proper validation?

PROBLEM WITH EXP-002 & EXP-003:
  - Both used ONLY OOS (2024-2025) trades
  - Kelly fraction calculated from OOS stats
  - Optimal $/step found from OOS brute-force
  - MC validated on same OOS data
  - This is CIRCULAR — fitting and testing on same data

PROPER APPROACH:
  1. Run backtest on TRAIN (2020-2023) -> train trades
  2. Run backtest on OOS (2024-2025) -> OOS trades
  3. Calculate Kelly from TRAIN stats
  4. Find optimal $/step from TRAIN via MC brute-force
  5. Test TRAIN-derived $/step on OOS trades via MC
  6. Compare: does train optimal work on OOS?

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
# LOAD TRADES — TRAIN and OOS separately
# ============================================================
config = load_config()

print("Loading TRAIN trades (2020-2023)...")
train_trades_raw = run_backtest(config, start="2020-01-01", end="2023-12-31")

print("Loading OOS trades (2024-2025)...")
oos_trades_raw = run_backtest(config, start="2024-01-01", end="2025-12-31")


def make_trade_data(trades_raw):
    """Convert TradeRecord list to trade_data dicts."""
    data = []
    for t in trades_raw:
        btc_price = t.entry_price
        qty_min = max(BINANCE_MIN_QTY,
                      math.ceil(BINANCE_MIN_NOTIONAL / btc_price / BINANCE_STEP_SIZE) * BINANCE_STEP_SIZE)
        data.append({
            'bps': t.net_profit_bps,
            'btc_price': btc_price,
            'qty_min': qty_min,
            'pos_min': qty_min * btc_price,
        })
    return data


train_data = make_trade_data(train_trades_raw)
oos_data = make_trade_data(oos_trades_raw)

train_returns = [td['bps'] for td in train_data]
oos_returns = [td['bps'] for td in oos_data]

train_wins = [r for r in train_returns if r > 0]
train_losses = [r for r in train_returns if r <= 0]
oos_wins = [r for r in oos_returns if r > 0]
oos_losses = [r for r in oos_returns if r <= 0]

print()
print("=" * 100)
print("L1-EXP-007: VALIDATED $/STEP AND KELLY — TRAIN/TEST SPLIT")
print("=" * 100)
print()
print(f"  TRAIN (2020-2023): {len(train_returns)} trades, {len(train_wins)/len(train_returns)*100:.1f}% win")
print(f"    Avg win: +{np.mean(train_wins):.1f} bps | Avg loss: {np.mean(train_losses):.1f} bps")
print(f"    Total: {sum(train_returns):+.0f} bps | Best: +{max(train_returns):.1f} | Worst: {min(train_returns):.1f}")
print(f"    Avg BTC: ${np.mean([td['btc_price'] for td in train_data]):,.0f}")
print()
print(f"  OOS (2024-2025): {len(oos_returns)} trades, {len(oos_wins)/len(oos_returns)*100:.1f}% win")
print(f"    Avg win: +{np.mean(oos_wins):.1f} bps | Avg loss: {np.mean(oos_losses):.1f} bps")
print(f"    Total: {sum(oos_returns):+.0f} bps | Best: +{max(oos_returns):.1f} | Worst: {min(oos_returns):.1f}")
print(f"    Avg BTC: ${np.mean([td['btc_price'] for td in oos_data]):,.0f}")
print()


# ============================================================
# HELPER FUNCTIONS
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


def simulate_scaling_qty(trade_list, dollars_per_step, capital=STARTING_CAPITAL):
    """Scale BTC qty with wallet size.
    qty = floor(wallet / dollars_per_step) * 0.001
    """
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


def run_mc(trade_list, step_val, n_sims=N_SIMS, seed=42):
    """Run MC simulation on given trade list."""
    np.random.seed(seed)
    finals = []
    max_dds = []
    ruin_count = 0

    for _ in range(n_sims):
        shuffled = list(trade_list)
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


# ============================================================
# PART 1: KELLY FROM TRAIN DATA
# ============================================================
print("=" * 100)
print("PART 1: KELLY FRACTION FROM TRAIN DATA (2020-2023)")
print("=" * 100)
print()

# Classic Kelly: f = p - q/b
p_train = len(train_wins) / len(train_returns)
q_train = 1 - p_train
b_train = abs(np.mean(train_wins)) / abs(np.mean(train_losses))
f_classic_train = p_train - q_train / b_train

print(f"  Classic Kelly from TRAIN:")
print(f"    p={p_train:.3f}, q={q_train:.3f}, b={b_train:.2f}")
print(f"    f = {p_train:.3f} - {q_train:.3f}/{b_train:.2f} = {f_classic_train:.4f}")
print()

# For comparison: Kelly from OOS (what EXP-003 used)
p_oos = len(oos_wins) / len(oos_returns)
q_oos = 1 - p_oos
b_oos = abs(np.mean(oos_wins)) / abs(np.mean(oos_losses))
f_classic_oos = p_oos - q_oos / b_oos

print(f"  Classic Kelly from OOS (for comparison):")
print(f"    p={p_oos:.3f}, q={q_oos:.3f}, b={b_oos:.2f}")
print(f"    f = {f_classic_oos:.4f}")
print()

# Convert Kelly fractions to $/step
avg_loss_frac_train = abs(np.mean(train_losses)) / 10000
avg_btc_train = np.mean([td['btc_price'] for td in train_data])


def kelly_to_step(fraction, avg_loss_frac, avg_price, wallet=STARTING_CAPITAL):
    """Convert Kelly fraction to $/step."""
    position = fraction * wallet / avg_loss_frac
    qty = position / avg_price
    steps = qty / BINANCE_STEP_SIZE
    if steps < 1:
        return wallet
    return wallet / steps


print(f"  TRAIN Kelly -> $/step (using train avg loss {abs(np.mean(train_losses)):.1f} bps, avg BTC ${avg_btc_train:,.0f}):")
for label, frac in [("Full", f_classic_train), ("Half", f_classic_train/2), ("Quarter", f_classic_train/4)]:
    step = kelly_to_step(frac, avg_loss_frac_train, avg_btc_train)
    print(f"    {label} Kelly: f={frac:.4f} -> ${step:.2f}/step")
print()

avg_loss_frac_oos = abs(np.mean(oos_losses)) / 10000
avg_btc_oos = np.mean([td['btc_price'] for td in oos_data])
print(f"  OOS Kelly -> $/step (for comparison, avg loss {abs(np.mean(oos_losses)):.1f} bps, avg BTC ${avg_btc_oos:,.0f}):")
for label, frac in [("Full", f_classic_oos), ("Half", f_classic_oos/2), ("Quarter", f_classic_oos/4)]:
    step = kelly_to_step(frac, avg_loss_frac_oos, avg_btc_oos)
    print(f"    {label} Kelly: f={frac:.4f} -> ${step:.2f}/step")
print()


# ============================================================
# PART 2: FIND OPTIMAL $/STEP ON TRAIN VIA MC
# ============================================================
print("=" * 100)
print("PART 2: BRUTE-FORCE OPTIMAL $/STEP ON TRAIN DATA")
print("=" * 100)
print()
print("  Sweeping $/step from $0.50 to $10.00 on TRAIN trades...")
print()

sweep_steps = np.arange(0.50, 20.25, 0.25)

print(f"  {'$/step':>8s} | {'GeoMean':>14s} | {'Median':>14s} | {'P5':>14s} | {'Ruin%':>6s} | {'AvgDD':>7s}")
print(f"  {'-'*80}")

train_mc_results = {}
for step in sweep_steps:
    r = run_mc(train_data, step)
    train_mc_results[step] = r
    # Print selected values
    if step in [1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00, 6.00, 8.00, 10.00, 15.00, 20.00]:
        print(f"  ${step:>6.2f} | ${r['geo_mean']:>12,.0f} | ${r['median']:>12,.0f} | ${r['p5']:>12,.0f} | {r['ruin_pct']:5.1f}% | {r['avg_dd']*100:5.1f}%")

# Find optimal on train
train_safe = {k: v for k, v in train_mc_results.items() if v['ruin_pct'] <= 1.0}
train_optimal = max(train_safe, key=lambda k: train_safe[k]['geo_mean']) if train_safe else None

print(f"  {'-'*80}")
if train_optimal:
    r = train_mc_results[train_optimal]
    print(f"  TRAIN OPTIMAL (ruin<=1%): ${train_optimal:.2f}/step")
    print(f"    GeoMean: ${r['geo_mean']:,.0f} | Median: ${r['median']:,.0f} | Ruin: {r['ruin_pct']:.1f}%")

# Fine sweep around train optimal
train_fine_results = {}
fine_best_step = None

if train_optimal:
    print()
    print(f"  FINE SWEEP around ${train_optimal:.2f}:")
    print()
    fine_steps = np.arange(max(0.50, train_optimal - 1.00), train_optimal + 1.25, 0.05)
    print(f"  {'$/step':>8s} | {'GeoMean':>14s} | {'Median':>14s} | {'Ruin%':>6s}")
    print(f"  {'-'*55}")

    fine_best_geo = None

    for step in fine_steps:
        step = round(step, 2)
        r = run_mc(train_data, step)
        train_fine_results[step] = r
        if r['ruin_pct'] <= 1.0 and (fine_best_geo is None or r['geo_mean'] > fine_best_geo):
            fine_best_geo = r['geo_mean']
            fine_best_step = step
        marker = " <-- BEST" if step == fine_best_step else ""
        print(f"  ${step:>6.2f} | ${r['geo_mean']:>12,.0f} | ${r['median']:>12,.0f} | {r['ruin_pct']:5.1f}%{marker}")

train_optimal_fine = fine_best_step if fine_best_step else train_optimal
print()
if train_optimal_fine:
    print(f"  TRAIN OPTIMAL (fine): ${train_optimal_fine:.2f}/step")
else:
    # No safe step found at all
    all_train = {**train_mc_results, **train_fine_results}
    least_ruin_step = min(all_train, key=lambda k: all_train[k]['ruin_pct'])
    print(f"  WARNING: No safe $/step (<=1% ruin) on TRAIN!")
    print(f"  Least ruin: ${least_ruin_step:.2f}/step ({all_train[least_ruin_step]['ruin_pct']:.1f}% ruin)")
    train_optimal_fine = least_ruin_step

# Also find train zero-ruin optimal
all_train_results = {**train_mc_results, **train_fine_results}
train_zero_ruin = {k: v for k, v in all_train_results.items() if v['ruin_pct'] == 0}
train_conservative = max(train_zero_ruin, key=lambda k: train_zero_ruin[k]['geo_mean']) if train_zero_ruin else None
if train_conservative:
    print(f"  TRAIN CONSERVATIVE (0% ruin): ${train_conservative:.2f}/step")
else:
    # No zero-ruin found, find lowest ruin
    train_low_ruin = {k: v for k, v in all_train_results.items() if v['ruin_pct'] <= 5.0}
    if train_low_ruin:
        train_conservative = max(train_low_ruin, key=lambda k: train_low_ruin[k]['geo_mean'])
        print(f"  TRAIN BEST LOW-RUIN (<=5%): ${train_conservative:.2f}/step ({all_train_results[train_conservative]['ruin_pct']:.1f}% ruin)")
    else:
        train_conservative = None
        print(f"  WARNING: No low-ruin $/step found on TRAIN!")
print()


# ============================================================
# PART 3: TEST TRAIN-DERIVED $/STEP ON OOS
# ============================================================
print("=" * 100)
print("PART 3: VALIDATE ON OOS — Test train-derived $/step on OOS trades")
print("=" * 100)
print()

# Configs to test on OOS
test_configs = []

# Train-derived values
if train_optimal_fine:
    test_configs.append(("TRAIN optimal (fine)", train_optimal_fine))
if train_optimal and train_optimal != train_optimal_fine:
    test_configs.append(("TRAIN optimal (broad)", train_optimal))
if train_conservative and train_conservative != train_optimal_fine:
    test_configs.append(("TRAIN conservative (0% ruin)", train_conservative))

# Train Kelly values
quarter_kelly_step = kelly_to_step(f_classic_train/4, avg_loss_frac_train, avg_btc_train)
half_kelly_step = kelly_to_step(f_classic_train/2, avg_loss_frac_train, avg_btc_train)
test_configs.append(("TRAIN Quarter-Kelly", round(quarter_kelly_step, 2)))
test_configs.append(("TRAIN Half-Kelly", round(half_kelly_step, 2)))

# Fixed references
test_configs.append(("Fixed $1.50", 1.50))
test_configs.append(("Fixed $1.75", 1.75))
test_configs.append(("Fixed $2.00", 2.00))
test_configs.append(("Fixed $2.50", 2.50))
test_configs.append(("Fixed $3.00", 3.00))

# Remove near-duplicates
unique_configs = []
seen = []
for label, step in sorted(test_configs, key=lambda x: x[1]):
    is_dup = False
    for s in seen:
        if abs(step - s) < 0.03:
            is_dup = True
            break
    if not is_dup:
        unique_configs.append((label, step))
        seen.append(step)

print(f"  Testing {len(unique_configs)} configs on OOS ({len(oos_returns)} trades):")
print()
print(f"  {'Config':>35s} | {'$/step':>8s} | {'Median':>14s} | {'GeoMean':>14s} | {'P5':>14s} | {'AvgDD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*115}")

oos_results = {}
for label, step in sorted(unique_configs, key=lambda x: x[1]):
    r = run_mc(oos_data, step)
    oos_results[(label, step)] = r
    print(f"  {label:>35s} | ${step:>6.2f} | ${r['median']:>12,.0f} | ${r['geo_mean']:>12,.0f} | ${r['p5']:>12,.0f} | {r['avg_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%")

print()


# ============================================================
# PART 4: OOS BRUTE-FORCE (for comparison — what WAS the OOS optimal?)
# ============================================================
print("=" * 100)
print("PART 4: OOS BRUTE-FORCE (for comparison only)")
print("=" * 100)
print()
print("  Finding what the OOS optimal $/step actually is...")
print()

oos_mc_all = {}
for step in sweep_steps:
    r = run_mc(oos_data, step)
    oos_mc_all[step] = r

oos_safe = {k: v for k, v in oos_mc_all.items() if v['ruin_pct'] <= 1.0}
oos_optimal = max(oos_safe, key=lambda k: oos_safe[k]['geo_mean']) if oos_safe else None
oos_zero_ruin = {k: v for k, v in oos_mc_all.items() if v['ruin_pct'] == 0}
oos_conservative = max(oos_zero_ruin, key=lambda k: oos_zero_ruin[k]['geo_mean']) if oos_zero_ruin else None

print(f"  {'$/step':>8s} | {'GeoMean':>14s} | {'Median':>14s} | {'Ruin%':>6s}")
print(f"  {'-'*55}")
for step in [1.50, 1.75, 2.00, 2.25, 2.50, 3.00, 4.00]:
    r = oos_mc_all[step]
    marker = ""
    if step == oos_optimal:
        marker = " <-- OOS OPTIMAL"
    elif step == oos_conservative:
        marker = " <-- OOS CONSERVATIVE"
    print(f"  ${step:>6.2f} | ${r['geo_mean']:>12,.0f} | ${r['median']:>12,.0f} | {r['ruin_pct']:5.1f}%{marker}")

if oos_optimal:
    print(f"\n  OOS OPTIMAL (ruin<=1%): ${oos_optimal:.2f}/step")
if oos_conservative:
    print(f"  OOS CONSERVATIVE (0% ruin): ${oos_conservative:.2f}/step")
print()


# ============================================================
# PART 5: COMPARISON — Train vs OOS
# ============================================================
print("=" * 100)
print("PART 5: TRAIN vs OOS COMPARISON")
print("=" * 100)
print()

print(f"  {'Metric':>30s} | {'TRAIN':>15s} | {'OOS':>15s} | {'Match?':>10s}")
print(f"  {'-'*80}")

print(f"  {'Trades':>30s} | {len(train_returns):>15d} | {len(oos_returns):>15d} |")
print(f"  {'Win rate':>30s} | {p_train*100:>14.1f}% | {p_oos*100:>14.1f}% |")
print(f"  {'Avg win (bps)':>30s} | {np.mean(train_wins):>+14.1f} | {np.mean(oos_wins):>+14.1f} |")
print(f"  {'Avg loss (bps)':>30s} | {np.mean(train_losses):>14.1f} | {np.mean(oos_losses):>14.1f} |")
print(f"  {'Payoff ratio':>30s} | {b_train:>15.2f} | {b_oos:>15.2f} |")
print(f"  {'Classic Kelly fraction':>30s} | {f_classic_train:>15.4f} | {f_classic_oos:>15.4f} |")

kelly_match = "YES" if abs(f_classic_train - f_classic_oos) / f_classic_oos < 0.3 else "NO"
print(f"  {'Kelly fractions similar?':>30s} | {'':>15s} | {'':>15s} | {kelly_match:>10s}")

if train_optimal_fine and oos_optimal:
    step_match = "YES" if abs(train_optimal_fine - oos_optimal) <= 0.50 else "NO"
    print(f"  {'Optimal $/step':>30s} | ${train_optimal_fine:>13.2f} | ${oos_optimal:>13.2f} | {step_match:>10s}")

if train_conservative and oos_conservative:
    cons_match = "YES" if abs(train_conservative - oos_conservative) <= 0.50 else "NO"
    print(f"  {'Conservative $/step':>30s} | ${train_conservative:>13.2f} | ${oos_conservative:>13.2f} | {cons_match:>10s}")

print()

# Performance of TRAIN-derived optimal on OOS
if train_optimal_fine:
    r_train_on_oos = run_mc(oos_data, train_optimal_fine)
    r_oos_on_oos = run_mc(oos_data, oos_optimal) if oos_optimal else None

    print(f"  TRAIN optimal (${train_optimal_fine:.2f}) tested on OOS:")
    print(f"    Median: ${r_train_on_oos['median']:,.0f} | GeoMean: ${r_train_on_oos['geo_mean']:,.0f} | Ruin: {r_train_on_oos['ruin_pct']:.1f}%")
    if r_oos_on_oos:
        print(f"  OOS optimal (${oos_optimal:.2f}) tested on OOS:")
        print(f"    Median: ${r_oos_on_oos['median']:,.0f} | GeoMean: ${r_oos_on_oos['geo_mean']:,.0f} | Ruin: {r_oos_on_oos['ruin_pct']:.1f}%")
        ratio = r_train_on_oos['median'] / r_oos_on_oos['median'] * 100
        print(f"  Train-derived achieves {ratio:.0f}% of OOS-optimal median")
print()


# ============================================================
# PART 6: RUIN CLIFF ANALYSIS
# ============================================================
print("=" * 100)
print("PART 6: RUIN CLIFF — Where does ruin spike?")
print("=" * 100)
print()

cliff_steps = np.arange(2.00, 8.00, 0.25)
print(f"  {'$/step':>8s} | {'TRAIN Ruin%':>12s} | {'OOS Ruin%':>12s}")
print(f"  {'-'*40}")

for step in cliff_steps:
    step = round(step, 2)
    r_train = run_mc(train_data, step)
    r_oos = run_mc(oos_data, step)
    marker = ""
    if r_train['ruin_pct'] > 0 and r_train['ruin_pct'] <= 1:
        marker = " <-- TRAIN edge"
    if r_oos['ruin_pct'] > 0 and r_oos['ruin_pct'] <= 1:
        marker += " <-- OOS edge"
    print(f"  ${step:>6.2f} | {r_train['ruin_pct']:>10.1f}% | {r_oos['ruin_pct']:>10.1f}%{marker}")

print()


# ============================================================
# PART 7: VERDICT
# ============================================================
print("=" * 100)
print("PART 7: VERDICT")
print("=" * 100)
print()

print(f"  TRAIN Stats: {len(train_returns)}t, {p_train*100:.1f}% win, PF {abs(np.mean(train_wins)*len(train_wins)) / abs(np.mean(train_losses)*len(train_losses)):.2f}")
print(f"  OOS Stats:   {len(oos_returns)}t, {p_oos*100:.1f}% win, PF {abs(np.mean(oos_wins)*len(oos_wins)) / abs(np.mean(oos_losses)*len(oos_losses)):.2f}")
print()
print(f"  TRAIN Classic Kelly: f={f_classic_train:.4f}")
print(f"  OOS Classic Kelly:   f={f_classic_oos:.4f}")
print()
if train_optimal_fine:
    print(f"  TRAIN optimal $/step: ${train_optimal_fine:.2f}")
if oos_optimal:
    print(f"  OOS optimal $/step:   ${oos_optimal:.2f}")
if train_conservative:
    print(f"  TRAIN conservative:   ${train_conservative:.2f}")
if oos_conservative:
    print(f"  OOS conservative:     ${oos_conservative:.2f}")
print()
print(f"  QUESTION: Does train-derived $/step work on OOS?")
if train_optimal_fine:
    r = run_mc(oos_data, train_optimal_fine)
    works = "YES" if r['ruin_pct'] <= 1.0 else "NO"
    print(f"  ANSWER: {works}")
    print(f"    Train optimal (${train_optimal_fine:.2f}) on OOS: median ${r['median']:,.0f}, ruin {r['ruin_pct']:.1f}%")
print()

# Final recommendation
print("  RECOMMENDATION:")
if train_optimal_fine and train_conservative:
    r_opt = run_mc(oos_data, train_optimal_fine)
    r_con = run_mc(oos_data, train_conservative)
    if r_opt['ruin_pct'] <= 1.0:
        print(f"    Aggressive: ${train_optimal_fine:.2f}/step (train optimal, OOS ruin {r_opt['ruin_pct']:.1f}%)")
    if r_con['ruin_pct'] == 0:
        print(f"    Conservative: ${train_conservative:.2f}/step (train 0% ruin, OOS ruin {r_con['ruin_pct']:.1f}%)")
    # Also test $2.00 and $2.50
    r_200 = run_mc(oos_data, 2.00)
    r_250 = run_mc(oos_data, 2.50)
    print(f"    Fixed $2.00: OOS median ${r_200['median']:,.0f}, ruin {r_200['ruin_pct']:.1f}%")
    print(f"    Fixed $2.50: OOS median ${r_250['median']:,.0f}, ruin {r_250['ruin_pct']:.1f}%")
print()
