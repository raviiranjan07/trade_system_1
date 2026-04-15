"""L1-EXP-002 Part 9: Risk-Based Margin — margin scales with equity

Tests 3 approaches:
  A) Isolated margin = X% of equity (5%, 10%, 15%, 20%)
  B) Tiered fixed margin (step function based on equity level)
  C) Cross margin, position scales with equity (not fixed $170)

All with hybrid switch to cross 20x scaling at threshold.
"""
import sys
sys.path.insert(0, "src")

import numpy as np
from engine.backtest import run_backtest
from engine.config.loader import load_config

STARTING_CAPITAL = 10.0
MIN_NOTIONAL = 170.0
MAX_LEVERAGE_SETTING = 125
N_SIMS = 1000

config = load_config()
trades = run_backtest(config)
returns = [t.net_profit_bps for t in trades]

print("=" * 120)
print("L1-EXP-002 PART 9: RISK-BASED MARGIN — Margin scales with equity")
print("=" * 120)
print(f"V1.3.2: {len(returns)} trades, {sum(1 for r in returns if r > 0)/len(returns)*100:.1f}% win")
print(f"Starting: ${STARTING_CAPITAL} | Position: $170 fixed in Phase 1")
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
# OPTION A: Isolated margin = X% of equity
# ============================================================
def simulate_pct_margin(rets, position_size, pct, capital=STARTING_CAPITAL):
    """Isolated margin = pct% of current equity per trade."""
    equity = [capital]
    skipped = 0
    liquidated = 0

    for r in rets:
        eq = equity[-1]
        margin = eq * pct

        if margin < 0.01:  # too small to trade
            equity.append(eq)
            skipped += 1
            continue

        pnl = position_size * (r / 10000)

        if pnl < -margin:
            equity.append(eq - margin)
            liquidated += 1
        else:
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


def simulate_pct_margin_hybrid(rets, position_size, pct, threshold, scale_lev, capital=STARTING_CAPITAL):
    """Phase 1: isolated pct margin. Phase 2: cross scaling."""
    equity = [capital]
    skipped = 0
    liquidated = 0

    for r in rets:
        eq = equity[-1]

        if eq < threshold:
            # Phase 1: isolated % margin
            margin = eq * pct
            if margin < 0.01:
                equity.append(eq)
                skipped += 1
                continue

            pnl = position_size * (r / 10000)
            if pnl < -margin:
                equity.append(eq - margin)
                liquidated += 1
            else:
                equity.append(max(eq + pnl, 0.01))
        else:
            # Phase 2: cross scaling
            pos = eq * scale_lev
            pnl = pos * (r / 10000)
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


# ============================================================
# OPTION B: Tiered fixed margin
# ============================================================
def get_tiered_margin(equity, tiers):
    """Return margin based on equity tier. tiers = [(threshold, margin), ...]"""
    for threshold, margin in sorted(tiers, reverse=True):
        if equity >= threshold:
            return margin
    return tiers[0][1]  # smallest tier


def simulate_tiered_margin(rets, position_size, tiers, capital=STARTING_CAPITAL):
    """Isolated margin with tiered levels based on equity."""
    equity = [capital]
    skipped = 0
    liquidated = 0

    for r in rets:
        eq = equity[-1]
        margin = get_tiered_margin(eq, tiers)

        if eq < margin:
            equity.append(eq)
            skipped += 1
            continue

        pnl = position_size * (r / 10000)

        if pnl < -margin:
            equity.append(eq - margin)
            liquidated += 1
        else:
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


def simulate_tiered_hybrid(rets, position_size, tiers, threshold, scale_lev, capital=STARTING_CAPITAL):
    """Phase 1: tiered isolated margin. Phase 2: cross scaling."""
    equity = [capital]
    skipped = 0
    liquidated = 0

    for r in rets:
        eq = equity[-1]

        if eq < threshold:
            margin = get_tiered_margin(eq, tiers)
            if eq < margin:
                equity.append(eq)
                skipped += 1
                continue

            pnl = position_size * (r / 10000)
            if pnl < -margin:
                equity.append(eq - margin)
                liquidated += 1
            else:
                equity.append(max(eq + pnl, 0.01))
        else:
            pos = eq * scale_lev
            pnl = pos * (r / 10000)
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped, liquidated


# ============================================================
# OPTION C: Cross margin, position scales with equity
# ============================================================
def simulate_cross_scaling_hybrid(rets, min_lev, threshold, scale_lev, capital=STARTING_CAPITAL):
    """Phase 1: cross margin, position = equity * min_lev (must meet $170 min).
       Phase 2: cross margin, position = equity * scale_lev."""
    equity = [capital]
    skipped = 0

    for r in rets:
        eq = equity[-1]

        if eq < threshold:
            # Phase 1: scaling at minimum viable leverage
            pos = eq * min_lev
            if pos < MIN_NOTIONAL:
                equity.append(eq)
                skipped += 1
                continue
            pnl = pos * (r / 10000)
            equity.append(max(eq + pnl, 0.01))
        else:
            # Phase 2: scaling at target leverage
            pos = eq * scale_lev
            pnl = pos * (r / 10000)
            equity.append(max(eq + pnl, 0.01))

    return equity, skipped


# Also need baselines
def simulate_fixed_position(rets, position_size, capital=STARTING_CAPITAL):
    equity = [capital]
    skipped = 0
    for r in rets:
        eq = equity[-1]
        if eq < position_size / MAX_LEVERAGE_SETTING:
            equity.append(eq)
            skipped += 1
            continue
        pnl = position_size * (r / 10000)
        equity.append(max(eq + pnl, 0.01))
    return equity, skipped


def simulate_fixed_iso(rets, position_size, margin, capital=STARTING_CAPITAL):
    equity = [capital]
    skipped = 0
    liquidated = 0
    for r in rets:
        eq = equity[-1]
        if eq < margin:
            equity.append(eq)
            skipped += 1
            continue
        pnl = position_size * (r / 10000)
        if pnl < -margin:
            equity.append(eq - margin)
            liquidated += 1
        else:
            equity.append(max(eq + pnl, 0.01))
    return equity, skipped, liquidated


def simulate_cross_hybrid(rets, position_size, threshold, scale_lev, capital=STARTING_CAPITAL):
    equity = [capital]
    skipped = 0
    for r in rets:
        eq = equity[-1]
        if eq < threshold:
            if eq < position_size / MAX_LEVERAGE_SETTING:
                equity.append(eq)
                skipped += 1
                continue
            pnl = position_size * (r / 10000)
            equity.append(max(eq + pnl, 0.01))
        else:
            pos = eq * scale_lev
            pnl = pos * (r / 10000)
            equity.append(max(eq + pnl, 0.01))
    return equity, skipped


# ============================================================
# PART 9A: ORIGINAL ORDER — All methods
# ============================================================
print("=" * 120)
print("PART 9A: ORIGINAL ORDER — All methods compared")
print("=" * 120)
print()

print(f"  {'Method':>50s} | {'Final':>10s} | {'MaxDD':>7s} | {'MinEq':>8s} | {'Skip':>5s} | {'Liq':>4s}")
print(f"  {'-'*105}")

# Baselines
eq, sk = simulate_fixed_position(returns, 170)
print(f"  {'[BASE] Cross Fixed $170':>50s} | ${eq[-1]:9.2f} | {calc_max_dd(eq)*100:5.1f}% | ${min(eq):6.2f} | {sk:>5d} | {'N/A':>4s}")

eq, sk, liq = simulate_fixed_iso(returns, 170, 2.0)
print(f"  {'[BASE] Iso Fixed $170, $2 margin':>50s} | ${eq[-1]:9.2f} | {calc_max_dd(eq)*100:5.1f}% | ${min(eq):6.2f} | {sk:>5d} | {liq:>4d}")

eq, sk = simulate_cross_hybrid(returns, 170, 25, 20)
print(f"  {'[BASE] Cross Hybrid $170->20x @$25':>50s} | ${eq[-1]:9.2f} | {calc_max_dd(eq)*100:5.1f}% | ${min(eq):6.2f} | {sk:>5d} | {'N/A':>4s}")

print()

# Option A: % margin
for pct in [0.05, 0.10, 0.15, 0.20, 0.30]:
    eq, sk, liq = simulate_pct_margin(returns, 170, pct)
    name = f"[A] Iso {pct*100:.0f}% margin"
    print(f"  {name:>50s} | ${eq[-1]:9.2f} | {calc_max_dd(eq)*100:5.1f}% | ${min(eq):6.2f} | {sk:>5d} | {liq:>4d}")

print()

# Option A: % margin hybrid
for pct in [0.05, 0.10, 0.15, 0.20, 0.30]:
    eq, sk, liq = simulate_pct_margin_hybrid(returns, 170, pct, 25, 20)
    name = f"[A] Iso {pct*100:.0f}% hybrid->20x @$25"
    print(f"  {name:>50s} | ${eq[-1]:9.2f} | {calc_max_dd(eq)*100:5.1f}% | ${min(eq):6.2f} | {sk:>5d} | {liq:>4d}")

print()

# Option B: Tiered
tier_configs = {
    'Conservative': [(0, 0.50), (5, 0.75), (10, 1.00), (15, 1.50), (20, 2.00)],
    'Moderate':     [(0, 0.75), (5, 1.00), (10, 1.50), (15, 2.00), (20, 2.50)],
    'Aggressive':   [(0, 1.00), (5, 1.50), (10, 2.00), (15, 2.50), (20, 3.00)],
}

for name, tiers in tier_configs.items():
    eq, sk, liq = simulate_tiered_margin(returns, 170, tiers)
    label = f"[B] Tiered {name}"
    print(f"  {label:>50s} | ${eq[-1]:9.2f} | {calc_max_dd(eq)*100:5.1f}% | ${min(eq):6.2f} | {sk:>5d} | {liq:>4d}")

print()

# Option B: Tiered hybrid
for name, tiers in tier_configs.items():
    eq, sk, liq = simulate_tiered_hybrid(returns, 170, tiers, 25, 20)
    label = f"[B] Tiered {name} hybrid->20x @$25"
    print(f"  {label:>50s} | ${eq[-1]:9.2f} | {calc_max_dd(eq)*100:5.1f}% | ${min(eq):6.2f} | {sk:>5d} | {liq:>4d}")

print()

# Option C: Cross scaling
for lev in [17, 18, 19, 20]:
    eq, sk = simulate_cross_scaling_hybrid(returns, lev, 25, 20)
    name = f"[C] Cross {lev}x->20x @$25"
    print(f"  {name:>50s} | ${eq[-1]:9.2f} | {calc_max_dd(eq)*100:5.1f}% | ${min(eq):6.2f} | {sk:>5d} | {'N/A':>4s}")


# ============================================================
# PART 9B: MONTE CARLO — All methods
# ============================================================
print()
print("=" * 120)
print("PART 9B: MONTE CARLO — 1000 shuffled paths for all methods")
print("=" * 120)
print()

np.random.seed(42)

def run_mc(name, sim_fn, n_sims=N_SIMS):
    """Run MC and return results dict."""
    finals = []
    max_dds = []
    ruined = 0

    for _ in range(n_sims):
        shuffled = list(returns)
        np.random.shuffle(shuffled)
        result = sim_fn(shuffled)
        eq = result[0]
        finals.append(eq[-1])
        dd = calc_max_dd(eq)
        max_dds.append(dd)
        if eq[-1] < 1.36:
            ruined += 1

    return {
        'name': name,
        'median': np.median(finals),
        'p5': np.percentile(finals, 5),
        'p25': np.percentile(finals, 25),
        'p75': np.percentile(finals, 75),
        'avg_dd': np.mean(max_dds),
        'p95_dd': np.percentile(max_dds, 95),
        'ruin_pct': ruined / n_sims * 100,
    }


all_results = []

# Baselines
print("  Running baselines...")
all_results.append(run_mc('[BASE] Cross Fixed $170',
    lambda r: simulate_fixed_position(r, 170)))
all_results.append(run_mc('[BASE] Iso Fixed $2',
    lambda r: simulate_fixed_iso(r, 170, 2.0)))
all_results.append(run_mc('[BASE] Cross Hybrid->20x @$25',
    lambda r: simulate_cross_hybrid(r, 170, 25, 20)))

# Option A: % margin (standalone)
print("  Running Option A (% margin)...")
for pct in [0.05, 0.10, 0.15, 0.20]:
    all_results.append(run_mc(f'[A] Iso {pct*100:.0f}%',
        lambda r, p=pct: simulate_pct_margin(r, 170, p)))

# Option A: % margin hybrid
print("  Running Option A hybrid...")
for pct in [0.05, 0.10, 0.15, 0.20]:
    all_results.append(run_mc(f'[A] {pct*100:.0f}% hyb->20x@25',
        lambda r, p=pct: simulate_pct_margin_hybrid(r, 170, p, 25, 20)))

# Option B: Tiered hybrid
print("  Running Option B (tiered)...")
for tname, tiers in tier_configs.items():
    all_results.append(run_mc(f'[B] {tname} hyb->20x@25',
        lambda r, t=tiers: simulate_tiered_hybrid(r, 170, t, 25, 20)))

# Option C: Cross scaling hybrid
print("  Running Option C (cross scaling)...")
for lev in [17, 18, 19, 20]:
    all_results.append(run_mc(f'[C] Cross {lev}x->20x@25',
        lambda r, l=lev: simulate_cross_scaling_hybrid(r, l, 25, 20)))

# Print all results
print()
print(f"  {'Method':>30s} | {'Median':>12s} | {'P5':>12s} | {'P25':>12s} | {'P75':>12s} | {'AvgDD':>7s} | {'P95DD':>7s} | {'Ruin%':>6s}")
print(f"  {'-'*120}")

for res in all_results:
    r = res
    print(f"  {r['name']:>30s} | ${r['median']:11.2f} | ${r['p5']:11.2f} | ${r['p25']:11.2f} | ${r['p75']:11.2f} | {r['avg_dd']*100:5.1f}% | {r['p95_dd']*100:5.1f}% | {r['ruin_pct']:5.1f}%")


# ============================================================
# PART 9C: Head-to-head comparison
# ============================================================
print()
print("=" * 120)
print("PART 9C: RANKING — Sorted by P5 (worst 5% luck)")
print("=" * 120)
print()

sorted_by_p5 = sorted(all_results, key=lambda x: x['p5'], reverse=True)

print(f"  {'Rank':>4s} | {'Method':>30s} | {'MC Median':>12s} | {'MC P5':>12s} | {'Ruin%':>6s} | {'AvgDD':>7s}")
print(f"  {'-'*90}")

for i, r in enumerate(sorted_by_p5):
    print(f"  {i+1:>4d} | {r['name']:>30s} | ${r['median']:11.2f} | ${r['p5']:11.2f} | {r['ruin_pct']:5.1f}% | {r['avg_dd']*100:5.1f}%")


# ============================================================
# PART 9D: Consecutive losses to ruin — worst case
# ============================================================
print()
print("=" * 120)
print("PART 9D: CONSECUTIVE LOSSES TO RUIN (worst case scenario)")
print("=" * 120)
print()

worst_loss_bps = min(returns)
avg_loss_bps = np.mean([r for r in returns if r < 0])

print(f"  Worst trade: {worst_loss_bps:.1f} bps (${170 * abs(worst_loss_bps) / 10000:.2f} on $170)")
print(f"  Avg loss:    {avg_loss_bps:.1f} bps (${170 * abs(avg_loss_bps) / 10000:.2f} on $170)")
print()

# Simulate consecutive worst/avg losses for each method
print(f"  {'Method':>35s} | {'Worst losses':>13s} | {'Avg losses':>11s}")
print(f"  {'-'*70}")

# Cross fixed $170
eq = STARTING_CAPITAL
for i in range(100):
    if eq < 170 / MAX_LEVERAGE_SETTING:
        break
    eq -= 170 * abs(worst_loss_bps) / 10000
print(f"  {'Cross Fixed $170 (worst)':>35s} | {i:>13d} | ", end="")

eq = STARTING_CAPITAL
for i in range(100):
    if eq < 170 / MAX_LEVERAGE_SETTING:
        break
    eq -= 170 * abs(avg_loss_bps) / 10000
print(f"{i:>11d}")

# Iso fixed $2
eq = STARTING_CAPITAL
for i in range(100):
    if eq < 2.0:
        break
    loss = 170 * abs(worst_loss_bps) / 10000
    if loss > 2.0:
        eq -= 2.0
    else:
        eq -= loss
print(f"  {'Iso Fixed $2 (worst)':>35s} | {i:>13d} | ", end="")

eq = STARTING_CAPITAL
for i in range(100):
    if eq < 2.0:
        break
    loss = 170 * abs(avg_loss_bps) / 10000
    if loss > 2.0:
        eq -= 2.0
    else:
        eq -= loss
print(f"{i:>11d}")

# % margin methods
for pct in [0.05, 0.10, 0.15, 0.20]:
    eq = STARTING_CAPITAL
    for i in range(100):
        margin = eq * pct
        if margin < 0.01:
            break
        loss = 170 * abs(worst_loss_bps) / 10000
        if loss > margin:
            eq -= margin
        else:
            eq -= loss
    worst_n = i

    eq = STARTING_CAPITAL
    for i in range(100):
        margin = eq * pct
        if margin < 0.01:
            break
        loss = 170 * abs(avg_loss_bps) / 10000
        if loss > margin:
            eq -= margin
        else:
            eq -= loss
    avg_n = i

    print(f"  {'Iso ' + f'{pct*100:.0f}% margin':>35s} | {worst_n:>13d} | {avg_n:>11d}")

# Tiered
for tname, tiers in tier_configs.items():
    eq = STARTING_CAPITAL
    for i in range(100):
        margin = get_tiered_margin(eq, tiers)
        if eq < margin:
            break
        loss = 170 * abs(worst_loss_bps) / 10000
        if loss > margin:
            eq -= margin
        else:
            eq -= loss
    worst_n = i

    eq = STARTING_CAPITAL
    for i in range(100):
        margin = get_tiered_margin(eq, tiers)
        if eq < margin:
            break
        loss = 170 * abs(avg_loss_bps) / 10000
        if loss > margin:
            eq -= margin
        else:
            eq -= loss
    avg_n = i

    print(f"  {'Tiered ' + tname:>35s} | {worst_n:>13d} | {avg_n:>11d}")

# Cross scaling
for lev in [17, 20]:
    eq = STARTING_CAPITAL
    for i in range(100):
        pos = eq * lev
        if pos < MIN_NOTIONAL:
            break
        eq -= pos * abs(worst_loss_bps) / 10000
    worst_n = i

    eq = STARTING_CAPITAL
    for i in range(100):
        pos = eq * lev
        if pos < MIN_NOTIONAL:
            break
        eq -= pos * abs(avg_loss_bps) / 10000
    avg_n = i

    print(f"  {'Cross Scaling ' + f'{lev}x':>35s} | {worst_n:>13d} | {avg_n:>11d}")


print()
print("=" * 120)
print("VERDICT")
print("=" * 120)
print()
print("  Best by P5 (bad luck protection):")
for i, r in enumerate(sorted_by_p5[:5]):
    print(f"    #{i+1}: {r['name']} — P5 ${r['p5']:.0f}, Median ${r['median']:.0f}, Ruin {r['ruin_pct']:.1f}%")

sorted_by_median = sorted(all_results, key=lambda x: x['median'], reverse=True)
print()
print("  Best by Median (typical outcome):")
for i, r in enumerate(sorted_by_median[:5]):
    print(f"    #{i+1}: {r['name']} — Median ${r['median']:.0f}, P5 ${r['p5']:.0f}, Ruin {r['ruin_pct']:.1f}%")

sorted_by_ruin = sorted(all_results, key=lambda x: (x['ruin_pct'], -x['median']))
print()
print("  Safest (lowest ruin, then highest median):")
for i, r in enumerate(sorted_by_ruin[:5]):
    print(f"    #{i+1}: {r['name']} — Ruin {r['ruin_pct']:.1f}%, Median ${r['median']:.0f}, P5 ${r['p5']:.0f}")
