"""L1R-002: $/step Sweep — Proper Train/Test Validation

QUESTION: What is the optimal $/step found on TRAIN, and does it hold on OOS?

Uses shared lib (no copy-paste of MC engine or trade loader).
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import json
import numpy as np
from pathlib import Path

from experiments.layer1.lib.trade_loader import load_enriched_trades
from experiments.layer1.lib.mc_engine import run_mc_fixed_step
from experiments.layer1.lib.metrics import print_mc_comparison, print_mc_comparison_train_oos
from experiments.layer1.lib.constants import DEFAULT_CAPITAL

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 100)
print("L1R-002: $/STEP SWEEP — PROPER TRAIN/TEST VALIDATION")
print("=" * 100)
print()

print("Loading trades...")
train = load_enriched_trades("train")
oos = load_enriched_trades("oos")
print(f"  TRAIN: {len(train)} trades | OOS: {len(oos)} trades")
print()

# Load stats from L1R-001
stats_path = Path("experiments/layer1/L1R-001")
with open(stats_path / "train_stats.json") as f:
    train_stats = json.load(f)
with open(stats_path / "oos_stats.json") as f:
    oos_stats = json.load(f)


# ============================================================
# PART 1: KELLY FROM TRAIN
# ============================================================
print("=" * 100)
print("PART 1: KELLY FRACTION FROM TRAIN")
print("=" * 100)
print()

f_kelly_train = train_stats['kelly_fraction']
f_kelly_oos = oos_stats['kelly_fraction']

print(f"  TRAIN Kelly: {f_kelly_train:.4f}")
print(f"  OOS Kelly:   {f_kelly_oos:.4f} (for comparison only)")
print()

# Convert to $/step
avg_loss_frac = abs(train_stats['avg_loss_bps']) / 10000
avg_btc = train_stats['avg_btc_price']

def kelly_to_step(fraction):
    """Convert Kelly fraction to $/step."""
    if fraction <= 0:
        return float('inf')
    position = fraction * DEFAULT_CAPITAL / avg_loss_frac
    qty = position / avg_btc
    from experiments.layer1.lib.constants import STEP_SIZE
    steps = qty / STEP_SIZE
    if steps < 1:
        return DEFAULT_CAPITAL
    return DEFAULT_CAPITAL / steps

for label, frac in [("Full Kelly", f_kelly_train),
                     ("Half Kelly", f_kelly_train / 2),
                     ("Quarter Kelly", f_kelly_train / 4)]:
    step = kelly_to_step(frac)
    print(f"  {label}: f={frac:.4f} -> ${step:.2f}/step")
print()


# ============================================================
# PART 2: BRUTE-FORCE SWEEP ON TRAIN
# ============================================================
print("=" * 100)
print("PART 2: $/STEP SWEEP ON TRAIN (find optimal)")
print("=" * 100)
print()

sweep_steps = list(np.arange(1.00, 20.25, 0.50))

print(f"  {'$/step':>8s} | {'Median':>14s} | {'GeoMean':>14s} | {'P5':>14s} | {'Ruin%':>6s} | {'AvgDD':>7s}")
print(f"  {'-'*80}")

train_results = {}
for step in sweep_steps:
    step = round(step, 2)
    r = run_mc_fixed_step(train, step)
    train_results[step] = r
    print(f"  ${step:>6.2f} | ${r.median:>12,.0f} | ${r.geo_mean:>12,.0f} | ${r.p5:>12,.0f} | {r.ruin_pct:5.1f}% | {r.avg_dd*100:5.1f}%")

# Find optimal on train (best geo mean with <=1% ruin)
train_safe = {k: v for k, v in train_results.items() if v.ruin_pct <= 1.0}
train_optimal = max(train_safe, key=lambda k: train_safe[k].geo_mean) if train_safe else None

# Find 0% ruin optimal
train_zero = {k: v for k, v in train_results.items() if v.ruin_pct == 0}
train_conservative = max(train_zero, key=lambda k: train_zero[k].geo_mean) if train_zero else None

print(f"  {'-'*80}")
if train_optimal:
    r = train_results[train_optimal]
    print(f"  TRAIN OPTIMAL (ruin<=1%): ${train_optimal:.2f}/step | GeoMean ${r.geo_mean:,.0f} | Ruin {r.ruin_pct:.1f}%")
else:
    print(f"  WARNING: No safe $/step (<=1% ruin) found on TRAIN!")
    # Find least ruin
    least_ruin = min(train_results, key=lambda k: train_results[k].ruin_pct)
    r = train_results[least_ruin]
    print(f"  LEAST RUIN: ${least_ruin:.2f}/step | GeoMean ${r.geo_mean:,.0f} | Ruin {r.ruin_pct:.1f}%")
    train_optimal = least_ruin

if train_conservative:
    r = train_results[train_conservative]
    print(f"  TRAIN CONSERVATIVE (0% ruin): ${train_conservative:.2f}/step | GeoMean ${r.geo_mean:,.0f}")
else:
    # Find lowest ruin
    low_ruin = {k: v for k, v in train_results.items() if v.ruin_pct <= 5.0}
    if low_ruin:
        train_conservative = max(low_ruin, key=lambda k: low_ruin[k].geo_mean)
        r = train_results[train_conservative]
        print(f"  TRAIN BEST LOW-RUIN (<=5%): ${train_conservative:.2f}/step | Ruin {r.ruin_pct:.1f}%")
print()


# ============================================================
# PART 3: SAME SWEEP ON OOS (for comparison)
# ============================================================
print("=" * 100)
print("PART 3: $/STEP SWEEP ON OOS (for comparison)")
print("=" * 100)
print()

print(f"  {'$/step':>8s} | {'Median':>14s} | {'GeoMean':>14s} | {'P5':>14s} | {'Ruin%':>6s} | {'AvgDD':>7s}")
print(f"  {'-'*80}")

oos_results = {}
for step in sweep_steps:
    step = round(step, 2)
    r = run_mc_fixed_step(oos, step)
    oos_results[step] = r
    print(f"  ${step:>6.2f} | ${r.median:>12,.0f} | ${r.geo_mean:>12,.0f} | ${r.p5:>12,.0f} | {r.ruin_pct:5.1f}% | {r.avg_dd*100:5.1f}%")

oos_safe = {k: v for k, v in oos_results.items() if v.ruin_pct <= 1.0}
oos_optimal = max(oos_safe, key=lambda k: oos_safe[k].geo_mean) if oos_safe else None
oos_zero = {k: v for k, v in oos_results.items() if v.ruin_pct == 0}
oos_conservative = max(oos_zero, key=lambda k: oos_zero[k].geo_mean) if oos_zero else None

print(f"  {'-'*80}")
if oos_optimal:
    r = oos_results[oos_optimal]
    print(f"  OOS OPTIMAL (ruin<=1%): ${oos_optimal:.2f}/step | GeoMean ${r.geo_mean:,.0f} | Ruin {r.ruin_pct:.1f}%")
if oos_conservative:
    r = oos_results[oos_conservative]
    print(f"  OOS CONSERVATIVE (0% ruin): ${oos_conservative:.2f}/step | GeoMean ${r.geo_mean:,.0f}")
print()


# ============================================================
# PART 4: VALIDATE — TRAIN-DERIVED ON OOS
# ============================================================
print("=" * 100)
print("PART 4: VALIDATE — Test train-derived $/step on OOS")
print("=" * 100)
print()

test_steps = set()
if train_optimal:
    test_steps.add(("TRAIN optimal", train_optimal))
if train_conservative:
    test_steps.add(("TRAIN conservative", train_conservative))
test_steps.add(("Quarter-Kelly (TRAIN)", round(kelly_to_step(f_kelly_train / 4), 2)))
for s in [2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 8.00, 10.00]:
    test_steps.add((f"Fixed ${s:.2f}", s))

print(f"  {'Config':>30s} | {'$/step':>8s} | {'TRAIN Median':>14s} {'Ruin':>6s} | {'OOS Median':>14s} {'Ruin':>6s}")
print(f"  {'-'*100}")

for label, step in sorted(test_steps, key=lambda x: x[1]):
    # Get or compute results
    tr = train_results.get(step) or run_mc_fixed_step(train, step)
    oo = oos_results.get(step) or run_mc_fixed_step(oos, step)
    print(f"  {label:>30s} | ${step:>6.2f} | ${tr.median:>12,.0f} {tr.ruin_pct:5.1f}% | ${oo.median:>12,.0f} {oo.ruin_pct:5.1f}%")

print()


# ============================================================
# PART 5: RUIN CLIFF — WHERE DOES IT SPIKE?
# ============================================================
print("=" * 100)
print("PART 5: RUIN CLIFF — Where does ruin spike?")
print("=" * 100)
print()

cliff_steps = list(np.arange(2.00, 10.25, 0.50))
print(f"  {'$/step':>8s} | {'TRAIN Ruin%':>12s} | {'OOS Ruin%':>12s}")
print(f"  {'-'*40}")

for step in cliff_steps:
    step = round(step, 2)
    tr = train_results.get(step) or run_mc_fixed_step(train, step)
    oo = oos_results.get(step) or run_mc_fixed_step(oos, step)
    marker = ""
    if 0 < tr.ruin_pct <= 1:
        marker += " <- TRAIN edge"
    if 0 < oo.ruin_pct <= 1:
        marker += " <- OOS edge"
    print(f"  ${step:>6.2f} | {tr.ruin_pct:>10.1f}% | {oo.ruin_pct:>10.1f}%{marker}")

print()


# ============================================================
# PART 6: VERDICT
# ============================================================
print("=" * 100)
print("PART 6: VERDICT")
print("=" * 100)
print()

print(f"  TRAIN optimal: ${train_optimal:.2f}/step" if train_optimal else "  TRAIN: no safe $/step")
print(f"  TRAIN conservative: ${train_conservative:.2f}/step" if train_conservative else "  TRAIN: no 0% ruin $/step")
print(f"  OOS optimal: ${oos_optimal:.2f}/step" if oos_optimal else "  OOS: no safe $/step")
print(f"  OOS conservative: ${oos_conservative:.2f}/step" if oos_conservative else "  OOS: no 0% ruin $/step")
print()

if train_optimal and oos_optimal:
    gap = abs(train_optimal - oos_optimal)
    print(f"  Gap between TRAIN and OOS optimal: ${gap:.2f}")
    if gap <= 1.0:
        print(f"  -> CLOSE MATCH: Train-derived $/step likely works on new data")
    else:
        print(f"  -> LARGE GAP: Train needs more conservative $/step")
        print(f"     This is because train worst loss ({train_stats['worst_loss_bps']:.0f} bps) is much worse than OOS ({oos_stats['worst_loss_bps']:.0f} bps)")
print()

# What the train-optimal achieves on OOS
if train_optimal:
    r = run_mc_fixed_step(oos, train_optimal)
    print(f"  TRAIN optimal (${train_optimal:.2f}) on OOS: median ${r.median:,.0f}, ruin {r.ruin_pct:.1f}%")
if train_conservative:
    r = run_mc_fixed_step(oos, train_conservative)
    print(f"  TRAIN conservative (${train_conservative:.2f}) on OOS: median ${r.median:,.0f}, ruin {r.ruin_pct:.1f}%")
print()
