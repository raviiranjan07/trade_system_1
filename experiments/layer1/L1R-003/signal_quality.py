"""L1R-003: Signal Quality Impact on Risk

QUESTION: Do validated bad conditions have worse loss profiles
          that justify different position sizing?

Uses EXP-004 validated conditions + shared lib.
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import json
import numpy as np
from pathlib import Path

from experiments.layer1.lib.trade_loader import load_enriched_trades
from experiments.layer1.lib.mc_engine import run_mc, run_mc_fixed_step, MCResult
from experiments.layer1.lib.metrics import print_mc_comparison
from experiments.layer1.lib.signal_quality import (
    score_signal, SignalQuality,
    DEFAULT_BAD_CONDITIONS, DEFAULT_STRONG_CONDITIONS,
)
from experiments.layer1.lib.constants import DEFAULT_CAPITAL, STEP_SIZE

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 100)
print("L1R-003: SIGNAL QUALITY IMPACT ON RISK")
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


# ============================================================
# PART 1: SCORE ALL TRADES
# ============================================================
print("=" * 100)
print("PART 1: SCORE ALL TRADES BY SIGNAL QUALITY")
print("=" * 100)
print()

def score_and_split(data, label):
    """Score trades and split into tiers."""
    tiers = {'STRONG': [], 'NORMAL': [], 'WEAK': []}
    for td in data:
        sq = score_signal(td)
        td['_quality'] = sq
        tiers[sq.tier].append(td)

    print(f"  {label}:")
    for tier_name in ['STRONG', 'NORMAL', 'WEAK']:
        t = tiers[tier_name]
        if not t:
            print(f"    {tier_name}: 0 trades")
            continue
        bps = [td['bps'] for td in t]
        wins = [b for b in bps if b > 0]
        losses = [b for b in bps if b <= 0]
        print(f"    {tier_name}: {len(t)}t, win {len(wins)/len(bps)*100:.1f}%, "
              f"avg {np.mean(bps):+.1f}, worst {min(bps):.1f}, "
              f"P5 {np.percentile(bps, 5):.1f}, total {sum(bps):+.0f}")
    print()
    return tiers

train_tiers = score_and_split(train, "TRAIN")
oos_tiers = score_and_split(oos, "OOS")


# ============================================================
# PART 2: DETAILED TIER COMPARISON
# ============================================================
print("=" * 100)
print("PART 2: DETAILED TIER STATS — TRAIN vs OOS")
print("=" * 100)
print()

print(f"  {'Metric':>20s} |   STRONG TRAIN |     STRONG OOS |   NORMAL TRAIN |     NORMAL OOS |     WEAK TRAIN |       WEAK OOS")
print(f"  {'-'*120}")

for tier in ['STRONG', 'NORMAL', 'WEAK']:
    tr = train_tiers[tier]
    oo = oos_tiers[tier]
    if not tr or not oo:
        continue

    tr_bps = [t['bps'] for t in tr]
    oo_bps = [t['bps'] for t in oo]
    tr_wins = [b for b in tr_bps if b > 0]
    oo_wins = [b for b in oo_bps if b > 0]
    tr_losses = [b for b in tr_bps if b <= 0]
    oo_losses = [b for b in oo_bps if b <= 0]

    if tier == 'STRONG':
        print(f"  {'Count':>20s} | {len(tr):>14d} | {len(oo):>14d} |", end="")
    elif tier == 'NORMAL':
        print(f" {len(train_tiers['NORMAL']):>14d} | {len(oos_tiers['NORMAL']):>14d} |", end="")
    elif tier == 'WEAK':
        print(f" {len(train_tiers['WEAK']):>14d} | {len(oos_tiers['WEAK']):>14d}")

# Rebuild properly as a clean table
print()
for tier in ['STRONG', 'NORMAL', 'WEAK']:
    tr = train_tiers[tier]
    oo = oos_tiers[tier]
    tr_bps = [t['bps'] for t in tr] if tr else [0]
    oo_bps = [t['bps'] for t in oo] if oo else [0]
    tr_wins = [b for b in tr_bps if b > 0]
    oo_wins = [b for b in oo_bps if b > 0]
    tr_losses = [b for b in tr_bps if b <= 0]
    oo_losses = [b for b in oo_bps if b <= 0]

    print(f"  --- {tier} ---")
    print(f"    TRAIN: {len(tr)}t, {len(tr_wins)/len(tr_bps)*100:.1f}% win, avg {np.mean(tr_bps):+.1f}, worst {min(tr_bps):.1f}, P5 {np.percentile(tr_bps, 5):.1f}")
    print(f"    OOS:   {len(oo)}t, {len(oo_wins)/len(oo_bps)*100:.1f}% win, avg {np.mean(oo_bps):+.1f}, worst {min(oo_bps):.1f}, P5 {np.percentile(oo_bps, 5):.1f}")

    # Do tiers hold?
    train_avg = np.mean(tr_bps)
    oos_avg = np.mean(oo_bps)
    if tier == 'WEAK':
        holds = "YES" if oos_avg < np.mean([t['bps'] for t in oos]) * 0.7 else "NO"
    elif tier == 'STRONG':
        holds = "YES" if oos_avg > np.mean([t['bps'] for t in oos]) * 1.3 else "NO"
    else:
        holds = "N/A"
    print(f"    Holds in OOS? {holds}")
    print()


# ============================================================
# PART 3: PER-TIER KELLY FRACTIONS (from TRAIN)
# ============================================================
print("=" * 100)
print("PART 3: PER-TIER KELLY FRACTIONS (from TRAIN)")
print("=" * 100)
print()

for tier in ['STRONG', 'NORMAL', 'WEAK']:
    tr = train_tiers[tier]
    if not tr:
        print(f"  {tier}: no trades")
        continue
    bps = [t['bps'] for t in tr]
    wins = [b for b in bps if b > 0]
    losses = [b for b in bps if b <= 0]
    if not wins or not losses:
        print(f"  {tier}: insufficient data (wins={len(wins)}, losses={len(losses)})")
        continue
    p = len(wins) / len(bps)
    q = 1 - p
    b = abs(np.mean(wins) / np.mean(losses))
    f = p - q / b if b > 0 else 0
    print(f"  {tier}: p={p:.3f}, b={b:.2f}, Kelly f={f:.4f}")
print()


# ============================================================
# PART 4: MC — UNIFORM vs TIER-SPECIFIC $/STEP
# ============================================================
print("=" * 100)
print("PART 4: MC — UNIFORM vs TIER-SPECIFIC $/STEP")
print("=" * 100)
print()

# Use a range of base steps to test
base_steps = [4.00, 5.00, 6.00, 8.00]

for base_step in base_steps:
    print(f"  --- Base $/step: ${base_step:.2f} ---")
    print()

    # Config 1: Uniform (same step for all)
    uniform_label = f"Uniform ${base_step:.2f}"

    # Config 2: Tier-specific (size down weak, size up strong)
    # WEAK: 2x base (half position)
    # STRONG: 0.7x base (larger position)
    weak_step = base_step * 2.0
    strong_step = base_step * 0.7

    def make_tier_fn(base, strong_s, weak_s):
        def tier_fn(wallet, trade, stats):
            sq = score_signal(trade)
            if sq.tier == "WEAK":
                step = weak_s
            elif sq.tier == "STRONG":
                step = strong_s
            else:
                step = base
            steps = max(1, int(wallet / step))
            qty = steps * STEP_SIZE
            qty = max(qty, trade['qty_min'])
            return qty, sq.tier
        return tier_fn

    tier_fn = make_tier_fn(base_step, strong_step, weak_step)

    # Run on TRAIN
    train_uniform = run_mc_fixed_step(train, base_step)
    train_tier = run_mc(train, tier_fn, {})

    # Run on OOS
    oos_uniform = run_mc_fixed_step(oos, base_step)
    oos_tier = run_mc(oos, tier_fn, {})

    print(f"    {'Config':>25s} | {'TRAIN Median':>14s} {'Ruin':>6s} | {'OOS Median':>14s} {'Ruin':>6s}")
    print(f"    {'-'*80}")
    print(f"    {uniform_label:>25s} | ${train_uniform.median:>12,.0f} {train_uniform.ruin_pct:5.1f}% | ${oos_uniform.median:>12,.0f} {oos_uniform.ruin_pct:5.1f}%")
    tier_label = f"Tier (W${weak_step:.0f}/N${base_step:.0f}/S${strong_step:.1f})"
    print(f"    {tier_label:>25s} | ${train_tier.median:>12,.0f} {train_tier.ruin_pct:5.1f}% | ${oos_tier.median:>12,.0f} {oos_tier.ruin_pct:5.1f}%")

    # Mode breakdown
    if train_tier.mode_counts:
        total = sum(train_tier.mode_counts.values())
        print(f"    Mode distribution: ", end="")
        for m, c in sorted(train_tier.mode_counts.items()):
            print(f"{m}={c/total*100:.0f}% ", end="")
        print()

    # Improvement
    if oos_uniform.geo_mean > 0:
        improv = (oos_tier.geo_mean / oos_uniform.geo_mean - 1) * 100
        print(f"    OOS improvement: {improv:+.1f}% GeoMean")
    print()


# ============================================================
# PART 5: WHAT MAKES WEAK TRADES WEAK? (Loss profile analysis)
# ============================================================
print("=" * 100)
print("PART 5: WHY ARE WEAK TRADES WEAK? (Loss profile)")
print("=" * 100)
print()

for period_label, tiers in [("TRAIN", train_tiers), ("OOS", oos_tiers)]:
    print(f"  {period_label}:")
    for tier in ['STRONG', 'NORMAL', 'WEAK']:
        tr = tiers[tier]
        if not tr:
            continue
        bps = [t['bps'] for t in tr]
        losses = [b for b in bps if b <= 0]
        wins = [b for b in bps if b > 0]
        print(f"    {tier} ({len(tr)}t):")
        print(f"      Win rate: {len(wins)/len(bps)*100:.1f}%")
        if losses:
            print(f"      Avg loss: {np.mean(losses):.1f} bps")
            print(f"      Worst loss: {min(losses):.1f} bps")
            print(f"      P5 loss: {np.percentile(bps, 5):.1f} bps")
            print(f"      Loss > 100 bps: {sum(1 for l in losses if l < -100)} trades")
        print(f"      Total: {sum(bps):+.0f} bps")
    print()


# ============================================================
# PART 6: REASON BREAKDOWN — WHY ARE TRADES WEAK?
# ============================================================
print("=" * 100)
print("PART 6: REASON BREAKDOWN — What triggers WEAK classification?")
print("=" * 100)
print()

for period_label, data in [("TRAIN", train), ("OOS", oos)]:
    weak_trades = [t for t in data if score_signal(t).tier == "WEAK"]
    reason_counts = {}
    for t in weak_trades:
        sq = score_signal(t)
        for r in sq.reasons:
            if not r.startswith('+'):
                reason_counts[r] = reason_counts.get(r, 0) + 1

    print(f"  {period_label} WEAK trades ({len(weak_trades)}):")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        # Also show avg bps for this reason
        reason_trades = [t for t in weak_trades if reason in score_signal(t).reasons]
        avg = np.mean([t['bps'] for t in reason_trades])
        print(f"    {reason}: {count}t ({count/len(weak_trades)*100:.0f}%), avg {avg:+.1f} bps")
    print()


# ============================================================
# PART 7: VERDICT
# ============================================================
print("=" * 100)
print("PART 7: VERDICT")
print("=" * 100)
print()

# Check: do WEAK trades have worse loss profile?
train_weak_bps = [t['bps'] for t in train_tiers['WEAK']]
train_rest_bps = [t['bps'] for t in train_tiers['STRONG'] + train_tiers['NORMAL']]

if train_weak_bps and train_rest_bps:
    print(f"  TRAIN WEAK avg: {np.mean(train_weak_bps):+.1f} bps (vs rest: {np.mean(train_rest_bps):+.1f} bps)")
    print(f"  TRAIN WEAK worst: {min(train_weak_bps):.1f} bps (vs rest worst: {min(train_rest_bps):.1f} bps)")
    print(f"  TRAIN WEAK P5: {np.percentile(train_weak_bps, 5):.1f} bps (vs rest P5: {np.percentile(train_rest_bps, 5):.1f} bps)")
    print()

oos_weak_bps = [t['bps'] for t in oos_tiers['WEAK']]
oos_rest_bps = [t['bps'] for t in oos_tiers['STRONG'] + oos_tiers['NORMAL']]

if oos_weak_bps and oos_rest_bps:
    print(f"  OOS WEAK avg: {np.mean(oos_weak_bps):+.1f} bps (vs rest: {np.mean(oos_rest_bps):+.1f} bps)")
    print(f"  OOS WEAK worst: {min(oos_weak_bps):.1f} bps (vs rest worst: {min(oos_rest_bps):.1f} bps)")
    print(f"  OOS WEAK P5: {np.percentile(oos_weak_bps, 5):.1f} bps (vs rest P5: {np.percentile(oos_rest_bps, 5):.1f} bps)")
    print()

print(f"  CONCLUSION: Does signal quality justify different sizing?")
if train_weak_bps:
    weak_worse = np.mean(train_weak_bps) < np.mean(train_rest_bps) * 0.5
    print(f"  -> WEAK underperforms rest by >{50 if weak_worse else '<50'}%: {'YES' if weak_worse else 'NO'}")
    print(f"  -> WEAK has worse tail risk: {'YES' if min(train_weak_bps) < min(train_rest_bps) else 'NO'}")
print()
