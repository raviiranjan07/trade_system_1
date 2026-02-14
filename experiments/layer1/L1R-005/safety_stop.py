"""L1R-005: Safety Stop Placement

QUESTION: Where should the exchange-level safety stop go, validated on TRAIN data?

EXP-006 found safety stop has zero impact on OOS (worst -182 bps, liq buffer 460 bps).
But TRAIN worst is -865 bps. Does safety stop matter on TRAIN?
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import json
import numpy as np
from pathlib import Path

from experiments.layer1.lib.trade_loader import load_enriched_trades
from experiments.layer1.lib.binance_math import (
    calc_min_qty, calc_margin, calc_liq_distance_bps,
)
from experiments.layer1.lib.mc_engine import run_mc
from experiments.layer1.lib.metrics import print_mc_comparison
from experiments.layer1.lib.constants import DEFAULT_CAPITAL, STEP_SIZE, LEVERAGE, MAINT_MARGIN_RATE

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 100)
print("L1R-005: SAFETY STOP PLACEMENT")
print("=" * 100)
print()

print("Loading trades...")
train = load_enriched_trades("train")
oos = load_enriched_trades("oos")
print(f"  TRAIN: {len(train)} trades | OOS: {len(oos)} trades")

stats_path = Path("experiments/layer1/L1R-001")
with open(stats_path / "train_stats.json") as f:
    train_stats = json.load(f)
print()


# ============================================================
# PART 1: LIQUIDATION DISTANCES AT DIFFERENT QTY LEVELS
# ============================================================
print("=" * 100)
print("PART 1: LIQUIDATION DISTANCES")
print("=" * 100)
print()

# At $10 wallet with different qty and BTC prices
print("  Wallet = $10, BTC $97K:")
for qty in [0.001, 0.002, 0.003, 0.005]:
    ld = calc_liq_distance_bps(10.0, qty, 97_000)
    pos = qty * 97_000
    print(f"    {qty:.3f} BTC (${pos:,.0f} pos): liq at {ld:.0f} bps")

print()
print("  Wallet = $10, BTC $100K:")
for qty in [0.001, 0.002, 0.003]:
    ld = calc_liq_distance_bps(10.0, qty, 100_000)
    pos = qty * 100_000
    print(f"    {qty:.3f} BTC (${pos:,.0f} pos): liq at {ld:.0f} bps")
print()


# ============================================================
# PART 2: SAFETY STOP LEVELS AND TRADES AFFECTED
# ============================================================
print("=" * 100)
print("PART 2: SAFETY STOP LEVELS — How many trades get stopped?")
print("=" * 100)
print()

# For each safety stop %, check which TRAIN trades would be affected
# Safety stop = X% of liq distance
# We need to consider actual wallet size during the trade
# For simplicity, check at fixed wallet=$10 with min qty

for price_label, btc_price in [("BTC $97K", 97_000), ("BTC $100K", 100_000)]:
    mq = calc_min_qty(btc_price)
    ld = calc_liq_distance_bps(10.0, mq, btc_price)

    print(f"  {price_label} — min qty {mq:.3f} BTC, liq distance {ld:.0f} bps")
    print()

    safety_pcts = [0.60, 0.70, 0.80, 0.90]
    print(f"    {'Safety %':>10s} | {'Stop (bps)':>12s} | {'TRAIN hit':>10s} | {'OOS hit':>10s} | {'TRAIN bps lost':>15s}")
    print(f"    {'-'*70}")

    for pct in safety_pcts:
        safety_bps = ld * pct

        # Count trades that would be stopped
        train_hit = [t for t in train if abs(t['bps']) > safety_bps]
        oos_hit = [t for t in oos if abs(t['bps']) > safety_bps]

        # How many bps would be lost/saved?
        # If safety stop catches a losing trade at -safety_bps instead of actual loss:
        train_saved = sum(abs(t['bps']) - safety_bps for t in train_hit if t['bps'] < 0)
        train_lost = sum(t['bps'] - safety_bps for t in train_hit if t['bps'] > 0)  # winners cut short

        print(f"    {pct*100:>8.0f}%   | {safety_bps:>10.0f}   | "
              f"{len(train_hit):>10d} | {len(oos_hit):>10d} | "
              f"saved {train_saved:+.0f}, lost {train_lost:+.0f}")
    print()


# ============================================================
# PART 3: MC WITH/WITHOUT SAFETY STOP
# ============================================================
print("=" * 100)
print("PART 3: MC — With vs Without Safety Stop")
print("=" * 100)
print()

# Fixed step sizing with and without safety stop
# Safety stop caps maximum loss at X bps (clamping bps to -safety_bps)

def make_safety_fn(step_val, safety_bps):
    """Sizing fn with safety stop (caps max loss)."""
    def fn(wallet, trade, stats):
        steps = max(1, int(wallet / step_val))
        qty = steps * STEP_SIZE
        qty = max(qty, trade['qty_min'])
        return qty, f"STEP_{step_val}"
    return fn

def cap_trades(data, safety_bps):
    """Create a copy of trades with losses capped at safety_bps."""
    capped = []
    for t in data:
        td = dict(t)
        if td['bps'] < -safety_bps:
            td['bps'] = -safety_bps
        capped.append(td)
    return capped

# Test at $6/step (the safest from L1R-002)
step = 6.00
for price_label, btc_price in [("BTC $97K", 97_000), ("BTC $100K", 100_000)]:
    mq = calc_min_qty(btc_price)
    ld = calc_liq_distance_bps(10.0, mq, btc_price)

    print(f"  {price_label} — $/step={step}, liq dist={ld:.0f} bps")
    print()

    fn = make_safety_fn(step, 0)

    # No safety
    r_none_train = run_mc(train, fn, {})
    r_none_oos = run_mc(oos, fn, {})

    print(f"    {'Config':>30s} | {'TRAIN Median':>14s} {'Ruin':>6s} | {'OOS Median':>14s} {'Ruin':>6s}")
    print(f"    {'-'*80}")
    print(f"    {'No safety stop':>30s} | ${r_none_train.median:>12,.0f} {r_none_train.ruin_pct:5.1f}% | ${r_none_oos.median:>12,.0f} {r_none_oos.ruin_pct:5.1f}%")

    for pct in [0.60, 0.70, 0.80, 0.90]:
        safety_bps = ld * pct
        train_capped = cap_trades(train, safety_bps)
        oos_capped = cap_trades(oos, safety_bps)

        r_train = run_mc(train_capped, fn, {})
        r_oos = run_mc(oos_capped, fn, {})

        label = f"Safety {pct*100:.0f}% ({safety_bps:.0f} bps)"
        print(f"    {label:>30s} | ${r_train.median:>12,.0f} {r_train.ruin_pct:5.1f}% | ${r_oos.median:>12,.0f} {r_oos.ruin_pct:5.1f}%")

    print()


# ============================================================
# PART 4: STRESS TEST
# ============================================================
print("=" * 100)
print("PART 4: STRESS TEST — Extreme scenarios")
print("=" * 100)
print()

train_worst = abs(train_stats['worst_loss_bps'])
print(f"  TRAIN worst: -{train_worst:.0f} bps")
print()

for price_label, btc_price in [("BTC $97K", 97_000), ("BTC $100K", 100_000)]:
    mq = calc_min_qty(btc_price)
    ld = calc_liq_distance_bps(10.0, mq, btc_price)
    safety_bps = ld * 0.80

    print(f"  {price_label} — liq {ld:.0f} bps, safety stop at {safety_bps:.0f} bps")
    print()
    print(f"    {'Scenario':>30s} | {'Loss (bps)':>12s} | {'Outcome':>15s}")
    print(f"    {'-'*65}")

    scenarios = [
        ("Worst historical", train_worst),
        ("1.5x worst", train_worst * 1.5),
        ("2x worst", train_worst * 2),
        ("3x worst", train_worst * 3),
        ("-5% flash crash", 500),
        ("-10% black swan", 1000),
    ]

    for label, loss in scenarios:
        if loss < safety_bps:
            outcome = "SAFE (within safety)"
        elif loss < ld:
            outcome = "STOPPED (safety hit)"
        else:
            outcome = "LIQUIDATED"
        print(f"    {label:>30s} | {loss:>10.0f}   | {outcome:>15s}")
    print()


# ============================================================
# VERDICT
# ============================================================
print("=" * 100)
print("VERDICT")
print("=" * 100)
print()

mq_97 = calc_min_qty(97_000)
mq_100 = calc_min_qty(100_000)
ld_97 = calc_liq_distance_bps(10.0, mq_97, 97_000)
ld_100 = calc_liq_distance_bps(10.0, mq_100, 100_000)

print(f"  1. At $10 wallet + min qty:")
print(f"     BTC $97K: liq distance {ld_97:.0f} bps, safety at {ld_97*0.80:.0f} bps")
print(f"     BTC $100K: liq distance {ld_100:.0f} bps, safety at {ld_100*0.80:.0f} bps")
print()
print(f"  2. TRAIN worst loss (-{train_worst:.0f} bps) vs safety stops:")
print(f"     BTC $97K: worst (-{train_worst:.0f}) {'<' if train_worst < ld_97*0.80 else '>'} safety ({ld_97*0.80:.0f}) -> {'NO TRIGGER' if train_worst < ld_97*0.80 else 'TRIGGERS'}")
print(f"     BTC $100K: worst (-{train_worst:.0f}) {'<' if train_worst < ld_100*0.80 else '>'} safety ({ld_100*0.80:.0f}) -> {'NO TRIGGER' if train_worst < ld_100*0.80 else 'TRIGGERS'}")
print()
print(f"  3. Safety stop = free insurance that protects against unprecedented moves")
print(f"  4. Recommended: 80% of liq distance (standard from EXP-006)")
print()
