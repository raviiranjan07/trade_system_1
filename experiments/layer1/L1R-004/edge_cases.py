"""L1R-004: BTC Price + Wallet Size Edge Cases

QUESTION: How does sizing behave across the full (BTC price x wallet size) matrix,
          and where are the danger zones?
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(line_buffering=True)

import json
import numpy as np
from pathlib import Path

from experiments.layer1.lib.trade_loader import load_enriched_trades
from experiments.layer1.lib.binance_math import (
    calc_min_qty, calc_margin, calc_liq_distance_bps, calc_risk_pct, calc_max_qty,
)
from experiments.layer1.lib.mc_engine import run_mc, run_mc_fixed_step
from experiments.layer1.lib.metrics import print_mc_comparison
from experiments.layer1.lib.constants import DEFAULT_CAPITAL, STEP_SIZE

# ============================================================
# LOAD DATA + STATS
# ============================================================
print("=" * 100)
print("L1R-004: BTC PRICE + WALLET SIZE EDGE CASES")
print("=" * 100)
print()

print("Loading trades...")
train = load_enriched_trades("train")
oos = load_enriched_trades("oos")
print(f"  TRAIN: {len(train)} trades | OOS: {len(oos)} trades")

stats_path = Path("experiments/layer1/L1R-001")
with open(stats_path / "train_stats.json") as f:
    train_stats = json.load(f)

train_worst = abs(train_stats['worst_loss_bps'])
train_p5 = abs(train_stats['p5_bps'])
print(f"  TRAIN worst: -{train_worst:.1f} bps | P5: -{train_p5:.1f} bps")
print()


# ============================================================
# PART 1: POSITION SIZE MATRIX (wallet x BTC price)
# ============================================================
print("=" * 100)
print("PART 1: MINIMUM POSITION MATRIX")
print("=" * 100)
print()

wallets = [5, 10, 15, 20, 30, 50, 100, 500, 1000]
btc_prices = [60_000, 80_000, 97_000, 100_000, 120_000]

print(f"  Min qty (BTC) at each price:")
print(f"  {'BTC':>10s} | {'Min Qty':>8s} | {'Min Pos':>10s} | {'Margin':>10s}")
print(f"  {'-'*45}")
for price in btc_prices:
    mq = calc_min_qty(price)
    print(f"  ${price:>8,d} | {mq:>8.3f} | ${mq*price:>8,.0f} | ${calc_margin(mq, price):>8.2f}")
print()
print("  NOTE: At $100K+, min qty drops to 0.001 (half of $97K). This is the sweet spot.")
print()


# ============================================================
# PART 2: RISK MATRIX (worst-case loss % of wallet)
# ============================================================
print("=" * 100)
print("PART 2: WORST-CASE RISK MATRIX (TRAIN worst = {:.0f} bps)".format(train_worst))
print("=" * 100)
print()

print(f"  {'Wallet':>10s} |", end="")
for price in btc_prices:
    print(f" ${price/1000:>5.0f}K", end=" |")
print()
print(f"  {'-' * (12 + 9 * len(btc_prices))}")

for w in wallets:
    print(f"  ${w:>8d} |", end="")
    for price in btc_prices:
        mq = calc_min_qty(price)
        risk = calc_risk_pct(w, mq, price, train_worst)
        if risk > 1.0:
            print(f" {risk*100:>5.0f}%!!", end=" |")
        elif risk > 0.5:
            print(f" {risk*100:>5.1f}%!", end=" |")
        elif risk > 0.2:
            print(f" {risk*100:>5.1f}% ", end=" |")
        else:
            print(f" {risk*100:>5.1f}% ", end=" |")
    print()

print()
print("  !! = >100% (INSTANT WIPE)")
print("  !  = >50% (DANGEROUS)")
print()


# ============================================================
# PART 3: TRANSITION THRESHOLDS — Where does "survival zone" end?
# ============================================================
print("=" * 100)
print("PART 3: TRANSITION THRESHOLDS")
print("=" * 100)
print()
print("  At what wallet size does min_qty risk drop below target?")
print()

target_risks = [0.50, 0.30, 0.20, 0.10, 0.05]

print(f"  {'Target Risk':>12s} |", end="")
for price in btc_prices:
    print(f" ${price/1000:>5.0f}K", end=" |")
print()
print(f"  {'-' * (14 + 9 * len(btc_prices))}")

for target in target_risks:
    print(f"  {target*100:>10.0f}%   |", end="")
    for price in btc_prices:
        mq = calc_min_qty(price)
        pos = mq * price
        loss_dollar = pos * train_worst / 10000
        wallet_needed = loss_dollar / target
        print(f"  ${wallet_needed:>5.0f}", end=" |")
    print()

print()
print("  Above these wallet sizes, you can size BELOW minimum (Kelly says trade less).")
print("  Below these sizes, min position exceeds your risk budget = 'survival zone'.")
print()


# ============================================================
# PART 4: KELLY-OPTIMAL QTY vs MINIMUM QTY
# ============================================================
print("=" * 100)
print("PART 4: KELLY QTY vs MINIMUM QTY — When can you size properly?")
print("=" * 100)
print()

# Using train Kelly and base $/step from L1R-002 findings
# L1R-002 found: no safe step on TRAIN. Best is ~$5.50 (6.8% ruin).
# Let's use both conservative ($6.00) and moderate ($4.00) as reference.
for step_label, step_val in [("Moderate $4/step", 4.00), ("Conservative $6/step", 6.00)]:
    print(f"  --- {step_label} ---")
    print(f"  {'Wallet':>10s} |", end="")
    for price in btc_prices:
        print(f" ${price/1000:>5.0f}K", end=" |")
    print()
    print(f"  {'-' * (12 + 9 * len(btc_prices))}")

    for w in wallets:
        print(f"  ${w:>8d} |", end="")
        for price in btc_prices:
            mq = calc_min_qty(price)
            steps = max(1, int(w / step_val))
            kelly_qty = steps * STEP_SIZE

            if kelly_qty >= mq:
                # Can size properly (Kelly >= min)
                mode = f" {kelly_qty:.3f}"
            else:
                # Forced to use minimum (survival zone)
                mode = f" {mq:.3f}*"
            print(f"{mode:>7s}", end=" |")
        print()

    print()
    print("  * = FORCED to min qty (survival zone: Kelly says trade less, but can't)")
    print()


# ============================================================
# PART 5: MC SIMULATION — SURVIVAL ZONE
# ============================================================
print("=" * 100)
print("PART 5: MC — What happens in the survival zone?")
print("=" * 100)
print()

# Test different starting capitals with fixed min qty
starting_caps = [5, 10, 15, 20, 30, 50]

def min_qty_fn_factory(btc_price_fixed):
    """Create a sizing fn that always trades min qty at a specific BTC price."""
    mq = calc_min_qty(btc_price_fixed)
    def fn(wallet, trade, stats):
        return mq, "MIN"
    return fn

# Test on TRAIN (the harder dataset)
print("  TRAIN data — fixed min qty at different BTC prices and starting capitals:")
print()

for price in [97_000, 100_000]:
    print(f"  BTC ${price/1000:.0f}K:")
    mq = calc_min_qty(price)
    print(f"    Min qty: {mq:.3f} BTC (${mq*price:,.0f} position)")
    print()
    print(f"    {'Start $':>10s} | {'Median':>14s} | {'P5':>14s} | {'Ruin%':>8s} | {'AvgDD':>8s}")
    print(f"    {'-'*65}")

    for cap in starting_caps:
        fn = min_qty_fn_factory(price)
        r = run_mc(train, fn, {}, capital=cap)
        print(f"    ${cap:>8d} | ${r.median:>12,.0f} | ${r.p5:>12,.0f} | {r.ruin_pct:>6.1f}% | {r.avg_dd*100:>6.1f}%")
    print()


# ============================================================
# PART 6: MC — OOS COMPARISON
# ============================================================
print("=" * 100)
print("PART 6: MC — OOS survival zone (for comparison)")
print("=" * 100)
print()

for price in [97_000, 100_000]:
    print(f"  BTC ${price/1000:.0f}K (OOS data):")
    mq = calc_min_qty(price)
    print(f"    Min qty: {mq:.3f} BTC")
    print()
    print(f"    {'Start $':>10s} | {'Median':>14s} | {'P5':>14s} | {'Ruin%':>8s} | {'AvgDD':>8s}")
    print(f"    {'-'*65}")

    for cap in starting_caps:
        fn = min_qty_fn_factory(price)
        r = run_mc(oos, fn, {}, capital=cap)
        print(f"    ${cap:>8d} | ${r.median:>12,.0f} | ${r.p5:>12,.0f} | {r.ruin_pct:>6.1f}% | {r.avg_dd*100:>6.1f}%")
    print()


# ============================================================
# PART 7: PROBABILITY OF SURVIVING TO GROWTH ZONE
# ============================================================
print("=" * 100)
print("PART 7: P(SURVIVE) — Probability of reaching growth zone wallet")
print("=" * 100)
print()

# Growth zone = wallet where Kelly qty >= min qty
# At $4/step + $97K BTC: need $4 * (0.002 / 0.001) = $8 per min_qty step
# Actually: steps = wallet / step_val, qty = steps * 0.001
# Need qty >= 0.002 at $97K -> steps >= 2 -> wallet >= 2 * step
# At $4/step -> wallet >= $8 (already above)
# But Kelly-optimal means wallet high enough that risk is acceptable

# Let's define growth zone as: risk_pct < 20% at worst-case
print(f"  Growth zone defined as: TRAIN worst-case loss < 20% of wallet")
print()

for price in [97_000, 100_000]:
    mq = calc_min_qty(price)
    pos = mq * price
    loss = pos * train_worst / 10000
    growth_wallet = loss / 0.20
    print(f"  BTC ${price/1000:.0f}K: need ${growth_wallet:,.0f} wallet for <20% risk")

    # MC: what % of paths reach this wallet from $10?
    fn = min_qty_fn_factory(price)
    rng = np.random.default_rng(42)
    reach_count = 0
    n_sims = 1000
    for _ in range(n_sims):
        shuffled = list(train)
        rng.shuffle(shuffled)
        eq = 10.0
        reached = False
        for td in shuffled:
            if eq <= 0.01:
                break
            position = mq * td['btc_price']
            margin = position / 125
            if eq < margin:
                continue
            maint = position * 0.004
            pnl = position * (td['bps'] / 10000)
            if pnl < -(eq - maint):
                eq = 0.01
            else:
                eq = max(eq + pnl, 0.01)
            if eq >= growth_wallet:
                reached = True
                break
        if reached:
            reach_count += 1

    print(f"    P(reach ${growth_wallet:,.0f} from $10 on TRAIN): {reach_count/n_sims*100:.1f}%")
    print()


# ============================================================
# VERDICT
# ============================================================
print("=" * 100)
print("VERDICT")
print("=" * 100)
print()
print("  1. BTC $100K+ is the SWEET SPOT: min position halves ($200 -> $100)")
print("  2. At $10 wallet: ALWAYS in survival zone regardless of BTC price")
print("  3. Train worst loss can wipe 87-168% of a $10 wallet in one trade")
print("  4. Growth zone (risk <20%) starts at $43-$84 depending on BTC price")
print("  5. The survival zone is unavoidable at small wallets — system must be honest about it")
print()
