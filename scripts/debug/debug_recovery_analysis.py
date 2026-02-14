"""
Recovery Analysis: When price goes below entry, what happens?

Run: .venv/Scripts/python.exe debug_recovery_analysis.py

3 CASES (when price goes below entry):

Case 1: Went below, never hit target within H, never hit even with more time (WRONG DIRECTION)
Case 2: Went below, but hit target WITHIN H bars (QUICK RECOVERY)
Case 3: Went below, didn't hit within H, but hit target with more time (SLOW RECOVERY - needed more time)

Key Question: Is failure due to wrong direction (Case 1) or insufficient time (Case 3)?
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60]
TARGETS = [8, 10, 15, 20, 25]  # bps
EXTENDED_H = 500  # Extended horizon to check if it eventually hits target

TRAIN_END = "2023-12-31"
SAMPLE_SIZE = 100000  # Reduced for speed with extended horizon check

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("RECOVERY ANALYSIS: 3 CASES")
print("=" * 70)
print("""
When price goes below entry:
  Case 1: Never hit target (even with extended time) = WRONG DIRECTION
  Case 2: Hit target within H bars = QUICK RECOVERY
  Case 3: Didn't hit within H, but hit later = SLOW RECOVERY (needed more time)
""")

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"Loaded {len(ohlcv):,} candles")

train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END]
print(f"TRAIN: {len(train_ohlcv):,} candles")

close = train_ohlcv['close'].values
high = train_ohlcv['high'].values
low = train_ohlcv['low'].values
n = len(train_ohlcv)

# Sample for speed
np.random.seed(42)
sample_idx = np.random.choice(n - EXTENDED_H, size=min(SAMPLE_SIZE, n - EXTENDED_H), replace=False)
print(f"Sampling {len(sample_idx):,} bars...")

# =============================================================================
# ANALYSIS FOR EACH HORIZON AND TARGET
# =============================================================================

for target_bps in TARGETS:
    print(f"\n{'='*80}")
    print(f"TARGET = {target_bps} bps")
    print(f"{'='*80}")

    target_pct = target_bps / 10000

    print(f"\n{'H':>4} | {'Went Below':>12} | {'Case 1':>12} | {'Case 2':>12} | {'Case 3':>12} | {'Recovery':>10}")
    print(f"{'':>4} | {'(Total)':>12} | {'Wrong Dir':>12} | {'Quick Rec':>12} | {'Slow Rec':>12} | {'Rate':>10}")
    print("-" * 80)

    for H in HORIZONS:
        case1 = 0  # Wrong direction (never hits even with extended time)
        case2 = 0  # Quick recovery (hits within H)
        case3 = 0  # Slow recovery (hits after H but within extended time)

        clean_win = 0  # For reference: hit target without going below

        for i in sample_idx:
            entry = close[i]
            target_price = entry * (1 + target_pct)

            went_below = False
            hit_within_H = False
            hit_extended = False

            # Check within H bars
            for j in range(i+1, min(i+1+H, n)):
                if low[j] < entry:
                    went_below = True
                if high[j] >= target_price:
                    hit_within_H = True
                    break

            # If went below and didn't hit within H, check extended time
            if went_below and not hit_within_H:
                for j in range(i+1+H, min(i+1+EXTENDED_H, n)):
                    if high[j] >= target_price:
                        hit_extended = True
                        break

            # Classify
            if not went_below and hit_within_H:
                clean_win += 1
            elif went_below and hit_within_H:
                case2 += 1  # Quick recovery
            elif went_below and not hit_within_H and hit_extended:
                case3 += 1  # Slow recovery
            elif went_below and not hit_within_H and not hit_extended:
                case1 += 1  # Wrong direction

        went_below_total = case1 + case2 + case3

        if went_below_total > 0:
            case1_pct = case1 / went_below_total * 100
            case2_pct = case2 / went_below_total * 100
            case3_pct = case3 / went_below_total * 100
            recovery_rate = (case2 + case3) / went_below_total * 100
        else:
            case1_pct = case2_pct = case3_pct = recovery_rate = 0

        print(f"H={H:<2} | {went_below_total:>12,} | {case1_pct:>11.1f}% | {case2_pct:>11.1f}% | {case3_pct:>11.1f}% | {recovery_rate:>9.1f}%")

# =============================================================================
# DETAILED VIEW: ABSOLUTE NUMBERS
# =============================================================================
print("\n" + "=" * 80)
print("ABSOLUTE NUMBERS (Count of trades)")
print("=" * 80)

for target_bps in [8, 15, 25]:
    print(f"\n--- TARGET = {target_bps} bps ---")
    print(f"{'H':>4} | {'Went Below':>12} | {'Case 1':>10} | {'Case 2':>10} | {'Case 3':>10}")
    print("-" * 60)

    target_pct = target_bps / 10000

    for H in HORIZONS:
        case1 = 0
        case2 = 0
        case3 = 0

        for i in sample_idx:
            entry = close[i]
            target_price = entry * (1 + target_pct)

            went_below = False
            hit_within_H = False
            hit_extended = False

            for j in range(i+1, min(i+1+H, n)):
                if low[j] < entry:
                    went_below = True
                if high[j] >= target_price:
                    hit_within_H = True
                    break

            if went_below and not hit_within_H:
                for j in range(i+1+H, min(i+1+EXTENDED_H, n)):
                    if high[j] >= target_price:
                        hit_extended = True
                        break

            if went_below and hit_within_H:
                case2 += 1
            elif went_below and not hit_within_H and hit_extended:
                case3 += 1
            elif went_below and not hit_within_H and not hit_extended:
                case1 += 1

        went_below_total = case1 + case2 + case3
        print(f"H={H:<2} | {went_below_total:>12,} | {case1:>10,} | {case2:>10,} | {case3:>10,}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
INTERPRETATION:

Case 1 (Wrong Direction): Price went against you and NEVER recovered
  → Your direction prediction was wrong
  → Stop loss would have saved you from further losses

Case 2 (Quick Recovery): Price went against you but recovered within H bars
  → Your direction was right, just had temporary drawdown
  → Tight stop would have killed a winning trade

Case 3 (Slow Recovery): Price went against you, didn't recover within H, but recovered later
  → Your direction was right, but H was too short
  → Either increase H or accept timeout as partial loss

KEY INSIGHT:
- If (Case 2 + Case 3) >> Case 1: Direction is often right, just timing/patience issue
- If Case 1 >> (Case 2 + Case 3): Direction prediction is the problem

IMPLICATION FOR STOP LOSS:
- High Case 2: Use wider stops (don't cut winners)
- High Case 3: Consider longer horizons
- High Case 1: Your signal quality needs improvement
""")
