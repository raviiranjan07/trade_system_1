"""
Recovery Analysis with MAE: How much did price go down for each case?

Run: .venv/Scripts/python.exe debug_recovery_mae.py

For each case (1, 2, 3), what was the MAE (max drawdown)?
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60]
TARGETS = [8, 15, 25]  # Key targets
EXTENDED_H = 500

TRAIN_END = "2023-12-31"
SAMPLE_SIZE = 100000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("RECOVERY ANALYSIS WITH MAE: How much did price go down for each case?")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"Loaded {len(ohlcv):,} candles")

train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END]
print(f"TRAIN: {len(train_ohlcv):,} candles")

close = train_ohlcv['close'].values
high = train_ohlcv['high'].values
low = train_ohlcv['low'].values
n = len(train_ohlcv)

np.random.seed(42)
sample_idx = np.random.choice(n - EXTENDED_H, size=min(SAMPLE_SIZE, n - EXTENDED_H), replace=False)
print(f"Sampling {len(sample_idx):,} bars...")

# =============================================================================
# ANALYSIS
# =============================================================================

for target_bps in TARGETS:
    print(f"\n{'='*80}")
    print(f"TARGET = {target_bps} bps")
    print(f"{'='*80}")

    target_pct = target_bps / 10000

    for H in HORIZONS:
        # Collect MAE for each case
        case1_mae = []  # Wrong direction
        case2_mae = []  # Quick recovery
        case3_mae = []  # Slow recovery

        for i in sample_idx:
            entry = close[i]
            target_price = entry * (1 + target_pct)

            went_below = False
            hit_within_H = False
            hit_extended = False
            max_adverse = 0  # Track MAE

            # Check within H bars
            for j in range(i+1, min(i+1+H, n)):
                # Track MAE
                adverse = (entry - low[j]) / entry * 10000  # in bps
                if adverse > max_adverse:
                    max_adverse = adverse

                if low[j] < entry:
                    went_below = True

                if high[j] >= target_price:
                    hit_within_H = True
                    break

            # If went below and didn't hit within H, check extended time
            # Continue tracking MAE
            if went_below and not hit_within_H:
                for j in range(i+1+H, min(i+1+EXTENDED_H, n)):
                    adverse = (entry - low[j]) / entry * 10000
                    if adverse > max_adverse:
                        max_adverse = adverse

                    if high[j] >= target_price:
                        hit_extended = True
                        break

            # Classify and record MAE
            if went_below and hit_within_H:
                case2_mae.append(max_adverse)  # Quick recovery
            elif went_below and not hit_within_H and hit_extended:
                case3_mae.append(max_adverse)  # Slow recovery
            elif went_below and not hit_within_H and not hit_extended:
                case1_mae.append(max_adverse)  # Wrong direction

        # Print results
        print(f"\n--- H = {H} bars ---")
        print(f"{'Case':<20} {'Count':>10} {'Median MAE':>12} {'75th MAE':>10} {'90th MAE':>10}")
        print("-" * 65)

        if len(case2_mae) > 0:
            arr = np.array(case2_mae)
            print(f"{'Case 2 (Quick Rec)':<20} {len(arr):>10,} {np.median(arr):>11.1f}bp {np.percentile(arr, 75):>9.1f}bp {np.percentile(arr, 90):>9.1f}bp")

        if len(case3_mae) > 0:
            arr = np.array(case3_mae)
            print(f"{'Case 3 (Slow Rec)':<20} {len(arr):>10,} {np.median(arr):>11.1f}bp {np.percentile(arr, 75):>9.1f}bp {np.percentile(arr, 90):>9.1f}bp")

        if len(case1_mae) > 0:
            arr = np.array(case1_mae)
            print(f"{'Case 1 (Wrong Dir)':<20} {len(arr):>10,} {np.median(arr):>11.1f}bp {np.percentile(arr, 75):>9.1f}bp {np.percentile(arr, 90):>9.1f}bp")

# =============================================================================
# COMPACT SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 80)
print("COMPACT SUMMARY: MAE by Case (Median)")
print("=" * 80)

print(f"\n{'Target':<10} {'H':<6} {'Case 2 MAE':>12} {'Case 3 MAE':>12} {'Case 1 MAE':>12}")
print("-" * 55)

for target_bps in TARGETS:
    target_pct = target_bps / 10000

    for H in HORIZONS:
        case1_mae = []
        case2_mae = []
        case3_mae = []

        for i in sample_idx:
            entry = close[i]
            target_price = entry * (1 + target_pct)

            went_below = False
            hit_within_H = False
            hit_extended = False
            max_adverse = 0

            for j in range(i+1, min(i+1+H, n)):
                adverse = (entry - low[j]) / entry * 10000
                if adverse > max_adverse:
                    max_adverse = adverse
                if low[j] < entry:
                    went_below = True
                if high[j] >= target_price:
                    hit_within_H = True
                    break

            if went_below and not hit_within_H:
                for j in range(i+1+H, min(i+1+EXTENDED_H, n)):
                    adverse = (entry - low[j]) / entry * 10000
                    if adverse > max_adverse:
                        max_adverse = adverse
                    if high[j] >= target_price:
                        hit_extended = True
                        break

            if went_below and hit_within_H:
                case2_mae.append(max_adverse)
            elif went_below and not hit_within_H and hit_extended:
                case3_mae.append(max_adverse)
            elif went_below and not hit_within_H and not hit_extended:
                case1_mae.append(max_adverse)

        c2 = np.median(case2_mae) if len(case2_mae) > 0 else 0
        c3 = np.median(case3_mae) if len(case3_mae) > 0 else 0
        c1 = np.median(case1_mae) if len(case1_mae) > 0 else 0

        print(f"{target_bps}bp{'':<6} H={H:<4} {c2:>11.1f}bp {c3:>11.1f}bp {c1:>11.1f}bp")

    print("-" * 55)

print("""
INTERPRETATION:
- Case 2 MAE: How much drawdown before quick recovery (within H)
- Case 3 MAE: How much drawdown before slow recovery (after H)
- Case 1 MAE: How much drawdown when wrong direction (never recovered)

Key: Case 1 MAE should be >> Case 2/3 MAE if wrong direction goes deeper.
""")
