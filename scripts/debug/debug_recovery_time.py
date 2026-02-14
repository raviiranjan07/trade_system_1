"""
Recovery Time Analysis: How long did Case 3 (Slow Recovery) take to recover?

Run: .venv/Scripts/python.exe debug_recovery_time.py

Case 3: Went below entry, didn't hit target within H, but hit target later.
Question: How many bars AFTER H did it take to finally hit target?
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
print("RECOVERY TIME ANALYSIS: How long did Case 3 (Slow Recovery) take?")
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
        recovery_times = []  # Bars AFTER H to hit target

        for i in sample_idx:
            entry = close[i]
            target_price = entry * (1 + target_pct)

            went_below = False
            hit_within_H = False

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
                        # How many bars AFTER H did it take?
                        bars_after_H = j - (i + H)
                        recovery_times.append(bars_after_H)
                        break

        if len(recovery_times) > 0:
            arr = np.array(recovery_times)
            print(f"\n--- H = {H} bars ---")
            print(f"Case 3 count: {len(arr):,}")
            print(f"Recovery time (bars after H):")
            print(f"  Median:      {np.median(arr):.0f} bars")
            print(f"  25th pct:    {np.percentile(arr, 25):.0f} bars")
            print(f"  75th pct:    {np.percentile(arr, 75):.0f} bars")
            print(f"  90th pct:    {np.percentile(arr, 90):.0f} bars")
            print(f"  Max:         {np.max(arr):.0f} bars")

            # Breakdown by time buckets
            print(f"\n  Time buckets:")
            buckets = [10, 30, 60, 120, 240, 500]
            prev = 0
            for b in buckets:
                count = np.sum((arr > prev) & (arr <= b))
                pct = count / len(arr) * 100
                print(f"    {prev+1:>3}-{b:<3} bars: {pct:>5.1f}% ({count:,})")
                prev = b

# =============================================================================
# COMPACT SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 80)
print("COMPACT SUMMARY: Recovery Time for Case 3 (Median bars after H)")
print("=" * 80)

print(f"\n{'Target':<10} {'H':<6} {'Count':>10} {'Median':>10} {'75th':>10} {'90th':>10}")
print("-" * 60)

for target_bps in TARGETS:
    target_pct = target_bps / 10000

    for H in HORIZONS:
        recovery_times = []

        for i in sample_idx:
            entry = close[i]
            target_price = entry * (1 + target_pct)

            went_below = False
            hit_within_H = False

            for j in range(i+1, min(i+1+H, n)):
                if low[j] < entry:
                    went_below = True
                if high[j] >= target_price:
                    hit_within_H = True
                    break

            if went_below and not hit_within_H:
                for j in range(i+1+H, min(i+1+EXTENDED_H, n)):
                    if high[j] >= target_price:
                        bars_after_H = j - (i + H)
                        recovery_times.append(bars_after_H)
                        break

        if len(recovery_times) > 0:
            arr = np.array(recovery_times)
            med = np.median(arr)
            p75 = np.percentile(arr, 75)
            p90 = np.percentile(arr, 90)
            print(f"{target_bps}bp{'':<6} H={H:<4} {len(arr):>10,} {med:>9.0f} {p75:>9.0f} {p90:>9.0f}")

    print("-" * 60)

# =============================================================================
# TOTAL TIME TO HIT TARGET (from entry, not from H)
# =============================================================================
print("\n" + "=" * 80)
print("TOTAL TIME TO HIT TARGET (from entry) for Case 3")
print("=" * 80)
print("This is H + recovery_time = total bars from entry to target hit")

print(f"\n{'Target':<10} {'H':<6} {'Median Total':>15} {'75th Total':>15} {'90th Total':>15}")
print("-" * 65)

for target_bps in TARGETS:
    target_pct = target_bps / 10000

    for H in HORIZONS:
        total_times = []

        for i in sample_idx:
            entry = close[i]
            target_price = entry * (1 + target_pct)

            went_below = False
            hit_within_H = False

            for j in range(i+1, min(i+1+H, n)):
                if low[j] < entry:
                    went_below = True
                if high[j] >= target_price:
                    hit_within_H = True
                    break

            if went_below and not hit_within_H:
                for j in range(i+1+H, min(i+1+EXTENDED_H, n)):
                    if high[j] >= target_price:
                        total_time = j - i  # Total from entry
                        total_times.append(total_time)
                        break

        if len(total_times) > 0:
            arr = np.array(total_times)
            med = np.median(arr)
            p75 = np.percentile(arr, 75)
            p90 = np.percentile(arr, 90)
            print(f"{target_bps}bp{'':<6} H={H:<4} {med:>14.0f} {p75:>14.0f} {p90:>14.0f}")

    print("-" * 65)

print("""
INTERPRETATION:
- Recovery time = bars AFTER H before hitting target
- Total time = H + recovery time = total bars from entry

Key insight: If median recovery is 50 bars after H=3,
then total time = 3 + 50 = 53 bars from entry.
This tells you how long you'd need to hold to eventually win.
""")
