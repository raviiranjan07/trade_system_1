"""
MAE Analysis: How much did price go below entry before hitting target?

Run: .venv/Scripts/python.exe debug_mae_analysis.py

Two considerations:
1. Per Horizon - MAE for each H separately
2. Overall - MAE across all trades that hit target (regardless of when)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60]
TARGETS = [8, 10, 15, 20, 25, 30, 40, 50]  # bps

TRAIN_END = "2023-12-31"
SAMPLE_SIZE = 200000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("MAE ANALYSIS: How much did price go below before hitting target?")
print("=" * 70)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END]
print(f"TRAIN: {len(train_ohlcv):,} candles")

close = train_ohlcv['close'].values
high = train_ohlcv['high'].values
low = train_ohlcv['low'].values
n = len(train_ohlcv)

# Sample for speed
np.random.seed(42)
max_h = max(HORIZONS)
sample_idx = np.random.choice(n - max_h, size=min(SAMPLE_SIZE, n - max_h), replace=False)
print(f"Sampling {len(sample_idx):,} bars...")

# =============================================================================
# CONSIDERATION 1: MAE PER HORIZON
# =============================================================================
print("\n" + "=" * 70)
print("CONSIDERATION 1: MAE PER HORIZON")
print("=" * 70)
print("\nFor each horizon H, among trades that HIT target within H bars,")
print("how much did price go below entry before hitting target?")

for target_bps in [8, 15, 25]:  # Key targets
    print(f"\n--- Target = {target_bps} bps ---")
    print(f"{'H':>6} {'Hit Count':>12} {'Median MAE':>12} {'75th MAE':>10} {'90th MAE':>10} {'Max MAE':>10}")
    print("-" * 65)

    target_pct = target_bps / 10000

    for H in HORIZONS:
        mae_list = []

        for i in sample_idx:
            entry = close[i]
            target_price = entry * (1 + target_pct)

            # Track MAE before hitting target
            max_adverse = 0
            hit_target = False

            for j in range(i+1, min(i+1+H, n)):
                # Track adverse excursion
                adverse = (entry - low[j]) / entry * 10000  # in bps
                if adverse > max_adverse:
                    max_adverse = adverse

                # Check if hit target
                if high[j] >= target_price:
                    hit_target = True
                    mae_list.append(max_adverse)
                    break

        if len(mae_list) > 0:
            mae_arr = np.array(mae_list)
            print(f"H={H:<4} {len(mae_list):>12,} {np.median(mae_arr):>11.1f}bp {np.percentile(mae_arr, 75):>9.1f}bp {np.percentile(mae_arr, 90):>9.1f}bp {np.max(mae_arr):>9.1f}bp")
        else:
            print(f"H={H:<4} {'0':>12} {'N/A':>12} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

# =============================================================================
# CONSIDERATION 2: OVERALL MAE (regardless of when target is hit)
# =============================================================================
print("\n" + "=" * 70)
print("CONSIDERATION 2: OVERALL MAE (regardless of when target is hit)")
print("=" * 70)
print("\nUsing H=60 (longest horizon) to capture trades that eventually hit target.")
print("This shows MAE for ALL trades that hit target within 60 bars.")

H = 60  # Use longest horizon for "overall"

print(f"\n{'Target':>8} {'Hit Count':>12} {'Median MAE':>12} {'75th MAE':>10} {'90th MAE':>10} {'Max MAE':>10}")
print("-" * 65)

for target_bps in TARGETS:
    target_pct = target_bps / 10000
    mae_list = []

    for i in sample_idx:
        entry = close[i]
        target_price = entry * (1 + target_pct)

        max_adverse = 0
        hit_target = False

        for j in range(i+1, min(i+1+H, n)):
            adverse = (entry - low[j]) / entry * 10000
            if adverse > max_adverse:
                max_adverse = adverse

            if high[j] >= target_price:
                hit_target = True
                mae_list.append(max_adverse)
                break

    if len(mae_list) > 0:
        mae_arr = np.array(mae_list)
        print(f"{target_bps:>7}bp {len(mae_list):>12,} {np.median(mae_arr):>11.1f}bp {np.percentile(mae_arr, 75):>9.1f}bp {np.percentile(mae_arr, 90):>9.1f}bp {np.max(mae_arr):>9.1f}bp")

# =============================================================================
# CONSIDERATION 2b: TRULY OVERALL (no horizon limit)
# =============================================================================
print("\n" + "-" * 70)
print("CONSIDERATION 2b: TRULY OVERALL (no horizon limit, max 500 bars)")
print("-" * 70)
print("\nWhat is the MAE for trades that EVENTUALLY hit target (within 500 bars)?")

H_max = 500  # Very long horizon
sample_size_small = 50000  # Smaller sample for speed
sample_idx_small = np.random.choice(n - H_max, size=min(sample_size_small, n - H_max), replace=False)

print(f"\n{'Target':>8} {'Hit Count':>12} {'Median MAE':>12} {'75th MAE':>10} {'90th MAE':>10}")
print("-" * 55)

for target_bps in [8, 15, 25, 50]:
    target_pct = target_bps / 10000
    mae_list = []

    for i in sample_idx_small:
        entry = close[i]
        target_price = entry * (1 + target_pct)

        max_adverse = 0

        for j in range(i+1, min(i+1+H_max, n)):
            adverse = (entry - low[j]) / entry * 10000
            if adverse > max_adverse:
                max_adverse = adverse

            if high[j] >= target_price:
                mae_list.append(max_adverse)
                break

    if len(mae_list) > 0:
        mae_arr = np.array(mae_list)
        print(f"{target_bps:>7}bp {len(mae_list):>12,} {np.median(mae_arr):>11.1f}bp {np.percentile(mae_arr, 75):>9.1f}bp {np.percentile(mae_arr, 90):>9.1f}bp")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
KEY INSIGHTS:

1. PER HORIZON:
   - Shorter H = smaller MAE (less time for price to go against you)
   - Longer H = larger MAE (more time = more drawdown before winning)

2. OVERALL:
   - Even trades that eventually win often go significantly against you
   - This tells you what stop you need to survive until target is hit

3. IMPLICATION FOR STOP PLACEMENT:
   - If 90th MAE = X, then stop must be > X to avoid stopping out 90% of winners
   - But wider stop = bigger losses when you're wrong
   - This is the fundamental tradeoff
""")
