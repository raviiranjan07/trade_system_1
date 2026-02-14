"""
ANALYSIS-8 COMPLETE: Case 3 MAE (Drawdown for Slow Recovery Trades)
Extend to include H=600

Question: For Case 3 trades (hit target AFTER H bars), how much drawdown did they experience?

Run: .venv/Scripts/python.exe scripts/debug/test_analysis8_complete.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60, 120, 240, 360, 480, 600]
TARGETS_BPS = [8, 12, 15, 25, 50, 100, 150, 200]
SAMPLE_SIZE = 100000
EXTENDED_H = 600

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS-8 COMPLETE: CASE 3 MAE")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data: {len(train):,} candles")

close = train['close'].values
high = train['high'].values
low = train['low'].values
n = len(train)

np.random.seed(42)
max_h = EXTENDED_H + 100
valid_start = 100
sample_idx = np.random.choice(range(valid_start, n - max_h),
                               size=min(SAMPLE_SIZE, n - max_h - valid_start),
                               replace=False)
print(f"Sample size: {len(sample_idx):,}")


# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================
def analyze_case3_mae(indices, H, target_pct):
    """
    Analyze MAE for Case 3 trades (Slow Recovery).

    Case 3 = Went below entry, didn't hit within H, but hit AFTER H with extended time.

    Returns: Median, 75th, Max MAE for Case 3 trades
    """
    case3_mae_list = []

    for i in indices:
        if i + EXTENDED_H >= n:
            continue

        entry = close[i]
        target_price = entry * (1 + target_pct)

        went_below = False
        hit_within_H = False
        max_adverse = 0

        # Check within H bars
        for j in range(1, H + 1):
            if i + j >= n:
                break

            # Track MAE
            if low[i + j] < entry:
                went_below = True
                adverse = (entry - low[i + j]) / entry * 10000
                max_adverse = max(max_adverse, adverse)

            # Check if hit target
            if high[i + j] >= target_price:
                hit_within_H = True
                break

        # If hit within H, it's not Case 3
        if hit_within_H:
            continue

        # If didn't go below, it's not Case 3 (would be noise or clean miss)
        if not went_below:
            continue

        # Now check extended horizon to see if it eventually hits (Case 3)
        hit_extended = False
        for j in range(H + 1, EXTENDED_H + 1):
            if i + j >= n:
                break

            # Continue tracking MAE during extended period
            if low[i + j] < entry:
                adverse = (entry - low[i + j]) / entry * 10000
                max_adverse = max(max_adverse, adverse)

            # Check if hit target
            if high[i + j] >= target_price:
                hit_extended = True
                break

        # Only include if it eventually hit (Case 3)
        if hit_extended:
            case3_mae_list.append(max_adverse)

    if not case3_mae_list:
        return None

    return {
        'median': np.median(case3_mae_list),
        'p75': np.percentile(case3_mae_list, 75),
        'max': np.max(case3_mae_list),
        'count': len(case3_mae_list)
    }


# =============================================================================
# GENERATE DATA FOR ALL HORIZONS
# =============================================================================
print("\nGenerating data for all horizons...")

all_results = {}

for H in HORIZONS:
    all_results[H] = {}
    for target_bps in TARGETS_BPS:
        target_pct = target_bps / 10000
        result = analyze_case3_mae(sample_idx, H, target_pct)
        if result:
            all_results[H][target_bps] = result
    print(f"  H={H} done")

print("\nData collection complete!")


# =============================================================================
# OUTPUT: MARKDOWN FORMAT FOR DOCUMENTATION
# =============================================================================
print("\n" + "=" * 80)
print("MARKDOWN OUTPUT FOR DOCUMENTATION")
print("=" * 80)

print("\n### TABLE B1: Case 3 MAE Median (bp drawdown before eventually hitting target)")
print("\n| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |")
print("|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|")

for target_bps in TARGETS_BPS:
    row = f"| {target_bps}bp |"
    for H in HORIZONS:
        if target_bps in all_results[H] and all_results[H][target_bps]:
            row += f" {all_results[H][target_bps]['median']:.0f} |"
        else:
            row += " - |"
    print(row)


print("\n### TABLE B2: Case 3 MAE 75th Percentile (worse 25% of slow recovery trades)")
print("\n| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |")
print("|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|")

for target_bps in TARGETS_BPS:
    row = f"| {target_bps}bp |"
    for H in HORIZONS:
        if target_bps in all_results[H] and all_results[H][target_bps]:
            row += f" {all_results[H][target_bps]['p75']:.0f} |"
        else:
            row += " - |"
    print(row)


print("\n### TABLE B3: Case 3 MAE MAX (worst case drawdown in bp)")
print("\n| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |")
print("|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|")

for target_bps in TARGETS_BPS:
    row = f"| {target_bps}bp |"
    for H in HORIZONS:
        if target_bps in all_results[H] and all_results[H][target_bps]:
            row += f" {all_results[H][target_bps]['max']:.0f} |"
        else:
            row += " - |"
    print(row)


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("\n1. Case 3 MAE increases with longer wait time:")
for target_bps in [15, 50, 100]:
    if target_bps in TARGETS_BPS:
        print(f"\n   Target {target_bps}bp:")
        for H in [3, 60, 240, 600]:
            if H in all_results and target_bps in all_results[H]:
                r = all_results[H][target_bps]
                print(f"     H={H:3d}: Median MAE = {r['median']:5.0f}bp, 75th = {r['p75']:5.0f}bp")

print("\n2. Larger targets have smaller relative MAE:")
print("   (Because they have more room to move)")
for H in [60, 240]:
    if H in all_results:
        print(f"\n   H={H}:")
        for target_bps in [15, 50, 100, 200]:
            if target_bps in all_results[H]:
                r = all_results[H][target_bps]
                ratio = r['median'] / target_bps if target_bps > 0 else 0
                print(f"     {target_bps}bp: Median MAE = {r['median']:.0f}bp ({ratio:.1f}x target)")

print("\n3. Sample sizes (Case 3 counts):")
for H in [3, 60, 240, 600]:
    if H in all_results and 15 in all_results[H]:
        r = all_results[H][15]
        print(f"   H={H:3d}, Target=15bp: {r['count']:,} Case 3 trades")

print("\n4. Maximum drawdowns are EXTREME:")
for H in [60, 240, 600]:
    if H in all_results and 15 in all_results[H]:
        r = all_results[H][15]
        print(f"   H={H:3d}, Target=15bp: Max MAE = {r['max']:.0f}bp ({r['max']/100:.1f}%)")
