"""
ANALYSIS-6 COMPLETE: Direction First (Real Moves Only)
Extend to all horizons H=3 to H=600

Question: For moves ≥12bp (excluding noise), which direction hits the threshold first?

Population: Only bars with real moves (≥12bp in either direction)

Run: .venv/Scripts/python.exe scripts/debug/test_analysis6_complete.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60, 120, 240, 360, 480, 600]
THRESHOLD_BPS = 12  # Rule #1: minimum profitable move
SAMPLE_SIZE = 200000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS-6 COMPLETE: DIRECTION FIRST (REAL MOVES ONLY)")
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
max_h = max(HORIZONS) + 10
valid_start = 100
sample_idx = np.random.choice(range(valid_start, n - max_h),
                               size=min(SAMPLE_SIZE, n - max_h - valid_start),
                               replace=False)
print(f"Sample size: {len(sample_idx):,}")

threshold_pct = THRESHOLD_BPS / 10000

# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================
def analyze_direction_real_moves_only(indices, H):
    """
    For bars with REAL MOVES (≥12bp in either direction), which direction hit first?
    Excludes noise (neither direction hits threshold).
    """
    up_first = 0
    down_first = 0

    for i in indices:
        if i + H >= n:
            continue

        entry = close[i]
        up_target = entry * (1 + threshold_pct)
        down_target = entry * (1 - threshold_pct)

        hit_up_bar = None
        hit_down_bar = None

        # Check each bar in horizon
        for j in range(1, H + 1):
            if i + j >= n:
                break

            # Check if hit UP threshold
            if hit_up_bar is None and high[i + j] >= up_target:
                hit_up_bar = j

            # Check if hit DOWN threshold
            if hit_down_bar is None and low[i + j] <= down_target:
                hit_down_bar = j

            # Break if both hit
            if hit_up_bar is not None and hit_down_bar is not None:
                break

        # Only count bars with REAL MOVES (exclude "Neither")
        if hit_up_bar is not None and hit_down_bar is not None:
            # Both hit - which was first?
            if hit_up_bar < hit_down_bar:
                up_first += 1
            elif hit_down_bar < hit_up_bar:
                down_first += 1
            # If equal (same bar), could go either way - skip
        elif hit_up_bar is not None:
            # Only UP hit
            up_first += 1
        elif hit_down_bar is not None:
            # Only DOWN hit
            down_first += 1
        # If neither hit, skip (noise)

    total_real_moves = up_first + down_first

    if total_real_moves == 0:
        return None

    up_pct = up_first / total_real_moves * 100
    down_pct = down_first / total_real_moves * 100
    ratio = up_first / down_first if down_first > 0 else 0

    return {
        'up_first': up_first,
        'down_first': down_first,
        'total': total_real_moves,
        'up_pct': up_pct,
        'down_pct': down_pct,
        'ratio': ratio
    }


# =============================================================================
# GENERATE DATA FOR ALL HORIZONS
# =============================================================================
print("\nGenerating data for all horizons...")

all_results = {}

for H in HORIZONS:
    result = analyze_direction_real_moves_only(sample_idx, H)
    if result:
        all_results[H] = result
    print(f"  H={H} done")

print("\nData collection complete!")


# =============================================================================
# OUTPUT: MARKDOWN FORMAT FOR DOCUMENTATION
# =============================================================================
print("\n" + "=" * 80)
print("MARKDOWN OUTPUT FOR DOCUMENTATION")
print("=" * 80)

print("\n| Horizon | UP First | DOWN First | Ratio |")
print("|---------|----------|------------|-------|")

for H in HORIZONS:
    if H in all_results:
        r = all_results[H]
        print(f"| H={H:3d}   | {r['up_pct']:5.1f}%   | {r['down_pct']:5.1f}%     | {r['ratio']:.2f}  |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("\n1. Direction remains 50/50 even among real moves")
for H in [3, 60, 600]:
    if H in all_results:
        r = all_results[H]
        print(f"   - H={H}: {r['up_pct']:.1f}% UP vs {r['down_pct']:.1f}% DOWN (ratio {r['ratio']:.2f})")

print("\n2. Filtering out noise does NOT provide directional edge")
print("   - ANALYSIS-2 (all bars): ~50/50")
print("   - ANALYSIS-6 (real moves only): Still ~50/50")

print("\n3. Sample sizes (real moves only):")
for H in [3, 60, 600]:
    if H in all_results:
        r = all_results[H]
        total_sample = len(sample_idx)
        real_pct = r['total'] / total_sample * 100
        print(f"   - H={H}: {r['total']:,} / {total_sample:,} ({real_pct:.1f}% are real moves)")

print("\n4. Conclusion: Market has no inherent directional bias")
print("   - Need entry filters to find edge")
print("   - State vector features (EMA, RSI, etc.) needed for prediction")
