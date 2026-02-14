"""
Bidirectional Analysis: LONG vs SHORT outcomes for each bar.

Run: .venv/Scripts/python.exe debug_bidirectional_analysis.py

For each bar, classify into one of 4 mutually exclusive categories:
1. LONG_FIRST: Hit +target BEFORE hitting -target
2. SHORT_FIRST: Hit -target BEFORE hitting +target
3. BOTH_SAME: Hit both targets on same bar (rare)
4. NEITHER: Didn't hit either target within H bars

This gives complete 100% breakdown for trading decisions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
TRAIN_END = "2023-12-31"
TARGETS = [8, 15, 25, 50]  # bps
HORIZONS = [3, 5, 10, 15, 30, 60]
SAMPLE_SIZE = 200000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("BIDIRECTIONAL ANALYSIS: LONG vs SHORT")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END]
print(f"Train data: {len(train_ohlcv):,} candles")

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
# BIDIRECTIONAL CLASSIFICATION
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS: Which direction hits first?")
print("=" * 80)

results = []

for target_bps in TARGETS:
    print(f"\n{'='*60}")
    print(f"TARGET = {target_bps} bps")
    print(f"{'='*60}")
    print(f"\n{'H':<6} {'LONG First':<14} {'SHORT First':<14} {'Both Same':<12} {'Neither':<12} {'Total':<8}")
    print("-" * 70)

    target_pct = target_bps / 10000

    for H in HORIZONS:
        long_first = 0
        short_first = 0
        both_same = 0
        neither = 0

        for i in sample_idx:
            entry = close[i]
            long_target = entry * (1 + target_pct)
            short_target = entry * (1 - target_pct)

            long_hit_bar = None
            short_hit_bar = None

            # Find when each target is hit
            for j in range(i + 1, min(i + 1 + H, n)):
                # Check LONG target (price goes UP to hit high)
                if long_hit_bar is None and high[j] >= long_target:
                    long_hit_bar = j - i

                # Check SHORT target (price goes DOWN to hit low)
                if short_hit_bar is None and low[j] <= short_target:
                    short_hit_bar = j - i

                # Early exit if both found
                if long_hit_bar is not None and short_hit_bar is not None:
                    break

            # Classify
            if long_hit_bar is None and short_hit_bar is None:
                neither += 1
            elif long_hit_bar is not None and short_hit_bar is None:
                long_first += 1
            elif long_hit_bar is None and short_hit_bar is not None:
                short_first += 1
            elif long_hit_bar < short_hit_bar:
                long_first += 1
            elif short_hit_bar < long_hit_bar:
                short_first += 1
            else:  # Same bar
                both_same += 1

        total = long_first + short_first + both_same + neither

        print(f"H={H:<4} {long_first/total*100:>12.1f}% {short_first/total*100:>12.1f}% "
              f"{both_same/total*100:>10.1f}% {neither/total*100:>10.1f}% {total:>8,}")

        results.append({
            "target_bps": target_bps,
            "horizon": H,
            "long_first_pct": long_first / total * 100,
            "short_first_pct": short_first / total * 100,
            "both_same_pct": both_same / total * 100,
            "neither_pct": neither / total * 100,
            "total": total,
        })

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Tradeable Opportunities (LONG or SHORT)")
print("=" * 80)

print(f"\n{'Target':<10} {'H':<6} {'Tradeable':<12} {'LONG':<10} {'SHORT':<10} {'Ratio':<10} {'Stagnation':<12}")
print("-" * 75)

for r in results:
    tradeable = r["long_first_pct"] + r["short_first_pct"] + r["both_same_pct"]
    ratio = r["long_first_pct"] / r["short_first_pct"] if r["short_first_pct"] > 0 else float('inf')

    print(f"{r['target_bps']}bp{'':<6} H={r['horizon']:<4} {tradeable:>10.1f}% "
          f"{r['long_first_pct']:>8.1f}% {r['short_first_pct']:>8.1f}% "
          f"{ratio:>8.2f}x {r['neither_pct']:>10.1f}%")

# =============================================================================
# KEY INSIGHT
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("""
1. TRADEABLE = LONG_FIRST + SHORT_FIRST + BOTH_SAME
   - This is the % of bars where a trade could have hit target
   - If you could predict direction, you'd capture these

2. STAGNATION = NEITHER
   - True noise - price didn't move enough in either direction
   - No trading opportunity regardless of direction prediction

3. LONG/SHORT RATIO ≈ 1.0
   - Confirms direction is ~50/50
   - No inherent LONG or SHORT bias in the market

4. IMPLICATION:
   - At H=60, Target=8bp: ~83% tradeable, ~17% stagnation
   - The "Never Hit LONG" (61.6% at H=3) breaks down to:
     - SHORT winners: ~38%
     - True stagnation: ~24%
""")

# =============================================================================
# DETAILED BREAKDOWN OF "NEVER HIT LONG"
# =============================================================================
print("\n" + "=" * 80)
print("RECONCILIATION: Breaking down 'Never Hit LONG'")
print("=" * 80)

print(f"\nFor Target=8bp, H=3:")
print(f"  From Clean Win table: Never Hit LONG = 61.6%")
print(f"  This breaks down to:")

# Find the specific result
for r in results:
    if r["target_bps"] == 8 and r["horizon"] == 3:
        print(f"    - SHORT winners: {r['short_first_pct']:.1f}%")
        print(f"    - True stagnation: {r['neither_pct']:.1f}%")
        print(f"    - Both same bar: {r['both_same_pct']:.1f}%")
        total_never_long = r["short_first_pct"] + r["neither_pct"] + r["both_same_pct"]
        print(f"    - TOTAL: {total_never_long:.1f}% (should ≈ 61.6%)")
        break

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
