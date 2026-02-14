"""
Comprehensive EMA Bounce Test - Data-Driven Approach

Test EMA support/resistance pattern WITHOUT assuming which parameters work.
Let the DATA tell us:
- Which EMAs act as support/resistance?
- At what "near" distance does the pattern appear?
- At what horizon does the effect show?

Simple measurement: Does price direction change after touching EMA?
- From BELOW EMA: Does price go UP? (bounce off support)
- From ABOVE EMA: Does price go DOWN? (bounce off resistance)

Run: .venv/Scripts/python.exe scripts/debug/test_ema_bounce_comprehensive.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION - Test EVERYTHING, let data decide
# =============================================================================
EMA_PERIODS = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300]  # 12 EMAs
NEAR_THRESHOLD_BPS = [3, 5, 8, 10, 15, 20, 30]  # 7 "near" distances
HORIZONS = [1, 2, 3, 5, 10, 15, 20, 30]  # 8 horizons

# Total combinations: 12 x 7 x 8 = 672

TRAIN_END = "2023-12-31"
SAMPLE_SIZE = 300000  # Large sample for statistical significance

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("COMPREHENSIVE EMA BOUNCE TEST - Data-Driven")
print("=" * 80)
print(f"\nTesting {len(EMA_PERIODS)} EMAs x {len(NEAR_THRESHOLD_BPS)} near-thresholds x {len(HORIZONS)} horizons")
print(f"Total combinations: {len(EMA_PERIODS) * len(NEAR_THRESHOLD_BPS) * len(HORIZONS)}")

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END].copy()
print(f"TRAIN: {len(train_ohlcv):,} candles (up to {TRAIN_END})")

# Calculate ALL EMAs
print(f"\nCalculating {len(EMA_PERIODS)} EMAs...")
for period in EMA_PERIODS:
    train_ohlcv[f'ema{period}'] = train_ohlcv['close'].ewm(span=period, adjust=False).mean()
print("EMAs calculated.")

close = train_ohlcv['close'].values
n = len(train_ohlcv)

# Sample for speed
np.random.seed(42)
max_h = max(HORIZONS)
max_ema = max(EMA_PERIODS)
valid_start = max_ema + 10  # Ensure EMAs are stable
sample_idx = np.random.choice(
    range(valid_start, n - max_h),
    size=min(SAMPLE_SIZE, n - max_h - valid_start),
    replace=False
)
print(f"Sampling {len(sample_idx):,} bars...")

# =============================================================================
# TEST: Simple Direction Change
# =============================================================================
print("\n" + "=" * 80)
print("Testing: When price is NEAR EMA, what direction does it go?")
print("=" * 80)
print("\nMeasurement: Simple direction (close[i+H] vs close[i])")
print("From BELOW EMA: Expect UP (EMA = support)")
print("From ABOVE EMA: Expect DOWN (EMA = resistance)")
print("\nRunning tests...")

results = []
total_tests = len(EMA_PERIODS) * len(NEAR_THRESHOLD_BPS) * len(HORIZONS)
test_count = 0

for ema_period in EMA_PERIODS:
    ema_col = f'ema{ema_period}'
    ema_values = train_ohlcv[ema_col].values

    for near_bps in NEAR_THRESHOLD_BPS:
        near_pct = near_bps / 10000

        for H in HORIZONS:
            test_count += 1
            if test_count % 100 == 0:
                print(f"  Progress: {test_count}/{total_tests} ({100*test_count/total_tests:.0f}%)")

            # Count outcomes when price is near EMA
            from_below_up = 0
            from_below_down = 0
            from_above_up = 0
            from_above_down = 0

            for i in sample_idx:
                price = close[i]
                ema = ema_values[i]

                # Calculate distance from EMA
                distance_pct = (price - ema) / ema

                # Check if price is "near" EMA
                if abs(distance_pct) > near_pct:
                    continue  # Not near EMA

                # Future price
                future_price = close[i + H]
                went_up = future_price > price

                # Determine if approaching from below or above
                if distance_pct < 0:  # Price is below EMA
                    if went_up:
                        from_below_up += 1
                    else:
                        from_below_down += 1
                elif distance_pct > 0:  # Price is above EMA
                    if went_up:
                        from_above_up += 1
                    else:
                        from_above_down += 1
                # distance_pct == 0 is rare, skip

            # Calculate percentages
            total_below = from_below_up + from_below_down
            total_above = from_above_up + from_above_down

            if total_below > 50 and total_above > 50:  # Minimum sample
                below_up_pct = 100 * from_below_up / total_below
                above_up_pct = 100 * from_above_up / total_above

                # Edge calculation
                # If EMA is support: from_below should go UP (UP% > 50%)
                # If EMA is resistance: from_above should go DOWN (UP% < 50%)
                below_edge = below_up_pct - 50  # Positive = supports hypothesis
                above_edge = 50 - above_up_pct  # Positive = supports hypothesis
                combined_edge = below_edge + above_edge  # Higher = stronger pattern

                results.append({
                    'ema': ema_period,
                    'near_bps': near_bps,
                    'horizon': H,
                    'from_below_count': total_below,
                    'from_below_up_pct': below_up_pct,
                    'from_above_count': total_above,
                    'from_above_up_pct': above_up_pct,
                    'below_edge': below_edge,
                    'above_edge': above_edge,
                    'combined_edge': combined_edge,
                })

print(f"\nCompleted {len(results)} valid tests (out of {total_tests} combinations)")

# =============================================================================
# RESULTS - Sorted by Edge Strength
# =============================================================================
print("\n" + "=" * 80)
print("TOP 30 RESULTS (Sorted by Combined Edge)")
print("=" * 80)

# Sort by combined edge (highest = strongest pattern)
results_sorted = sorted(results, key=lambda x: x['combined_edge'], reverse=True)

print(f"\n{'EMA':<6} {'Near':<6} {'H':<4} {'Below->UP%':<12} {'Above->UP%':<12} {'Below Edge':<12} {'Above Edge':<12} {'Combined':<10}")
print("-" * 90)

for r in results_sorted[:30]:
    print(f"EMA{r['ema']:<3} {r['near_bps']:<6} H={r['horizon']:<2} "
          f"{r['from_below_up_pct']:<12.1f} {r['from_above_up_pct']:<12.1f} "
          f"{r['below_edge']:>+11.1f} {r['above_edge']:>+11.1f} {r['combined_edge']:>+9.1f}")

# =============================================================================
# BOTTOM 30 (Opposite Pattern)
# =============================================================================
print("\n" + "=" * 80)
print("BOTTOM 30 RESULTS (Opposite Pattern - EMA acts as 'magnet'?)")
print("=" * 80)

print(f"\n{'EMA':<6} {'Near':<6} {'H':<4} {'Below->UP%':<12} {'Above->UP%':<12} {'Below Edge':<12} {'Above Edge':<12} {'Combined':<10}")
print("-" * 90)

for r in results_sorted[-30:]:
    print(f"EMA{r['ema']:<3} {r['near_bps']:<6} H={r['horizon']:<2} "
          f"{r['from_below_up_pct']:<12.1f} {r['from_above_up_pct']:<12.1f} "
          f"{r['below_edge']:>+11.1f} {r['above_edge']:>+11.1f} {r['combined_edge']:>+9.1f}")

# =============================================================================
# SUMMARY BY EMA
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY BY EMA (Average Edge Across All Horizons & Near-Thresholds)")
print("=" * 80)

ema_summary = {}
for r in results:
    ema = r['ema']
    if ema not in ema_summary:
        ema_summary[ema] = {'edges': [], 'counts': []}
    ema_summary[ema]['edges'].append(r['combined_edge'])
    ema_summary[ema]['counts'].append(r['from_below_count'] + r['from_above_count'])

print(f"\n{'EMA':<8} {'Avg Edge':<12} {'Max Edge':<12} {'Min Edge':<12} {'Avg Count':<12}")
print("-" * 60)

for ema in sorted(ema_summary.keys()):
    edges = ema_summary[ema]['edges']
    counts = ema_summary[ema]['counts']
    print(f"EMA{ema:<5} {np.mean(edges):>+10.2f} {max(edges):>+10.2f} {min(edges):>+10.2f} {np.mean(counts):>10.0f}")

# =============================================================================
# SUMMARY BY HORIZON
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY BY HORIZON (Average Edge Across All EMAs & Near-Thresholds)")
print("=" * 80)

h_summary = {}
for r in results:
    h = r['horizon']
    if h not in h_summary:
        h_summary[h] = {'edges': [], 'counts': []}
    h_summary[h]['edges'].append(r['combined_edge'])
    h_summary[h]['counts'].append(r['from_below_count'] + r['from_above_count'])

print(f"\n{'Horizon':<10} {'Avg Edge':<12} {'Max Edge':<12} {'Min Edge':<12} {'Avg Count':<12}")
print("-" * 60)

for h in sorted(h_summary.keys()):
    edges = h_summary[h]['edges']
    counts = h_summary[h]['counts']
    print(f"H={h:<7} {np.mean(edges):>+10.2f} {max(edges):>+10.2f} {min(edges):>+10.2f} {np.mean(counts):>10.0f}")

# =============================================================================
# SUMMARY BY NEAR THRESHOLD
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY BY NEAR-THRESHOLD (Average Edge Across All EMAs & Horizons)")
print("=" * 80)

near_summary = {}
for r in results:
    near = r['near_bps']
    if near not in near_summary:
        near_summary[near] = {'edges': [], 'counts': []}
    near_summary[near]['edges'].append(r['combined_edge'])
    near_summary[near]['counts'].append(r['from_below_count'] + r['from_above_count'])

print(f"\n{'Near (bp)':<12} {'Avg Edge':<12} {'Max Edge':<12} {'Min Edge':<12} {'Avg Count':<12}")
print("-" * 60)

for near in sorted(near_summary.keys()):
    edges = near_summary[near]['edges']
    counts = near_summary[near]['counts']
    print(f"{near:<12} {np.mean(edges):>+10.2f} {max(edges):>+10.2f} {min(edges):>+10.2f} {np.mean(counts):>10.0f}")

# =============================================================================
# OVERALL VERDICT
# =============================================================================
print("\n" + "=" * 80)
print("OVERALL VERDICT")
print("=" * 80)

all_edges = [r['combined_edge'] for r in results]
avg_edge = np.mean(all_edges)
max_edge = max(all_edges)
min_edge = min(all_edges)
positive_edge_count = sum(1 for e in all_edges if e > 2)  # >2% edge
strong_edge_count = sum(1 for e in all_edges if e > 5)    # >5% edge

print(f"\nTotal combinations tested: {len(results)}")
print(f"Average combined edge: {avg_edge:+.2f}%")
print(f"Max combined edge: {max_edge:+.2f}%")
print(f"Min combined edge: {min_edge:+.2f}%")
print(f"\nCombinations with edge > 2%: {positive_edge_count} ({100*positive_edge_count/len(results):.1f}%)")
print(f"Combinations with edge > 5%: {strong_edge_count} ({100*strong_edge_count/len(results):.1f}%)")

if avg_edge > 2:
    print("\n*** PATTERN EXISTS: EMA acts as support/resistance on average ***")
elif avg_edge > 0.5:
    print("\n*** WEAK PATTERN: Small effect, may not overcome fees ***")
elif avg_edge < -2:
    print("\n*** REVERSE PATTERN: EMA acts as 'magnet' (mean-reversion) ***")
else:
    print("\n*** NO PATTERN: EMA proximity does not predict direction ***")

# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================
results_df = pd.DataFrame(results)
results_df.to_csv("experiments/ema_bounce_comprehensive_results.csv", index=False)
print(f"\nResults saved to: experiments/ema_bounce_comprehensive_results.csv")
