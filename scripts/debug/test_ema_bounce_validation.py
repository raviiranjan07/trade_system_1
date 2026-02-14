"""
EMA Bounce Test - OUT-OF-SAMPLE VALIDATION (2024 data)

Purpose: Validate that the EMA support/resistance pattern found in training data
also holds in unseen 2024 data.

Run: .venv/Scripts/python.exe scripts/debug/test_ema_bounce_validation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION - Same parameters as training test
# =============================================================================
EMA_PERIODS = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300]
NEAR_THRESHOLD_BPS = [3, 5, 8, 10, 15, 20, 30]
HORIZONS = [1, 2, 3, 5, 10, 15, 20, 30]

TEST_START = "2024-01-01"  # Out-of-sample data
SAMPLE_SIZE = 300000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("EMA BOUNCE TEST - OUT-OF-SAMPLE VALIDATION (2024+)")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

# Use TEST data only (2024+)
test_ohlcv = ohlcv[ohlcv.index >= TEST_START].copy()
print(f"TEST: {len(test_ohlcv):,} candles (from {TEST_START})")

# Calculate EMAs
print(f"\nCalculating {len(EMA_PERIODS)} EMAs...")
for period in EMA_PERIODS:
    test_ohlcv[f'ema{period}'] = test_ohlcv['close'].ewm(span=period, adjust=False).mean()
print("EMAs calculated.")

close = test_ohlcv['close'].values
n = len(test_ohlcv)

# Sample for speed
np.random.seed(42)
max_h = max(HORIZONS)
max_ema = max(EMA_PERIODS)
valid_start = max_ema + 10
sample_idx = np.random.choice(
    range(valid_start, n - max_h),
    size=min(SAMPLE_SIZE, n - max_h - valid_start),
    replace=False
)
print(f"Sampling {len(sample_idx):,} bars from 2024 data...")

# =============================================================================
# TEST
# =============================================================================
print("\n" + "=" * 80)
print("Testing EMA Support/Resistance on 2024 DATA (Out-of-Sample)")
print("=" * 80)

results = []
total_tests = len(EMA_PERIODS) * len(NEAR_THRESHOLD_BPS) * len(HORIZONS)
test_count = 0

for ema_period in EMA_PERIODS:
    ema_col = f'ema{ema_period}'
    ema_values = test_ohlcv[ema_col].values

    for near_bps in NEAR_THRESHOLD_BPS:
        near_pct = near_bps / 10000

        for H in HORIZONS:
            test_count += 1
            if test_count % 100 == 0:
                print(f"  Progress: {test_count}/{total_tests} ({100*test_count/total_tests:.0f}%)")

            from_below_up = 0
            from_below_down = 0
            from_above_up = 0
            from_above_down = 0

            for i in sample_idx:
                price = close[i]
                ema = ema_values[i]
                distance_pct = (price - ema) / ema

                if abs(distance_pct) > near_pct:
                    continue

                future_price = close[i + H]
                went_up = future_price > price

                if distance_pct < 0:
                    if went_up:
                        from_below_up += 1
                    else:
                        from_below_down += 1
                elif distance_pct > 0:
                    if went_up:
                        from_above_up += 1
                    else:
                        from_above_down += 1

            total_below = from_below_up + from_below_down
            total_above = from_above_up + from_above_down

            if total_below > 50 and total_above > 50:
                below_up_pct = 100 * from_below_up / total_below
                above_up_pct = 100 * from_above_up / total_above
                below_edge = below_up_pct - 50
                above_edge = 50 - above_up_pct
                combined_edge = below_edge + above_edge

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

print(f"\nCompleted {len(results)} valid tests")

# =============================================================================
# COMPARE WITH TRAINING RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON: TRAIN vs TEST")
print("=" * 80)

all_edges = [r['combined_edge'] for r in results]
avg_edge_test = np.mean(all_edges)
max_edge_test = max(all_edges)
min_edge_test = min(all_edges)

print(f"\n{'Metric':<30} {'TRAIN (2020-2023)':<20} {'TEST (2024)':<20} {'Diff':<10}")
print("-" * 80)
print(f"{'Average combined edge':<30} {'+2.08%':<20} {f'{avg_edge_test:+.2f}%':<20} {f'{avg_edge_test - 2.08:+.2f}%':<10}")
print(f"{'Max combined edge':<30} {'+6.33%':<20} {f'{max_edge_test:+.2f}%':<20} {f'{max_edge_test - 6.33:+.2f}%':<10}")
print(f"{'Min combined edge':<30} {'-1.55%':<20} {f'{min_edge_test:+.2f}%':<20}")

# =============================================================================
# TOP 10 RESULTS (2024)
# =============================================================================
print("\n" + "=" * 80)
print("TOP 10 RESULTS (2024 Out-of-Sample)")
print("=" * 80)

results_sorted = sorted(results, key=lambda x: x['combined_edge'], reverse=True)

print(f"\n{'EMA':<6} {'Near':<6} {'H':<4} {'Below->UP%':<12} {'Above->UP%':<12} {'Combined Edge':<15}")
print("-" * 70)

for r in results_sorted[:10]:
    print(f"EMA{r['ema']:<3} {r['near_bps']:<6} H={r['horizon']:<2} "
          f"{r['from_below_up_pct']:<12.1f} {r['from_above_up_pct']:<12.1f} {r['combined_edge']:>+13.1f}")

# =============================================================================
# SUMMARY BY PARAMETER (2024)
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY BY EMA (2024)")
print("=" * 80)

ema_summary = {}
for r in results:
    ema = r['ema']
    if ema not in ema_summary:
        ema_summary[ema] = []
    ema_summary[ema].append(r['combined_edge'])

print(f"\n{'EMA':<10} {'Avg Edge (2024)':<18} {'Avg Edge (Train)':<18} {'Diff':<10}")
print("-" * 60)

train_ema_edges = {10: 2.34, 15: 2.54, 20: 2.58, 25: 2.53, 30: 2.54, 40: 2.53,
                   50: 2.32, 75: 1.61, 100: 1.73, 150: 1.74, 200: 1.36, 300: 1.14}

for ema in sorted(ema_summary.keys()):
    test_avg = np.mean(ema_summary[ema])
    train_avg = train_ema_edges.get(ema, 0)
    diff = test_avg - train_avg
    print(f"EMA{ema:<7} {f'{test_avg:+.2f}%':<18} {f'+{train_avg:.2f}%':<18} {f'{diff:+.2f}%':<10}")

print("\n" + "=" * 80)
print("SUMMARY BY HORIZON (2024)")
print("=" * 80)

h_summary = {}
for r in results:
    h = r['horizon']
    if h not in h_summary:
        h_summary[h] = []
    h_summary[h].append(r['combined_edge'])

train_h_edges = {1: 0.11, 2: 0.71, 3: 1.19, 5: 1.86, 10: 2.58, 15: 3.08, 20: 3.34, 30: 3.77}

print(f"\n{'Horizon':<10} {'Avg Edge (2024)':<18} {'Avg Edge (Train)':<18} {'Diff':<10}")
print("-" * 60)

for h in sorted(h_summary.keys()):
    test_avg = np.mean(h_summary[h])
    train_avg = train_h_edges.get(h, 0)
    diff = test_avg - train_avg
    print(f"H={h:<7} {f'{test_avg:+.2f}%':<18} {f'+{train_avg:.2f}%':<18} {f'{diff:+.2f}%':<10}")

# =============================================================================
# VERDICT
# =============================================================================
print("\n" + "=" * 80)
print("VALIDATION VERDICT")
print("=" * 80)

positive_edge_count = sum(1 for e in all_edges if e > 2)
strong_edge_count = sum(1 for e in all_edges if e > 5)

print(f"\nTotal combinations: {len(results)}")
print(f"Combinations with edge > 2%: {positive_edge_count} ({100*positive_edge_count/len(results):.1f}%)")
print(f"Combinations with edge > 5%: {strong_edge_count} ({100*strong_edge_count/len(results):.1f}%)")

if avg_edge_test > 1.5:
    print(f"\n*** VALIDATED: Pattern holds in 2024 (avg edge {avg_edge_test:+.2f}%) ***")
    print("The EMA support/resistance effect is REAL, not overfit.")
elif avg_edge_test > 0.5:
    print(f"\n*** PARTIALLY VALIDATED: Pattern weaker in 2024 (avg edge {avg_edge_test:+.2f}%) ***")
    print("Effect exists but reduced compared to training data.")
else:
    print(f"\n*** NOT VALIDATED: Pattern does not hold in 2024 (avg edge {avg_edge_test:+.2f}%) ***")
    print("The pattern was likely overfit to training data.")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("experiments/ema_bounce_validation_2024.csv", index=False)
print(f"\nResults saved to: experiments/ema_bounce_validation_2024.csv")
