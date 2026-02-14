"""
EMA Bounce Test - 2025 DATA ONLY

Purpose: Validate EMA pattern on most recent data (2025)

Run: .venv/Scripts/python.exe scripts/debug/test_ema_bounce_validation_2025.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_PERIODS = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300]
NEAR_THRESHOLD_BPS = [3, 5, 8, 10, 15, 20, 30]
HORIZONS = [1, 2, 3, 5, 10, 15, 20, 30]

TEST_START = "2025-01-01"
SAMPLE_SIZE = 300000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("EMA BOUNCE TEST - 2025 DATA ONLY")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

test_ohlcv = ohlcv[ohlcv.index >= TEST_START].copy()
print(f"TEST: {len(test_ohlcv):,} candles (from {TEST_START})")

print(f"\nCalculating {len(EMA_PERIODS)} EMAs...")
for period in EMA_PERIODS:
    test_ohlcv[f'ema{period}'] = test_ohlcv['close'].ewm(span=period, adjust=False).mean()
print("EMAs calculated.")

close = test_ohlcv['close'].values
n = len(test_ohlcv)

np.random.seed(42)
max_h = max(HORIZONS)
max_ema = max(EMA_PERIODS)
valid_start = max_ema + 10
available_samples = n - max_h - valid_start
sample_size = min(SAMPLE_SIZE, available_samples)
sample_idx = np.random.choice(
    range(valid_start, n - max_h),
    size=sample_size,
    replace=False
)
print(f"Sampling {len(sample_idx):,} bars from 2025 data...")

# =============================================================================
# TEST
# =============================================================================
print("\nTesting EMA Support/Resistance on 2025 DATA...")

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
                print(f"  Progress: {test_count}/{total_tests}")

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

            if total_below > 30 and total_above > 30:
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
# RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON: TRAIN vs 2024 vs 2025")
print("=" * 80)

all_edges = [r['combined_edge'] for r in results]
avg_edge_2025 = np.mean(all_edges) if all_edges else 0
max_edge_2025 = max(all_edges) if all_edges else 0

# Previous values
train_avg = 2.08
val_2024_avg = 0.70

print(f"\n{'Period':<20} {'Avg Edge':<20} {'vs Train':<15}")
print("-" * 55)
print(f"{'Train (2020-2023)':<20} {'+2.08%':<20} {'baseline':<15}")
print(f"{'Test 2024':<20} {'+0.70%':<20} {'-66%':<15}")
print(f"{'Test 2025':<20} {f'{avg_edge_2025:+.2f}%':<20} {f'{100*(avg_edge_2025/train_avg - 1):.0f}%':<15}")

# Top results
print("\n" + "=" * 80)
print("TOP 10 EMA RESULTS (2025)")
print("=" * 80)

if results:
    results_sorted = sorted(results, key=lambda x: x['combined_edge'], reverse=True)
    print(f"\n{'EMA':<6} {'Near':<6} {'H':<4} {'Below->UP%':<12} {'Above->UP%':<12} {'Combined Edge':<15}")
    print("-" * 70)
    for r in results_sorted[:10]:
        print(f"EMA{r['ema']:<3} {r['near_bps']:<6} H={r['horizon']:<2} "
              f"{r['from_below_up_pct']:<12.1f} {r['from_above_up_pct']:<12.1f} {r['combined_edge']:>+13.1f}")

# Verdict
print("\n" + "=" * 80)
print("VERDICT: EMA PATTERN TREND")
print("=" * 80)

print(f"\n{'Year':<15} {'Edge':<15} {'Trend':<20}")
print("-" * 50)
print(f"{'Train':<15} {'+2.08%':<15} {'Baseline':<20}")
print(f"{'2024':<15} {'+0.70%':<15} {'Decayed 66%':<20}")
trend_2025 = "Decayed" if avg_edge_2025 < val_2024_avg else "Recovered"
decay_from_2024 = 100 * (1 - avg_edge_2025/val_2024_avg) if val_2024_avg > 0 else 0
print(f"{'2025':<15} {f'{avg_edge_2025:+.2f}%':<15} {f'{trend_2025} {abs(decay_from_2024):.0f}% from 2024':<20}")

if avg_edge_2025 > 1.5:
    print("\n*** EMA PATTERN RECOVERED in 2025 ***")
elif avg_edge_2025 > 0.5:
    print("\n*** EMA STILL WEAK in 2025 ***")
elif avg_edge_2025 > 0:
    print("\n*** EMA VERY WEAK in 2025 ***")
else:
    print("\n*** EMA PATTERN DEAD in 2025 ***")

# Compare with RSI
print("\n" + "=" * 80)
print("EMA vs RSI COMPARISON (All Periods)")
print("=" * 80)

print(f"\n{'Feature':<20} {'Train':<12} {'2024':<12} {'2025':<12} {'Decay':<12}")
print("-" * 70)
print(f"{'EMA Proximity':<20} {'+2.08%':<12} {'+0.70%':<12} {f'{avg_edge_2025:+.2f}%':<12} {'66%':<12}")
print(f"{'RSI Combined':<20} {'+5.32%':<12} {'+3.58%':<12} {'+2.97%':<12} {'44%':<12}")
