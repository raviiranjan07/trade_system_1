"""
RSI Validation Test - OUT-OF-SAMPLE (2024 data)

Purpose: Validate that RSI pattern found in training data
also holds in unseen 2024 data.

Run: .venv/Scripts/python.exe scripts/debug/test_rsi_validation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION - Same parameters as training test
# =============================================================================
RSI_PERIODS = [7, 10, 14, 20, 30]
OVERSOLD_LEVELS = [20, 25, 30, 35, 40]
OVERBOUGHT_LEVELS = [60, 65, 70, 75, 80]
HORIZONS = [1, 2, 3, 5, 10, 15, 20, 30]

TEST_START = "2024-01-01"
SAMPLE_SIZE = 300000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("RSI VALIDATION TEST - OUT-OF-SAMPLE (2024+)")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

test_ohlcv = ohlcv[ohlcv.index >= TEST_START].copy()
print(f"TEST: {len(test_ohlcv):,} candles (from {TEST_START})")

def calculate_rsi(close, period):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

print(f"\nCalculating {len(RSI_PERIODS)} RSI periods...")
for period in RSI_PERIODS:
    test_ohlcv[f'rsi{period}'] = calculate_rsi(test_ohlcv['close'], period)
print("RSI calculated.")

close = test_ohlcv['close'].values
n = len(test_ohlcv)

np.random.seed(42)
max_h = max(HORIZONS)
max_rsi_period = max(RSI_PERIODS)
valid_start = max_rsi_period + 10
sample_idx = np.random.choice(
    range(valid_start, n - max_h),
    size=min(SAMPLE_SIZE, n - max_h - valid_start),
    replace=False
)
print(f"Sampling {len(sample_idx):,} bars from 2024 data...")

# =============================================================================
# TEST OVERSOLD
# =============================================================================
print("\n" + "=" * 80)
print("Testing OVERSOLD on 2024 DATA")
print("=" * 80)

oversold_results = []
total_tests = len(RSI_PERIODS) * len(OVERSOLD_LEVELS) * len(HORIZONS)
test_count = 0

for rsi_period in RSI_PERIODS:
    rsi_col = f'rsi{rsi_period}'
    rsi_values = test_ohlcv[rsi_col].values

    for oversold_level in OVERSOLD_LEVELS:
        for H in HORIZONS:
            test_count += 1
            if test_count % 50 == 0:
                print(f"  Progress: {test_count}/{total_tests}")

            up_count = 0
            down_count = 0

            for i in sample_idx:
                rsi = rsi_values[i]
                if np.isnan(rsi) or rsi >= oversold_level:
                    continue
                price = close[i]
                future_price = close[i + H]
                if future_price > price:
                    up_count += 1
                else:
                    down_count += 1

            total = up_count + down_count
            if total > 50:
                up_pct = 100 * up_count / total
                edge = up_pct - 50

                oversold_results.append({
                    'rsi_period': rsi_period,
                    'level': oversold_level,
                    'horizon': H,
                    'count': total,
                    'up_pct': up_pct,
                    'edge': edge,
                })

# =============================================================================
# TEST OVERBOUGHT
# =============================================================================
print("\n" + "=" * 80)
print("Testing OVERBOUGHT on 2024 DATA")
print("=" * 80)

overbought_results = []
test_count = 0

for rsi_period in RSI_PERIODS:
    rsi_col = f'rsi{rsi_period}'
    rsi_values = test_ohlcv[rsi_col].values

    for overbought_level in OVERBOUGHT_LEVELS:
        for H in HORIZONS:
            test_count += 1
            if test_count % 50 == 0:
                print(f"  Progress: {test_count}/{total_tests}")

            up_count = 0
            down_count = 0

            for i in sample_idx:
                rsi = rsi_values[i]
                if np.isnan(rsi) or rsi <= overbought_level:
                    continue
                price = close[i]
                future_price = close[i + H]
                if future_price > price:
                    up_count += 1
                else:
                    down_count += 1

            total = up_count + down_count
            if total > 50:
                down_pct = 100 * down_count / total
                edge = down_pct - 50

                overbought_results.append({
                    'rsi_period': rsi_period,
                    'level': overbought_level,
                    'horizon': H,
                    'count': total,
                    'down_pct': down_pct,
                    'edge': edge,
                })

# =============================================================================
# COMPARISON: TRAIN vs TEST
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON: TRAIN (2020-2023) vs TEST (2024)")
print("=" * 80)

all_oversold_edges = [r['edge'] for r in oversold_results]
all_overbought_edges = [r['edge'] for r in overbought_results]

avg_oversold_test = np.mean(all_oversold_edges) if all_oversold_edges else 0
avg_overbought_test = np.mean(all_overbought_edges) if all_overbought_edges else 0
combined_avg_test = (avg_oversold_test + avg_overbought_test) / 2

# Training values from previous test
avg_oversold_train = 5.72
avg_overbought_train = 4.91
combined_avg_train = 5.32

print(f"\n{'Metric':<30} {'TRAIN (2020-2023)':<20} {'TEST (2024)':<20} {'Diff':<10}")
print("-" * 80)
print(f"{'Oversold avg edge':<30} {f'+{avg_oversold_train:.2f}%':<20} {f'{avg_oversold_test:+.2f}%':<20} {f'{avg_oversold_test - avg_oversold_train:+.2f}%':<10}")
print(f"{'Overbought avg edge':<30} {f'+{avg_overbought_train:.2f}%':<20} {f'{avg_overbought_test:+.2f}%':<20} {f'{avg_overbought_test - avg_overbought_train:+.2f}%':<10}")
print(f"{'Combined avg edge':<30} {f'+{combined_avg_train:.2f}%':<20} {f'{combined_avg_test:+.2f}%':<20} {f'{combined_avg_test - combined_avg_train:+.2f}%':<10}")

# =============================================================================
# TOP RESULTS (2024)
# =============================================================================
print("\n" + "=" * 80)
print("TOP 10 OVERSOLD RESULTS (2024)")
print("=" * 80)

oversold_sorted = sorted(oversold_results, key=lambda x: x['edge'], reverse=True)
print(f"\n{'RSI':<6} {'Level':<8} {'H':<4} {'Count':<10} {'UP%':<10} {'Edge':<10}")
print("-" * 60)
for r in oversold_sorted[:10]:
    print(f"RSI{r['rsi_period']:<3} <{r['level']:<6} H={r['horizon']:<2} "
          f"{r['count']:<10} {r['up_pct']:<10.1f} {r['edge']:>+9.1f}")

print("\n" + "=" * 80)
print("TOP 10 OVERBOUGHT RESULTS (2024)")
print("=" * 80)

overbought_sorted = sorted(overbought_results, key=lambda x: x['edge'], reverse=True)
print(f"\n{'RSI':<6} {'Level':<8} {'H':<4} {'Count':<10} {'DOWN%':<10} {'Edge':<10}")
print("-" * 60)
for r in overbought_sorted[:10]:
    print(f"RSI{r['rsi_period']:<3} >{r['level']:<6} H={r['horizon']:<2} "
          f"{r['count']:<10} {r['down_pct']:<10.1f} {r['edge']:>+9.1f}")

# =============================================================================
# VERDICT
# =============================================================================
print("\n" + "=" * 80)
print("VALIDATION VERDICT")
print("=" * 80)

positive_oversold = sum(1 for e in all_oversold_edges if e > 0)
positive_overbought = sum(1 for e in all_overbought_edges if e > 0)

print(f"\nOversold combinations with positive edge: {positive_oversold}/{len(all_oversold_edges)}")
print(f"Overbought combinations with positive edge: {positive_overbought}/{len(all_overbought_edges)}")

if combined_avg_test > 3:
    print(f"\n*** VALIDATED: RSI pattern STRONG in 2024 (avg edge {combined_avg_test:+.2f}%) ***")
elif combined_avg_test > 1.5:
    print(f"\n*** VALIDATED: RSI pattern holds in 2024 (avg edge {combined_avg_test:+.2f}%) ***")
elif combined_avg_test > 0.5:
    print(f"\n*** PARTIALLY VALIDATED: RSI pattern weaker in 2024 (avg edge {combined_avg_test:+.2f}%) ***")
else:
    print(f"\n*** NOT VALIDATED: RSI pattern does not hold in 2024 (avg edge {combined_avg_test:+.2f}%) ***")

decay_pct = 100 * (1 - combined_avg_test / combined_avg_train) if combined_avg_train > 0 else 0
print(f"\nDecay from training: {decay_pct:.0f}%")

# Compare with EMA
print("\n" + "=" * 80)
print("COMPARISON: RSI vs EMA (Out-of-Sample 2024)")
print("=" * 80)

print(f"\n{'Feature':<25} {'Train':<15} {'Test (2024)':<15} {'Decay':<10}")
print("-" * 65)
print(f"{'EMA Proximity':<25} {'+2.08%':<15} {'+0.70%':<15} {'-66%':<10}")
print(f"{'RSI Combined':<25} {f'+{combined_avg_train:.2f}%':<15} {f'{combined_avg_test:+.2f}%':<15} {f'{-decay_pct:.0f}%':<10}")

# Save
oversold_df = pd.DataFrame(oversold_results)
overbought_df = pd.DataFrame(overbought_results)
oversold_df.to_csv("experiments/rsi_oversold_validation_2024.csv", index=False)
overbought_df.to_csv("experiments/rsi_overbought_validation_2024.csv", index=False)
print(f"\nResults saved to experiments/rsi_*_validation_2024.csv")
