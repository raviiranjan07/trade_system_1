"""
RSI Validation Test - 2025 DATA ONLY

Purpose: Validate RSI pattern on most recent data (2025)

Run: .venv/Scripts/python.exe scripts/debug/test_rsi_validation_2025.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
RSI_PERIODS = [7, 10, 14, 20, 30]
OVERSOLD_LEVELS = [20, 25, 30, 35, 40]
OVERBOUGHT_LEVELS = [60, 65, 70, 75, 80]
HORIZONS = [1, 2, 3, 5, 10, 15, 20, 30]

TEST_START = "2025-01-01"
SAMPLE_SIZE = 300000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("RSI VALIDATION TEST - 2025 DATA ONLY")
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

print(f"\nCalculating RSI...")
for period in RSI_PERIODS:
    test_ohlcv[f'rsi{period}'] = calculate_rsi(test_ohlcv['close'], period)

close = test_ohlcv['close'].values
n = len(test_ohlcv)

np.random.seed(42)
max_h = max(HORIZONS)
max_rsi_period = max(RSI_PERIODS)
valid_start = max_rsi_period + 10
available_samples = n - max_h - valid_start
sample_size = min(SAMPLE_SIZE, available_samples)
sample_idx = np.random.choice(
    range(valid_start, n - max_h),
    size=sample_size,
    replace=False
)
print(f"Sampling {len(sample_idx):,} bars from 2025 data...")

# =============================================================================
# TEST OVERSOLD
# =============================================================================
print("\nTesting OVERSOLD...")
oversold_results = []

for rsi_period in RSI_PERIODS:
    rsi_col = f'rsi{rsi_period}'
    rsi_values = test_ohlcv[rsi_col].values

    for oversold_level in OVERSOLD_LEVELS:
        for H in HORIZONS:
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
            if total > 30:
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
print("Testing OVERBOUGHT...")
overbought_results = []

for rsi_period in RSI_PERIODS:
    rsi_col = f'rsi{rsi_period}'
    rsi_values = test_ohlcv[rsi_col].values

    for overbought_level in OVERBOUGHT_LEVELS:
        for H in HORIZONS:
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
            if total > 30:
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
# RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON: TRAIN vs 2024 vs 2025")
print("=" * 80)

all_oversold_edges = [r['edge'] for r in oversold_results]
all_overbought_edges = [r['edge'] for r in overbought_results]

avg_oversold_2025 = np.mean(all_oversold_edges) if all_oversold_edges else 0
avg_overbought_2025 = np.mean(all_overbought_edges) if all_overbought_edges else 0
combined_avg_2025 = (avg_oversold_2025 + avg_overbought_2025) / 2

# Previous values
train_combined = 5.32
val_2024_combined = 3.58

print(f"\n{'Period':<20} {'Combined Avg Edge':<20} {'vs Train':<15}")
print("-" * 55)
print(f"{'Train (2020-2023)':<20} {'+5.32%':<20} {'baseline':<15}")
print(f"{'Test 2024':<20} {'+3.58%':<20} {'-33%':<15}")
print(f"{'Test 2025':<20} {f'{combined_avg_2025:+.2f}%':<20} {f'{100*(combined_avg_2025/train_combined - 1):.0f}%':<15}")

print(f"\n{'Breakdown':<20} {'Train':<12} {'2024':<12} {'2025':<12}")
print("-" * 55)
print(f"{'Oversold edge':<20} {'+5.72%':<12} {'+4.08%':<12} {f'{avg_oversold_2025:+.2f}%':<12}")
print(f"{'Overbought edge':<20} {'+4.91%':<12} {'+3.08%':<12} {f'{avg_overbought_2025:+.2f}%':<12}")

# Top results
print("\n" + "=" * 80)
print("TOP 10 OVERSOLD RESULTS (2025)")
print("=" * 80)

if oversold_results:
    oversold_sorted = sorted(oversold_results, key=lambda x: x['edge'], reverse=True)
    print(f"\n{'RSI':<6} {'Level':<8} {'H':<4} {'Count':<10} {'UP%':<10} {'Edge':<10}")
    print("-" * 60)
    for r in oversold_sorted[:10]:
        print(f"RSI{r['rsi_period']:<3} <{r['level']:<6} H={r['horizon']:<2} "
              f"{r['count']:<10} {r['up_pct']:<10.1f} {r['edge']:>+9.1f}")

print("\n" + "=" * 80)
print("TOP 10 OVERBOUGHT RESULTS (2025)")
print("=" * 80)

if overbought_results:
    overbought_sorted = sorted(overbought_results, key=lambda x: x['edge'], reverse=True)
    print(f"\n{'RSI':<6} {'Level':<8} {'H':<4} {'Count':<10} {'DOWN%':<10} {'Edge':<10}")
    print("-" * 60)
    for r in overbought_sorted[:10]:
        print(f"RSI{r['rsi_period']:<3} >{r['level']:<6} H={r['horizon']:<2} "
              f"{r['count']:<10} {r['down_pct']:<10.1f} {r['edge']:>+9.1f}")

# Verdict
print("\n" + "=" * 80)
print("VERDICT: RSI PATTERN TREND")
print("=" * 80)

print(f"\n{'Year':<15} {'Edge':<15} {'Trend':<20}")
print("-" * 50)
print(f"{'Train':<15} {'+5.32%':<15} {'Baseline':<20}")
print(f"{'2024':<15} {'+3.58%':<15} {'Decayed 33%':<20}")
trend_2025 = "Decayed" if combined_avg_2025 < val_2024_combined else "Recovered"
decay_from_2024 = 100 * (1 - combined_avg_2025/val_2024_combined) if val_2024_combined > 0 else 0
print(f"{'2025':<15} {f'{combined_avg_2025:+.2f}%':<15} {f'{trend_2025} {abs(decay_from_2024):.0f}% from 2024':<20}")

if combined_avg_2025 > 3:
    print("\n*** RSI STILL STRONG in 2025 ***")
elif combined_avg_2025 > 1.5:
    print("\n*** RSI MODERATE in 2025 ***")
elif combined_avg_2025 > 0.5:
    print("\n*** RSI WEAK in 2025 ***")
else:
    print("\n*** RSI PATTERN DEAD in 2025 ***")
