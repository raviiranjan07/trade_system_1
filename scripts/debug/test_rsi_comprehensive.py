"""
Comprehensive RSI Test - Data-Driven Approach

Test RSI as a directional predictor.
Let the DATA tell us if RSI predicts direction.

Hypothesis:
- RSI < 30 (oversold) -> Price should go UP (mean reversion)
- RSI > 70 (overbought) -> Price should go DOWN (mean reversion)

Run: .venv/Scripts/python.exe scripts/debug/test_rsi_comprehensive.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION - Test EVERYTHING, let data decide
# =============================================================================
RSI_PERIODS = [7, 10, 14, 20, 30]  # 5 RSI periods
OVERSOLD_LEVELS = [20, 25, 30, 35, 40]  # 5 oversold thresholds
OVERBOUGHT_LEVELS = [60, 65, 70, 75, 80]  # 5 overbought thresholds
HORIZONS = [1, 2, 3, 5, 10, 15, 20, 30]  # 8 horizons

TRAIN_END = "2023-12-31"
SAMPLE_SIZE = 300000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("COMPREHENSIVE RSI TEST - Data-Driven")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END].copy()
print(f"TRAIN: {len(train_ohlcv):,} candles (up to {TRAIN_END})")

# Calculate RSI for all periods
print(f"\nCalculating {len(RSI_PERIODS)} RSI periods...")

def calculate_rsi(close, period):
    """Calculate RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

for period in RSI_PERIODS:
    train_ohlcv[f'rsi{period}'] = calculate_rsi(train_ohlcv['close'], period)
print("RSI calculated.")

close = train_ohlcv['close'].values
n = len(train_ohlcv)

# Sample for speed
np.random.seed(42)
max_h = max(HORIZONS)
max_rsi_period = max(RSI_PERIODS)
valid_start = max_rsi_period + 10
sample_idx = np.random.choice(
    range(valid_start, n - max_h),
    size=min(SAMPLE_SIZE, n - max_h - valid_start),
    replace=False
)
print(f"Sampling {len(sample_idx):,} bars...")

# =============================================================================
# TEST 1: Oversold (RSI < threshold) -> Does price go UP?
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: OVERSOLD (RSI < threshold) -> Does price go UP?")
print("=" * 80)

oversold_results = []
total_tests = len(RSI_PERIODS) * len(OVERSOLD_LEVELS) * len(HORIZONS)
test_count = 0

for rsi_period in RSI_PERIODS:
    rsi_col = f'rsi{rsi_period}'
    rsi_values = train_ohlcv[rsi_col].values

    for oversold_level in OVERSOLD_LEVELS:
        for H in HORIZONS:
            test_count += 1
            if test_count % 50 == 0:
                print(f"  Progress: {test_count}/{total_tests} ({100*test_count/total_tests:.0f}%)")

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
            if total > 100:
                up_pct = 100 * up_count / total
                edge = up_pct - 50  # Positive = supports hypothesis

                oversold_results.append({
                    'rsi_period': rsi_period,
                    'level': oversold_level,
                    'horizon': H,
                    'count': total,
                    'up_pct': up_pct,
                    'edge': edge,
                })

print(f"\nCompleted {len(oversold_results)} oversold tests")

# =============================================================================
# TEST 2: Overbought (RSI > threshold) -> Does price go DOWN?
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: OVERBOUGHT (RSI > threshold) -> Does price go DOWN?")
print("=" * 80)

overbought_results = []
test_count = 0

for rsi_period in RSI_PERIODS:
    rsi_col = f'rsi{rsi_period}'
    rsi_values = train_ohlcv[rsi_col].values

    for overbought_level in OVERBOUGHT_LEVELS:
        for H in HORIZONS:
            test_count += 1
            if test_count % 50 == 0:
                print(f"  Progress: {test_count}/{total_tests} ({100*test_count/total_tests:.0f}%)")

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
            if total > 100:
                down_pct = 100 * down_count / total
                edge = down_pct - 50  # Positive = supports hypothesis (price goes DOWN)

                overbought_results.append({
                    'rsi_period': rsi_period,
                    'level': overbought_level,
                    'horizon': H,
                    'count': total,
                    'down_pct': down_pct,
                    'edge': edge,
                })

print(f"\nCompleted {len(overbought_results)} overbought tests")

# =============================================================================
# RESULTS - OVERSOLD
# =============================================================================
print("\n" + "=" * 80)
print("TOP 20 OVERSOLD RESULTS (RSI < level -> Price UP?)")
print("=" * 80)

oversold_sorted = sorted(oversold_results, key=lambda x: x['edge'], reverse=True)

print(f"\n{'RSI':<6} {'Level':<8} {'H':<4} {'Count':<10} {'UP%':<10} {'Edge':<10}")
print("-" * 60)

for r in oversold_sorted[:20]:
    print(f"RSI{r['rsi_period']:<3} <{r['level']:<6} H={r['horizon']:<2} "
          f"{r['count']:<10} {r['up_pct']:<10.1f} {r['edge']:>+9.1f}")

# =============================================================================
# RESULTS - OVERBOUGHT
# =============================================================================
print("\n" + "=" * 80)
print("TOP 20 OVERBOUGHT RESULTS (RSI > level -> Price DOWN?)")
print("=" * 80)

overbought_sorted = sorted(overbought_results, key=lambda x: x['edge'], reverse=True)

print(f"\n{'RSI':<6} {'Level':<8} {'H':<4} {'Count':<10} {'DOWN%':<10} {'Edge':<10}")
print("-" * 60)

for r in overbought_sorted[:20]:
    print(f"RSI{r['rsi_period']:<3} >{r['level']:<6} H={r['horizon']:<2} "
          f"{r['count']:<10} {r['down_pct']:<10.1f} {r['edge']:>+9.1f}")

# =============================================================================
# SUMMARY BY RSI PERIOD
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY BY RSI PERIOD")
print("=" * 80)

print("\nOVERSOLD (RSI < level):")
print(f"{'RSI Period':<12} {'Avg Edge':<12} {'Max Edge':<12} {'Min Edge':<12}")
print("-" * 50)

for period in RSI_PERIODS:
    period_results = [r for r in oversold_results if r['rsi_period'] == period]
    if period_results:
        edges = [r['edge'] for r in period_results]
        print(f"RSI{period:<9} {np.mean(edges):>+10.2f} {max(edges):>+10.2f} {min(edges):>+10.2f}")

print("\nOVERBOUGHT (RSI > level):")
print(f"{'RSI Period':<12} {'Avg Edge':<12} {'Max Edge':<12} {'Min Edge':<12}")
print("-" * 50)

for period in RSI_PERIODS:
    period_results = [r for r in overbought_results if r['rsi_period'] == period]
    if period_results:
        edges = [r['edge'] for r in period_results]
        print(f"RSI{period:<9} {np.mean(edges):>+10.2f} {max(edges):>+10.2f} {min(edges):>+10.2f}")

# =============================================================================
# SUMMARY BY HORIZON
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY BY HORIZON")
print("=" * 80)

print("\nOVERSOLD (RSI < level):")
print(f"{'Horizon':<12} {'Avg Edge':<12} {'Max Edge':<12} {'Min Edge':<12}")
print("-" * 50)

for h in HORIZONS:
    h_results = [r for r in oversold_results if r['horizon'] == h]
    if h_results:
        edges = [r['edge'] for r in h_results]
        print(f"H={h:<10} {np.mean(edges):>+10.2f} {max(edges):>+10.2f} {min(edges):>+10.2f}")

print("\nOVERBOUGHT (RSI > level):")
print(f"{'Horizon':<12} {'Avg Edge':<12} {'Max Edge':<12} {'Min Edge':<12}")
print("-" * 50)

for h in HORIZONS:
    h_results = [r for r in overbought_results if r['horizon'] == h]
    if h_results:
        edges = [r['edge'] for r in h_results]
        print(f"H={h:<10} {np.mean(edges):>+10.2f} {max(edges):>+10.2f} {min(edges):>+10.2f}")

# =============================================================================
# OVERALL VERDICT
# =============================================================================
print("\n" + "=" * 80)
print("OVERALL VERDICT")
print("=" * 80)

all_oversold_edges = [r['edge'] for r in oversold_results]
all_overbought_edges = [r['edge'] for r in overbought_results]

avg_oversold = np.mean(all_oversold_edges)
avg_overbought = np.mean(all_overbought_edges)
combined_avg = (avg_oversold + avg_overbought) / 2

print(f"\nOversold (RSI < level -> UP):")
print(f"  Average edge: {avg_oversold:+.2f}%")
print(f"  Max edge: {max(all_oversold_edges):+.2f}%")
print(f"  Positive edge combinations: {sum(1 for e in all_oversold_edges if e > 0)}/{len(all_oversold_edges)}")

print(f"\nOverbought (RSI > level -> DOWN):")
print(f"  Average edge: {avg_overbought:+.2f}%")
print(f"  Max edge: {max(all_overbought_edges):+.2f}%")
print(f"  Positive edge combinations: {sum(1 for e in all_overbought_edges if e > 0)}/{len(all_overbought_edges)}")

print(f"\nCombined average edge: {combined_avg:+.2f}%")

if combined_avg > 2:
    print("\n*** RSI HAS EDGE: Mean reversion pattern exists ***")
elif combined_avg > 0.5:
    print("\n*** WEAK RSI EDGE: Small effect, may not overcome fees ***")
elif combined_avg < -2:
    print("\n*** REVERSE RSI PATTERN: Momentum, not mean reversion ***")
else:
    print("\n*** NO RSI EDGE: RSI does not predict direction ***")

# Compare with EMA
print("\n" + "=" * 80)
print("COMPARISON WITH EMA PROXIMITY (from ANALYSIS-15)")
print("=" * 80)

print(f"\n{'Feature':<25} {'Avg Edge (Train)':<20}")
print("-" * 50)
print(f"{'EMA Proximity':<25} {'+2.08%':<20}")
print(f"{'RSI Oversold':<25} {f'{avg_oversold:+.2f}%':<20}")
print(f"{'RSI Overbought':<25} {f'{avg_overbought:+.2f}%':<20}")
print(f"{'RSI Combined':<25} {f'{combined_avg:+.2f}%':<20}")

# Save results
oversold_df = pd.DataFrame(oversold_results)
overbought_df = pd.DataFrame(overbought_results)
oversold_df.to_csv("experiments/rsi_oversold_results.csv", index=False)
overbought_df.to_csv("experiments/rsi_overbought_results.csv", index=False)
print(f"\nResults saved to experiments/rsi_*.csv")
