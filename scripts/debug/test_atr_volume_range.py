"""
ATR, Volume, Range Position Test - Data-Driven Approach

Test these features as directional predictors.
Let the DATA tell us if they predict direction.

Hypotheses:
1. ATR: High volatility vs Low volatility - does it predict direction?
2. Volume: High volume vs Low volume - does it predict direction?
3. Range Position: Near high (resistance) vs Near low (support)

Run: .venv/Scripts/python.exe scripts/debug/test_atr_volume_range.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
ATR_PERIOD = 14
VOLUME_WINDOW = 50
RANGE_LOOKBACK = 50

# Thresholds to test
ATR_PERCENTILES = [10, 20, 30, 70, 80, 90]  # Low vs High volatility
VOLUME_PERCENTILES = [10, 20, 30, 70, 80, 90]  # Low vs High volume
RANGE_THRESHOLDS = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]  # Position in range (0=low, 1=high)

HORIZONS = [1, 2, 3, 5, 10, 15, 20, 30]

TRAIN_END = "2023-12-31"
TEST_2024_START = "2024-01-01"
TEST_2024_END = "2024-12-31"
TEST_2025_START = "2025-01-01"

SAMPLE_SIZE = 300000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ATR, VOLUME, RANGE POSITION TEST - Data-Driven")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# Calculate features
print("\nCalculating features...")

# ATR
high_low = ohlcv['high'] - ohlcv['low']
high_close = abs(ohlcv['high'] - ohlcv['close'].shift(1))
low_close = abs(ohlcv['low'] - ohlcv['close'].shift(1))
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
ohlcv['atr'] = tr.rolling(window=ATR_PERIOD).mean()
ohlcv['atr_percentile'] = ohlcv['atr'].rolling(window=500).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
)

# Volume percentile
ohlcv['volume_percentile'] = ohlcv['volume'].rolling(window=VOLUME_WINDOW).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
)

# Range Position (where price is relative to recent high/low)
ohlcv['range_high'] = ohlcv['high'].rolling(window=RANGE_LOOKBACK).max()
ohlcv['range_low'] = ohlcv['low'].rolling(window=RANGE_LOOKBACK).min()
ohlcv['range_position'] = (ohlcv['close'] - ohlcv['range_low']) / (ohlcv['range_high'] - ohlcv['range_low'])

print("Features calculated.")

# Split data
train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END].copy()
test_2024 = ohlcv[(ohlcv.index >= TEST_2024_START) & (ohlcv.index <= TEST_2024_END)].copy()
test_2025 = ohlcv[ohlcv.index >= TEST_2025_START].copy()

print(f"TRAIN: {len(train_ohlcv):,} candles (up to {TRAIN_END})")
print(f"TEST 2024: {len(test_2024):,} candles")
print(f"TEST 2025: {len(test_2025):,} candles")


def test_feature(df, feature_col, thresholds, horizons, sample_size, feature_name, is_percentile=True):
    """Test if a feature predicts direction."""
    close = df['close'].values
    feature_values = df[feature_col].values
    n = len(df)

    np.random.seed(42)
    max_h = max(horizons)
    valid_start = 600  # Ensure enough warmup
    available = n - max_h - valid_start
    actual_sample = min(sample_size, available)

    if actual_sample < 1000:
        return []

    sample_idx = np.random.choice(
        range(valid_start, n - max_h),
        size=actual_sample,
        replace=False
    )

    results = []

    for threshold in thresholds:
        for H in horizons:
            # Test LOW condition (feature < threshold)
            low_up = 0
            low_down = 0

            # Test HIGH condition (feature > threshold)
            high_up = 0
            high_down = 0

            for i in sample_idx:
                feat = feature_values[i]
                if np.isnan(feat):
                    continue

                price = close[i]
                future_price = close[i + H]
                went_up = future_price > price

                # For percentile-based features
                if is_percentile:
                    thresh_val = threshold / 100  # Convert to 0-1 range
                else:
                    thresh_val = threshold

                if feat < thresh_val:
                    if went_up:
                        low_up += 1
                    else:
                        low_down += 1
                elif feat > (1 - thresh_val) if is_percentile else feat > thresh_val:
                    if went_up:
                        high_up += 1
                    else:
                        high_down += 1

            total_low = low_up + low_down
            total_high = high_up + high_down

            if total_low > 100:
                low_up_pct = 100 * low_up / total_low
                low_edge = low_up_pct - 50
                results.append({
                    'feature': feature_name,
                    'condition': f'< {threshold}' if is_percentile else f'< {threshold:.1f}',
                    'threshold': threshold,
                    'horizon': H,
                    'count': total_low,
                    'up_pct': low_up_pct,
                    'edge': low_edge,
                    'type': 'low'
                })

            if total_high > 100:
                high_up_pct = 100 * high_up / total_high
                high_edge = high_up_pct - 50
                results.append({
                    'feature': feature_name,
                    'condition': f'> {100-threshold}' if is_percentile else f'> {threshold:.1f}',
                    'threshold': threshold,
                    'horizon': H,
                    'count': total_high,
                    'up_pct': high_up_pct,
                    'edge': high_edge,
                    'type': 'high'
                })

    return results


def test_range_position(df, thresholds, horizons, sample_size):
    """Test if range position predicts direction (support/resistance)."""
    close = df['close'].values
    range_pos = df['range_position'].values
    n = len(df)

    np.random.seed(42)
    max_h = max(horizons)
    valid_start = 600
    available = n - max_h - valid_start
    actual_sample = min(sample_size, available)

    if actual_sample < 1000:
        return []

    sample_idx = np.random.choice(
        range(valid_start, n - max_h),
        size=actual_sample,
        replace=False
    )

    results = []

    for threshold in thresholds:
        for H in horizons:
            # Near LOW of range (support) - expect UP
            near_low_up = 0
            near_low_down = 0

            # Near HIGH of range (resistance) - expect DOWN
            near_high_up = 0
            near_high_down = 0

            for i in sample_idx:
                rp = range_pos[i]
                if np.isnan(rp):
                    continue

                price = close[i]
                future_price = close[i + H]
                went_up = future_price > price

                if rp < threshold:  # Near low
                    if went_up:
                        near_low_up += 1
                    else:
                        near_low_down += 1
                elif rp > (1 - threshold):  # Near high
                    if went_up:
                        near_high_up += 1
                    else:
                        near_high_down += 1

            total_low = near_low_up + near_low_down
            total_high = near_high_up + near_high_down

            if total_low > 100:
                low_up_pct = 100 * near_low_up / total_low
                # Hypothesis: near low -> should go UP
                low_edge = low_up_pct - 50
                results.append({
                    'feature': 'Range Position',
                    'condition': f'Near Low (<{threshold:.1f})',
                    'threshold': threshold,
                    'horizon': H,
                    'count': total_low,
                    'up_pct': low_up_pct,
                    'edge': low_edge,
                    'expected': 'UP',
                    'type': 'support'
                })

            if total_high > 100:
                high_up_pct = 100 * near_high_up / total_high
                # Hypothesis: near high -> should go DOWN
                high_edge = 50 - high_up_pct  # Positive if DOWN more than 50%
                results.append({
                    'feature': 'Range Position',
                    'condition': f'Near High (>{1-threshold:.1f})',
                    'threshold': threshold,
                    'horizon': H,
                    'count': total_high,
                    'up_pct': high_up_pct,
                    'down_pct': 100 - high_up_pct,
                    'edge': high_edge,
                    'expected': 'DOWN',
                    'type': 'resistance'
                })

    return results


# =============================================================================
# TEST 1: ATR (Volatility)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: ATR (Volatility) - Does volatility predict direction?")
print("=" * 80)

print("\nTesting on TRAIN data...")
atr_train = test_feature(train_ohlcv, 'atr_percentile', ATR_PERCENTILES, HORIZONS, SAMPLE_SIZE, 'ATR')

print("Testing on 2024 data...")
atr_2024 = test_feature(test_2024, 'atr_percentile', ATR_PERCENTILES, HORIZONS, SAMPLE_SIZE, 'ATR')

print("Testing on 2025 data...")
atr_2025 = test_feature(test_2025, 'atr_percentile', ATR_PERCENTILES, HORIZONS, SAMPLE_SIZE, 'ATR')

# Summarize ATR
all_atr_edges_train = [r['edge'] for r in atr_train]
all_atr_edges_2024 = [r['edge'] for r in atr_2024]
all_atr_edges_2025 = [r['edge'] for r in atr_2025]

avg_atr_train = np.mean([abs(e) for e in all_atr_edges_train]) if all_atr_edges_train else 0
avg_atr_2024 = np.mean([abs(e) for e in all_atr_edges_2024]) if all_atr_edges_2024 else 0
avg_atr_2025 = np.mean([abs(e) for e in all_atr_edges_2025]) if all_atr_edges_2025 else 0

print(f"\nATR Summary (absolute avg edge):")
print(f"  Train: {avg_atr_train:.2f}%")
print(f"  2024:  {avg_atr_2024:.2f}%")
print(f"  2025:  {avg_atr_2025:.2f}%")

# =============================================================================
# TEST 2: Volume
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: VOLUME - Does volume predict direction?")
print("=" * 80)

print("\nTesting on TRAIN data...")
vol_train = test_feature(train_ohlcv, 'volume_percentile', VOLUME_PERCENTILES, HORIZONS, SAMPLE_SIZE, 'Volume')

print("Testing on 2024 data...")
vol_2024 = test_feature(test_2024, 'volume_percentile', VOLUME_PERCENTILES, HORIZONS, SAMPLE_SIZE, 'Volume')

print("Testing on 2025 data...")
vol_2025 = test_feature(test_2025, 'volume_percentile', VOLUME_PERCENTILES, HORIZONS, SAMPLE_SIZE, 'Volume')

# Summarize Volume
all_vol_edges_train = [r['edge'] for r in vol_train]
all_vol_edges_2024 = [r['edge'] for r in vol_2024]
all_vol_edges_2025 = [r['edge'] for r in vol_2025]

avg_vol_train = np.mean([abs(e) for e in all_vol_edges_train]) if all_vol_edges_train else 0
avg_vol_2024 = np.mean([abs(e) for e in all_vol_edges_2024]) if all_vol_edges_2024 else 0
avg_vol_2025 = np.mean([abs(e) for e in all_vol_edges_2025]) if all_vol_edges_2025 else 0

print(f"\nVolume Summary (absolute avg edge):")
print(f"  Train: {avg_vol_train:.2f}%")
print(f"  2024:  {avg_vol_2024:.2f}%")
print(f"  2025:  {avg_vol_2025:.2f}%")

# =============================================================================
# TEST 3: Range Position (Support/Resistance)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: RANGE POSITION - Does position in range predict direction?")
print("=" * 80)
print("Hypothesis: Near Low = Support (UP), Near High = Resistance (DOWN)")

print("\nTesting on TRAIN data...")
range_train = test_range_position(train_ohlcv, RANGE_THRESHOLDS, HORIZONS, SAMPLE_SIZE)

print("Testing on 2024 data...")
range_2024 = test_range_position(test_2024, RANGE_THRESHOLDS, HORIZONS, SAMPLE_SIZE)

print("Testing on 2025 data...")
range_2025 = test_range_position(test_2025, RANGE_THRESHOLDS, HORIZONS, SAMPLE_SIZE)

# Summarize Range Position
support_edges_train = [r['edge'] for r in range_train if r['type'] == 'support']
resistance_edges_train = [r['edge'] for r in range_train if r['type'] == 'resistance']
support_edges_2024 = [r['edge'] for r in range_2024 if r['type'] == 'support']
resistance_edges_2024 = [r['edge'] for r in range_2024 if r['type'] == 'resistance']
support_edges_2025 = [r['edge'] for r in range_2025 if r['type'] == 'support']
resistance_edges_2025 = [r['edge'] for r in range_2025 if r['type'] == 'resistance']

avg_support_train = np.mean(support_edges_train) if support_edges_train else 0
avg_resistance_train = np.mean(resistance_edges_train) if resistance_edges_train else 0
avg_support_2024 = np.mean(support_edges_2024) if support_edges_2024 else 0
avg_resistance_2024 = np.mean(resistance_edges_2024) if resistance_edges_2024 else 0
avg_support_2025 = np.mean(support_edges_2025) if support_edges_2025 else 0
avg_resistance_2025 = np.mean(resistance_edges_2025) if resistance_edges_2025 else 0

combined_range_train = (avg_support_train + avg_resistance_train) / 2
combined_range_2024 = (avg_support_2024 + avg_resistance_2024) / 2
combined_range_2025 = (avg_support_2025 + avg_resistance_2025) / 2

print(f"\nRange Position Summary:")
print(f"  Train - Support (near low->UP): {avg_support_train:+.2f}%")
print(f"  Train - Resistance (near high->DOWN): {avg_resistance_train:+.2f}%")
print(f"  Train - Combined: {combined_range_train:+.2f}%")
print(f"  2024 - Support: {avg_support_2024:+.2f}%, Resistance: {avg_resistance_2024:+.2f}%, Combined: {combined_range_2024:+.2f}%")
print(f"  2025 - Support: {avg_support_2025:+.2f}%, Resistance: {avg_resistance_2025:+.2f}%, Combined: {combined_range_2025:+.2f}%")

# =============================================================================
# OVERALL COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("OVERALL FEATURE COMPARISON")
print("=" * 80)

print(f"\n{'Feature':<20} {'Train':<12} {'2024':<12} {'2025':<12} {'Verdict':<15}")
print("-" * 70)

# ATR
atr_verdict = "No Edge" if avg_atr_train < 1 else "Weak" if avg_atr_train < 2 else "Moderate"
print(f"{'ATR (Volatility)':<20} {f'{avg_atr_train:.2f}%':<12} {f'{avg_atr_2024:.2f}%':<12} {f'{avg_atr_2025:.2f}%':<12} {atr_verdict:<15}")

# Volume
vol_verdict = "No Edge" if avg_vol_train < 1 else "Weak" if avg_vol_train < 2 else "Moderate"
print(f"{'Volume':<20} {f'{avg_vol_train:.2f}%':<12} {f'{avg_vol_2024:.2f}%':<12} {f'{avg_vol_2025:.2f}%':<12} {vol_verdict:<15}")

# Range Position
range_verdict = "No Edge" if combined_range_train < 1 else "Weak" if combined_range_train < 2 else "Moderate"
print(f"{'Range Position':<20} {f'{combined_range_train:+.2f}%':<12} {f'{combined_range_2024:+.2f}%':<12} {f'{combined_range_2025:+.2f}%':<12} {range_verdict:<15}")

# Compare with EMA and RSI
print("\n" + "-" * 70)
print("Comparison with previously tested features:")
print(f"{'EMA Proximity':<20} {'+2.08%':<12} {'+0.70%':<12} {'+0.39%':<12} {'Very Weak':<15}")
print(f"{'RSI Combined':<20} {'+5.32%':<12} {'+3.58%':<12} {'+2.97%':<12} {'Best so far':<15}")

# =============================================================================
# TOP RESULTS FOR RANGE POSITION
# =============================================================================
print("\n" + "=" * 80)
print("TOP 10 RANGE POSITION RESULTS (2025)")
print("=" * 80)

if range_2025:
    range_sorted = sorted(range_2025, key=lambda x: x['edge'], reverse=True)
    print(f"\n{'Condition':<25} {'H':<4} {'Count':<10} {'UP%':<10} {'Edge':<10}")
    print("-" * 65)
    for r in range_sorted[:10]:
        print(f"{r['condition']:<25} H={r['horizon']:<2} {r['count']:<10} {r['up_pct']:<10.1f} {r['edge']:>+9.2f}")

# =============================================================================
# VERDICT
# =============================================================================
print("\n" + "=" * 80)
print("VERDICT: REMAINING FEATURES")
print("=" * 80)

print("""
Feature Analysis Summary:

1. ATR (Volatility):
   - Tests if high/low volatility predicts direction
   - Expected: No directional edge (volatility != direction)

2. Volume:
   - Tests if high/low volume predicts direction
   - Expected: No directional edge (volume != direction)

3. Range Position:
   - Tests support (near low -> UP) and resistance (near high -> DOWN)
   - This is the only one with a potential edge
""")

if combined_range_2025 > 2:
    print("*** RANGE POSITION shows meaningful edge ***")
elif combined_range_2025 > 1:
    print("*** RANGE POSITION shows weak edge ***")
else:
    print("*** RANGE POSITION shows NO significant edge ***")

print("\nConclusion: RSI remains the best feature for direction prediction.")
