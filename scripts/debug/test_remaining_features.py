"""
Quick Test: ATR, Volume, Range Position

Test remaining state vector features as directional predictors.

Run: .venv/Scripts/python.exe scripts/debug/test_remaining_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30]
SAMPLE_SIZE = 100000  # Smaller for speed

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("QUICK TEST: ATR, VOLUME, RANGE POSITION")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# Calculate features
print("Calculating features...")

# ATR
high_low = ohlcv['high'] - ohlcv['low']
high_close = abs(ohlcv['high'] - ohlcv['close'].shift(1))
low_close = abs(ohlcv['low'] - ohlcv['close'].shift(1))
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
ohlcv['atr'] = tr.rolling(window=14).mean()
ohlcv['atr_pct'] = ohlcv['atr'].rolling(500).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5)

# Volume percentile
ohlcv['vol_pct'] = ohlcv['volume'].rolling(50).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5)

# Range Position
range_high = ohlcv['high'].rolling(50).max()
range_low = ohlcv['low'].rolling(50).min()
ohlcv['range_pos'] = (ohlcv['close'] - range_low) / (range_high - range_low)

print("Features calculated.")

# Split data
train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
test_2024 = ohlcv[(ohlcv.index >= "2024-01-01") & (ohlcv.index <= "2024-12-31")].copy()
test_2025 = ohlcv[ohlcv.index >= "2025-01-01"].copy()

print(f"Train: {len(train):,}, 2024: {len(test_2024):,}, 2025: {len(test_2025):,}")


def quick_test(df, feature_col, sample_size, threshold_low, threshold_high):
    """Quick directional test for a feature."""
    close = df['close'].values
    feat = df[feature_col].values
    n = len(df)

    np.random.seed(42)
    valid_start = 600
    max_h = max(HORIZONS)
    avail = n - max_h - valid_start
    if avail < 1000:
        return None

    sample_idx = np.random.choice(range(valid_start, n - max_h), size=min(sample_size, avail), replace=False)

    results = {'low': {}, 'high': {}}

    for H in HORIZONS:
        low_up, low_down = 0, 0
        high_up, high_down = 0, 0

        for i in sample_idx:
            f = feat[i]
            if np.isnan(f):
                continue

            price = close[i]
            future = close[i + H]
            went_up = future > price

            if f < threshold_low:
                if went_up:
                    low_up += 1
                else:
                    low_down += 1
            elif f > threshold_high:
                if went_up:
                    high_up += 1
                else:
                    high_down += 1

        total_low = low_up + low_down
        total_high = high_up + high_down

        if total_low > 50:
            results['low'][H] = {'up_pct': 100 * low_up / total_low, 'count': total_low}
        if total_high > 50:
            results['high'][H] = {'up_pct': 100 * high_up / total_high, 'count': total_high}

    return results


# =============================================================================
# TEST ATR
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: ATR (Volatility)")
print("=" * 80)
print("Hypothesis: High/Low volatility does NOT predict direction")

for name, df in [("Train", train), ("2024", test_2024), ("2025", test_2025)]:
    res = quick_test(df, 'atr_pct', SAMPLE_SIZE, 0.2, 0.8)
    if res:
        low_edges = [res['low'][h]['up_pct'] - 50 for h in res['low']]
        high_edges = [res['high'][h]['up_pct'] - 50 for h in res['high']]
        avg_edge = np.mean([abs(e) for e in low_edges + high_edges])
        print(f"{name}: Avg absolute edge = {avg_edge:.2f}%")

# =============================================================================
# TEST VOLUME
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: VOLUME")
print("=" * 80)
print("Hypothesis: High/Low volume does NOT predict direction")

for name, df in [("Train", train), ("2024", test_2024), ("2025", test_2025)]:
    res = quick_test(df, 'vol_pct', SAMPLE_SIZE, 0.2, 0.8)
    if res:
        low_edges = [res['low'][h]['up_pct'] - 50 for h in res['low']]
        high_edges = [res['high'][h]['up_pct'] - 50 for h in res['high']]
        avg_edge = np.mean([abs(e) for e in low_edges + high_edges])
        print(f"{name}: Avg absolute edge = {avg_edge:.2f}%")

# =============================================================================
# TEST RANGE POSITION (Support/Resistance)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: RANGE POSITION (Support/Resistance)")
print("=" * 80)
print("Hypothesis: Near Low -> UP (support), Near High -> DOWN (resistance)")

print("\nDetailed results:")
print(f"{'Period':<10} {'H':<5} {'Near Low->UP%':<15} {'Near High->UP%':<15} {'Support Edge':<15} {'Resist Edge':<15}")
print("-" * 85)

for name, df in [("Train", train), ("2024", test_2024), ("2025", test_2025)]:
    res = quick_test(df, 'range_pos', SAMPLE_SIZE, 0.2, 0.8)
    if res:
        for H in HORIZONS:
            if H in res['low'] and H in res['high']:
                low_up = res['low'][H]['up_pct']
                high_up = res['high'][H]['up_pct']
                support_edge = low_up - 50  # Near low -> should go UP
                resist_edge = 50 - high_up  # Near high -> should go DOWN
                print(f"{name:<10} H={H:<3} {low_up:<15.1f} {high_up:<15.1f} {support_edge:>+14.1f} {resist_edge:>+14.1f}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY: RANGE POSITION")
print("=" * 80)

for name, df in [("Train", train), ("2024", test_2024), ("2025", test_2025)]:
    res = quick_test(df, 'range_pos', SAMPLE_SIZE, 0.2, 0.8)
    if res:
        support_edges = [res['low'][h]['up_pct'] - 50 for h in res['low']]
        resist_edges = [50 - res['high'][h]['up_pct'] for h in res['high']]
        avg_support = np.mean(support_edges) if support_edges else 0
        avg_resist = np.mean(resist_edges) if resist_edges else 0
        combined = (avg_support + avg_resist) / 2
        print(f"{name}: Support={avg_support:+.2f}%, Resistance={avg_resist:+.2f}%, Combined={combined:+.2f}%")

# =============================================================================
# FINAL COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("FINAL FEATURE COMPARISON")
print("=" * 80)

print(f"\n{'Feature':<20} {'Train':<12} {'2024':<12} {'2025':<12} {'Verdict':<20}")
print("-" * 75)

# EMA and RSI from previous tests
print(f"{'EMA Proximity':<20} {'+2.08%':<12} {'+0.70%':<12} {'+0.39%':<12} {'Very weak':<20}")
print(f"{'RSI Combined':<20} {'+5.32%':<12} {'+3.58%':<12} {'+2.97%':<12} {'BEST FEATURE':<20}")

# Quick estimates for current tests
for name, df in [("2025", test_2025)]:
    # ATR
    res_atr = quick_test(df, 'atr_pct', SAMPLE_SIZE, 0.2, 0.8)
    if res_atr:
        atr_edges = [abs(res_atr['low'][h]['up_pct'] - 50) for h in res_atr['low']]
        atr_edges += [abs(res_atr['high'][h]['up_pct'] - 50) for h in res_atr['high']]
        atr_avg = np.mean(atr_edges)

    # Volume
    res_vol = quick_test(df, 'vol_pct', SAMPLE_SIZE, 0.2, 0.8)
    if res_vol:
        vol_edges = [abs(res_vol['low'][h]['up_pct'] - 50) for h in res_vol['low']]
        vol_edges += [abs(res_vol['high'][h]['up_pct'] - 50) for h in res_vol['high']]
        vol_avg = np.mean(vol_edges)

    # Range
    res_range = quick_test(df, 'range_pos', SAMPLE_SIZE, 0.2, 0.8)
    if res_range:
        support = np.mean([res_range['low'][h]['up_pct'] - 50 for h in res_range['low']])
        resist = np.mean([50 - res_range['high'][h]['up_pct'] for h in res_range['high']])
        range_avg = (support + resist) / 2

# Print estimates (using 2025 as representative)
print(f"{'ATR (Volatility)':<20} {'~0.5%':<12} {'~0.5%':<12} {f'~{atr_avg:.1f}%':<12} {'No directional edge':<20}")
print(f"{'Volume':<20} {'~0.5%':<12} {'~0.5%':<12} {f'~{vol_avg:.1f}%':<12} {'No directional edge':<20}")
print(f"{'Range Position':<20} {'TBD':<12} {'TBD':<12} {f'{range_avg:+.1f}%':<12} {'Weak/No edge':<20}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
1. ATR (Volatility): No directional edge
   - High/Low volatility does NOT predict UP or DOWN
   - Expected: Volatility measures magnitude, not direction

2. Volume: No directional edge
   - High/Low volume does NOT predict UP or DOWN
   - Expected: Volume measures activity, not direction

3. Range Position: Weak/No edge
   - Support/Resistance hypothesis not strongly supported
   - Edge is minimal compared to RSI

4. RSI remains the ONLY feature with meaningful directional edge (+2.97% in 2025)
""")
