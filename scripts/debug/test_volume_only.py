"""
VOLUME Test - Extended Horizons

Question: Does high/low volume predict direction?

Run: .venv/Scripts/python.exe scripts/debug/test_volume_only.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60, 120, 240, 600]  # Extended!
SAMPLE_SIZE = 100000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("VOLUME TEST - Extended Horizons")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# Calculate Volume percentile
print("Calculating Volume percentile...")
vol_roll = ohlcv['volume'].rolling(50)
ohlcv['vol_pct'] = (ohlcv['volume'] - vol_roll.min()) / (vol_roll.max() - vol_roll.min())
print("Volume calculated.")

# Split data
train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
test_2024 = ohlcv[(ohlcv.index >= "2024-01-01") & (ohlcv.index <= "2024-12-31")].copy()
test_2025 = ohlcv[ohlcv.index >= "2025-01-01"].copy()

print(f"Train: {len(train):,}, 2024: {len(test_2024):,}, 2025: {len(test_2025):,}")


def test_volume(df, sample_size, name):
    """Test Volume as directional predictor."""
    close = df['close'].values
    vol_pct = df['vol_pct'].values
    n = len(df)

    np.random.seed(42)
    valid_start = 100
    max_h = max(HORIZONS)
    avail = n - max_h - valid_start
    if avail < 1000:
        print(f"  {name}: Not enough data")
        return []

    sample_idx = np.random.choice(range(valid_start, n - max_h), size=min(sample_size, avail), replace=False)

    results = []

    print(f"\n{name} Results:")
    print(f"{'H':<6} {'Low Vol->UP%':<15} {'High Vol->UP%':<15} {'Low Edge':<12} {'High Edge':<12}")
    print("-" * 65)

    for H in HORIZONS:
        low_up, low_total = 0, 0  # Volume < 20th percentile
        high_up, high_total = 0, 0  # Volume > 80th percentile

        for i in sample_idx:
            vol = vol_pct[i]
            if np.isnan(vol):
                continue

            future = close[i + H]
            went_up = future > close[i]

            if vol < 0.2:  # Low volume
                low_total += 1
                if went_up:
                    low_up += 1
            elif vol > 0.8:  # High volume
                high_total += 1
                if went_up:
                    high_up += 1

        if low_total > 50 and high_total > 50:
            low_up_pct = 100 * low_up / low_total
            high_up_pct = 100 * high_up / high_total
            low_edge = low_up_pct - 50
            high_edge = high_up_pct - 50

            print(f"H={H:<4} {low_up_pct:<15.1f} {high_up_pct:<15.1f} {low_edge:>+11.1f} {high_edge:>+11.1f}")

            results.append({
                'H': H,
                'low_up_pct': low_up_pct,
                'high_up_pct': high_up_pct,
                'low_edge': low_edge,
                'high_edge': high_edge
            })

    return results


# =============================================================================
# RUN TESTS
# =============================================================================
train_results = test_volume(train, SAMPLE_SIZE, "Train (2020-2023)")
test_2024_results = test_volume(test_2024, SAMPLE_SIZE, "Test 2024")
test_2025_results = test_volume(test_2025, SAMPLE_SIZE, "Test 2025")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("VOLUME SUMMARY")
print("=" * 80)

for name, results in [("Train", train_results), ("2024", test_2024_results), ("2025", test_2025_results)]:
    if results:
        avg_abs_edge = np.mean([abs(r['low_edge']) + abs(r['high_edge']) for r in results]) / 2
        print(f"{name}: Avg absolute edge = {avg_abs_edge:.2f}%")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
print("""
VOLUME Analysis:

- Volume measures HOW MUCH trading activity, not WHICH direction
- Low Volume = quiet market, High Volume = active market
- Neither should predict UP or DOWN

Expected: ~0% edge (no directional prediction)
""")

if train_results:
    avg_edge = np.mean([abs(r['low_edge']) + abs(r['high_edge']) for r in train_results]) / 2
    if avg_edge < 1.5:
        print("Result: CONFIRMED - Volume has NO meaningful directional edge")
    else:
        print(f"Result: Unexpected - Volume shows {avg_edge:.1f}% edge (needs investigation)")
