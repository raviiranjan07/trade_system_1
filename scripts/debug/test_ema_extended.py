"""
EMA Proximity Test - Extended Horizons (up to H=600)

Run: .venv/Scripts/python.exe scripts/debug/test_ema_extended.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60, 120, 240, 600]  # Extended!
EMA_PERIOD = 50  # Standard EMA
NEAR_BPS = 20  # Near threshold
SAMPLE_SIZE = 100000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("EMA PROXIMITY TEST - Extended Horizons")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# Calculate EMA
print("Calculating EMA...")
ohlcv['ema'] = ohlcv['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
print("EMA calculated.")

# Split data
train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
test_2024 = ohlcv[(ohlcv.index >= "2024-01-01") & (ohlcv.index <= "2024-12-31")].copy()
test_2025 = ohlcv[ohlcv.index >= "2025-01-01"].copy()

print(f"Train: {len(train):,}, 2024: {len(test_2024):,}, 2025: {len(test_2025):,}")


def test_ema(df, sample_size, name):
    """Test EMA proximity as directional predictor."""
    close = df['close'].values
    ema = df['ema'].values
    n = len(df)
    near_pct = NEAR_BPS / 10000

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
    print(f"{'H':<6} {'Below EMA->UP%':<16} {'Above EMA->DOWN%':<18} {'Below Edge':<12} {'Above Edge':<12} {'Combined':<10}")
    print("-" * 90)

    for H in HORIZONS:
        below_up, below_total = 0, 0  # Price below EMA -> expect UP
        above_down, above_total = 0, 0  # Price above EMA -> expect DOWN

        for i in sample_idx:
            price = close[i]
            e = ema[i]
            if np.isnan(e):
                continue

            distance_pct = (price - e) / e
            if abs(distance_pct) > near_pct:
                continue

            future = close[i + H]
            went_up = future > price

            if distance_pct < 0:  # Below EMA -> expect UP
                below_total += 1
                if went_up:
                    below_up += 1
            elif distance_pct > 0:  # Above EMA -> expect DOWN
                above_total += 1
                if not went_up:
                    above_down += 1

        if below_total > 50 and above_total > 50:
            below_up_pct = 100 * below_up / below_total
            above_down_pct = 100 * above_down / above_total

            below_edge = below_up_pct - 50  # Positive = good
            above_edge = above_down_pct - 50  # Positive = good
            combined = (below_edge + above_edge) / 2

            print(f"H={H:<4} {below_up_pct:<16.1f} {above_down_pct:<18.1f} {below_edge:>+11.1f} {above_edge:>+11.1f} {combined:>+9.1f}")

            results.append({
                'H': H,
                'below_up_pct': below_up_pct,
                'above_down_pct': above_down_pct,
                'below_edge': below_edge,
                'above_edge': above_edge,
                'combined': combined
            })

    return results


# =============================================================================
# RUN TESTS
# =============================================================================
train_results = test_ema(train, SAMPLE_SIZE, "Train (2020-2023)")
test_2024_results = test_ema(test_2024, SAMPLE_SIZE, "Test 2024")
test_2025_results = test_ema(test_2025, SAMPLE_SIZE, "Test 2025")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("EMA SUMMARY")
print("=" * 80)

for name, results in [("Train", train_results), ("2024", test_2024_results), ("2025", test_2025_results)]:
    if results:
        avg_below = np.mean([r['below_edge'] for r in results])
        avg_above = np.mean([r['above_edge'] for r in results])
        avg_combined = np.mean([r['combined'] for r in results])
        print(f"{name}: Below={avg_below:+.2f}%, Above={avg_above:+.2f}%, Combined={avg_combined:+.2f}%")

# =============================================================================
# OPTIMAL HORIZON ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("OPTIMAL HORIZON (2025 data)")
print("=" * 80)

if test_2025_results:
    print(f"\n{'H':<8} {'Combined Edge':<15}")
    print("-" * 25)
    for r in test_2025_results:
        print(f"H={r['H']:<5} {r['combined']:>+14.2f}%")

    best = max(test_2025_results, key=lambda x: x['combined'])
    print(f"\nBest horizon: H={best['H']} with {best['combined']:+.2f}% combined edge")
