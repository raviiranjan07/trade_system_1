"""
RSI Test - Extended Horizons (up to H=600)

Run: .venv/Scripts/python.exe scripts/debug/test_rsi_extended.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60, 120, 240, 600]  # Extended!
RSI_PERIOD = 14  # Standard RSI
SAMPLE_SIZE = 100000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("RSI TEST - Extended Horizons")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# Calculate RSI
print("Calculating RSI...")
delta = ohlcv['close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=RSI_PERIOD).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
rs = gain / loss
ohlcv['rsi'] = 100 - (100 / (1 + rs))
print("RSI calculated.")

# Split data
train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
test_2024 = ohlcv[(ohlcv.index >= "2024-01-01") & (ohlcv.index <= "2024-12-31")].copy()
test_2025 = ohlcv[ohlcv.index >= "2025-01-01"].copy()

print(f"Train: {len(train):,}, 2024: {len(test_2024):,}, 2025: {len(test_2025):,}")


def test_rsi(df, sample_size, name):
    """Test RSI as directional predictor."""
    close = df['close'].values
    rsi = df['rsi'].values
    n = len(df)

    np.random.seed(42)
    valid_start = 50
    max_h = max(HORIZONS)
    avail = n - max_h - valid_start
    if avail < 1000:
        print(f"  {name}: Not enough data")
        return []

    sample_idx = np.random.choice(range(valid_start, n - max_h), size=min(sample_size, avail), replace=False)

    results = []

    print(f"\n{name} Results:")
    print(f"{'H':<6} {'Oversold->UP%':<15} {'Overbought->DOWN%':<18} {'Oversold Edge':<15} {'Overbought Edge':<15} {'Combined':<10}")
    print("-" * 95)

    for H in HORIZONS:
        oversold_up, oversold_total = 0, 0  # RSI < 30
        overbought_down, overbought_total = 0, 0  # RSI > 70

        for i in sample_idx:
            r = rsi[i]
            if np.isnan(r):
                continue

            future = close[i + H]
            went_up = future > close[i]

            if r < 30:  # Oversold -> expect UP
                oversold_total += 1
                if went_up:
                    oversold_up += 1
            elif r > 70:  # Overbought -> expect DOWN
                overbought_total += 1
                if not went_up:
                    overbought_down += 1

        if oversold_total > 50 and overbought_total > 50:
            oversold_up_pct = 100 * oversold_up / oversold_total
            overbought_down_pct = 100 * overbought_down / overbought_total

            oversold_edge = oversold_up_pct - 50  # Positive = good
            overbought_edge = overbought_down_pct - 50  # Positive = good
            combined = (oversold_edge + overbought_edge) / 2

            print(f"H={H:<4} {oversold_up_pct:<15.1f} {overbought_down_pct:<18.1f} {oversold_edge:>+14.1f} {overbought_edge:>+14.1f} {combined:>+9.1f}")

            results.append({
                'H': H,
                'oversold_up_pct': oversold_up_pct,
                'overbought_down_pct': overbought_down_pct,
                'oversold_edge': oversold_edge,
                'overbought_edge': overbought_edge,
                'combined': combined
            })

    return results


# =============================================================================
# RUN TESTS
# =============================================================================
train_results = test_rsi(train, SAMPLE_SIZE, "Train (2020-2023)")
test_2024_results = test_rsi(test_2024, SAMPLE_SIZE, "Test 2024")
test_2025_results = test_rsi(test_2025, SAMPLE_SIZE, "Test 2025")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("RSI SUMMARY")
print("=" * 80)

for name, results in [("Train", train_results), ("2024", test_2024_results), ("2025", test_2025_results)]:
    if results:
        avg_oversold = np.mean([r['oversold_edge'] for r in results])
        avg_overbought = np.mean([r['overbought_edge'] for r in results])
        avg_combined = np.mean([r['combined'] for r in results])
        print(f"{name}: Oversold={avg_oversold:+.2f}%, Overbought={avg_overbought:+.2f}%, Combined={avg_combined:+.2f}%")

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
