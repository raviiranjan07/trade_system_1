"""
ATR (Volatility) Test - Extended Horizons

Question: Does high/low volatility predict direction?

Run: .venv/Scripts/python.exe scripts/debug/test_atr_only.py
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
print("ATR (VOLATILITY) TEST - Extended Horizons")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# Calculate ATR
print("Calculating ATR...")
high_low = ohlcv['high'] - ohlcv['low']
high_close = abs(ohlcv['high'] - ohlcv['close'].shift(1))
low_close = abs(ohlcv['low'] - ohlcv['close'].shift(1))
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
ohlcv['atr'] = tr.rolling(14).mean()

# ATR percentile (where current ATR sits in recent range)
atr_roll = ohlcv['atr'].rolling(200)
ohlcv['atr_pct'] = (ohlcv['atr'] - atr_roll.min()) / (atr_roll.max() - atr_roll.min())
print("ATR calculated.")

# Split data
train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
test_2024 = ohlcv[(ohlcv.index >= "2024-01-01") & (ohlcv.index <= "2024-12-31")].copy()
test_2025 = ohlcv[ohlcv.index >= "2025-01-01"].copy()

print(f"Train: {len(train):,}, 2024: {len(test_2024):,}, 2025: {len(test_2025):,}")


def test_atr(df, sample_size, name):
    """Test ATR as directional predictor."""
    close = df['close'].values
    atr_pct = df['atr_pct'].values
    n = len(df)

    np.random.seed(42)
    valid_start = 300
    max_h = max(HORIZONS)
    avail = n - max_h - valid_start
    if avail < 1000:
        print(f"  {name}: Not enough data")
        return []

    sample_idx = np.random.choice(range(valid_start, n - max_h), size=min(sample_size, avail), replace=False)

    results = []

    print(f"\n{name} Results:")
    print(f"{'H':<6} {'Low ATR->UP%':<15} {'High ATR->UP%':<15} {'Low Edge':<12} {'High Edge':<12}")
    print("-" * 65)

    for H in HORIZONS:
        low_up, low_total = 0, 0  # ATR < 20th percentile
        high_up, high_total = 0, 0  # ATR > 80th percentile

        for i in sample_idx:
            atr = atr_pct[i]
            if np.isnan(atr):
                continue

            future = close[i + H]
            went_up = future > close[i]

            if atr < 0.2:  # Low volatility
                low_total += 1
                if went_up:
                    low_up += 1
            elif atr > 0.8:  # High volatility
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
train_results = test_atr(train, SAMPLE_SIZE, "Train (2020-2023)")
test_2024_results = test_atr(test_2024, SAMPLE_SIZE, "Test 2024")
test_2025_results = test_atr(test_2025, SAMPLE_SIZE, "Test 2025")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("ATR SUMMARY")
print("=" * 80)

for name, results in [("Train", train_results), ("2024", test_2024_results), ("2025", test_2025_results)]:
    if results:
        avg_abs_edge = np.mean([abs(r['low_edge']) + abs(r['high_edge']) for r in results]) / 2
        print(f"{name}: Avg absolute edge = {avg_abs_edge:.2f}%")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
print("""
ATR (Volatility) Analysis:

- ATR measures HOW MUCH price moves, not WHICH direction
- Low ATR = calm market, High ATR = volatile market
- Neither should predict UP or DOWN

Expected: ~0% edge (no directional prediction)
""")

if train_results:
    avg_edge = np.mean([abs(r['low_edge']) + abs(r['high_edge']) for r in train_results]) / 2
    if avg_edge < 1.0:
        print("Result: CONFIRMED - ATR has NO directional edge")
    else:
        print(f"Result: Unexpected - ATR shows {avg_edge:.1f}% edge (needs investigation)")
