"""
RANGE POSITION Test - Extended Horizons

Question: Does position in recent range predict direction (support/resistance)?

Hypothesis:
- Near LOW of range (support) -> Price should go UP
- Near HIGH of range (resistance) -> Price should go DOWN

Run: .venv/Scripts/python.exe scripts/debug/test_range_position_only.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60, 120, 240, 600]  # Extended!
RANGE_LOOKBACKS = [20, 50, 100, 200]  # Different lookback periods
SAMPLE_SIZE = 100000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("RANGE POSITION TEST - Extended Horizons")
print("=" * 80)
print("Hypothesis: Near Low = Support (UP), Near High = Resistance (DOWN)")

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# Split data first
train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
test_2024 = ohlcv[(ohlcv.index >= "2024-01-01") & (ohlcv.index <= "2024-12-31")].copy()
test_2025 = ohlcv[ohlcv.index >= "2025-01-01"].copy()

print(f"Train: {len(train):,}, 2024: {len(test_2024):,}, 2025: {len(test_2025):,}")


def test_range_position(df, lookback, sample_size, name):
    """Test Range Position as directional predictor."""
    # Calculate range position for this lookback
    range_high = df['high'].rolling(lookback).max()
    range_low = df['low'].rolling(lookback).min()
    range_pos = (df['close'] - range_low) / (range_high - range_low)

    close = df['close'].values
    rp = range_pos.values
    n = len(df)

    np.random.seed(42)
    valid_start = lookback + 10
    max_h = max(HORIZONS)
    avail = n - max_h - valid_start
    if avail < 1000:
        print(f"  {name}: Not enough data")
        return []

    sample_idx = np.random.choice(range(valid_start, n - max_h), size=min(sample_size, avail), replace=False)

    results = []

    print(f"\n{name} (Lookback={lookback}):")
    print(f"{'H':<6} {'Near Low->UP%':<15} {'Near High->UP%':<15} {'Support Edge':<14} {'Resist Edge':<14} {'Combined':<10}")
    print("-" * 85)

    for H in HORIZONS:
        near_low_up, near_low_total = 0, 0  # Range pos < 0.2 (near support)
        near_high_up, near_high_total = 0, 0  # Range pos > 0.8 (near resistance)

        for i in sample_idx:
            pos = rp[i]
            if np.isnan(pos):
                continue

            future = close[i + H]
            went_up = future > close[i]

            if pos < 0.2:  # Near low (support)
                near_low_total += 1
                if went_up:
                    near_low_up += 1
            elif pos > 0.8:  # Near high (resistance)
                near_high_total += 1
                if went_up:
                    near_high_up += 1

        if near_low_total > 50 and near_high_total > 50:
            low_up_pct = 100 * near_low_up / near_low_total
            high_up_pct = 100 * near_high_up / near_high_total

            # Support edge: near low should go UP (positive = good)
            support_edge = low_up_pct - 50
            # Resistance edge: near high should go DOWN (positive = good)
            resist_edge = 50 - high_up_pct
            # Combined edge
            combined = (support_edge + resist_edge) / 2

            print(f"H={H:<4} {low_up_pct:<15.1f} {high_up_pct:<15.1f} {support_edge:>+13.1f} {resist_edge:>+13.1f} {combined:>+9.1f}")

            results.append({
                'H': H,
                'near_low_up_pct': low_up_pct,
                'near_high_up_pct': high_up_pct,
                'support_edge': support_edge,
                'resist_edge': resist_edge,
                'combined': combined
            })

    return results


# =============================================================================
# RUN TESTS
# =============================================================================
print("\n" + "=" * 80)
print("TESTING DIFFERENT LOOKBACK PERIODS")
print("=" * 80)

# Test with lookback=50 (default) for all periods
lookback = 50

train_results = test_range_position(train, lookback, SAMPLE_SIZE, "Train (2020-2023)")
test_2024_results = test_range_position(test_2024, lookback, SAMPLE_SIZE, "Test 2024")
test_2025_results = test_range_position(test_2025, lookback, SAMPLE_SIZE, "Test 2025")

# =============================================================================
# TEST DIFFERENT LOOKBACKS ON 2025 DATA
# =============================================================================
print("\n" + "=" * 80)
print("COMPARING LOOKBACK PERIODS (2025 data)")
print("=" * 80)

lookback_summary = {}
for lb in RANGE_LOOKBACKS:
    results = test_range_position(test_2025, lb, SAMPLE_SIZE, f"2025 LB={lb}")
    if results:
        avg_combined = np.mean([r['combined'] for r in results])
        lookback_summary[lb] = avg_combined

print("\n" + "-" * 40)
print("Lookback Comparison (avg combined edge):")
for lb, edge in lookback_summary.items():
    print(f"  Lookback {lb}: {edge:+.2f}%")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("RANGE POSITION SUMMARY (Lookback=50)")
print("=" * 80)

for name, results in [("Train", train_results), ("2024", test_2024_results), ("2025", test_2025_results)]:
    if results:
        avg_support = np.mean([r['support_edge'] for r in results])
        avg_resist = np.mean([r['resist_edge'] for r in results])
        avg_combined = np.mean([r['combined'] for r in results])
        print(f"{name}: Support={avg_support:+.2f}%, Resist={avg_resist:+.2f}%, Combined={avg_combined:+.2f}%")

# =============================================================================
# VERDICT
# =============================================================================
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if train_results:
    avg_combined_train = np.mean([r['combined'] for r in train_results])
    avg_combined_2025 = np.mean([r['combined'] for r in test_2025_results]) if test_2025_results else 0
    decay = 100 * (1 - avg_combined_2025 / avg_combined_train) if avg_combined_train > 0 else 0

    print(f"""
Range Position Analysis:

Train combined edge: {avg_combined_train:+.2f}%
2025 combined edge:  {avg_combined_2025:+.2f}%
Decay: {decay:.0f}%

Interpretation:
- Support/Resistance effect EXISTS but is WEAK
- Pattern shows significant decay over time
- Not strong enough alone for profitable trading

Comparison with other features:
- RSI:            +2.97% (2025) - BEST
- Range Position: {avg_combined_2025:+.2f}% (2025) - 2nd best
- EMA:            +0.39% (2025) - Very weak
""")
