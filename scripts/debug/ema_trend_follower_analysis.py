"""
Analyze which EMA period is the best trend follower

Test EMA periods: 5, 7, 10, 15, 20, 30
Measure:
- Number of crosses (fewer = cleaner trend)
- Average bars between crosses (longer = better)
- Time spent above vs below (balanced?)

Run: python scripts/debug/ema_trend_follower_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EMA TREND FOLLOWER ANALYSIS")
print("=" * 80)

# Load data
ohlcv_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
data = pd.read_parquet(ohlcv_path)

# Use test data (2024-2025)
test_data = data[data.index >= "2024-01-01"].copy()
print(f"\nAnalyzing: {len(test_data):,} candles ({test_data.index[0]} to {test_data.index[-1]})")

# EMA periods to test
ema_periods = [5, 7, 10, 15, 20, 30]

print("\nCalculating EMAs...")
results = []

for period in ema_periods:
    # Calculate EMA
    ema_col = f'ema{period}'
    test_data[ema_col] = test_data['close'].ewm(span=period, adjust=False).mean()

    # Detect crosses
    test_data[f'above_{period}'] = test_data['close'] > test_data[ema_col]
    test_data[f'cross_above_{period}'] = (test_data[f'above_{period}'] == True) & (test_data[f'above_{period}'].shift(1) == False)
    test_data[f'cross_below_{period}'] = (test_data[f'above_{period}'] == False) & (test_data[f'above_{period}'].shift(1) == True)

    # Count crosses
    crosses_above = test_data[f'cross_above_{period}'].sum()
    crosses_below = test_data[f'cross_below_{period}'].sum()
    total_crosses = crosses_above + crosses_below

    # Calculate average bars between crosses
    if total_crosses > 0:
        avg_bars_between = len(test_data) / total_crosses
    else:
        avg_bars_between = len(test_data)

    # Time spent above vs below
    time_above = test_data[f'above_{period}'].sum()
    time_below = len(test_data) - time_above
    pct_above = time_above / len(test_data) * 100
    pct_below = time_below / len(test_data) * 100

    # Average distance from EMA (as percentage)
    avg_distance_bps = (abs(test_data['close'] - test_data[ema_col]) / test_data['close'] * 10000).mean()

    results.append({
        'EMA': period,
        'Total Crosses': total_crosses,
        'Crosses Above': crosses_above,
        'Crosses Below': crosses_below,
        'Avg Bars Between': avg_bars_between,
        'Time Above (%)': pct_above,
        'Time Below (%)': pct_below,
        'Avg Distance (bps)': avg_distance_bps
    })

results_df = pd.DataFrame(results)

# Display results
print("\n" + "=" * 80)
print("RESULTS - Which EMA is the best trend follower?")
print("=" * 80)
print("\nMETRICS:")
print(results_df.to_string(index=False))

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("=" * 80)
print("- Fewer crosses = Cleaner trend identification")
print("- Longer bars between crosses = Better trend follower")
print("- Balanced time above/below = Not biased")
print("- Lower avg distance = More relevant to price")

# Find best trend follower
best_by_crosses = results_df.loc[results_df['Total Crosses'].idxmin()]
best_by_bars_between = results_df.loc[results_df['Avg Bars Between'].idxmax()]

print(f"\nFewest crosses: EMA{int(best_by_crosses['EMA'])} ({int(best_by_crosses['Total Crosses'])} crosses)")
print(f"Longest between crosses: EMA{int(best_by_bars_between['EMA'])} ({best_by_bars_between['Avg Bars Between']:.1f} bars)")

# Save results
output_path = Path("experiments/ema/ema_trend_follower_comparison.csv")
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

print("\n" + "=" * 80)
