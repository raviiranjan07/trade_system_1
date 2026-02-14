"""
Visualize different EMA periods to compare trend-following behavior

Show price with EMA 5, 7, 10, 15, 20, 30 on separate subplots
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

print("Creating EMA comparison visualization...")

# Load data
ohlcv_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
data = pd.read_parquet(ohlcv_path)

# Use a representative period - 1 month from 2024
data = data[(data.index >= "2024-01-01") & (data.index < "2024-02-01")].copy()

# Calculate all EMAs
ema_periods = [5, 7, 10, 15, 20, 30]
for period in ema_periods:
    data[f'ema{period}'] = data['close'].ewm(span=period, adjust=False).mean()

# Create 6 subplots (one for each EMA)
fig, axes = plt.subplots(6, 1, figsize=(24, 24))
fig.suptitle('EMA Trend Follower Comparison - January 2024\nWhich EMA follows the trend best?',
             fontsize=18, fontweight='bold')

for idx, period in enumerate(ema_periods):
    ax = axes[idx]

    # Plot price
    ax.plot(data.index, data['close'], label='Price', color='black', linewidth=1.5, alpha=0.7)

    # Plot EMA
    ema_col = f'ema{period}'
    ax.plot(data.index, data[ema_col], label=f'EMA{period}',
            color='blue', linewidth=2.5, linestyle='--', alpha=0.9)

    # Count crosses for this period
    above = data['close'] > data[ema_col]
    cross_above = (above == True) & (above.shift(1) == False)
    cross_below = (above == False) & (above.shift(1) == True)
    total_crosses = cross_above.sum() + cross_below.sum()
    avg_bars_between = len(data) / total_crosses if total_crosses > 0 else len(data)

    # Calculate average distance
    avg_distance_bps = (abs(data['close'] - data[ema_col]) / data['close'] * 10000).mean()

    # Title with metrics
    ax.set_title(f'EMA{period}: {total_crosses} crosses, Avg {avg_bars_between:.1f} bars between, Distance {avg_distance_bps:.1f} bps',
                fontsize=14, pad=10, fontweight='bold')

    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=11)

# Format x-axis for bottom plot
axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()

# Save
output_path = Path("experiments/ema/ema_comparison_jan2024.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[OK] Saved: {output_path}")
print("\nCompare the charts:")
print("- Which EMA has cleaner trend periods?")
print("- Which EMA stays relevant to price?")
print("- Which EMA do you visually prefer?")
