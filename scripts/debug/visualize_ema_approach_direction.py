"""
Visualize EMA Approach Direction Strategy

Tests CORRECT logic:
- LONG: Price was ABOVE EMA, pulled back to touch it (support test)
- SHORT: Price was BELOW EMA, rallied to touch it (resistance test)

Run: python scripts/debug/visualize_ema_approach_direction.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numba import njit
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("VISUALIZING EMA APPROACH DIRECTION STRATEGY")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
START_DATE = "2024-01-01"
END_DATE = "2024-01-07"  # 1 week sample
HORIZON = 10
LOOKBACK = 1  # How many bars back to check if price was above/below EMA

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/4] Loading data...")
ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)

# Resample to 15-minute
ohlcv = ohlcv_1m.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Filter to sample period
data = ohlcv[(ohlcv.index >= START_DATE) & (ohlcv.index < END_DATE)].copy()
print(f"Sample period: {data.index[0]} to {data.index[-1]}")
print(f"Total candles: {len(data)}")

# =============================================================================
# CALCULATE INDICATORS
# =============================================================================
print("\n[2/4] Calculating indicators...")
data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()

# Touch detection
data['dist_from_ema7'] = abs(data['close'] - data['ema7']) / data['ema7'] * 100
data['touch'] = data['dist_from_ema7'] <= 0.2

# Check approach direction
data['was_above_ema'] = data['close'].shift(LOOKBACK) > data['ema7'].shift(LOOKBACK)
data['was_below_ema'] = data['close'].shift(LOOKBACK) < data['ema7'].shift(LOOKBACK)

# Determine entry direction based on approach
data['entry_direction'] = 0  # 0 = no entry, 1 = LONG, -1 = SHORT

# LONG: Was above EMA, now touching (support test)
data.loc[data['touch'] & data['was_above_ema'], 'entry_direction'] = 1

# SHORT: Was below EMA, now touching (resistance test)
data.loc[data['touch'] & data['was_below_ema'], 'entry_direction'] = -1

# Calculate forward MFE
@njit
def calculate_forward_mfe(highs, lows, close_prices, horizon):
    n = len(close_prices)
    mfe_long = np.full(n, np.nan)
    mfe_short = np.full(n, np.nan)

    for i in range(n - horizon):
        entry = close_prices[i]
        future_highs = highs[i+1:i+1+horizon]
        future_lows = lows[i+1:i+1+horizon]

        # LONG
        max_high = np.max(future_highs)
        mfe_long[i] = (max_high - entry) / entry * 10000

        # SHORT
        min_low = np.min(future_lows)
        mfe_short[i] = (entry - min_low) / entry * 10000

    return mfe_long, mfe_short

mfe_long, mfe_short = calculate_forward_mfe(
    data['high'].values,
    data['low'].values,
    data['close'].values,
    HORIZON
)

data['mfe_long'] = mfe_long
data['mfe_short'] = mfe_short

# Filter entries
long_entries = data[data['entry_direction'] == 1].copy()
short_entries = data[data['entry_direction'] == -1].copy()

# Drop NaN
long_entries = long_entries.dropna(subset=['mfe_long'])
short_entries = short_entries.dropna(subset=['mfe_short'])

# Determine success
long_entries['success'] = long_entries['mfe_long'] >= 12
short_entries['success'] = short_entries['mfe_short'] >= 12

print(f"\nLONG entries (from above): {len(long_entries)}")
print(f"  Successful: {long_entries['success'].sum()} ({long_entries['success'].mean()*100:.1f}%)")
print(f"\nSHORT entries (from below): {len(short_entries)}")
print(f"  Successful: {short_entries['success'].sum()} ({short_entries['success'].mean()*100:.1f}%)")

# =============================================================================
# CREATE VISUALIZATION
# =============================================================================
print("\n[3/4] Creating visualization...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
fig.suptitle(f'EMA7 Approach Direction Strategy\n{START_DATE} to {END_DATE}\nLONG = From Above | SHORT = From Below',
             fontsize=16, fontweight='bold')

# ----- CHART 1: Price + EMA + Entry Points -----
ax1.plot(data.index, data['close'], label='Price', color='black', linewidth=1, alpha=0.7)
ax1.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=2)

# Mark LONG entries (from above)
long_success = long_entries[long_entries['success']]
long_fail = long_entries[~long_entries['success']]

ax1.scatter(long_success.index, long_success['close'],
           color='green', s=150, marker='^', label='LONG Success (from above)', zorder=5, alpha=0.8, edgecolors='darkgreen', linewidths=2)
ax1.scatter(long_fail.index, long_fail['close'],
           color='lightgreen', s=150, marker='^', label='LONG Failed', zorder=5, alpha=0.5, edgecolors='green', linewidths=1)

# Mark SHORT entries (from below)
short_success = short_entries[short_entries['success']]
short_fail = short_entries[~short_entries['success']]

ax1.scatter(short_success.index, short_success['close'],
           color='red', s=150, marker='v', label='SHORT Success (from below)', zorder=5, alpha=0.8, edgecolors='darkred', linewidths=2)
ax1.scatter(short_fail.index, short_fail['close'],
           color='lightcoral', s=150, marker='v', label='SHORT Failed', zorder=5, alpha=0.5, edgecolors='red', linewidths=1)

ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.set_title('Price Chart: Entry Points Based on Approach Direction', fontsize=14)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# ----- CHART 2: Forward Paths -----
ax2.plot(data.index, data['close'], label='Price', color='gray', linewidth=1, alpha=0.3)
ax2.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=1, alpha=0.3)

# Plot forward paths for LONG entries
for idx, row in long_entries.iterrows():
    touch_idx = data.index.get_loc(idx)
    if touch_idx + HORIZON >= len(data):
        continue

    future_data = data.iloc[touch_idx : touch_idx + HORIZON + 1]
    color = 'green' if row['success'] else 'lightgreen'
    alpha = 0.7 if row['success'] else 0.3
    linewidth = 2 if row['success'] else 1

    ax2.plot(future_data.index, future_data['close'],
            color=color, linewidth=linewidth, alpha=alpha, zorder=3)
    ax2.scatter(idx, row['close'], color=color, s=80, marker='^', zorder=4, alpha=0.8)

# Plot forward paths for SHORT entries
for idx, row in short_entries.iterrows():
    touch_idx = data.index.get_loc(idx)
    if touch_idx + HORIZON >= len(data):
        continue

    future_data = data.iloc[touch_idx : touch_idx + HORIZON + 1]
    color = 'red' if row['success'] else 'lightcoral'
    alpha = 0.7 if row['success'] else 0.3
    linewidth = 2 if row['success'] else 1

    ax2.plot(future_data.index, future_data['close'],
            color=color, linewidth=linewidth, alpha=alpha, zorder=3)
    ax2.scatter(idx, row['close'], color=color, s=80, marker='v', zorder=4, alpha=0.8)

ax2.set_xlabel('Date/Time', fontsize=12)
ax2.set_ylabel('Price (USD)', fontsize=12)
ax2.set_title(f'Forward Price Paths (Next {HORIZON} bars = 2.5 hours)', fontsize=14)
ax2.grid(True, alpha=0.3)

# Format x-axis
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
plt.xticks(rotation=45)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='green', linewidth=2, marker='^', markersize=8,
           label=f'LONG from above - Success ({long_success.shape[0]})'),
    Line2D([0], [0], color='lightgreen', linewidth=1, marker='^', markersize=8,
           label=f'LONG from above - Failed ({long_fail.shape[0]})'),
    Line2D([0], [0], color='red', linewidth=2, marker='v', markersize=8,
           label=f'SHORT from below - Success ({short_success.shape[0]})'),
    Line2D([0], [0], color='lightcoral', linewidth=1, marker='v', markersize=8,
           label=f'SHORT from below - Failed ({short_fail.shape[0]})'),
]
ax2.legend(handles=legend_elements, loc='best', fontsize=10)

plt.tight_layout()

# Save
output_dir = Path("experiments/ema")
output_path = output_dir / "ema_approach_direction_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization: {output_path}")

# =============================================================================
# CREATE DETAILED EXAMPLES
# =============================================================================
print("\n[4/4] Creating detailed examples...")

if len(long_success) > 0 and len(short_success) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle('Detailed Examples: LONG (from above) vs SHORT (from below)',
                 fontsize=16, fontweight='bold')

    # Example 1: LONG entry (from above)
    example_long = long_success.iloc[0]
    ax = axes[0]

    touch_time = example_long.name
    touch_idx = data.index.get_loc(touch_time)

    context_before = 20
    context_after = HORIZON + 5
    start_idx = max(0, touch_idx - context_before)
    end_idx = min(len(data), touch_idx + context_after)

    context_data = data.iloc[start_idx:end_idx]

    ax.plot(context_data.index, context_data['close'], label='Price', color='black', linewidth=2)
    ax.plot(context_data.index, context_data['ema7'], label='EMA7', color='blue', linewidth=2, linestyle='--')

    # Mark touch point
    ax.scatter(touch_time, example_long['close'], color='green', s=300, marker='^',
              label='LONG Entry (from above)', zorder=10, edgecolors='black', linewidths=2)

    # Show where price came from
    prev_idx = max(0, touch_idx - 3)
    prev_data = data.iloc[prev_idx:touch_idx+1]
    ax.plot(prev_data.index, prev_data['close'], color='green', linewidth=3,
            alpha=0.5, label='Approach from above')

    # TP level
    entry_price = example_long['close']
    tp_level = entry_price * 1.0012
    ax.axhline(y=entry_price, color='orange', linestyle=':', linewidth=2, label='Entry', alpha=0.7)
    ax.axhline(y=tp_level, color='green', linestyle=':', linewidth=2, label='12bp Target', alpha=0.7)

    # Highlight forward period
    forward_data = data.iloc[touch_idx:touch_idx+HORIZON+1]
    ax.axvspan(forward_data.index[0], forward_data.index[-1],
              alpha=0.2, color='green', label=f'Forward {HORIZON} bars')

    textstr = f"LONG Entry\nApproached from ABOVE\nMFE: {example_long['mfe_long']:.1f} bps\nSuccess: YES"
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.set_title('LONG: Price Pulled Back from Above (Support Test)', fontsize=14, fontweight='bold', color='green')
    ax.set_xlabel('Date/Time', fontsize=11)
    ax.set_ylabel('Price (USD)', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Example 2: SHORT entry (from below)
    example_short = short_success.iloc[0]
    ax = axes[1]

    touch_time = example_short.name
    touch_idx = data.index.get_loc(touch_time)

    start_idx = max(0, touch_idx - context_before)
    end_idx = min(len(data), touch_idx + context_after)

    context_data = data.iloc[start_idx:end_idx]

    ax.plot(context_data.index, context_data['close'], label='Price', color='black', linewidth=2)
    ax.plot(context_data.index, context_data['ema7'], label='EMA7', color='blue', linewidth=2, linestyle='--')

    # Mark touch point
    ax.scatter(touch_time, example_short['close'], color='red', s=300, marker='v',
              label='SHORT Entry (from below)', zorder=10, edgecolors='black', linewidths=2)

    # Show where price came from
    prev_idx = max(0, touch_idx - 3)
    prev_data = data.iloc[prev_idx:touch_idx+1]
    ax.plot(prev_data.index, prev_data['close'], color='red', linewidth=3,
            alpha=0.5, label='Approach from below')

    # TP level
    entry_price = example_short['close']
    tp_level = entry_price * 0.9988
    ax.axhline(y=entry_price, color='orange', linestyle=':', linewidth=2, label='Entry', alpha=0.7)
    ax.axhline(y=tp_level, color='red', linestyle=':', linewidth=2, label='12bp Target', alpha=0.7)

    # Highlight forward period
    forward_data = data.iloc[touch_idx:touch_idx+HORIZON+1]
    ax.axvspan(forward_data.index[0], forward_data.index[-1],
              alpha=0.2, color='red', label=f'Forward {HORIZON} bars')

    textstr = f"SHORT Entry\nApproached from BELOW\nMFE: {example_short['mfe_short']:.1f} bps\nSuccess: YES"
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    ax.set_title('SHORT: Price Rallied from Below (Resistance Test)', fontsize=14, fontweight='bold', color='red')
    ax.set_xlabel('Date/Time', fontsize=11)
    ax.set_ylabel('Price (USD)', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    detail_path = output_dir / "ema_approach_direction_detailed.png"
    plt.savefig(detail_path, dpi=150, bbox_inches='tight')
    print(f"Saved detailed examples: {detail_path}")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print(f"\nFiles created:")
print(f"  1. {output_path}")
print(f"  2. {detail_path}")
print(f"\nThis shows YOUR logic - approach direction matters!")
