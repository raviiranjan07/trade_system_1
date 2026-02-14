"""
Visualize EMA Analysis - What "77% success" actually looks like

Shows:
- Price chart with EMA7
- Touch points
- Forward price paths (next 2.5 hours)
- Color-coded by whether MFE reached 12 bps

Run: python scripts/debug/visualize_ema_analysis.py
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
print("VISUALIZING EMA ANALYSIS - What 77% Success Means")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
# Sample a short time period to visualize (otherwise too crowded)
START_DATE = "2024-01-01"
END_DATE = "2024-01-07"  # 1 week sample
HORIZON = 10  # H=10 bars forward

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
data['ema50'] = data['close'].ewm(span=50, adjust=False).mean()
data['ema200'] = data['close'].ewm(span=200, adjust=False).mean()
data['trend'] = np.where(data['ema50'] > data['ema200'], 1, -1)

# Touch detection
data['dist_from_ema7'] = abs(data['close'] - data['ema7']) / data['ema7'] * 100
data['touch'] = data['dist_from_ema7'] <= 0.2

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

# Determine success for each touch
touches = data[data['touch']].copy()
touches = touches.dropna(subset=['mfe_long', 'mfe_short'])

# Success = MFE >= 12 bps in trend direction
touches['success'] = False
touches.loc[touches['trend'] == 1, 'success'] = touches.loc[touches['trend'] == 1, 'mfe_long'] >= 12
touches.loc[touches['trend'] == -1, 'success'] = touches.loc[touches['trend'] == -1, 'mfe_short'] >= 12

print(f"Touches found: {len(touches)}")
print(f"Successful: {touches['success'].sum()} ({touches['success'].mean()*100:.1f}%)")
print(f"Failed: {(~touches['success']).sum()} ({(~touches['success']).mean()*100:.1f}%)")

# =============================================================================
# CREATE VISUALIZATION
# =============================================================================
print("\n[3/4] Creating visualization...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
fig.suptitle(f'EMA7 Bounce Analysis Visualization\n{START_DATE} to {END_DATE}',
             fontsize=16, fontweight='bold')

# ----- CHART 1: Price + EMA + Touch Points -----
ax1.plot(data.index, data['close'], label='Price', color='black', linewidth=1, alpha=0.7)
ax1.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=2)

# Mark touch points
success_touches = touches[touches['success']]
fail_touches = touches[~touches['success']]

ax1.scatter(success_touches.index, success_touches['close'],
           color='green', s=100, marker='^', label='Success (MFE >= 12bp)', zorder=5, alpha=0.7)
ax1.scatter(fail_touches.index, fail_touches['close'],
           color='red', s=100, marker='v', label='Failed (MFE < 12bp)', zorder=5, alpha=0.7)

ax1.set_ylabel('Price (USD)', fontsize=12)
ax1.set_title('Price Chart with EMA7 and Touch Points', fontsize=14)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# ----- CHART 2: Forward Paths After Touch -----
ax2.plot(data.index, data['close'], label='Price', color='gray', linewidth=1, alpha=0.3)

# Plot forward paths for each touch
for idx, row in touches.iterrows():
    touch_idx = data.index.get_loc(idx)

    if touch_idx + HORIZON >= len(data):
        continue

    # Get forward data
    future_data = data.iloc[touch_idx : touch_idx + HORIZON + 1]

    # Color based on success
    color = 'green' if row['success'] else 'red'
    alpha = 0.6 if row['success'] else 0.4
    linewidth = 2 if row['success'] else 1

    ax2.plot(future_data.index, future_data['close'],
            color=color, linewidth=linewidth, alpha=alpha, zorder=3)

    # Mark entry point
    ax2.scatter(idx, row['close'], color=color, s=80, marker='o', zorder=4, alpha=0.8)

ax2.set_xlabel('Date/Time', fontsize=12)
ax2.set_ylabel('Price (USD)', fontsize=12)
ax2.set_title(f'Forward Price Paths (Next {HORIZON} bars = 2.5 hours after each touch)', fontsize=14)
ax2.grid(True, alpha=0.3)

# Format x-axis
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
plt.xticks(rotation=45)

# Add legend manually
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='green', linewidth=2, label=f'Success: MFE reached 12+ bps ({success_touches.shape[0]} touches)'),
    Line2D([0], [0], color='red', linewidth=1, label=f'Failed: MFE < 12 bps ({fail_touches.shape[0]} touches)')
]
ax2.legend(handles=legend_elements, loc='best', fontsize=10)

plt.tight_layout()

# Save
output_dir = Path("experiments/ema")
output_path = output_dir / "ema_analysis_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization: {output_path}")

# =============================================================================
# CREATE DETAILED EXAMPLE
# =============================================================================
print("\n[4/4] Creating detailed example...")

# Pick one successful and one failed touch to zoom in
if len(success_touches) > 0 and len(fail_touches) > 0:
    example_success = success_touches.iloc[0]
    example_fail = fail_touches.iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle('Detailed Examples: Success vs Failure', fontsize=16, fontweight='bold')

    for i, (example, title, color) in enumerate([
        (example_success, 'SUCCESS: MFE Reached 12+ bps', 'green'),
        (example_fail, 'FAILED: MFE < 12 bps', 'red')
    ]):
        ax = axes[i]

        # Get index
        touch_time = example.name
        touch_idx = data.index.get_loc(touch_time)

        # Plot context (before + after)
        context_before = 20
        context_after = HORIZON + 5
        start_idx = max(0, touch_idx - context_before)
        end_idx = min(len(data), touch_idx + context_after)

        context_data = data.iloc[start_idx:end_idx]

        # Plot price and EMA
        ax.plot(context_data.index, context_data['close'], label='Price', color='black', linewidth=2)
        ax.plot(context_data.index, context_data['ema7'], label='EMA7', color='blue', linewidth=2, linestyle='--')

        # Mark touch point
        ax.scatter(touch_time, example['close'], color=color, s=300, marker='*',
                  label='Touch Point', zorder=10, edgecolors='black', linewidths=2)

        # Draw TP/SL levels
        entry_price = example['close']
        tp_level = entry_price * 1.0012 if example['trend'] == 1 else entry_price * 0.9988

        ax.axhline(y=entry_price, color='orange', linestyle=':', linewidth=2, label='Entry', alpha=0.7)
        ax.axhline(y=tp_level, color='green', linestyle=':', linewidth=2, label='12bp Target', alpha=0.7)

        # Highlight forward period
        forward_data = data.iloc[touch_idx:touch_idx+HORIZON+1]
        ax.axvspan(forward_data.index[0], forward_data.index[-1],
                  alpha=0.2, color=color, label=f'Forward {HORIZON} bars')

        # Add text annotations
        direction = "LONG" if example['trend'] == 1 else "SHORT"
        mfe_value = example['mfe_long'] if example['trend'] == 1 else example['mfe_short']

        textstr = f"Direction: {direction}\nMFE: {mfe_value:.1f} bps\nSuccess: {'YES' if example['success'] else 'NO'}"
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_title(title, fontsize=14, fontweight='bold', color=color)
        ax.set_xlabel('Date/Time', fontsize=11)
        ax.set_ylabel('Price (USD)', fontsize=11)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    detail_path = output_dir / "ema_analysis_detailed_examples.png"
    plt.savefig(detail_path, dpi=150, bbox_inches='tight')
    print(f"Saved detailed examples: {detail_path}")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print(f"\nFiles created:")
print(f"  1. {output_path}")
print(f"  2. {detail_path}")
print(f"\nOpen these images to see what the 77% success rate actually means!")
