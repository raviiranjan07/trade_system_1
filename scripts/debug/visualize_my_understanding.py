"""
Visualize MY UNDERSTANDING of EMA Bounce Strategy

Shows the specific example you pointed out:
- ~1:00 AM: LONG entry (from above) → exit at peak
- ~2:30-3:00 AM: SHORT entry (from below) → bounce down

Run: python scripts/debug/visualize_my_understanding.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch, Rectangle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("VISUALIZING MY UNDERSTANDING")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading data...")
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

# Get the specific time period you mentioned (Jan 1, 2024)
start = "2024-01-01 00:00"
end = "2024-01-01 06:00"
data = ohlcv[(ohlcv.index >= start) & (ohlcv.index < end)].copy()

# Calculate EMA7
data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()

print(f"Time period: {data.index[0]} to {data.index[-1]}")
print(f"Total bars: {len(data)}")

# =============================================================================
# CREATE VISUALIZATION
# =============================================================================
print("\nCreating visualization...")

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
fig.suptitle('MY UNDERSTANDING: EMA7 as Dynamic Support/Resistance\nSame Level, Different Approaches',
             fontsize=18, fontweight='bold')

# Plot price and EMA
ax.plot(data.index, data['close'], label='Price (Close)', color='black', linewidth=2.5, zorder=3)
ax.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=3, linestyle='--', zorder=2)

# Fill area to show above/below EMA
ax.fill_between(data.index, data['close'], data['ema7'],
                where=(data['close'] > data['ema7']),
                alpha=0.2, color='green', label='Price ABOVE EMA')
ax.fill_between(data.index, data['close'], data['ema7'],
                where=(data['close'] <= data['ema7']),
                alpha=0.2, color='red', label='Price BELOW EMA')

# ============================================================================
# SCENARIO 1: ~1:00 AM - LONG Entry (from above)
# ============================================================================

# Find approximate time ~1:00
long_entry_time = pd.Timestamp('2024-01-01 01:00:00+00:00')
# Find closest bar
long_idx = data.index.get_indexer([long_entry_time], method='nearest')[0]
long_bar = data.iloc[long_idx]

# Mark the approach (previous bars showing price coming from above)
approach_start_idx = max(0, long_idx - 3)
approach_data = data.iloc[approach_start_idx:long_idx+1]

# Draw approach path
ax.plot(approach_data.index, approach_data['close'],
        color='green', linewidth=5, alpha=0.7, zorder=5,
        label='Approach from ABOVE')

# Mark entry point
ax.scatter(long_bar.name, long_bar['close'],
          color='green', s=500, marker='^',
          edgecolors='darkgreen', linewidths=3,
          label='LONG Entry', zorder=10)

# Add annotation for LONG
ax.annotate('LONG ENTRY\n(Price came from ABOVE)\nEMA = SUPPORT',
           xy=(long_bar.name, long_bar['close']),
           xytext=(long_bar.name, long_bar['close'] + 150),
           fontsize=13, fontweight='bold', color='darkgreen',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', edgecolor='darkgreen', linewidth=2),
           arrowprops=dict(arrowstyle='->', color='darkgreen', lw=3),
           ha='center', zorder=15)

# Show expected bounce UP
if long_idx + 4 < len(data):
    bounce_data = data.iloc[long_idx:long_idx+5]
    ax.plot(bounce_data.index, bounce_data['close'],
            color='green', linewidth=4, linestyle=':', alpha=0.8, zorder=4)

    # Mark exit at peak
    peak_idx = bounce_data['close'].idxmax()
    peak_price = bounce_data.loc[peak_idx, 'close']
    ax.scatter(peak_idx, peak_price,
              color='gold', s=400, marker='*',
              edgecolors='darkgreen', linewidths=2,
              label='Exit (Peak)', zorder=10)

# ============================================================================
# SCENARIO 2: ~2:30-3:00 AM - SHORT Entry (from below)
# ============================================================================

# Find approximate time ~2:45
short_entry_time = pd.Timestamp('2024-01-01 02:45:00+00:00')
# Find closest bar
short_idx = data.index.get_indexer([short_entry_time], method='nearest')[0]
short_bar = data.iloc[short_idx]

# Mark the approach (previous bars showing price coming from below)
approach_start_idx = max(0, short_idx - 3)
approach_data = data.iloc[approach_start_idx:short_idx+1]

# Draw approach path
ax.plot(approach_data.index, approach_data['close'],
        color='red', linewidth=5, alpha=0.7, zorder=5,
        label='Approach from BELOW')

# Mark entry point
ax.scatter(short_bar.name, short_bar['close'],
          color='red', s=500, marker='v',
          edgecolors='darkred', linewidths=3,
          label='SHORT Entry', zorder=10)

# Add annotation for SHORT
ax.annotate('SHORT ENTRY\n(Price came from BELOW)\nEMA = RESISTANCE',
           xy=(short_bar.name, short_bar['close']),
           xytext=(short_bar.name, short_bar['close'] - 150),
           fontsize=13, fontweight='bold', color='darkred',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcoral', edgecolor='darkred', linewidth=2),
           arrowprops=dict(arrowstyle='->', color='darkred', lw=3),
           ha='center', zorder=15)

# Show expected bounce DOWN
if short_idx + 4 < len(data):
    bounce_data = data.iloc[short_idx:short_idx+5]
    ax.plot(bounce_data.index, bounce_data['close'],
            color='red', linewidth=4, linestyle=':', alpha=0.8, zorder=4)

# ============================================================================
# KEY INSIGHT BOX
# ============================================================================

# Add text box with key insight
insight_text = (
    "KEY INSIGHT:\n\n"
    "Same EMA7 level (~$42,300)\n"
    "Acts as BOTH:\n"
    "• SUPPORT (when approached from above) → LONG\n"
    "• RESISTANCE (when approached from below) → SHORT\n\n"
    "Direction matters!\n"
    "NOT just trend (EMA50 vs EMA200)"
)

ax.text(0.02, 0.98, insight_text,
       transform=ax.transAxes,
       fontsize=12,
       verticalalignment='top',
       bbox=dict(boxstyle='round,pad=1',
                facecolor='yellow',
                edgecolor='black',
                linewidth=2,
                alpha=0.9),
       zorder=20)

# ============================================================================
# FORMATTING
# ============================================================================

ax.set_xlabel('Time', fontsize=14, fontweight='bold')
ax.set_ylabel('Price (USD)', fontsize=14, fontweight='bold')
ax.set_title('Same EMA Level = Dynamic Support/Resistance', fontsize=16, pad=20)

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.xticks(rotation=45)

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

# Grid
ax.grid(True, alpha=0.3, linestyle='--')

# Tight layout
plt.tight_layout()

# Save
output_dir = Path("experiments/ema")
output_path = output_dir / "my_understanding_ema_strategy.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')

print(f"\n[OK] Saved visualization: {output_path}")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print("\nThis shows MY UNDERSTANDING:")
print("1. ~1:00 AM: Price from above -> LONG -> Exit at peak")
print("2. ~2:45 AM: Price from below -> SHORT -> Bounce down")
print("3. Same EMA7 level works as BOTH support AND resistance")
print("4. Approach direction determines the trade direction")
print("=" * 80)
