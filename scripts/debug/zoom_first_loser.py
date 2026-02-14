"""
Zoom into the FIRST LOSER trade to see WHY it took LONG when price was falling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

print("Creating zoomed view of first losing trade...")

# Load 15-min OHLCV
ohlcv_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
data = pd.read_parquet(ohlcv_path)

# Calculate EMA7
data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()

# Focus on the period around first loser: 2024-01-01 02:45
# Show 2 hours before and 3 hours after
start_time = "2024-01-01 00:00:00"
end_time = "2024-01-01 08:00:00"

period_data = data[(data.index >= start_time) & (data.index < end_time)].copy()

# Detect crosses
period_data['above_ema'] = period_data['close'] > period_data['ema7']
period_data['below_ema'] = period_data['close'] < period_data['ema7']

period_data['cross_above'] = (period_data['above_ema'] == True) & (period_data['above_ema'].shift(1) == False)
period_data['cross_below'] = (period_data['below_ema'] == True) & (period_data['below_ema'].shift(1) == False)

# Find the specific entry at 02:45
entry_time = pd.Timestamp("2024-01-01 02:45:00+00:00")
entry_price = 42581.10
exit_time = pd.Timestamp("2024-01-01 05:15:00+00:00")
exit_price = 42297.91

print(f"\nEntry: {entry_time}")
print(f"Entry price: ${entry_price:.2f}")
print(f"Exit: {exit_time}")
print(f"Exit price: ${exit_price:.2f}")

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
fig.suptitle('FIRST LOSER - Why did it take LONG when price was falling?\n2024-01-01 00:00 to 08:00',
             fontsize=16, fontweight='bold')

# Plot price and EMA
ax.plot(period_data.index, period_data['close'], label='Price', color='black', linewidth=3, zorder=2, marker='o', markersize=4)
ax.plot(period_data.index, period_data['ema7'], label='EMA7', color='blue', linewidth=3, linestyle='--', zorder=2, marker='s', markersize=3)

# Mark all crosses in this period
for idx, row in period_data.iterrows():
    if row['cross_above']:
        ax.scatter(idx, row['close'], color='green', s=400, marker='^',
                  edgecolors='black', linewidths=3, zorder=10, alpha=0.8)
        ax.annotate('CROSS\nABOVE', xy=(idx, row['close']),
                   xytext=(0, 20), textcoords='offset points',
                   fontsize=9, fontweight='bold', color='green',
                   ha='center', bbox=dict(boxstyle='round,pad=0.5',
                                         facecolor='lightgreen', edgecolor='green', linewidth=2))
    elif row['cross_below']:
        ax.scatter(idx, row['close'], color='red', s=400, marker='v',
                  edgecolors='black', linewidths=3, zorder=10, alpha=0.8)
        ax.annotate('CROSS\nBELOW', xy=(idx, row['close']),
                   xytext=(0, -30), textcoords='offset points',
                   fontsize=9, fontweight='bold', color='red',
                   ha='center', bbox=dict(boxstyle='round,pad=0.5',
                                         facecolor='lightcoral', edgecolor='red', linewidth=2))

# Highlight the losing trade
ax.plot([entry_time, exit_time], [entry_price, exit_price],
       color='red', linewidth=4, alpha=0.8, linestyle='-', zorder=8, label='Trade Path')

# Entry marker
ax.scatter(entry_time, entry_price, color='red', s=600, marker='^',
          edgecolors='yellow', linewidths=4, zorder=12, alpha=1.0)

# Exit marker
ax.scatter(exit_time, exit_price, color='red', s=600, marker='X',
          edgecolors='yellow', linewidths=4, zorder=12, alpha=1.0)

# Add annotation for the trade
ax.annotate('ENTRY: LONG\n(Price crossed ABOVE EMA7)',
           xy=(entry_time, entry_price),
           xytext=(30, 60), textcoords='offset points',
           fontsize=12, fontweight='bold', color='darkred',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow',
                    edgecolor='red', linewidth=3),
           arrowprops=dict(arrowstyle='->', color='red', linewidth=3),
           ha='left', zorder=20)

ax.annotate('EXIT: -74.5 bps\n(Held 10 bars)',
           xy=(exit_time, exit_price),
           xytext=(30, -60), textcoords='offset points',
           fontsize=12, fontweight='bold', color='darkred',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow',
                    edgecolor='red', linewidth=3),
           arrowprops=dict(arrowstyle='->', color='red', linewidth=3),
           ha='left', zorder=20)

# Show where price was BEFORE the cross
prev_5_bars = period_data[period_data.index < entry_time].tail(5)
for idx, row in prev_5_bars.iterrows():
    ax.scatter(idx, row['close'], color='orange', s=200, marker='o',
              edgecolors='black', linewidths=2, zorder=9, alpha=0.7)

# Formatting
ax.set_xlabel('Time', fontsize=14, fontweight='bold')
ax.set_ylabel('Price (USD)', fontsize=14, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)

ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)

# Add text box explaining what happened
explanation = """WHY DID IT TAKE LONG?

Strategy rule: If price crosses ABOVE EMA7 → LONG

At 02:45:
• Price crossed from BELOW to ABOVE EMA7
• Strategy mechanically took LONG
• But price was in DOWNTREND!

Problem: Strategy only looks at the CROSS,
not the broader TREND or MOMENTUM!

This is a FALSE BREAKOUT - price touched
EMA7 briefly then continued falling."""

ax.text(0.02, 0.98, explanation,
       transform=ax.transAxes, fontsize=11,
       verticalalignment='top', horizontalalignment='left',
       bbox=dict(boxstyle='round,pad=1.0', facecolor='lightyellow',
                edgecolor='black', linewidth=3, alpha=0.95),
       zorder=25, family='monospace', fontweight='bold')

plt.tight_layout()

# Save
output_path = Path("experiments/ema/zoom_first_loser_why_long.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[OK] Saved: {output_path}")
print("\nThis shows WHY the strategy took LONG when price was falling:")
print("  - Strategy only looks at EMA7 cross")
print("  - Doesn't check broader trend/momentum")
print("  - This was a FALSE BREAKOUT!")
