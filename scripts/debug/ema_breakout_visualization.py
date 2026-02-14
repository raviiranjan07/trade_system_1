"""
EMA7 BREAKOUT Strategy (NOT bounce)

Entry: Price crosses EMA7
Exit: 12 bps profit target

Run: python scripts/debug/ema_breakout_visualization.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

print("Creating EMA7 BREAKOUT visualization...")

# Load 15-min OHLCV
ohlcv_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
data = pd.read_parquet(ohlcv_path)

# Filter to Jan 1, 2024
data = data[(data.index >= "2024-01-01") & (data.index < "2024-01-02")].copy()
data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()

print(f"Price data: {len(data)} candles")

# Detect crosses
data['above_ema'] = data['close'] > data['ema7']
data['below_ema'] = data['close'] < data['ema7']

# Cross detection
data['cross_above'] = (data['above_ema'] == True) & (data['above_ema'].shift(1) == False)  # Was below, now above
data['cross_below'] = (data['below_ema'] == True) & (data['below_ema'].shift(1) == False)  # Was above, now below

# Find trades
trades = []
TP_BPS = 12

for idx in range(len(data)):
    if idx + 10 >= len(data):
        break

    row = data.iloc[idx]
    entry_time = row.name
    entry_price = row['close']

    # LONG entry (cross above)
    if row['cross_above']:
        direction = 'LONG'
        tp_price = entry_price * (1 + TP_BPS / 10000)

        # Find exit
        for j in range(idx + 1, min(idx + 11, len(data))):
            future_bar = data.iloc[j]
            if future_bar['high'] >= tp_price:
                exit_time = future_bar.name
                exit_price = tp_price
                bars_held = j - idx
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'direction': direction,
                    'bars_held': bars_held
                })
                break

    # SHORT entry (cross below)
    elif row['cross_below']:
        direction = 'SHORT'
        tp_price = entry_price * (1 - TP_BPS / 10000)

        # Find exit
        for j in range(idx + 1, min(idx + 11, len(data))):
            future_bar = data.iloc[j]
            if future_bar['low'] <= tp_price:
                exit_time = future_bar.name
                exit_price = tp_price
                bars_held = j - idx
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'direction': direction,
                    'bars_held': bars_held
                })
                break

trades_df = pd.DataFrame(trades)
print(f"\nFound {len(trades_df)} breakout trades that hit 12bp target")
print(f"  LONG: {(trades_df['direction'] == 'LONG').sum()}")
print(f"  SHORT: {(trades_df['direction'] == 'SHORT').sum()}")

# Create visualization
fig, ax = plt.subplots(1, 1, figsize=(24, 10))
fig.suptitle('EMA7 BREAKOUT Strategy - 2024-01-01\nEntry: Cross EMA7 | Exit: 12bp TP',
             fontsize=16, fontweight='bold')

# Plot price and EMA
ax.plot(data.index, data['close'], label='Price', color='black', linewidth=2, zorder=2)
ax.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=2.5, linestyle='--', zorder=2)

# Plot trades
for _, trade in trades_df.iterrows():
    entry_time = trade['entry_time']
    entry_price = trade['entry_price']
    exit_time = trade['exit_time']
    exit_price = trade['exit_price']
    direction = trade['direction']

    color = 'green' if direction == 'LONG' else 'red'
    marker = '^' if direction == 'LONG' else 'v'

    # Entry marker
    ax.scatter(entry_time, entry_price, color=color, s=250, marker=marker,
              edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)

    # Exit marker
    ax.scatter(exit_time, exit_price, color=color, s=350, marker='*',
              edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)

    # Line
    ax.plot([entry_time, exit_time], [entry_price, exit_price],
           color=color, linewidth=2.5, alpha=0.7, linestyle='-', zorder=5)

    # Annotation
    mid_time = entry_time + (exit_time - entry_time) / 2
    mid_price = (entry_price + exit_price) / 2
    pnl = 12 - 8  # Gross 12bp - 8bp fees

    ax.annotate(f"{pnl}bp", xy=(mid_time, mid_price),
               fontsize=10, fontweight='bold', color='darkgreen' if direction == 'LONG' else 'darkred',
               bbox=dict(boxstyle='round,pad=0.4',
                        facecolor='lightgreen' if direction == 'LONG' else 'lightcoral',
                        edgecolor='darkgreen' if direction == 'LONG' else 'darkred',
                        linewidth=2),
               ha='center', zorder=15)

# Formatting
ax.set_xlabel('Time', fontsize=13, fontweight='bold')
ax.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax.set_title('Cross ABOVE EMA7 = LONG | Cross BELOW EMA7 = SHORT', fontsize=14, pad=15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.xticks(rotation=45)

# Legend
legend_elements = [
    plt.Line2D([0], [0], color='black', linewidth=2, label='Price'),
    plt.Line2D([0], [0], color='blue', linewidth=2.5, linestyle='--', label='EMA7'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=12,
              label='LONG (cross above)', markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=12,
              label='SHORT (cross below)', markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=14,
              label='Exit (12bp TP)', markeredgecolor='black', markeredgewidth=2),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

# Stats box
long_count = (trades_df['direction'] == 'LONG').sum()
short_count = (trades_df['direction'] == 'SHORT').sum()

stats_text = f"BREAKOUT STRATEGY\n\n"
stats_text += f"Total: {len(trades_df)}\n"
stats_text += f"LONG: {long_count}\n"
stats_text += f"SHORT: {short_count}\n\n"
stats_text += f"Entry: Cross EMA7\n"
stats_text += f"Exit: 12bp TP\n"
stats_text += f"Net: 4bp per trade"

ax.text(0.98, 0.97, stats_text,
       transform=ax.transAxes, fontsize=11,
       verticalalignment='top', horizontalalignment='right',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                edgecolor='black', linewidth=2, alpha=0.9),
       zorder=20, family='monospace')

plt.tight_layout()

# Save
output_path = Path("experiments/ema/ema_breakout_strategy_2024_01_01.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[OK] Saved: {output_path}")
print("\nThis shows EMA7 BREAKOUT strategy (cross above/below), NOT bounce!")
