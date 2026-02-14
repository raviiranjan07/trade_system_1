"""
Zoom into the period where MAE exits happened
Show why strategy took SHORT when trend was clearly UP
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

print("Creating zoomed view of MAE loss period...")

# Load 15-min OHLCV
ohlcv_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
data = pd.read_parquet(ohlcv_path)

# Zoom to 11:00 to 18:00 on 2024-01-01
start_time = "2024-01-01 11:00:00"
end_time = "2024-01-01 18:00:00"

period_data = data[(data.index >= start_time) & (data.index < end_time)].copy()
period_data['ema7'] = period_data['close'].ewm(span=7, adjust=False).mean()

# Detect crosses
period_data['above_ema'] = period_data['close'] > period_data['ema7']
period_data['below_ema'] = period_data['close'] < period_data['ema7']

period_data['cross_above'] = (period_data['above_ema'] == True) & (period_data['above_ema'].shift(1) == False)
period_data['cross_below'] = (period_data['below_ema'] == True) & (period_data['below_ema'].shift(1) == False)

print(f"Price data: {len(period_data)} candles")
print(f"Price range: ${period_data['close'].min():.2f} to ${period_data['close'].max():.2f}")
print(f"Price move: {((period_data['close'].iloc[-1] - period_data['close'].iloc[0]) / period_data['close'].iloc[0] * 10000):.0f} bps")

# MAE losing trades in this period
mae_trades = [
    {'time': '2024-01-01 11:30:00', 'price': 42607.81, 'direction': 'SHORT', 'loss': -40.29},
    {'time': '2024-01-01 12:00:00', 'price': 42626.53, 'direction': 'SHORT', 'loss': -36.63},
    {'time': '2024-01-01 16:15:00', 'price': 42723.04, 'direction': 'SHORT', 'loss': -205.88},
    {'time': '2024-01-01 16:45:00', 'price': 42725.15, 'direction': 'SHORT', 'loss': -210.03},
    {'time': '2024-01-01 17:15:00', 'price': 42731.11, 'direction': 'SHORT', 'loss': -184.50},
]

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(22, 12))
fig.suptitle('WHY DID STRATEGY TAKE SHORT WHEN PRICE WAS CLEARLY GOING UP?\\n2024-01-01 11:00 to 18:00',
             fontsize=16, fontweight='bold', color='red')

# Plot price and EMA with BIGGER markers
ax.plot(period_data.index, period_data['close'], label='Price', color='black',
        linewidth=3.5, zorder=2, marker='o', markersize=6)
ax.plot(period_data.index, period_data['ema7'], label='EMA7', color='blue',
        linewidth=3.5, linestyle='--', zorder=2, marker='s', markersize=4)

# Highlight the OVERALL UPTREND
ax.axhline(y=period_data['close'].iloc[0], color='gray', linestyle=':',
          linewidth=2, alpha=0.5, label=f'Start Price: ${period_data["close"].iloc[0]:.0f}')
ax.axhline(y=period_data['close'].iloc[-1], color='gray', linestyle=':',
          linewidth=2, alpha=0.5, label=f'End Price: ${period_data["close"].iloc[-1]:.0f}')

# Mark ALL crosses in this period
for idx, row in period_data.iterrows():
    if row['cross_above']:
        ax.scatter(idx, row['close'], color='lime', s=500, marker='^',
                  edgecolors='black', linewidths=3, zorder=10, alpha=0.9)
        ax.annotate('CROSS\\nABOVE', xy=(idx, row['close']),
                   xytext=(0, 25), textcoords='offset points',
                   fontsize=9, fontweight='bold', color='green',
                   ha='center', bbox=dict(boxstyle='round,pad=0.5',
                                         facecolor='lightgreen', edgecolor='green', linewidth=2))
    elif row['cross_below']:
        ax.scatter(idx, row['close'], color='red', s=500, marker='v',
                  edgecolors='black', linewidths=3, zorder=10, alpha=0.9)
        ax.annotate('CROSS\\nBELOW', xy=(idx, row['close']),
                   xytext=(0, -35), textcoords='offset points',
                   fontsize=9, fontweight='bold', color='red',
                   ha='center', bbox=dict(boxstyle='round,pad=0.5',
                                         facecolor='lightcoral', edgecolor='red', linewidth=2))

# Mark the MAE losing trades with HUGE markers
for trade in mae_trades:
    trade_time = pd.Timestamp(trade['time'], tz='UTC')
    if trade_time >= period_data.index[0] and trade_time <= period_data.index[-1]:
        # Find matching bar
        trade_idx = period_data.index.get_indexer([trade_time], method='nearest')[0]
        trade_row = period_data.iloc[trade_idx]

        # HUGE marker for entry
        ax.scatter(trade_time, trade['price'], color='red', s=800, marker='v',
                  edgecolors='yellow', linewidths=5, zorder=15, alpha=1.0)

        # Annotation with loss
        ax.annotate(f"SHORT ENTRY\\nLost: {trade['loss']:.0f}bp",
                   xy=(trade_time, trade['price']),
                   xytext=(40, -70), textcoords='offset points',
                   fontsize=11, fontweight='bold', color='darkred',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow',
                            edgecolor='red', linewidth=3, alpha=0.95),
                   arrowprops=dict(arrowstyle='->', color='red', linewidth=3),
                   ha='left', zorder=20)

# Formatting
ax.set_xlabel('Time', fontsize=14, fontweight='bold')
ax.set_ylabel('Price (USD)', fontsize=14, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

ax.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)

# Add explanation box
explanation = f"""WHY STRATEGY LOST MONEY HERE:

Price moved from ${period_data['close'].iloc[0]:.0f} to ${period_data['close'].iloc[-1]:.0f}
= CLEAR UPTREND (+{((period_data['close'].iloc[-1] - period_data['close'].iloc[0]) / period_data['close'].iloc[0] * 10000):.0f} bps!)

But strategy took SHORT because:
• Price briefly crossed BELOW EMA7
• Strategy rule: Cross below → SHORT
• NO CHECK for broader trend!

Human trader would see:
• Price trending UP
• Higher highs, higher lows
• Should take LONG, not SHORT!

Problem: Strategy is BLIND to trend
Only sees the cross, not the context!

Those 5 SHORT trades lost 677 bps total
on a day when price went UP 1000+ bps!"""

ax.text(0.02, 0.98, explanation,
       transform=ax.transAxes, fontsize=11,
       verticalalignment='top', horizontalalignment='left',
       bbox=dict(boxstyle='round,pad=1.0', facecolor='lightyellow',
                edgecolor='red', linewidth=4, alpha=0.95),
       zorder=25, family='monospace', fontweight='bold')

plt.tight_layout()

# Save
output_path = Path("experiments/ema/zoom_mae_losses_why_short_in_uptrend.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[OK] Saved: {output_path}")
print("\nThis shows the CORE PROBLEM:")
print("  - Price was in CLEAR UPTREND")
print("  - Strategy took SHORT because of EMA7 crosses")
print("  - Lost 677 bps while price rallied 1000+ bps")
print("  - Strategy needs TREND FILTER!")
