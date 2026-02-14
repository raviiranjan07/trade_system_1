"""
Show ONLY trades that hit 12bp TP (successful bounces)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

print("Creating visualization with ONLY TP exits (successful bounces)...")

# Load trades
trades_df = pd.read_csv("experiments/ema/ema7_backtest_trades.csv")
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

# Load 15-min OHLCV
ohlcv_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
data = pd.read_parquet(ohlcv_path)

# Filter to Jan 1, 2024
data = data[(data.index >= "2024-01-01") & (data.index < "2024-01-02")].copy()
data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()

# Filter trades: Only TP exits on Jan 1
period_trades = trades_df[
    (trades_df['entry_time'] >= "2024-01-01") &
    (trades_df['entry_time'] < "2024-01-02") &
    (trades_df['exit_reason'] == 'TP')  # ONLY TP exits!
].copy()

print(f"\nTotal trades on 2024-01-01: {len(trades_df[(trades_df['entry_time'] >= '2024-01-01') & (trades_df['entry_time'] < '2024-01-02')])}")
print(f"TP exits only: {len(period_trades)}")

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(24, 10))
fig.suptitle('ONLY TP Exits (12bp Target Hit) - 2024-01-01\nClean EMA7 Bounces',
             fontsize=16, fontweight='bold')

# Plot price and EMA
ax.plot(data.index, data['close'], label='Price', color='black', linewidth=2, zorder=2)
ax.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=2.5, linestyle='--', zorder=2)

# Plot ONLY TP exit trades
for _, trade in period_trades.iterrows():
    entry_time = trade['entry_time']
    entry_price = trade['entry_price']

    # Calculate exit time when TP hit
    bars_held = int(trade['bars_held'])
    entry_idx = data.index.get_indexer([entry_time], method='nearest')[0]
    exit_idx = min(entry_idx + bars_held, len(data) - 1)
    exit_time = data.index[exit_idx]

    exit_price = trade['exit_price']
    direction_str = trade['direction']

    # All are green (TP hit)
    color = 'green'
    marker = '^' if direction_str == 'LONG' else 'v'

    # Entry marker
    ax.scatter(entry_time, entry_price, color=color, s=250, marker=marker,
              edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)

    # Exit marker
    ax.scatter(exit_time, exit_price, color=color, s=350, marker='*',
              edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)

    # Line connecting entry to exit
    ax.plot([entry_time, exit_time], [entry_price, exit_price],
           color=color, linewidth=2.5, alpha=0.7, linestyle='-', zorder=5)

    # Add annotation
    pnl = trade['pnl_bps']
    mid_time = entry_time + (exit_time - entry_time) / 2
    mid_price = (entry_price + exit_price) / 2

    ax.annotate(f"{pnl:.0f}bp", xy=(mid_time, mid_price),
               fontsize=10, fontweight='bold', color='darkgreen',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen',
                        edgecolor='darkgreen', linewidth=2),
               ha='center', zorder=15)

# Formatting
ax.set_xlabel('Time', fontsize=13, fontweight='bold')
ax.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax.set_title('Only showing trades that HIT 12bp target', fontsize=14, pad=15, color='green')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.xticks(rotation=45)

# Legend
legend_elements = [
    plt.Line2D([0], [0], color='black', linewidth=2, label='Price'),
    plt.Line2D([0], [0], color='blue', linewidth=2.5, linestyle='--', label='EMA7'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=12,
              label='LONG Entry (TP)', markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='green', markersize=12,
              label='SHORT Entry (TP)', markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=14,
              label='TP Exit (12bp)', markeredgecolor='black', markeredgewidth=2),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

# Stats box
long_count = (period_trades['direction'] == 'LONG').sum()
short_count = (period_trades['direction'] == 'SHORT').sum()
avg_pnl = period_trades['pnl_bps'].mean()
avg_bars = period_trades['bars_held'].mean()

stats_text = f"TP EXITS ONLY\n\n"
stats_text += f"Total: {len(period_trades)}\n"
stats_text += f"LONG: {long_count}\n"
stats_text += f"SHORT: {short_count}\n\n"
stats_text += f"Avg P&L: {avg_pnl:.1f} bps\n"
stats_text += f"Avg Bars: {avg_bars:.1f}\n\n"
stats_text += f"All hit 12bp target!"

ax.text(0.98, 0.97, stats_text,
       transform=ax.transAxes, fontsize=11,
       verticalalignment='top', horizontalalignment='right',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen',
                edgecolor='darkgreen', linewidth=2, alpha=0.9),
       zorder=20, family='monospace')

plt.tight_layout()

# Save
output_path = Path("experiments/ema/trades_TP_ONLY_2024_01_01.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[OK] Saved: {output_path}")
print(f"\nShowing {len(period_trades)} clean bounces that hit 12bp target.")
