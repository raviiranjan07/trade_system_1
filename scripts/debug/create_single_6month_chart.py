"""
Create single PNG for first 6 months (2024 H1: Jan-Jun)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

print("Creating 2024 H1 (Jan-Jun) visualization...")

# Load trades
print("\n[1/3] Loading trades...")
trades_df = pd.read_csv("experiments/ema/ema7_backtest_trades.csv")
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
print(f"Loaded {len(trades_df):,} trades")

# Load 15-min OHLCV (FAST!)
print("\n[2/3] Loading 15-min OHLCV...")
ohlcv_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
data = pd.read_parquet(ohlcv_path)

# Filter to Jan-Jun 2024
data = data[(data.index >= "2024-01-01") & (data.index < "2024-07-01")].copy()
data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()

print(f"Price data: {len(data):,} candles from {data.index[0]} to {data.index[-1]}")

# Filter trades for Jan-Jun 2024
period_trades = trades_df[
    (trades_df['entry_time'] >= "2024-01-01") &
    (trades_df['entry_time'] < "2024-07-01")
].copy()

print(f"Trades in period: {len(period_trades):,}")

# Create figure
print("\n[3/3] Creating visualization...")
fig, ax = plt.subplots(1, 1, figsize=(24, 10))
fig.suptitle('Trade Entries and Exits - 2024 H1 (Jan-Jun)\nEMA7 Approach Direction Strategy',
             fontsize=16, fontweight='bold')

# Plot price and EMA
ax.plot(data.index, data['close'], label='Price', color='black', linewidth=1.5, zorder=2)
ax.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=2, linestyle='--', zorder=2)

# Plot trades
print(f"Plotting {len(period_trades)} trades...")
for _, trade in period_trades.iterrows():
    entry_time = trade['entry_time']
    entry_price = trade['entry_price']

    # Calculate exit time
    bars_held = int(trade['bars_held'])
    entry_idx = data.index.get_indexer([entry_time], method='nearest')[0]
    exit_idx = min(entry_idx + bars_held, len(data) - 1)
    exit_time = data.index[exit_idx]

    exit_price = trade['exit_price']
    direction_str = trade['direction']
    profit = trade['winner']

    color = 'green' if profit else 'red'
    marker = '^' if direction_str == 'LONG' else 'v'
    alpha = 0.7 if profit else 0.5

    ax.scatter(entry_time, entry_price, color=color, s=150, marker=marker,
              edgecolors='black', linewidths=1.5, zorder=10, alpha=alpha)
    ax.scatter(exit_time, exit_price, color=color, s=250, marker='*',
              edgecolors='black', linewidths=1.5, zorder=10, alpha=alpha)
    ax.plot([entry_time, exit_time], [entry_price, exit_price],
           color=color, linewidth=1, alpha=0.3, linestyle='-', zorder=5)

# Formatting
ax.set_xlabel('Date/Time', fontsize=13, fontweight='bold')
ax.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax.set_title('2024-01-01 to 2024-07-01', fontsize=14, pad=15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

# Legend
legend_elements = [
    plt.Line2D([0], [0], color='black', linewidth=1.5, label='Price'),
    plt.Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='EMA7'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='LONG Entry', markeredgecolor='black'),
    plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, label='SHORT Entry', markeredgecolor='black'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=12, label='Exit (Profit)', markeredgecolor='black'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=12, label='Exit (Loss)', markeredgecolor='black'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

# Stats box
winners = period_trades['winner'].sum()
total = len(period_trades)
win_rate = (winners / total * 100) if total > 0 else 0
avg_pnl = period_trades['pnl_bps'].mean()
long_count = (period_trades['direction'] == 'LONG').sum()
short_count = (period_trades['direction'] == 'SHORT').sum()
tp_count = (period_trades['exit_reason'] == 'TP').sum()
time_count = (period_trades['exit_reason'] == 'TIME').sum()

stats_text = f"PERIOD: 2024 H1 (Jan-Jun)\n\n"
stats_text += f"Total Trades: {total:,}\n"
stats_text += f"Winners: {winners:,} ({win_rate:.1f}%)\n"
stats_text += f"Losers: {total - winners:,}\n\n"
stats_text += f"LONG: {long_count:,}\n"
stats_text += f"SHORT: {short_count:,}\n\n"
stats_text += f"TP Exits: {tp_count:,}\n"
stats_text += f"TIME Exits: {time_count:,}\n\n"
stats_text += f"Avg P&L: {avg_pnl:.1f} bps"

ax.text(0.98, 0.97, stats_text,
       transform=ax.transAxes, fontsize=10,
       verticalalignment='top', horizontalalignment='right',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                edgecolor='black', linewidth=1.5, alpha=0.9),
       zorder=20, family='monospace')

plt.tight_layout()

# Save
output_dir = Path("experiments/ema")
output_path = output_dir / "trades_2024_H1_Jan_Jun.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[OK] Saved: {output_path}")
print("\nDone! Check the PNG file to see 6 months of trades.")
