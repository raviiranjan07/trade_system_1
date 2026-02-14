"""
Visualize Hold Longer Strategy

Show the difference between:
- 92.6% that hit TP (dirty wins that recovered!)
- 7.4% that hit TIME (massive losers)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

print("Creating hold longer visualization...")

# Load trades
trades_df = pd.read_csv("experiments/ema/hold_longer_trades.csv")
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

# Load 15-min OHLCV
ohlcv_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
data = pd.read_parquet(ohlcv_path)

# Filter to Jan 1, 2024
data = data[(data.index >= "2024-01-01") & (data.index < "2024-01-02")].copy()
data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()

# Filter trades for Jan 1, 2024
period_trades = trades_df[
    (trades_df['entry_time'] >= "2024-01-01") &
    (trades_df['entry_time'] < "2024-01-02")
].copy()

print(f"\nTrades on 2024-01-01:")
print(f"  Total: {len(period_trades)}")
print(f"  TP exits: {(period_trades['exit_reason'] == 'TP').sum()}")
print(f"  TIME exits: {(period_trades['exit_reason'] == 'TIME').sum()}")

# Split by exit reason
tp_trades = period_trades[period_trades['exit_reason'] == 'TP']
time_trades = period_trades[period_trades['exit_reason'] == 'TIME']

# Create visualization - 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 16))
fig.suptitle('HOLD LONGER Strategy (60 bars = 15 hours) - 2024-01-01\\nNo MAE exit - Let dirty wins recover!',
             fontsize=16, fontweight='bold')

# =============================================================================
# TOP PLOT: TP EXITS (Winners - 92.6%)
# =============================================================================
ax1.plot(data.index, data['close'], label='Price', color='black', linewidth=2, zorder=2)
ax1.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=2.5, linestyle='--', zorder=2)

for _, trade in tp_trades.iterrows():
    entry_time = trade['entry_time']
    entry_price = trade['entry_price']

    bars_held = int(trade['bars_held'])
    entry_idx = data.index.get_indexer([entry_time], method='nearest')[0]
    exit_idx = min(entry_idx + bars_held, len(data) - 1)
    exit_time = data.index[exit_idx]

    exit_price = trade['exit_price']
    direction_str = trade['direction']
    pnl = trade['pnl_bps']
    mae = trade['mae_bps']

    color = 'green'
    marker = '^' if direction_str == 'LONG' else 'v'

    # Entry marker
    ax1.scatter(entry_time, entry_price, color=color, s=200, marker=marker,
               edgecolors='black', linewidths=2, zorder=10, alpha=0.8)

    # Exit marker
    ax1.scatter(exit_time, exit_price, color=color, s=300, marker='*',
               edgecolors='black', linewidths=2, zorder=10, alpha=0.8)

    # Line
    ax1.plot([entry_time, exit_time], [entry_price, exit_price],
            color=color, linewidth=2, alpha=0.6, linestyle='-', zorder=5)

    # Show MAE (max drawdown) for winners
    if bars_held > 10:  # Only annotate if held longer
        mid_time = entry_time + (exit_time - entry_time) / 2
        mid_price = (entry_price + exit_price) / 2

        ax1.annotate(f"Bars:{bars_held}\\nMAE:{mae:.0f}", xy=(mid_time, mid_price),
                    fontsize=7, color='darkgreen',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen',
                             edgecolor='darkgreen', linewidth=1),
                    ha='center', zorder=15, alpha=0.7)

ax1.set_title('WINNERS (92.6%): Eventually hit 12bp target - "Dirty wins" recovered!',
             fontsize=14, pad=15, color='green', fontweight='bold')
ax1.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax1.grid(True, alpha=0.3, linestyle='--')

stats_text_tp = f"TP EXITS (Winners)\\n\\n"
stats_text_tp += f"Total: {len(tp_trades)}\\n"
stats_text_tp += f"Avg P&L: +{tp_trades['pnl_bps'].mean():.1f} bps\\n"
stats_text_tp += f"Avg Bars: {tp_trades['bars_held'].mean():.1f}\\n"
stats_text_tp += f"Avg MAE: {tp_trades['mae_bps'].mean():.1f} bps\\n\\n"
stats_text_tp += f"These survived drawdown!"

ax1.text(0.98, 0.97, stats_text_tp,
        transform=ax1.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen',
                 edgecolor='darkgreen', linewidth=2, alpha=0.95),
        zorder=20, family='monospace')

# =============================================================================
# BOTTOM PLOT: TIME EXITS (Losers - 7.4%)
# =============================================================================
ax2.plot(data.index, data['close'], label='Price', color='black', linewidth=2, zorder=2)
ax2.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=2.5, linestyle='--', zorder=2)

for _, trade in time_trades.iterrows():
    entry_time = trade['entry_time']
    entry_price = trade['entry_price']

    bars_held = int(trade['bars_held'])
    entry_idx = data.index.get_indexer([entry_time], method='nearest')[0]
    exit_idx = min(entry_idx + bars_held, len(data) - 1)
    exit_time = data.index[exit_idx]

    exit_price = trade['exit_price']
    direction_str = trade['direction']
    pnl = trade['pnl_bps']
    mae = trade['mae_bps']

    color = 'red'
    marker = '^' if direction_str == 'LONG' else 'v'

    # Entry marker
    ax2.scatter(entry_time, entry_price, color=color, s=300, marker=marker,
               edgecolors='black', linewidths=3, zorder=10, alpha=0.9)

    # Exit marker
    ax2.scatter(exit_time, exit_price, color=color, s=400, marker='X',
               edgecolors='black', linewidths=3, zorder=10, alpha=0.9)

    # Line
    ax2.plot([entry_time, exit_time], [entry_price, exit_price],
            color=color, linewidth=3, alpha=0.7, linestyle='-', zorder=5)

    # Annotation
    mid_time = entry_time + (exit_time - entry_time) / 2
    mid_price = (entry_price + exit_price) / 2

    ax2.annotate(f"{pnl:.0f}bp\\nMAE:{mae:.0f}\\n60 bars", xy=(mid_time, mid_price),
                fontsize=10, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral',
                         edgecolor='darkred', linewidth=2),
                ha='center', zorder=15)

ax2.set_title('LOSERS (7.4%): Held 60 bars, never recovered - Wrong direction!',
             fontsize=14, pad=15, color='red', fontweight='bold')
ax2.set_xlabel('Time', fontsize=13, fontweight='bold')
ax2.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
ax2.grid(True, alpha=0.3, linestyle='--')

stats_text_time = f"TIME EXITS (Losers)\\n\\n"
if len(time_trades) > 0:
    stats_text_time += f"Total: {len(time_trades)}\\n"
    stats_text_time += f"Avg P&L: {time_trades['pnl_bps'].mean():.1f} bps\\n"
    stats_text_time += f"Avg MAE: {time_trades['mae_bps'].mean():.1f} bps\\n\\n"
    stats_text_time += f"These never recovered\\nWrong direction!"
else:
    stats_text_time += "No TIME exits\\non this day!"

ax2.text(0.98, 0.97, stats_text_time,
        transform=ax2.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcoral',
                 edgecolor='darkred', linewidth=2, alpha=0.95),
        zorder=20, family='monospace')

# Legend
legend_elements = [
    plt.Line2D([0], [0], color='black', linewidth=2, label='Price'),
    plt.Line2D([0], [0], color='blue', linewidth=2.5, linestyle='--', label='EMA7'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=12,
              label='LONG Entry', markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=12,
              label='SHORT Entry', markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=14,
              label='TP Exit (recovered!)', markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=14,
              label='TIME Exit (60 bars)', markeredgecolor='black', markeredgewidth=2),
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)

plt.tight_layout()

# Save
output_path = Path("experiments/ema/hold_longer_visualization_2024_01_01.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[OK] Saved: {output_path}")
print("\nTop chart = Winners (hit TP after holding through drawdown)")
print("Bottom chart = Losers (held 60 bars, never recovered)")
