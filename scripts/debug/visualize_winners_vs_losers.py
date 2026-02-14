"""
Visualize EMA7 Breakout: Winners vs Losers

Show what happens with winning trades (TP) vs losing trades (TIME exits)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

print("Creating visualization showing winners vs losers...")

# Load trades from TEST data
trades_df = pd.read_csv("experiments/ema/ema_breakout_trades.csv")
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
print(f"  Winners (TP): {(period_trades['exit_reason'] == 'TP').sum()}")
print(f"  Losers (TIME): {(period_trades['exit_reason'] == 'TIME').sum()}")

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 16))
fig.suptitle('EMA7 BREAKOUT: Winners (TP) vs Losers (TIME Exits) - 2024-01-01',
             fontsize=16, fontweight='bold')

# =============================================================================
# TOP PLOT: WINNERS ONLY (TP exits)
# =============================================================================
winners = period_trades[period_trades['exit_reason'] == 'TP']

ax1.plot(data.index, data['close'], label='Price', color='black', linewidth=2, zorder=2)
ax1.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=2.5, linestyle='--', zorder=2)

for _, trade in winners.iterrows():
    entry_time = trade['entry_time']
    entry_price = trade['entry_price']

    bars_held = int(trade['bars_held'])
    entry_idx = data.index.get_indexer([entry_time], method='nearest')[0]
    exit_idx = min(entry_idx + bars_held, len(data) - 1)
    exit_time = data.index[exit_idx]

    exit_price = trade['exit_price']
    direction_str = trade['direction']
    pnl = trade['pnl_bps']

    color = 'green'
    marker = '^' if direction_str == 'LONG' else 'v'

    # Entry marker
    ax1.scatter(entry_time, entry_price, color=color, s=250, marker=marker,
               edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)

    # Exit marker
    ax1.scatter(exit_time, exit_price, color=color, s=350, marker='*',
               edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)

    # Line
    ax1.plot([entry_time, exit_time], [entry_price, exit_price],
            color=color, linewidth=2.5, alpha=0.7, linestyle='-', zorder=5)

    # Annotation
    mid_time = entry_time + (exit_time - entry_time) / 2
    mid_price = (entry_price + exit_price) / 2

    ax1.annotate(f"+{pnl:.0f}bp", xy=(mid_time, mid_price),
                fontsize=9, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen',
                         edgecolor='darkgreen', linewidth=2),
                ha='center', zorder=15)

ax1.set_title('WINNERS ONLY: Hit 12bp Target (TP Exits)', fontsize=14, pad=15, color='green', fontweight='bold')
ax1.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax1.grid(True, alpha=0.3, linestyle='--')

# Stats box for winners
stats_text_winners = f"WINNERS\\n\\n"
stats_text_winners += f"Total: {len(winners)}\\n"
stats_text_winners += f"Avg P&L: +{winners['pnl_bps'].mean():.1f} bps\\n"
stats_text_winners += f"Avg Bars: {winners['bars_held'].mean():.1f}\\n\\n"
stats_text_winners += f"Exit: TP (12bp hit)"

ax1.text(0.98, 0.97, stats_text_winners,
        transform=ax1.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen',
                 edgecolor='darkgreen', linewidth=2, alpha=0.9),
        zorder=20, family='monospace')

# =============================================================================
# BOTTOM PLOT: LOSERS ONLY (TIME exits)
# =============================================================================
losers = period_trades[period_trades['exit_reason'] == 'TIME']

ax2.plot(data.index, data['close'], label='Price', color='black', linewidth=2, zorder=2)
ax2.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=2.5, linestyle='--', zorder=2)

for _, trade in losers.iterrows():
    entry_time = trade['entry_time']
    entry_price = trade['entry_price']

    bars_held = int(trade['bars_held'])
    entry_idx = data.index.get_indexer([entry_time], method='nearest')[0]
    exit_idx = min(entry_idx + bars_held, len(data) - 1)
    exit_time = data.index[exit_idx]

    exit_price = trade['exit_price']
    direction_str = trade['direction']
    pnl = trade['pnl_bps']

    color = 'red'
    marker = '^' if direction_str == 'LONG' else 'v'

    # Entry marker
    ax2.scatter(entry_time, entry_price, color=color, s=250, marker=marker,
               edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)

    # Exit marker
    ax2.scatter(exit_time, exit_price, color=color, s=350, marker='X',
               edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)

    # Line
    ax2.plot([entry_time, exit_time], [entry_price, exit_price],
            color=color, linewidth=2.5, alpha=0.7, linestyle='-', zorder=5)

    # Annotation
    mid_time = entry_time + (exit_time - entry_time) / 2
    mid_price = (entry_price + exit_price) / 2

    ax2.annotate(f"{pnl:.0f}bp", xy=(mid_time, mid_price),
                fontsize=9, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral',
                         edgecolor='darkred', linewidth=2),
                ha='center', zorder=15)

ax2.set_title('LOSERS ONLY: Max Time Reached (TIME Exits)', fontsize=14, pad=15, color='red', fontweight='bold')
ax2.set_xlabel('Time', fontsize=13, fontweight='bold')
ax2.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
ax2.grid(True, alpha=0.3, linestyle='--')

# Stats box for losers
stats_text_losers = f"LOSERS\\n\\n"
stats_text_losers += f"Total: {len(losers)}\\n"
stats_text_losers += f"Avg P&L: {losers['pnl_bps'].mean():.1f} bps\\n"
stats_text_losers += f"Avg Bars: {losers['bars_held'].mean():.1f}\\n\\n"
stats_text_losers += f"Exit: TIME (10 bars)"

ax2.text(0.98, 0.97, stats_text_losers,
        transform=ax2.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcoral',
                 edgecolor='darkred', linewidth=2, alpha=0.9),
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
              label='TP Exit (+4bp)', markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=14,
              label='TIME Exit (loss)', markeredgecolor='black', markeredgewidth=2),
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)

plt.tight_layout()

# Save
output_path = Path("experiments/ema/winners_vs_losers_2024_01_01.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[OK] Saved: {output_path}")
print("\nTop chart = Winners (hit 12bp target)")
print("Bottom chart = Losers (held 10 bars, didn't hit target)")
