"""
Visualize Two-Stage Exit Strategy

Show how trades behave with two-stage exit logic:
- Stage 1: Wait through drawdown
- Stage 2: Exit if MAE > threshold
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

print("Creating two-stage exit visualization...")

# Load trades
trades_df = pd.read_csv("experiments/ema/two_stage_exit_trades.csv")
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
print(f"  MAE exits: {(period_trades['exit_reason'] == 'MAE').sum()}")
print(f"  TIME exits: {(period_trades['exit_reason'] == 'TIME').sum()}")

# Split by exit reason
tp_trades = period_trades[period_trades['exit_reason'] == 'TP']
mae_trades = period_trades[period_trades['exit_reason'] == 'MAE']
time_trades = period_trades[period_trades['exit_reason'] == 'TIME']

# =============================================================================
# CREATE VISUALIZATION - 3 SUBPLOTS
# =============================================================================

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 18))
fig.suptitle('Two-Stage Exit Strategy - 2024-01-01\\nStage 1: Wait 20 bars | Stage 2: Exit if MAE > 50bp',
             fontsize=16, fontweight='bold')

# =============================================================================
# TOP PLOT: TP EXITS (Winners)
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
    ax1.scatter(entry_time, entry_price, color=color, s=250, marker=marker,
               edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)

    # Exit marker
    ax1.scatter(exit_time, exit_price, color=color, s=350, marker='*',
               edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)

    # Line
    ax1.plot([entry_time, exit_time], [entry_price, exit_price],
            color=color, linewidth=2.5, alpha=0.7, linestyle='-', zorder=5)

    # Annotation with MAE
    mid_time = entry_time + (exit_time - entry_time) / 2
    mid_price = (entry_price + exit_price) / 2

    ax1.annotate(f"+{pnl:.0f}bp\\nMAE:{mae:.0f}", xy=(mid_time, mid_price),
                fontsize=8, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen',
                         edgecolor='darkgreen', linewidth=2),
                ha='center', zorder=15)

ax1.set_title('TP EXITS: Hit 12bp target (before Stage 2)', fontsize=14, pad=15,
             color='green', fontweight='bold')
ax1.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax1.grid(True, alpha=0.3, linestyle='--')

stats_text_tp = f"TP EXITS\\n\\n"
stats_text_tp += f"Total: {len(tp_trades)}\\n"
stats_text_tp += f"Avg P&L: +{tp_trades['pnl_bps'].mean():.1f} bps\\n"
stats_text_tp += f"Avg Bars: {tp_trades['bars_held'].mean():.1f}\\n"
stats_text_tp += f"Avg MAE: {tp_trades['mae_bps'].mean():.1f} bps"

ax1.text(0.98, 0.97, stats_text_tp,
        transform=ax1.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen',
                 edgecolor='darkgreen', linewidth=2, alpha=0.95),
        zorder=20, family='monospace')

# =============================================================================
# MIDDLE PLOT: MAE EXITS (Stage 2 triggered)
# =============================================================================
ax2.plot(data.index, data['close'], label='Price', color='black', linewidth=2, zorder=2)
ax2.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=2.5, linestyle='--', zorder=2)

for _, trade in mae_trades.iterrows():
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
    ax2.scatter(entry_time, entry_price, color=color, s=250, marker=marker,
               edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)

    # Exit marker
    ax2.scatter(exit_time, exit_price, color=color, s=350, marker='X',
               edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)

    # Line
    ax2.plot([entry_time, exit_time], [entry_price, exit_price],
            color=color, linewidth=2.5, alpha=0.7, linestyle='-', zorder=5)

    # Annotation with MAE
    mid_time = entry_time + (exit_time - entry_time) / 2
    mid_price = (entry_price + exit_price) / 2

    ax2.annotate(f"{pnl:.0f}bp\\nMAE:{mae:.0f}", xy=(mid_time, mid_price),
                fontsize=8, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral',
                         edgecolor='darkred', linewidth=2),
                ha='center', zorder=15)

ax2.set_title('MAE EXITS: MAE > 50bp after 20 bars (Stage 2 triggered)', fontsize=14, pad=15,
             color='red', fontweight='bold')
ax2.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax2.grid(True, alpha=0.3, linestyle='--')

stats_text_mae = f"MAE EXITS\\n\\n"
stats_text_mae += f"Total: {len(mae_trades)}\\n"
stats_text_mae += f"Avg P&L: {mae_trades['pnl_bps'].mean():.1f} bps\\n"
stats_text_mae += f"Avg Bars: {mae_trades['bars_held'].mean():.1f}\\n"
stats_text_mae += f"Avg MAE: {mae_trades['mae_bps'].mean():.1f} bps"

ax2.text(0.98, 0.97, stats_text_mae,
        transform=ax2.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcoral',
                 edgecolor='darkred', linewidth=2, alpha=0.95),
        zorder=20, family='monospace')

# =============================================================================
# BOTTOM PLOT: TIME EXITS (Max holding reached)
# =============================================================================
ax3.plot(data.index, data['close'], label='Price', color='black', linewidth=2, zorder=2)
ax3.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=2.5, linestyle='--', zorder=2)

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

    color = 'purple'
    marker = '^' if direction_str == 'LONG' else 'v'

    # Entry marker
    ax3.scatter(entry_time, entry_price, color=color, s=250, marker=marker,
               edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)

    # Exit marker
    ax3.scatter(exit_time, exit_price, color=color, s=350, marker='s',
               edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)

    # Line
    ax3.plot([entry_time, exit_time], [entry_price, exit_price],
            color=color, linewidth=2.5, alpha=0.7, linestyle='-', zorder=5)

    # Annotation with MAE
    mid_time = entry_time + (exit_time - entry_time) / 2
    mid_price = (entry_price + exit_price) / 2

    ax3.annotate(f"{pnl:.0f}bp\\nMAE:{mae:.0f}", xy=(mid_time, mid_price),
                fontsize=8, fontweight='bold', color='purple',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='plum',
                         edgecolor='purple', linewidth=2),
                ha='center', zorder=15)

ax3.set_title('TIME EXITS: Max 40 bars reached (MAE < 50bp)', fontsize=14, pad=15,
             color='purple', fontweight='bold')
ax3.set_xlabel('Time', fontsize=13, fontweight='bold')
ax3.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
ax3.grid(True, alpha=0.3, linestyle='--')

stats_text_time = f"TIME EXITS\\n\\n"
if len(time_trades) > 0:
    stats_text_time += f"Total: {len(time_trades)}\\n"
    stats_text_time += f"Avg P&L: {time_trades['pnl_bps'].mean():.1f} bps\\n"
    stats_text_time += f"Avg Bars: {time_trades['bars_held'].mean():.1f}\\n"
    stats_text_time += f"Avg MAE: {time_trades['mae_bps'].mean():.1f} bps"
else:
    stats_text_time += "No TIME exits\\non this day"

ax3.text(0.98, 0.97, stats_text_time,
        transform=ax3.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='plum',
                 edgecolor='purple', linewidth=2, alpha=0.95),
        zorder=20, family='monospace')

# Legend
legend_elements = [
    plt.Line2D([0], [0], color='black', linewidth=2, label='Price'),
    plt.Line2D([0], [0], color='blue', linewidth=2.5, linestyle='--', label='EMA7'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=12,
              label='LONG Entry', markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=12,
              label='SHORT Entry', markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=14,
              label='TP Exit (+4bp)', markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=14,
              label='MAE Exit (>50bp)', markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=12,
              label='TIME Exit (40 bars)', markeredgecolor='black', markeredgewidth=2),
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)

plt.tight_layout()

# Save
output_path = Path("experiments/ema/two_stage_exit_visualization_2024_01_01.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[OK] Saved: {output_path}")
print("\nTop chart = TP exits (winners)")
print("Middle chart = MAE exits (Stage 2 triggered)")
print("Bottom chart = TIME exits (max 40 bars, MAE still < 50bp)")
