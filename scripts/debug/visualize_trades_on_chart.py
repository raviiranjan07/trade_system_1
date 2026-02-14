"""
Visualize Trades on Price Chart

Shows actual price + EMA7 with:
- Entry points (triangles)
- Exit points (stars)
- Lines connecting entry to exit
- Green = profitable trade
- Red = losing trade

Run: python scripts/debug/visualize_trades_on_chart.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("VISUALIZING TRADES ON PRICE CHART")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
TAKE_PROFIT_BPS = 12
MAX_HOLDING_BARS = 10
FEES_BPS = 8

# Sample period for visualization (not too crowded)
SAMPLE_START = "2024-01-01"
SAMPLE_END = "2024-01-03"  # 2 days sample

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/3] Loading data...")
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

# Filter sample period
data = ohlcv[(ohlcv.index >= SAMPLE_START) & (ohlcv.index < SAMPLE_END)].copy()
print(f"Sample period: {data.index[0]} to {data.index[-1]}")
print(f"Total candles: {len(data)}")

# =============================================================================
# CALCULATE INDICATORS AND FIND ENTRIES
# =============================================================================
print("\n[2/3] Finding trades...")
data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()

# Touch detection
data['dist_from_ema7'] = abs(data['close'] - data['ema7']) / data['ema7'] * 100
data['touch'] = data['dist_from_ema7'] <= 0.2

# Approach direction (CORRECT LOGIC)
LOOKBACK = 1
data['was_above_ema'] = data['close'].shift(LOOKBACK) > data['ema7'].shift(LOOKBACK)
data['was_below_ema'] = data['close'].shift(LOOKBACK) < data['ema7'].shift(LOOKBACK)

# Determine entry direction
data['entry_direction'] = 0
data.loc[data['touch'] & data['was_above_ema'], 'entry_direction'] = 1   # LONG
data.loc[data['touch'] & data['was_below_ema'], 'entry_direction'] = -1  # SHORT

# Simulate trades
trades = []

entry_indices = data[data['entry_direction'] != 0].index.tolist()

for entry_time in entry_indices:
    entry_idx = data.index.get_loc(entry_time)

    # Check if we have enough future bars
    if entry_idx + MAX_HOLDING_BARS >= len(data):
        continue

    entry_row = data.iloc[entry_idx]
    entry_price = entry_row['close']
    direction = entry_row['entry_direction']

    # Find exit (TP or TIME)
    exit_idx = entry_idx
    exit_reason = 'TIME'

    for i in range(1, MAX_HOLDING_BARS + 1):
        if entry_idx + i >= len(data):
            break

        future_bar = data.iloc[entry_idx + i]

        if direction == 1:  # LONG
            # Check if TP hit
            if future_bar['high'] >= entry_price * (1 + TAKE_PROFIT_BPS / 10000):
                exit_idx = entry_idx + i
                exit_reason = 'TP'
                break
        else:  # SHORT
            # Check if TP hit
            if future_bar['low'] <= entry_price * (1 - TAKE_PROFIT_BPS / 10000):
                exit_idx = entry_idx + i
                exit_reason = 'TP'
                break

    # If no TP hit, exit at max holding
    if exit_reason == 'TIME':
        exit_idx = entry_idx + MAX_HOLDING_BARS

    exit_time = data.index[exit_idx]
    exit_price = data.iloc[exit_idx]['close']

    # Calculate P&L
    if direction == 1:  # LONG
        pnl_bps = (exit_price - entry_price) / entry_price * 10000
    else:  # SHORT
        pnl_bps = (entry_price - exit_price) / entry_price * 10000

    pnl_bps -= FEES_BPS

    trades.append({
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'direction': direction,
        'exit_reason': exit_reason,
        'pnl_bps': pnl_bps,
        'profit': pnl_bps > 0
    })

trades_df = pd.DataFrame(trades)
print(f"Total trades: {len(trades_df)}")
print(f"Winners: {trades_df['profit'].sum()} ({trades_df['profit'].mean()*100:.1f}%)")
print(f"Losers: {(~trades_df['profit']).sum()} ({(~trades_df['profit']).mean()*100:.1f}%)")

# =============================================================================
# CREATE VISUALIZATION
# =============================================================================
print("\n[3/3] Creating visualization...")

fig, ax = plt.subplots(1, 1, figsize=(24, 10))
fig.suptitle('Trade Entries and Exits on Price Chart\nEMA7 Approach Direction Strategy',
             fontsize=16, fontweight='bold')

# Plot price and EMA
ax.plot(data.index, data['close'], label='Price', color='black', linewidth=2, zorder=2)
ax.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=2.5, linestyle='--', zorder=2)

# Plot each trade
for idx, trade in trades_df.iterrows():
    entry_time = trade['entry_time']
    entry_price = trade['entry_price']
    exit_time = trade['exit_time']
    exit_price = trade['exit_price']
    direction = trade['direction']
    profit = trade['profit']

    # Color based on profit/loss
    color = 'green' if profit else 'red'
    alpha = 0.8 if profit else 0.6

    # Entry marker
    if direction == 1:  # LONG
        marker = '^'
        entry_label = 'LONG Entry' if idx == 0 else None
    else:  # SHORT
        marker = 'v'
        entry_label = 'SHORT Entry' if idx == 0 else None

    ax.scatter(entry_time, entry_price,
              color=color, s=200, marker=marker,
              edgecolors='black', linewidths=2,
              label=entry_label, zorder=10, alpha=alpha)

    # Exit marker
    exit_label = 'Exit (Profit)' if profit and idx == 0 else ('Exit (Loss)' if not profit and idx == 0 else None)
    ax.scatter(exit_time, exit_price,
              color=color, s=300, marker='*',
              edgecolors='black', linewidths=2,
              label=exit_label, zorder=10, alpha=alpha)

    # Draw line from entry to exit
    ax.plot([entry_time, exit_time], [entry_price, exit_price],
           color=color, linewidth=2, alpha=0.5, linestyle='-', zorder=5)

    # Add P&L text annotation
    mid_time = entry_time + (exit_time - entry_time) / 2
    mid_price = (entry_price + exit_price) / 2

    pnl_text = f"{trade['pnl_bps']:.0f}bp"
    ax.annotate(pnl_text,
               xy=(mid_time, mid_price),
               fontsize=8,
               color=color,
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.7),
               ha='center',
               zorder=15)

# Formatting
ax.set_xlabel('Date/Time', fontsize=13, fontweight='bold')
ax.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax.set_title(f'{SAMPLE_START} to {SAMPLE_END} - Entry/Exit Points', fontsize=14, pad=15)

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
plt.xticks(rotation=45)

# Legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

# Grid
ax.grid(True, alpha=0.3, linestyle='--')

# Add statistics box
stats_text = f"Period: {SAMPLE_START} to {SAMPLE_END}\n"
stats_text += f"Total Trades: {len(trades_df)}\n"
stats_text += f"Winners: {trades_df['profit'].sum()} ({trades_df['profit'].mean()*100:.1f}%)\n"
stats_text += f"Losers: {(~trades_df['profit']).sum()} ({(~trades_df['profit']).mean()*100:.1f}%)\n"
stats_text += f"Avg P&L: {trades_df['pnl_bps'].mean():.1f} bps\n"
stats_text += f"TP Exits: {(trades_df['exit_reason'] == 'TP').sum()}\n"
stats_text += f"TIME Exits: {(trades_df['exit_reason'] == 'TIME').sum()}"

ax.text(0.98, 0.97, stats_text,
       transform=ax.transAxes,
       fontsize=10,
       verticalalignment='top',
       horizontalalignment='right',
       bbox=dict(boxstyle='round,pad=0.8',
                facecolor='lightyellow',
                edgecolor='black',
                linewidth=1.5,
                alpha=0.9),
       zorder=20,
       family='monospace')

plt.tight_layout()

# Save
output_dir = Path("experiments/ema")
output_path = output_dir / "trades_on_chart.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] Saved visualization: {output_path}")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print("\nGreen = Profitable trade")
print("Red = Losing trade")
print("Triangle = Entry, Star = Exit")
print("Lines connect entry to exit")
