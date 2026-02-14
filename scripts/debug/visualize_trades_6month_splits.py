"""
Visualize Trades on Price Chart - 6-Month Splits

Creates 4 PNG files (one per 6-month period):
1. 2024 H1 (Jan-Jun)
2. 2024 H2 (Jul-Dec)
3. 2025 H1 (Jan-Jun)
4. 2025 H2 (Jul-Dec)

Each chart shows:
- Price + EMA7
- Entry/Exit points
- Green = profit, Red = loss

Run: python scripts/debug/visualize_trades_6month_splits.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("VISUALIZING TRADES - 6-MONTH SPLITS")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
TAKE_PROFIT_BPS = 12
MAX_HOLDING_BARS = 10
FEES_BPS = 8

# Define 6-month periods
PERIODS = [
    {"name": "2024_H1", "start": "2024-01-01", "end": "2024-07-01", "label": "2024 H1 (Jan-Jun)"},
    {"name": "2024_H2", "start": "2024-07-01", "end": "2025-01-01", "label": "2024 H2 (Jul-Dec)"},
    {"name": "2025_H1", "start": "2025-01-01", "end": "2025-07-01", "label": "2025 H1 (Jan-Jun)"},
    {"name": "2025_H2", "start": "2025-07-01", "end": "2026-01-01", "label": "2025 H2 (Jul-Dec)"},
]

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/4] Loading data...")
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

# Filter to 2024-2025
data = ohlcv[(ohlcv.index >= "2024-01-01") & (ohlcv.index < "2026-01-01")].copy()
print(f"Full data: {data.index[0]} to {data.index[-1]}")
print(f"Total candles: {len(data):,}")

# =============================================================================
# CALCULATE INDICATORS AND FIND ALL TRADES
# =============================================================================
print("\n[2/4] Finding all trades...")
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

# Simulate ALL trades
trades = []
entry_indices = data[data['entry_direction'] != 0].index.tolist()

print(f"Processing {len(entry_indices):,} entries...")

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
print(f"Total trades: {len(trades_df):,}")
print(f"Winners: {trades_df['profit'].sum():,} ({trades_df['profit'].mean()*100:.1f}%)")

# =============================================================================
# CREATE VISUALIZATIONS FOR EACH PERIOD
# =============================================================================
print("\n[3/4] Creating visualizations...")

output_dir = Path("experiments/ema/6month_splits")
output_dir.mkdir(parents=True, exist_ok=True)

created_files = []

for period in PERIODS:
    print(f"\n  Creating chart: {period['label']}...")

    # Filter data for this period
    period_data = data[(data.index >= period['start']) & (data.index < period['end'])].copy()

    if len(period_data) == 0:
        print(f"    No data for {period['label']}, skipping...")
        continue

    # Filter trades for this period
    period_trades = trades_df[
        (trades_df['entry_time'] >= period['start']) &
        (trades_df['entry_time'] < period['end'])
    ].copy()

    if len(period_trades) == 0:
        print(f"    No trades for {period['label']}, skipping...")
        continue

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(24, 10))
    fig.suptitle(f'Trade Entries and Exits - {period["label"]}\nEMA7 Approach Direction Strategy',
                 fontsize=16, fontweight='bold')

    # Plot price and EMA
    ax.plot(period_data.index, period_data['close'], label='Price', color='black', linewidth=1.5, zorder=2)
    ax.plot(period_data.index, period_data['ema7'], label='EMA7', color='blue', linewidth=2, linestyle='--', zorder=2)

    # Plot each trade
    for idx, trade in period_trades.iterrows():
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        exit_time = trade['exit_time']
        exit_price = trade['exit_price']
        direction = trade['direction']
        profit = trade['profit']

        # Color based on profit/loss
        color = 'green' if profit else 'red'
        alpha = 0.7 if profit else 0.5

        # Entry marker
        if direction == 1:  # LONG
            marker = '^'
        else:  # SHORT
            marker = 'v'

        ax.scatter(entry_time, entry_price,
                  color=color, s=150, marker=marker,
                  edgecolors='black', linewidths=1.5,
                  zorder=10, alpha=alpha)

        # Exit marker
        ax.scatter(exit_time, exit_price,
                  color=color, s=250, marker='*',
                  edgecolors='black', linewidths=1.5,
                  zorder=10, alpha=alpha)

        # Draw line from entry to exit
        ax.plot([entry_time, exit_time], [entry_price, exit_price],
               color=color, linewidth=1, alpha=0.3, linestyle='-', zorder=5)

    # Formatting
    ax.set_xlabel('Date/Time', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
    ax.set_title(f'{period["start"]} to {period["end"]}', fontsize=14, pad=15)

    # Format x-axis
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

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add statistics box
    winners = period_trades['profit'].sum()
    total = len(period_trades)
    win_rate = (winners / total * 100) if total > 0 else 0
    avg_pnl = period_trades['pnl_bps'].mean()

    long_count = (period_trades['direction'] == 1).sum()
    short_count = (period_trades['direction'] == -1).sum()

    tp_count = (period_trades['exit_reason'] == 'TP').sum()
    time_count = (period_trades['exit_reason'] == 'TIME').sum()

    stats_text = f"PERIOD: {period['label']}\n\n"
    stats_text += f"Total Trades: {total:,}\n"
    stats_text += f"Winners: {winners:,} ({win_rate:.1f}%)\n"
    stats_text += f"Losers: {total - winners:,}\n\n"
    stats_text += f"LONG: {long_count:,}\n"
    stats_text += f"SHORT: {short_count:,}\n\n"
    stats_text += f"TP Exits: {tp_count:,}\n"
    stats_text += f"TIME Exits: {time_count:,}\n\n"
    stats_text += f"Avg P&L: {avg_pnl:.1f} bps"

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
    output_path = output_dir / f"trades_{period['name']}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    created_files.append(output_path)
    print(f"    [OK] Saved: {output_path}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n[4/4] Summary...")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print(f"\nCreated {len(created_files)} PNG files:")
for f in created_files:
    print(f"  - {f}")

print("\nEach chart shows:")
print("  - Black line = Price")
print("  - Blue dashed line = EMA7")
print("  - Triangle (up/down) = Entry point (LONG/SHORT)")
print("  - Star = Exit point")
print("  - Green = Profitable trade")
print("  - Red = Losing trade")
print("  - Lines connect entry to exit")

print("\n" + "=" * 80)
