"""
Test visualization with 1 month only
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

print("Creating 1-month test visualization...")

# Load trades
trades_df = pd.read_csv("experiments/ema/ema7_backtest_trades.csv")
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
print(f"Loaded {len(trades_df):,} trades")

# Load 1-min OHLCV (this is the slow part)
print("Loading OHLCV data (this may take a minute)...")
ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)
print("Resampling to 15-min...")

# Resample to 15-min
ohlcv = ohlcv_1m.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Filter to Jan 2024 only (1 month test)
data = ohlcv[(ohlcv.index >= "2024-01-01") & (ohlcv.index < "2024-02-01")].copy()
data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()

print(f"Price data: {len(data)} candles from {data.index[0]} to {data.index[-1]}")

# Filter trades for Jan 2024
period_trades = trades_df[
    (trades_df['entry_time'] >= "2024-01-01") &
    (trades_df['entry_time'] < "2024-02-01")
].copy()

print(f"Trades in period: {len(period_trades)}")

# Create figure
print("Creating visualization...")
fig, ax = plt.subplots(1, 1, figsize=(24, 10))
fig.suptitle('Test: Jan 2024 Trades',
             fontsize=16, fontweight='bold')

# Plot price and EMA
ax.plot(data.index, data['close'], label='Price', color='black', linewidth=1.5, zorder=2)
ax.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=2, linestyle='--', zorder=2)

# Plot trades
for idx, trade in period_trades.iterrows():
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

    ax.scatter(entry_time, entry_price, color=color, s=150, marker=marker,
              edgecolors='black', linewidths=1.5, zorder=10, alpha=0.7)
    ax.scatter(exit_time, exit_price, color=color, s=250, marker='*',
              edgecolors='black', linewidths=1.5, zorder=10, alpha=0.7)
    ax.plot([entry_time, exit_time], [entry_price, exit_price],
           color=color, linewidth=1, alpha=0.3, linestyle='-', zorder=5)

ax.set_xlabel('Date/Time', fontsize=13)
ax.set_ylabel('Price (USD)', fontsize=13)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.xticks(rotation=45)

plt.tight_layout()

# Save
output_dir = Path("experiments/ema")
output_path = output_dir / "test_jan2024_trades.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[OK] Saved: {output_path}")
print("Test complete!")
