"""
Visualize Stop-and-Reverse Strategy

Show how positions reverse when SL is hit
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numba import njit
import warnings
warnings.filterwarnings('ignore')

print("Creating stop-and-reverse visualization...")

# =============================================================================
# CONFIGURATION
# =============================================================================
TP_BPS = 12
SL_BPS = 12
MAX_HOLDING_BARS = 10

# =============================================================================
# LOAD DATA
# =============================================================================
# Load 15-min OHLCV
ohlcv_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
data = pd.read_parquet(ohlcv_path)

# Filter to Jan 1, 2024
data = data[(data.index >= "2024-01-01") & (data.index < "2024-01-02")].copy()
data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()

# Detect crosses
data['above_ema'] = data['close'] > data['ema7']
data['below_ema'] = data['close'] < data['ema7']

data['cross_above'] = (data['above_ema'] == True) & (data['above_ema'].shift(1) == False)
data['cross_below'] = (data['below_ema'] == True) & (data['below_ema'].shift(1) == False)

print(f"Price data: {len(data)} candles")

# =============================================================================
# SIMULATE TRADES WITH DETAILED TRACKING
# =============================================================================

@njit
def simulate_with_tracking(entry_price, initial_direction, future_highs, future_lows,
                           tp_bps, sl_bps, max_bars):
    """
    Track all reversals for visualization
    Returns list of (bar_idx, event_type, price, direction)
    event_type: 0=entry, 1=TP, 2=SL_reverse, 3=TIME
    """
    events = []
    current_price = entry_price
    direction = initial_direction

    # Record initial entry
    events.append((0, 0, entry_price, direction))  # (bar=0, event=entry, price, dir)

    for i in range(min(len(future_highs), max_bars)):
        high = future_highs[i]
        low = future_lows[i]

        if direction == 1:  # LONG
            tp_price = current_price * (1 + tp_bps / 10000)
            sl_price = current_price * (1 - sl_bps / 10000)

            if high >= tp_price:
                events.append((i + 1, 1, tp_price, direction))  # TP
                return events

            if low <= sl_price:
                events.append((i + 1, 2, sl_price, direction))  # SL - reverse
                current_price = sl_price
                direction = -1
                continue

        else:  # SHORT
            tp_price = current_price * (1 - tp_bps / 10000)
            sl_price = current_price * (1 + sl_bps / 10000)

            if low <= tp_price:
                events.append((i + 1, 1, tp_price, direction))  # TP
                return events

            if high >= sl_price:
                events.append((i + 1, 2, sl_price, direction))  # SL - reverse
                current_price = sl_price
                direction = 1
                continue

    # TIME exit
    if i < len(future_lows):
        final_close = (future_highs[i] + future_lows[i]) / 2
    else:
        final_close = current_price
    events.append((max_bars, 3, final_close, direction))  # TIME
    return events

# Find all trades on Jan 1
trades_with_events = []

for idx in range(len(data)):
    if idx + MAX_HOLDING_BARS >= len(data):
        continue

    row = data.iloc[idx]
    entry_time = row.name
    entry_price = row['close']

    if row['cross_above']:
        direction = 1  # LONG
    elif row['cross_below']:
        direction = -1  # SHORT
    else:
        continue

    # Get future data
    future_data = data.iloc[idx + 1 : idx + 1 + MAX_HOLDING_BARS]
    future_highs = future_data['high'].values
    future_lows = future_data['low'].values

    # Simulate and track events
    events = simulate_with_tracking(
        entry_price,
        direction,
        future_highs,
        future_lows,
        TP_BPS,
        SL_BPS,
        MAX_HOLDING_BARS
    )

    trades_with_events.append({
        'entry_time': entry_time,
        'entry_idx': idx,
        'events': events
    })

print(f"\nFound {len(trades_with_events)} trades on 2024-01-01")

# Count trades with reversals
trades_with_reversals = [t for t in trades_with_events if any(e[1] == 2 for e in t['events'])]
print(f"Trades with reversals: {len(trades_with_reversals)}")

# =============================================================================
# CREATE VISUALIZATION
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(24, 12))
fig.suptitle('Stop-and-Reverse Strategy - 2024-01-01\\nTP: 12bp | SL: 12bp (Reverse Position)',
             fontsize=16, fontweight='bold')

# Plot price and EMA
ax.plot(data.index, data['close'], label='Price', color='black', linewidth=2.5, zorder=2)
ax.plot(data.index, data['ema7'], label='EMA7', color='blue', linewidth=2.5, linestyle='--', zorder=2)

# Plot each trade with its events
for trade in trades_with_events:
    entry_time = trade['entry_time']
    entry_idx = trade['entry_idx']
    events = trade['events']

    # Plot each segment between events
    for i in range(len(events) - 1):
        bar_idx, event_type, price, direction = events[i]
        next_bar_idx, next_event_type, next_price, next_direction = events[i + 1]

        # Calculate times
        start_time = data.index[entry_idx + bar_idx]
        end_time = data.index[min(entry_idx + next_bar_idx, len(data) - 1)]

        # Color based on direction
        if direction == 1:  # LONG
            color = 'green'
            marker = '^'
        else:  # SHORT
            color = 'red'
            marker = 'v'

        # Draw line segment
        ax.plot([start_time, end_time], [price, next_price],
               color=color, linewidth=3, alpha=0.7, zorder=5)

        # Mark the event
        if event_type == 0:  # Initial entry
            ax.scatter(start_time, price, color=color, s=250, marker=marker,
                      edgecolors='black', linewidths=2.5, zorder=10, alpha=0.9)
        elif event_type == 2:  # SL - reverse
            # Mark reversal with special symbol
            ax.scatter(end_time, next_price, color='orange', s=400, marker='X',
                      edgecolors='black', linewidths=3, zorder=11, alpha=1.0)
            ax.annotate('REV', xy=(end_time, next_price),
                       xytext=(0, -20 if direction == 1 else 20), textcoords='offset points',
                       fontsize=8, fontweight='bold', color='orange',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                edgecolor='orange', linewidth=2),
                       ha='center', zorder=15)

    # Mark final exit
    final_bar_idx, final_event_type, final_price, final_direction = events[-1]
    final_time = data.index[min(entry_idx + final_bar_idx, len(data) - 1)]

    if final_event_type == 1:  # TP
        ax.scatter(final_time, final_price, color='gold', s=400, marker='*',
                  edgecolors='black', linewidths=2.5, zorder=12, alpha=1.0)
    else:  # TIME
        ax.scatter(final_time, final_price, color='purple', s=300, marker='s',
                  edgecolors='black', linewidths=2.5, zorder=12, alpha=0.8)

# Formatting
ax.set_xlabel('Time', fontsize=13, fontweight='bold')
ax.set_ylabel('Price (USD)', fontsize=13, fontweight='bold')
ax.set_title('Green = LONG | Red = SHORT | Orange X = Reversal | Gold Star = TP | Purple Square = TIME',
            fontsize=12, pad=15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.xticks(rotation=45)

# Legend
legend_elements = [
    plt.Line2D([0], [0], color='black', linewidth=2.5, label='Price'),
    plt.Line2D([0], [0], color='blue', linewidth=2.5, linestyle='--', label='EMA7'),
    plt.Line2D([0], [0], color='green', linewidth=3, label='LONG position'),
    plt.Line2D([0], [0], color='red', linewidth=3, label='SHORT position'),
    plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='orange', markersize=14,
              label='Reversal (SL hit)', markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=16,
              label='TP Exit (+4bp net)', markeredgecolor='black', markeredgewidth=2),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=12,
              label='TIME Exit', markeredgecolor='black', markeredgewidth=2),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')

# Stats box
total_trades = len(trades_with_events)
with_reversals = len(trades_with_reversals)
no_reversals = total_trades - with_reversals

stats_text = f"STOP & REVERSE\\n\\n"
stats_text += f"Total trades: {total_trades}\\n"
stats_text += f"No reversals: {no_reversals}\\n"
stats_text += f"With reversals: {with_reversals}\\n\\n"
stats_text += f"TP: 12 bps\\n"
stats_text += f"SL: 12 bps (reverse)\\n"
stats_text += f"Max time: 10 bars"

ax.text(0.98, 0.97, stats_text,
       transform=ax.transAxes, fontsize=11,
       verticalalignment='top', horizontalalignment='right',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                edgecolor='black', linewidth=2, alpha=0.95),
       zorder=20, family='monospace')

plt.tight_layout()

# Save
output_path = Path("experiments/ema/stop_and_reverse_visualization_2024_01_01.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[OK] Saved: {output_path}")
print("\nLegend:")
print("  Green lines = LONG positions")
print("  Red lines = SHORT positions")
print("  Orange X = Reversal (SL hit, flipped direction)")
print("  Gold star = TP exit")
print("  Purple square = TIME exit")
