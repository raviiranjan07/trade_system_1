"""
EXP-004: Price Action Visualization of 200 SMA Filter
Shows KEPT vs SKIPPED signals on actual price chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
PLOTS_PATH = Path("experiments/rsi/EXP-004/plots")

RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
HORIZON = 10

print("="*70)
print("200 SMA FILTER - PRICE ACTION VISUALIZATION")
print("="*70)

# Load and prepare data
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calculate_rsi(df, RSI_PERIOD)
df['sma_200'] = df['close'].rolling(200).mean()
df['oversold'] = df['rsi'] < RSI_OVERSOLD
df['overbought'] = df['rsi'] > RSI_OVERBOUGHT
df['rsi_oversold_cross'] = (df['oversold'] == True) & (df['oversold'].shift(1) == False)
df['rsi_overbought_cross'] = (df['overbought'] == True) & (df['overbought'].shift(1) == False)

# OOS data
oos_data = df['2024-01-01':'2025-12-30'].copy()

# Classify all signals
def classify_all_signals(data):
    results = []

    # LONG signals
    long_signals = data[data['rsi_oversold_cross'] == True].copy()
    for idx in long_signals.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + HORIZON >= len(data):
            continue

        entry_price = data.iloc[idx_pos]['close']
        sma_200 = data.iloc[idx_pos]['sma_200']
        future_data = data.iloc[idx_pos + 1 : idx_pos + 1 + HORIZON]

        if len(future_data) == 0:
            continue

        mfe = (future_data['high'].max() - entry_price) / entry_price * 10000
        mae = (entry_price - future_data['low'].min()) / entry_price * 10000

        is_bull = entry_price > sma_200
        is_success = mfe > 0

        # With filter: LONG only in BULL
        kept = is_bull

        results.append({
            'timestamp': idx,
            'price': entry_price,
            'sma_200': sma_200,
            'type': 'LONG',
            'market': 'BULL' if is_bull else 'BEAR',
            'success': is_success,
            'kept': kept,
            'mfe': mfe,
            'mae': mae
        })

    # SHORT signals
    short_signals = data[data['rsi_overbought_cross'] == True].copy()
    for idx in short_signals.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + HORIZON >= len(data):
            continue

        entry_price = data.iloc[idx_pos]['close']
        sma_200 = data.iloc[idx_pos]['sma_200']
        future_data = data.iloc[idx_pos + 1 : idx_pos + 1 + HORIZON]

        if len(future_data) == 0:
            continue

        mfe = (entry_price - future_data['low'].min()) / entry_price * 10000
        mae = (future_data['high'].max() - entry_price) / entry_price * 10000

        is_bull = entry_price > sma_200
        is_success = mfe > 0

        # With filter: SHORT only in BEAR
        kept = not is_bull

        results.append({
            'timestamp': idx,
            'price': entry_price,
            'sma_200': sma_200,
            'type': 'SHORT',
            'market': 'BULL' if is_bull else 'BEAR',
            'success': is_success,
            'kept': kept,
            'mfe': mfe,
            'mae': mae
        })

    return pd.DataFrame(results)

signals_df = classify_all_signals(oos_data)

# Resample to daily for cleaner chart
daily_data = oos_data.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last'
}).dropna()

daily_sma200 = daily_data['close'].rolling(200).mean()

# =============================================================================
# VISUALIZATION 1: Full chart with KEPT vs SKIPPED
# =============================================================================
print("\n[1/3] Creating full price chart with KEPT vs SKIPPED signals...")

fig, ax = plt.subplots(figsize=(20, 12))

# Plot price and SMA200
ax.plot(daily_data.index, daily_data['close'], 'b-', linewidth=1, alpha=0.7, label='BTC Price')
ax.plot(daily_data.index, daily_sma200, 'orange', linewidth=2.5, label='SMA 200')

# Shade BULL vs BEAR zones
for i in range(1, len(daily_data)):
    sma_val = daily_sma200.iloc[i] if not pd.isna(daily_sma200.iloc[i]) else 0
    if daily_data['close'].iloc[i] > sma_val and sma_val > 0:
        ax.axvspan(daily_data.index[i-1], daily_data.index[i], alpha=0.1, color='green')
    elif sma_val > 0:
        ax.axvspan(daily_data.index[i-1], daily_data.index[i], alpha=0.1, color='red')

# Plot KEPT signals (will be traded)
long_kept = signals_df[(signals_df['type'] == 'LONG') & (signals_df['kept'] == True)]
short_kept = signals_df[(signals_df['type'] == 'SHORT') & (signals_df['kept'] == True)]

ax.scatter(long_kept['timestamp'], long_kept['price'],
           color='green', marker='^', s=100, alpha=0.8,
           label=f'LONG KEPT ({len(long_kept)})', zorder=5)
ax.scatter(short_kept['timestamp'], short_kept['price'],
           color='red', marker='v', s=100, alpha=0.8,
           label=f'SHORT KEPT ({len(short_kept)})', zorder=5)

# Plot SKIPPED signals (won't be traded) - smaller, gray
long_skipped = signals_df[(signals_df['type'] == 'LONG') & (signals_df['kept'] == False)]
short_skipped = signals_df[(signals_df['type'] == 'SHORT') & (signals_df['kept'] == False)]

ax.scatter(long_skipped['timestamp'], long_skipped['price'],
           color='gray', marker='^', s=40, alpha=0.4,
           label=f'LONG SKIPPED ({len(long_skipped)})', zorder=4)
ax.scatter(short_skipped['timestamp'], short_skipped['price'],
           color='gray', marker='v', s=40, alpha=0.4,
           label=f'SHORT SKIPPED ({len(short_skipped)})', zorder=4)

# Mark failures in KEPT signals
long_kept_fail = signals_df[(signals_df['type'] == 'LONG') & (signals_df['kept'] == True) & (signals_df['success'] == False)]
short_kept_fail = signals_df[(signals_df['type'] == 'SHORT') & (signals_df['kept'] == True) & (signals_df['success'] == False)]

if len(long_kept_fail) > 0:
    ax.scatter(long_kept_fail['timestamp'], long_kept_fail['price'],
               color='green', marker='^', s=300, edgecolors='red', linewidths=3,
               label=f'LONG KEPT FAIL ({len(long_kept_fail)})', zorder=6)

if len(short_kept_fail) > 0:
    ax.scatter(short_kept_fail['timestamp'], short_kept_fail['price'],
               color='red', marker='v', s=300, edgecolors='yellow', linewidths=3,
               label=f'SHORT KEPT FAIL ({len(short_kept_fail)})', zorder=6)

ax.set_ylabel('Price ($)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.set_title('200 SMA Filter: KEPT vs SKIPPED Signals on Price Chart\n'
             'Green zone = BULL (LONG kept, SHORT skipped) | Red zone = BEAR (SHORT kept, LONG skipped)',
             fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

plt.tight_layout()
plt.savefig(PLOTS_PATH / '14_sma_filter_price_kept_skipped.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_PATH / '14_sma_filter_price_kept_skipped.png'}")

# =============================================================================
# VISUALIZATION 2: Zoomed views - BULL period and BEAR period
# =============================================================================
print("\n[2/3] Creating zoomed views...")

fig, axes = plt.subplots(2, 1, figsize=(18, 14))

# BULL period zoom (Feb-Apr 2024 - ETF rally)
ax1 = axes[0]
zoom_start = pd.Timestamp('2024-02-01')
zoom_end = pd.Timestamp('2024-04-30')
zoom_data = oos_data[zoom_start:zoom_end]
zoom_signals = signals_df[(signals_df['timestamp'] >= zoom_start) & (signals_df['timestamp'] <= zoom_end)]

ax1.plot(zoom_data.index, zoom_data['close'], 'b-', linewidth=1.5, label='Price')
ax1.plot(zoom_data.index, zoom_data['sma_200'], 'orange', linewidth=2, label='SMA 200')

# KEPT
long_kept_z = zoom_signals[(zoom_signals['type'] == 'LONG') & (zoom_signals['kept'] == True)]
short_kept_z = zoom_signals[(zoom_signals['type'] == 'SHORT') & (zoom_signals['kept'] == True)]
ax1.scatter(long_kept_z['timestamp'], long_kept_z['price'], color='green', marker='^', s=150, label=f'LONG KEPT ({len(long_kept_z)})')
ax1.scatter(short_kept_z['timestamp'], short_kept_z['price'], color='red', marker='v', s=150, label=f'SHORT KEPT ({len(short_kept_z)})')

# SKIPPED
long_skip_z = zoom_signals[(zoom_signals['type'] == 'LONG') & (zoom_signals['kept'] == False)]
short_skip_z = zoom_signals[(zoom_signals['type'] == 'SHORT') & (zoom_signals['kept'] == False)]
ax1.scatter(long_skip_z['timestamp'], long_skip_z['price'], color='gray', marker='^', s=60, alpha=0.5, label=f'LONG SKIPPED ({len(long_skip_z)})')
ax1.scatter(short_skip_z['timestamp'], short_skip_z['price'], color='gray', marker='v', s=60, alpha=0.5, label=f'SHORT SKIPPED ({len(short_skip_z)})')

# Mark account killer that was SKIPPED
killer_date = pd.Timestamp('2024-02-26')
killer_signal = zoom_signals[(zoom_signals['timestamp'].dt.date == killer_date.date()) & (zoom_signals['type'] == 'SHORT')]
if len(killer_signal) > 0:
    k = killer_signal.iloc[0]
    ax1.annotate('ACCOUNT KILLER\n(SKIPPED!)', xy=(k['timestamp'], k['price']),
                xytext=(k['timestamp'] + pd.Timedelta(days=10), k['price'] * 1.05),
                fontsize=11, color='darkgreen', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))

ax1.set_title('BULL Market Period (Feb-Apr 2024): SHORT signals SKIPPED\nAccount killer at Feb 26 would be SKIPPED by filter', fontsize=12)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# BEAR period zoom (if any exists)
ax2 = axes[1]
# Find a period where price < SMA200
bear_periods = oos_data[oos_data['close'] < oos_data['sma_200']]
if len(bear_periods) > 100:
    # Get first significant bear period
    zoom_start2 = bear_periods.index[0]
    zoom_end2 = zoom_start2 + pd.Timedelta(days=60)
    zoom_data2 = oos_data[zoom_start2:zoom_end2]
    zoom_signals2 = signals_df[(signals_df['timestamp'] >= zoom_start2) & (signals_df['timestamp'] <= zoom_end2)]

    ax2.plot(zoom_data2.index, zoom_data2['close'], 'b-', linewidth=1.5, label='Price')
    ax2.plot(zoom_data2.index, zoom_data2['sma_200'], 'orange', linewidth=2, label='SMA 200')

    # KEPT
    long_kept_z2 = zoom_signals2[(zoom_signals2['type'] == 'LONG') & (zoom_signals2['kept'] == True)]
    short_kept_z2 = zoom_signals2[(zoom_signals2['type'] == 'SHORT') & (zoom_signals2['kept'] == True)]
    ax2.scatter(long_kept_z2['timestamp'], long_kept_z2['price'], color='green', marker='^', s=150, label=f'LONG KEPT ({len(long_kept_z2)})')
    ax2.scatter(short_kept_z2['timestamp'], short_kept_z2['price'], color='red', marker='v', s=150, label=f'SHORT KEPT ({len(short_kept_z2)})')

    # SKIPPED
    long_skip_z2 = zoom_signals2[(zoom_signals2['type'] == 'LONG') & (zoom_signals2['kept'] == False)]
    short_skip_z2 = zoom_signals2[(zoom_signals2['type'] == 'SHORT') & (zoom_signals2['kept'] == False)]
    ax2.scatter(long_skip_z2['timestamp'], long_skip_z2['price'], color='gray', marker='^', s=60, alpha=0.5, label=f'LONG SKIPPED ({len(long_skip_z2)})')
    ax2.scatter(short_skip_z2['timestamp'], short_skip_z2['price'], color='gray', marker='v', s=60, alpha=0.5, label=f'SHORT SKIPPED ({len(short_skip_z2)})')

    ax2.set_title(f'BEAR Market Period ({zoom_start2.strftime("%Y-%m-%d")}): LONG signals SKIPPED', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.tight_layout()
plt.savefig(PLOTS_PATH / '15_sma_filter_zoomed_periods.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_PATH / '15_sma_filter_zoomed_periods.png'}")

# =============================================================================
# VISUALIZATION 3: Summary comparison
# =============================================================================
print("\n[3/3] Creating summary comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Before filter
ax1 = axes[0]
long_all = signals_df[signals_df['type'] == 'LONG']
short_all = signals_df[signals_df['type'] == 'SHORT']

long_success_all = len(long_all[long_all['success'] == True])
long_fail_all = len(long_all[long_all['success'] == False])
short_success_all = len(short_all[short_all['success'] == True])
short_fail_all = len(short_all[short_all['success'] == False])

categories = ['LONG\nSuccess', 'LONG\nFail', 'SHORT\nSuccess', 'SHORT\nFail']
values = [long_success_all, long_fail_all, short_success_all, short_fail_all]
colors = ['green', 'lightcoral', 'green', 'lightcoral']

bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_title('BEFORE Filter: All Signals', fontsize=14)
ax1.set_ylabel('Count')
ax1.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)

# After filter
ax2 = axes[1]
long_kept_df = signals_df[(signals_df['type'] == 'LONG') & (signals_df['kept'] == True)]
short_kept_df = signals_df[(signals_df['type'] == 'SHORT') & (signals_df['kept'] == True)]

long_success_kept = len(long_kept_df[long_kept_df['success'] == True])
long_fail_kept = len(long_kept_df[long_kept_df['success'] == False])
short_success_kept = len(short_kept_df[short_kept_df['success'] == True])
short_fail_kept = len(short_kept_df[short_kept_df['success'] == False])

values_after = [long_success_kept, long_fail_kept, short_success_kept, short_fail_kept]

bars = ax2.bar(categories, values_after, color=colors, alpha=0.7, edgecolor='black')
ax2.set_title('AFTER Filter: Only KEPT Signals', fontsize=14)
ax2.set_ylabel('Count')
ax2.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax2.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)

# Add summary text
total_before = len(signals_df)
total_after = len(signals_df[signals_df['kept'] == True])
fail_before = long_fail_all + short_fail_all
fail_after = long_fail_kept + short_fail_kept

fig.text(0.5, 0.01,
         f'BEFORE: {total_before} signals, {fail_before} failures | '
         f'AFTER: {total_after} signals, {fail_after} failures | '
         f'Signals kept: {total_after/total_before*100:.1f}%, Failures avoided: {(fail_before-fail_after)/fail_before*100:.1f}%',
         ha='center', fontsize=12, style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig(PLOTS_PATH / '16_sma_filter_before_after.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_PATH / '16_sma_filter_before_after.png'}")

print("\n" + "="*70)
print("PRICE ACTION VISUALIZATIONS COMPLETE")
print("="*70)
print("\nFiles created:")
print("  14_sma_filter_price_kept_skipped.png - Full price chart with KEPT vs SKIPPED")
print("  15_sma_filter_zoomed_periods.png     - Zoomed BULL and BEAR periods")
print("  16_sma_filter_before_after.png       - Before vs After comparison")
