"""
EXP-004: Price-Based Visualizations
Shows WHERE on the price chart signals worked vs failed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
RESULTS_PATH = Path("experiments/rsi/EXP-004")
PLOTS_PATH = RESULTS_PATH / "plots"
PLOTS_PATH.mkdir(exist_ok=True)

RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
HORIZON = 10

print("="*70)
print("EXP-004: PRICE-BASED VISUALIZATIONS")
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
def classify_signals(data, signal_col, signal_type):
    signals = data[data[signal_col] == True].copy()
    results = []

    for idx in signals.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + HORIZON >= len(data):
            continue

        entry_price = data.iloc[idx_pos]['close']
        sma_200 = data.iloc[idx_pos]['sma_200']
        future_data = data.iloc[idx_pos + 1 : idx_pos + 1 + HORIZON]

        if len(future_data) == 0:
            continue

        if signal_type == 'long':
            mfe = (future_data['high'].max() - entry_price) / entry_price * 10000
            mae = (entry_price - future_data['low'].min()) / entry_price * 10000
        else:
            mfe = (entry_price - future_data['low'].min()) / entry_price * 10000
            mae = (future_data['high'].max() - entry_price) / entry_price * 10000

        market = 'BULL' if entry_price > sma_200 else 'BEAR'
        success = mfe > 0

        results.append({
            'timestamp': idx,
            'price': entry_price,
            'sma_200': sma_200,
            'market': market,
            'mfe': mfe,
            'mae': mae,
            'success': success,
            'type': signal_type
        })

    return pd.DataFrame(results)

long_signals = classify_signals(oos_data, 'rsi_oversold_cross', 'long')
short_signals = classify_signals(oos_data, 'rsi_overbought_cross', 'short')

# =============================================================================
# VISUALIZATION 1: Full Price Chart with ALL Signals
# =============================================================================
print("\n[1/5] Creating full price chart with all signals...")

# Resample to daily for cleaner chart
daily_data = oos_data.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

daily_sma200 = daily_data['close'].rolling(200).mean()

fig, axes = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [3, 1]})

# Price chart
ax1 = axes[0]
ax1.plot(daily_data.index, daily_data['close'], 'b-', linewidth=1, alpha=0.7, label='BTC Price')
ax1.plot(daily_data.index, daily_sma200, 'orange', linewidth=2, alpha=0.8, label='SMA 200')

# Plot LONG signals
long_success = long_signals[long_signals['success'] == True]
long_fail = long_signals[long_signals['success'] == False]

ax1.scatter(long_success['timestamp'], long_success['price'],
            color='green', marker='^', s=50, alpha=0.6, label=f'LONG Success ({len(long_success)})')
ax1.scatter(long_fail['timestamp'], long_fail['price'],
            color='lime', marker='^', s=150, edgecolors='red', linewidths=2,
            label=f'LONG Failure ({len(long_fail)})')

# Plot SHORT signals
short_success = short_signals[short_signals['success'] == True]
short_fail = short_signals[short_signals['success'] == False]

ax1.scatter(short_success['timestamp'], short_success['price'],
            color='red', marker='v', s=50, alpha=0.6, label=f'SHORT Success ({len(short_success)})')
ax1.scatter(short_fail['timestamp'], short_fail['price'],
            color='darkred', marker='v', s=150, edgecolors='yellow', linewidths=2,
            label=f'SHORT Failure ({len(short_fail)})')

# Mark account killers
killers = short_fail[short_fail['mae'] > 300]
for _, k in killers.iterrows():
    ax1.annotate(f'KILLER\n{k["mae"]:.0f}bps', xy=(k['timestamp'], k['price']),
                xytext=(k['timestamp'], k['price'] * 1.05),
                fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))

ax1.set_ylabel('Price ($)', fontsize=12)
ax1.set_title('BTC Price with RSI Signals (2024-2025 OOS)\n▲ = LONG (oversold), ▼ = SHORT (overbought)', fontsize=14)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# RSI subplot
ax2 = axes[1]
# Resample RSI to daily (use close value)
daily_rsi = oos_data['rsi'].resample('1D').last().dropna()
ax2.plot(daily_rsi.index, daily_rsi, 'purple', linewidth=1)
ax2.axhline(y=RSI_OVERSOLD, color='green', linestyle='--', alpha=0.7, label=f'Oversold ({RSI_OVERSOLD})')
ax2.axhline(y=RSI_OVERBOUGHT, color='red', linestyle='--', alpha=0.7, label=f'Overbought ({RSI_OVERBOUGHT})')
ax2.fill_between(daily_rsi.index, RSI_OVERSOLD, 0, alpha=0.2, color='green')
ax2.fill_between(daily_rsi.index, RSI_OVERBOUGHT, 100, alpha=0.2, color='red')
ax2.set_ylabel('RSI', fontsize=12)
ax2.set_ylim(0, 100)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

plt.tight_layout()
plt.savefig(PLOTS_PATH / '08_price_all_signals.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_PATH / '08_price_all_signals.png'}")

# =============================================================================
# VISUALIZATION 2: Account Killer Zoomed (Feb 2024)
# =============================================================================
print("\n[2/5] Creating Account Killer zoom view...")

killer_date = pd.Timestamp('2024-02-26')
zoom_start = killer_date - pd.Timedelta(days=10)
zoom_end = killer_date + pd.Timedelta(days=30)

zoom_data = oos_data[zoom_start:zoom_end].copy()

fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

# Price chart
ax1 = axes[0]
ax1.plot(zoom_data.index, zoom_data['close'], 'b-', linewidth=1.5, label='BTC Price')
ax1.plot(zoom_data.index, zoom_data['sma_200'], 'orange', linewidth=2, label='SMA 200')

# Mark the killer signal
killer_signal = short_signals[short_signals['timestamp'].dt.date == killer_date.date()]
if len(killer_signal) > 0:
    k = killer_signal.iloc[0]
    ax1.scatter(k['timestamp'], k['price'], color='red', marker='v', s=300,
                edgecolors='yellow', linewidths=3, zorder=5, label='SHORT Entry (KILLER)')

    # Draw the loss zone
    entry_price = k['price']
    max_price = zoom_data[zoom_data.index > k['timestamp']]['high'].max()
    ax1.axhline(y=entry_price, color='red', linestyle='--', alpha=0.5)
    ax1.fill_between(zoom_data.index, entry_price, max_price,
                     where=zoom_data.index > k['timestamp'],
                     alpha=0.3, color='red', label='Loss Zone')

    # Annotate
    ax1.annotate(f'ENTRY: ${entry_price:,.0f}\nRSI said "overbought"',
                xy=(k['timestamp'], entry_price),
                xytext=(k['timestamp'] - pd.Timedelta(days=5), entry_price * 0.95),
                fontsize=11, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

    ax1.annotate(f'Price kept rising!\nBTC ETF Rally',
                xy=(zoom_end - pd.Timedelta(days=5), max_price * 0.98),
                fontsize=12, color='darkred', fontweight='bold')

ax1.set_ylabel('Price ($)', fontsize=12)
ax1.set_title('ACCOUNT KILLER: 2024-02-26 SHORT Signal\nRSI said "overbought" but price kept rising (ETF Rally)', fontsize=14)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# RSI subplot
ax2 = axes[1]
ax2.plot(zoom_data.index, zoom_data['rsi'], 'purple', linewidth=1.5)
ax2.axhline(y=RSI_OVERBOUGHT, color='red', linestyle='--', alpha=0.7)
ax2.axhline(y=RSI_OVERSOLD, color='green', linestyle='--', alpha=0.7)
ax2.fill_between(zoom_data.index, RSI_OVERBOUGHT, 100, alpha=0.2, color='red')

if len(killer_signal) > 0:
    ax2.axvline(x=killer_signal.iloc[0]['timestamp'], color='red', linewidth=2, label='Signal triggered')
    ax2.annotate('RSI crossed 80\n(Overbought)',
                xy=(killer_signal.iloc[0]['timestamp'], 82),
                xytext=(killer_signal.iloc[0]['timestamp'] + pd.Timedelta(days=3), 90),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='purple'))

ax2.set_ylabel('RSI', fontsize=12)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.tight_layout()
plt.savefig(PLOTS_PATH / '09_account_killer_zoom.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_PATH / '09_account_killer_zoom.png'}")

# =============================================================================
# VISUALIZATION 3: BULL vs BEAR Market Zones with Failures
# =============================================================================
print("\n[3/5] Creating BULL vs BEAR market zones...")

fig, ax = plt.subplots(figsize=(18, 10))

# Plot price and SMA
ax.plot(daily_data.index, daily_data['close'], 'b-', linewidth=1.5, label='BTC Price')
ax.plot(daily_data.index, daily_sma200, 'orange', linewidth=2.5, label='SMA 200')

# Shade BULL vs BEAR zones
for i in range(1, len(daily_data)):
    if daily_data['close'].iloc[i] > daily_sma200.iloc[i] if not pd.isna(daily_sma200.iloc[i]) else False:
        ax.axvspan(daily_data.index[i-1], daily_data.index[i], alpha=0.1, color='green')
    else:
        ax.axvspan(daily_data.index[i-1], daily_data.index[i], alpha=0.1, color='red')

# Plot SHORT failures by market
short_fail_bull = short_fail[short_fail['market'] == 'BULL']
short_fail_bear = short_fail[short_fail['market'] == 'BEAR']

ax.scatter(short_fail_bull['timestamp'], short_fail_bull['price'],
           color='red', marker='v', s=200, edgecolors='yellow', linewidths=2,
           label=f'SHORT Fail in BULL ({len(short_fail_bull)}) - AVOID', zorder=5)

ax.scatter(short_fail_bear['timestamp'], short_fail_bear['price'],
           color='blue', marker='v', s=200, edgecolors='white', linewidths=2,
           label=f'SHORT Fail in BEAR ({len(short_fail_bear)}) - OK', zorder=5)

# Plot LONG failures by market
long_fail_bull = long_fail[long_fail['market'] == 'BULL']
long_fail_bear = long_fail[long_fail['market'] == 'BEAR']

ax.scatter(long_fail_bear['timestamp'], long_fail_bear['price'],
           color='lime', marker='^', s=200, edgecolors='red', linewidths=2,
           label=f'LONG Fail in BEAR ({len(long_fail_bear)}) - AVOID', zorder=5)

# Add annotations for killers
for _, k in killers.iterrows():
    ax.annotate(f'KILLER #{killers.index.get_loc(_)+1}\n-{k["mae"]:.0f}bps',
               xy=(k['timestamp'], k['price']),
               xytext=(k['timestamp'], k['price'] * 1.08),
               fontsize=10, color='darkred', fontweight='bold', ha='center',
               arrowprops=dict(arrowstyle='->', color='red'))

ax.set_ylabel('Price ($)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.set_title('Market Zones: Green = BULL (price > SMA200), Red = BEAR (price < SMA200)\n'
             'Rule: Skip SHORT in BULL, Skip LONG in BEAR → Avoids ALL dangerous failures', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

plt.tight_layout()
plt.savefig(PLOTS_PATH / '10_bull_bear_zones.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_PATH / '10_bull_bear_zones.png'}")

# =============================================================================
# VISUALIZATION 4: All SHORT Failures Zoomed
# =============================================================================
print("\n[4/5] Creating individual SHORT failure views...")

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for i, (_, fail) in enumerate(short_fail.iterrows()):
    if i >= 12:
        break

    ax = axes[i]

    # Get zoom window
    fail_date = fail['timestamp']
    zoom_start = fail_date - pd.Timedelta(days=3)
    zoom_end = fail_date + pd.Timedelta(days=7)

    zoom_data = oos_data[zoom_start:zoom_end].copy()

    if len(zoom_data) == 0:
        continue

    # Plot price
    ax.plot(zoom_data.index, zoom_data['close'], 'b-', linewidth=1)
    ax.plot(zoom_data.index, zoom_data['sma_200'], 'orange', linewidth=1.5, alpha=0.7)

    # Mark entry
    ax.scatter(fail['timestamp'], fail['price'], color='red', marker='v', s=150,
               edgecolors='yellow', linewidths=2, zorder=5)
    ax.axhline(y=fail['price'], color='red', linestyle='--', alpha=0.5)

    # Title with info
    market_color = 'red' if fail['market'] == 'BULL' else 'blue'
    killer_text = ' KILLER!' if fail['mae'] > 300 else ''
    ax.set_title(f"{fail['timestamp'].strftime('%Y-%m-%d')}\n"
                 f"Market: {fail['market']} | MAE: {fail['mae']:.0f}bps{killer_text}",
                 fontsize=10, color=market_color)

    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.grid(True, alpha=0.3)

# Hide empty subplots
for j in range(i+1, 12):
    axes[j].axis('off')

plt.suptitle('All 11 SHORT Failures - Zoomed View\nRed title = BULL market (should skip), Blue = BEAR market',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_PATH / '11_short_failures_zoomed.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_PATH / '11_short_failures_zoomed.png'}")

# =============================================================================
# VISUALIZATION 5: LONG Failures Zoomed with Recovery Path
# =============================================================================
print("\n[5/5] Creating LONG failure recovery paths...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, (_, fail) in enumerate(long_fail.iterrows()):
    if i >= 2:
        break

    ax = axes[i]

    # Get extended zoom window to see recovery
    fail_date = fail['timestamp']
    zoom_start = fail_date - pd.Timedelta(days=2)
    zoom_end = fail_date + pd.Timedelta(days=10)

    zoom_data = oos_data[zoom_start:zoom_end].copy()

    # Plot price
    ax.plot(zoom_data.index, zoom_data['close'], 'b-', linewidth=1.5, label='Price')
    ax.plot(zoom_data.index, zoom_data['sma_200'], 'orange', linewidth=2, label='SMA 200')

    # Mark entry
    ax.scatter(fail['timestamp'], fail['price'], color='green', marker='^', s=200,
               edgecolors='red', linewidths=2, zorder=5, label='LONG Entry')
    ax.axhline(y=fail['price'], color='green', linestyle='--', alpha=0.5, label='Entry Price')

    # Shade loss zone
    min_price = zoom_data[zoom_data.index > fail['timestamp']]['low'].min()
    ax.fill_between(zoom_data.index, fail['price'], min_price,
                   where=zoom_data.index > fail['timestamp'],
                   alpha=0.3, color='red', label='Drawdown Zone')

    # Find recovery point (when price goes back above entry)
    recovery_data = zoom_data[zoom_data.index > fail['timestamp']]
    recovery_idx = recovery_data[recovery_data['high'] > fail['price']].index
    if len(recovery_idx) > 0:
        ax.axvline(x=recovery_idx[0], color='green', linewidth=2, linestyle=':', label='Recovery')

    ax.set_title(f"LONG Failure #{i+1}: {fail['timestamp'].strftime('%Y-%m-%d')}\n"
                f"Market: BEAR | MAE: {fail['mae']:.0f}bps | RECOVERED",
                fontsize=12, color='orange')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.suptitle('Both LONG Failures - Price Path and Recovery\nBoth in BEAR market (price < SMA200), Both RECOVERED',
             fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS_PATH / '12_long_failures_recovery.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_PATH / '12_long_failures_recovery.png'}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("PRICE-BASED VISUALIZATIONS COMPLETE")
print("="*70)
print(f"\nSaved to: {PLOTS_PATH}")
print("\nNew files created:")
print("  08_price_all_signals.png   - Full price chart with all signals")
print("  09_account_killer_zoom.png - Account killer zoomed (Feb 2024)")
print("  10_bull_bear_zones.png     - BULL vs BEAR market zones")
print("  11_short_failures_zoomed.png - All 11 SHORT failures zoomed")
print("  12_long_failures_recovery.png - LONG failures with recovery path")
