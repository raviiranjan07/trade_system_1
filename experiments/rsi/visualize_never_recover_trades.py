"""
VISUALIZATION: The 15 trades that NEVER recover
Show actual price paths with SMA200 crossover
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
PLOTS_PATH = Path("experiments/rsi/plots")
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
SMA_PERIOD = 200
FEES_BPS = 8

df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calculate_rsi(df, RSI_PERIOD)
df['sma_200'] = df['close'].rolling(SMA_PERIOD).mean()
df['rsi_oversold_cross'] = (df['rsi'] < RSI_OVERSOLD) & (df['rsi'].shift(1) >= RSI_OVERSOLD)
df['rsi_overbought_cross'] = (df['rsi'] > RSI_OVERBOUGHT) & (df['rsi'].shift(1) <= RSI_OVERBOUGHT)
df['bull_market'] = df['close'] > df['sma_200']
df['bear_market'] = df['close'] < df['sma_200']

oos_data = df['2024-01-01':'2025-12-31'].copy()

# Find the never-recover trades
never_recover = []

for idx in oos_data.index:
    idx_pos = oos_data.index.get_loc(idx)
    if idx_pos + 210 >= len(oos_data):
        continue

    current = oos_data.iloc[idx_pos]
    direction = None

    if current['rsi_oversold_cross'] and current['bull_market']:
        direction = 'LONG'
    elif current['rsi_overbought_cross'] and current['bear_market']:
        direction = 'SHORT'
    else:
        continue

    entry_price = oos_data.iloc[idx_pos + 1]['open']

    ever_profit = False
    for bar in range(1, 201):
        future_idx = idx_pos + 1 + bar
        if future_idx >= len(oos_data):
            break
        future_bar = oos_data.iloc[future_idx]
        if direction == 'LONG':
            pnl = (future_bar['close'] - entry_price) / entry_price * 10000
        else:
            pnl = (entry_price - future_bar['close']) / entry_price * 10000
        if pnl - FEES_BPS > 0:
            ever_profit = True
            break

    if not ever_profit:
        never_recover.append({
            'idx': idx,
            'idx_pos': idx_pos,
            'direction': direction,
            'entry_price': entry_price,
            'sma200': current['sma_200'],
            'distance_pct': (entry_price - current['sma_200']) / current['sma_200'] * 100
        })

print(f"Found {len(never_recover)} never-recover trades")

# =============================================================================
# FIGURE 1: Show 4 worst examples with price + SMA200
# =============================================================================

# Pick 4 representative examples
examples = [never_recover[0], never_recover[3], never_recover[7], never_recover[8]]
if len(never_recover) >= 5:
    examples = [never_recover[0], never_recover[4], never_recover[7], never_recover[8]]

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

for ax_idx, trade in enumerate(examples):
    ax = axes[ax_idx // 2][ax_idx % 2]

    idx_pos = trade['idx_pos']
    direction = trade['direction']

    # Show 20 bars before entry and 60 bars after
    start = max(0, idx_pos - 20)
    end = min(len(oos_data), idx_pos + 60)

    plot_data = oos_data.iloc[start:end]

    # Plot price and SMA200
    x = range(len(plot_data))
    ax.plot(x, plot_data['close'].values, 'b-', linewidth=1.5, label='Price')
    ax.plot(x, plot_data['sma_200'].values, 'orange', linewidth=2, linestyle='--', label='SMA 200')

    # Mark entry point
    entry_x = idx_pos - start
    entry_price = trade['entry_price']
    ax.scatter([entry_x], [entry_price], color='green' if direction == 'LONG' else 'red',
              s=200, zorder=5, marker='^' if direction == 'LONG' else 'v')

    # Mark SMA200 crossover area
    for i in range(len(plot_data) - 1):
        price1 = plot_data.iloc[i]['close']
        sma1 = plot_data.iloc[i]['sma_200']
        price2 = plot_data.iloc[i + 1]['close']
        sma2 = plot_data.iloc[i + 1]['sma_200']

        # Detect cross
        if (price1 > sma1 and price2 < sma2) or (price1 < sma1 and price2 > sma2):
            if i >= entry_x:  # Only after entry
                ax.axvline(i, color='red', linestyle=':', alpha=0.7, linewidth=2)
                ax.annotate('SMA200\nCROSSED!', xy=(i, plot_data.iloc[i]['close']),
                           xytext=(i + 3, plot_data.iloc[i]['close'] + (plot_data['close'].max() - plot_data['close'].min()) * 0.15),
                           fontsize=10, color='red', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='red'))
                break

    # Entry annotation
    ax.annotate(f'{direction} ENTRY\n${entry_price:,.0f}\nSMA200: ${trade["sma200"]:,.0f}\nDist: {trade["distance_pct"]:+.2f}%',
               xy=(entry_x, entry_price),
               xytext=(entry_x - 8, entry_price + (plot_data['close'].max() - plot_data['close'].min()) * 0.2),
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
               arrowprops=dict(arrowstyle='->', color='black'))

    # Shade the "WRONG SIDE" area after cross
    ax.set_title(f'Trade: {trade["idx"].strftime("%Y-%m-%d")} | {direction} | Dist from SMA: {trade["distance_pct"]:+.2f}%',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Bars (15 min each)')
    ax.set_ylabel('Price ($)')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add x-axis labels showing relative bars
    tick_positions = list(range(0, len(plot_data), 10))
    tick_labels = [str(i - entry_x) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Bars from Entry')

plt.suptitle('NEVER-RECOVER TRADES: Price was TOO CLOSE to SMA200\nSMA200 crossed right after entry = trend reversed!',
            fontsize=16, fontweight='bold', color='red')
plt.tight_layout()
plt.savefig(PLOTS_PATH / 'never_recover_examples.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_PATH / 'never_recover_examples.png'}")


# =============================================================================
# FIGURE 2: Distance from SMA200 for ALL trades (recover vs never)
# =============================================================================

# Collect distance data for ALL trades
all_trades_data = []

for idx in oos_data.index:
    idx_pos = oos_data.index.get_loc(idx)
    if idx_pos + 210 >= len(oos_data):
        continue

    current = oos_data.iloc[idx_pos]
    direction = None

    if current['rsi_oversold_cross'] and current['bull_market']:
        direction = 'LONG'
    elif current['rsi_overbought_cross'] and current['bear_market']:
        direction = 'SHORT'
    else:
        continue

    entry_price = oos_data.iloc[idx_pos + 1]['open']
    distance = abs((entry_price - current['sma_200']) / current['sma_200'] * 100)

    # Check if it ever profits
    ever_profit = False
    for bar in range(1, 201):
        future_idx = idx_pos + 1 + bar
        if future_idx >= len(oos_data):
            break
        future_bar = oos_data.iloc[future_idx]
        if direction == 'LONG':
            pnl = (future_bar['close'] - entry_price) / entry_price * 10000
        else:
            pnl = (entry_price - future_bar['close']) / entry_price * 10000
        if pnl - FEES_BPS > 0:
            ever_profit = True
            break

    all_trades_data.append({
        'direction': direction,
        'distance_from_sma_pct': distance,
        'ever_profit': ever_profit
    })

all_df = pd.DataFrame(all_trades_data)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# LEFT: Scatter plot showing distance vs outcome
ax1 = axes[0]

recovers = all_df[all_df['ever_profit'] == True]
never = all_df[all_df['ever_profit'] == False]

ax1.scatter(range(len(recovers)), recovers['distance_from_sma_pct'].values,
           color='green', alpha=0.5, s=30, label=f'Recovers ({len(recovers)})')
ax1.scatter(range(len(recovers), len(recovers) + len(never)),
           never['distance_from_sma_pct'].values,
           color='red', alpha=0.8, s=100, marker='X', label=f'NEVER recovers ({len(never)})')

ax1.axhline(2, color='orange', linestyle='--', linewidth=2, label='2% threshold')
ax1.set_xlabel('Trade #')
ax1.set_ylabel('Distance from SMA200 (%)')
ax1.set_title('Distance from SMA200 at Entry\n(Red X = NEVER recovers)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# RIGHT: Histogram comparison
ax2 = axes[1]

bins = np.arange(0, 15, 0.5)
ax2.hist(recovers['distance_from_sma_pct'], bins=bins, alpha=0.6, color='green',
        label=f'Recovers (n={len(recovers)})', density=True)
ax2.hist(never['distance_from_sma_pct'], bins=bins, alpha=0.8, color='red',
        label=f'Never recovers (n={len(never)})', density=True)

ax2.axvline(2, color='orange', linestyle='--', linewidth=2, label='2% threshold')

ax2.set_xlabel('Distance from SMA200 (%)')
ax2.set_ylabel('Density')
ax2.set_title('Distribution: Where do never-recover trades cluster?', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Count how many never-recover would be filtered
close_never = never[never['distance_from_sma_pct'] < 2]
far_never = never[never['distance_from_sma_pct'] >= 2]
close_recovers = recovers[recovers['distance_from_sma_pct'] < 2]

ax2.text(0.95, 0.95, f'Below 2%:\n  Never recover: {len(close_never)}/{len(never)}\n  Recovers: {len(close_recovers)}/{len(recovers)}',
        transform=ax2.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle('FINDING: Never-Recover Trades Are CLOSE to SMA200',
            fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_PATH / 'never_recover_distance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_PATH / 'never_recover_distance.png'}")


# =============================================================================
# FIGURE 3: Simple visual explanation
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# LEFT: SAFE trade (far from SMA200)
ax1 = axes[0]
bars = np.arange(0, 30)
sma = np.ones(30) * 100
price_safe = [108, 107, 105, 104, 103, 102, 101, 100.5, 100, 99.5,
              100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
              110, 111, 112, 111, 110, 109, 108, 107, 108, 109]

ax1.plot(bars, price_safe, 'b-', linewidth=2, label='Price')
ax1.plot(bars, sma, 'orange', linewidth=2, linestyle='--', label='SMA 200')

# Entry point
ax1.scatter([2], [105], color='red', s=200, zorder=5, marker='v')
ax1.annotate('SHORT ENTRY\n$105\n(5% below SMA)', xy=(2, 105), xytext=(5, 112),
            fontsize=11, fontweight='bold', arrowprops=dict(arrowstyle='->', color='black'),
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Show distance
ax1.annotate('', xy=(0, 100), xytext=(0, 108),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax1.text(-0.5, 104, '5%\nFAR', fontsize=11, ha='right', color='green', fontweight='bold')

ax1.set_title('SAFE: Price far from SMA200\nPrice dips, then RECOVERS', fontsize=13, fontweight='bold', color='green')
ax1.set_xlabel('Bars')
ax1.set_ylabel('Price')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(95, 115)

# RIGHT: DANGEROUS trade (close to SMA200)
ax2 = axes[1]
price_danger = [101, 100.8, 100.5, 100.2, 100, 99.5, 99, 98.5, 98, 97,
                96, 95, 94, 93, 92, 91, 90, 89, 88, 87,
                86, 85, 84, 83, 82, 81, 80, 79, 78, 77]

ax2.plot(bars, price_danger, 'b-', linewidth=2, label='Price')
ax2.plot(bars, sma, 'orange', linewidth=2, linestyle='--', label='SMA 200')

# Entry point
ax2.scatter([1], [100.8], color='red', s=200, zorder=5, marker='v')
ax2.annotate('SHORT ENTRY\n$100.8\n(0.8% below SMA)', xy=(1, 100.8), xytext=(5, 108),
            fontsize=11, fontweight='bold', arrowprops=dict(arrowstyle='->', color='black'),
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# SMA cross
ax2.axvline(4, color='red', linestyle=':', alpha=0.7, linewidth=2)
ax2.annotate('SMA200 CROSSED!\nTrend REVERSED!', xy=(4, 100), xytext=(8, 108),
            fontsize=11, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red'),
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Show distance
ax2.annotate('', xy=(0, 100), xytext=(0, 101),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax2.text(-0.5, 100.5, '0.8%\nTOO\nCLOSE!', fontsize=11, ha='right', color='red', fontweight='bold')

ax2.set_title('DANGER: Price close to SMA200\nSMA crosses -> price keeps going AGAINST you', fontsize=13, fontweight='bold', color='red')
ax2.set_xlabel('Bars')
ax2.set_ylabel('Price')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(75, 115)

plt.suptitle('WHY Distance from SMA200 Matters for NEVER-RECOVER Trades',
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_PATH / 'never_recover_explanation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_PATH / 'never_recover_explanation.png'}")

print("\nDone! Check experiments/rsi/plots/ for all visualizations.")
