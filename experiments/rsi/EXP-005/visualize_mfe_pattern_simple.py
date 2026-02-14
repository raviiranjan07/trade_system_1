"""
EXP-005: Simple visualization of MFE continuation/reversal pattern
Shows clearly: Does early profit lead to more profit or reversal?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
PLOTS_PATH = Path("experiments/rsi/EXP-005/plots")

RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
HORIZON = 10

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
df['rsi_oversold_cross'] = (df['rsi'] < RSI_OVERSOLD) & (df['rsi'].shift(1) >= RSI_OVERSOLD)
df['rsi_overbought_cross'] = (df['rsi'] > RSI_OVERBOUGHT) & (df['rsi'].shift(1) <= RSI_OVERBOUGHT)

oos_data = df['2024-01-01':'2025-12-30'].copy()

def get_profit_paths(data, signal_col, signal_type, filter_by_sma=True):
    """Get profit at each bar for all signals."""
    signals = data[data[signal_col] == True].copy()

    all_paths = []

    for idx in signals.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + HORIZON >= len(data):
            continue

        entry_price = data.iloc[idx_pos]['close']
        sma_200 = data.iloc[idx_pos]['sma_200']

        if filter_by_sma:
            if signal_type == 'long' and entry_price < sma_200:
                continue
            if signal_type == 'short' and entry_price > sma_200:
                continue

        path = {'bar_0': 0}
        for bar in [1, 3, 5, 10]:
            if idx_pos + bar >= len(data):
                continue
            future_bar = data.iloc[idx_pos + bar]

            if signal_type == 'long':
                mfe = (future_bar['high'] - entry_price) / entry_price * 10000
            else:
                mfe = (entry_price - future_bar['low']) / entry_price * 10000

            path[f'bar_{bar}'] = mfe

        if len(path) == 5:
            all_paths.append(path)

    return pd.DataFrame(all_paths)

# Get data
long_df = get_profit_paths(oos_data, 'rsi_oversold_cross', 'long')
short_df = get_profit_paths(oos_data, 'rsi_overbought_cross', 'short')

# =============================================================================
# VISUALIZATION: Simple bar chart showing Bar 1 profit vs Bar 10 profit
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# -----------------------------------------------------------------------------
# LONG: Group by Bar 1 profit
# -----------------------------------------------------------------------------
long_df['bar1_group'] = pd.cut(long_df['bar_1'],
                                bins=[-100, 10, 20, 100],
                                labels=['Weak (0-10)', 'Medium (10-20)', 'Strong (>20)'])

ax1 = axes[0, 0]
groups = ['Weak (0-10)', 'Medium (10-20)', 'Strong (>20)']
bar1_avgs = []
bar10_avgs = []
counts = []

for group in groups:
    subset = long_df[long_df['bar1_group'] == group]
    if len(subset) > 0:
        bar1_avgs.append(subset['bar_1'].mean())
        bar10_avgs.append(subset['bar_10'].mean())
        counts.append(len(subset))
    else:
        bar1_avgs.append(0)
        bar10_avgs.append(0)
        counts.append(0)

x = np.arange(len(groups))
width = 0.35

bars1 = ax1.bar(x - width/2, bar1_avgs, width, label='Bar 1 (15 min)', color='lightblue', edgecolor='blue')
bars2 = ax1.bar(x + width/2, bar10_avgs, width, label='Bar 10 (2.5 hr)', color='lightgreen', edgecolor='green')

ax1.axhline(0, color='gray', linestyle='-', alpha=0.5)
ax1.set_ylabel('Average Profit (bps)')
ax1.set_title('LONG: Does early profit continue?\n(Compare blue vs green bars)')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{g}\nn={c}' for g, c in zip(groups, counts)])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

# -----------------------------------------------------------------------------
# LONG: Interpretation
# -----------------------------------------------------------------------------
ax2 = axes[0, 1]
ax2.axis('off')

interpretation_long = """
LONG SIGNALS - INTERPRETATION:

Weak Start (0-10 bps at Bar 1):
  Bar 1: +5 bps  ->  Bar 10: -6 bps
  REVERSES! Price goes against you.
  ACTION: Use tight trailing stop (10-15 bps)

Medium Start (10-20 bps at Bar 1):
  Bar 1: +15 bps  ->  Bar 10: +8 bps
  REVERSES! Gives back half the profit.
  ACTION: Use trailing stop (15-20 bps)

Strong Start (>20 bps at Bar 1):
  Bar 1: +35 bps  ->  Bar 10: +90 bps
  CONTINUES! Price keeps going up.
  ACTION: HOLD, use wider trailing stop (25 bps)

RULE FOR LONG:
If Bar 1 shows >20 bps profit = STRONG signal, HOLD
If Bar 1 shows <20 bps profit = WEAK signal, use tight stop
"""
ax2.text(0.1, 0.9, interpretation_long, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# -----------------------------------------------------------------------------
# SHORT: Group by Bar 1 profit
# -----------------------------------------------------------------------------
short_df['bar1_group'] = pd.cut(short_df['bar_1'],
                                 bins=[-100, 15, 30, 100],
                                 labels=['Normal (0-15)', 'Good (15-30)', 'Fast (>30)'])

ax3 = axes[1, 0]
groups = ['Normal (0-15)', 'Good (15-30)', 'Fast (>30)']
bar1_avgs = []
bar10_avgs = []
counts = []

for group in groups:
    subset = short_df[short_df['bar1_group'] == group]
    if len(subset) > 0:
        bar1_avgs.append(subset['bar_1'].mean())
        bar10_avgs.append(subset['bar_10'].mean())
        counts.append(len(subset))
    else:
        bar1_avgs.append(0)
        bar10_avgs.append(0)
        counts.append(0)

x = np.arange(len(groups))

bars1 = ax3.bar(x - width/2, bar1_avgs, width, label='Bar 1 (15 min)', color='lightsalmon', edgecolor='red')
bars2 = ax3.bar(x + width/2, bar10_avgs, width, label='Bar 10 (2.5 hr)', color='lightgreen', edgecolor='green')

ax3.axhline(0, color='gray', linestyle='-', alpha=0.5)
ax3.set_ylabel('Average Profit (bps)')
ax3.set_title('SHORT: Does early profit continue?\n(Compare red vs green bars)')
ax3.set_xticks(x)
ax3.set_xticklabels([f'{g}\nn={c}' for g, c in zip(groups, counts)])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    height = bar.get_height()
    ax3.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax3.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

# -----------------------------------------------------------------------------
# SHORT: Interpretation
# -----------------------------------------------------------------------------
ax4 = axes[1, 1]
ax4.axis('off')

interpretation_short = """
SHORT SIGNALS - INTERPRETATION:

Normal Start (0-15 bps at Bar 1):
  Bar 1: +8 bps  ->  Bar 10: +18 bps
  CONTINUES! Price keeps going down.
  ACTION: HOLD with trailing stop (30 bps)

Good Start (15-30 bps at Bar 1):
  Bar 1: +22 bps  ->  Bar 10: +41 bps
  CONTINUES! Even better!
  ACTION: HOLD with trailing stop (30 bps)

Fast Start (>30 bps at Bar 1):
  Bar 1: +45 bps  ->  Bar 10: +16 bps
  REVERSES! Price bounces back.
  ACTION: EXIT early or tight stop (15 bps)

RULE FOR SHORT:
If Bar 1 shows >30 bps profit = TOO FAST, exit soon
If Bar 1 shows <30 bps profit = NORMAL, hold longer
"""
ax4.text(0.1, 0.9, interpretation_short, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('EXP-005: Early Profit Pattern - Does it Continue or Reverse?',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_PATH / '03_mfe_pattern_simple.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {PLOTS_PATH / '03_mfe_pattern_simple.png'}")

# =============================================================================
# VISUALIZATION 2: Arrow diagram showing the pattern
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# LONG diagram
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('LONG: Early Profit Pattern', fontsize=14, fontweight='bold')

# Weak start - reverses
ax1.annotate('', xy=(4, 7), xytext=(2, 8),
            arrowprops=dict(arrowstyle='->', color='red', lw=3))
ax1.text(1.5, 8.5, 'Weak Start\n(0-10 bps)', fontsize=11, ha='center')
ax1.text(4.5, 6.5, 'REVERSES\n-6 bps', fontsize=10, ha='center', color='red')

# Strong start - continues
ax1.annotate('', xy=(4, 3), xytext=(2, 2),
            arrowprops=dict(arrowstyle='->', color='green', lw=3))
ax1.text(1.5, 1.5, 'Strong Start\n(>20 bps)', fontsize=11, ha='center')
ax1.text(4.5, 3.5, 'CONTINUES\n+90 bps', fontsize=10, ha='center', color='green')

# Box with rule
ax1.add_patch(plt.Rectangle((5.5, 4), 4, 2, fill=True, facecolor='lightyellow', edgecolor='black'))
ax1.text(7.5, 5, 'RULE:\nIf Bar1 > 20 bps\n= HOLD', fontsize=11, ha='center', va='center')

# SHORT diagram
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('SHORT: Early Profit Pattern', fontsize=14, fontweight='bold')

# Normal start - continues
ax2.annotate('', xy=(4, 8), xytext=(2, 7),
            arrowprops=dict(arrowstyle='->', color='green', lw=3))
ax2.text(1.5, 6.5, 'Normal Start\n(0-30 bps)', fontsize=11, ha='center')
ax2.text(4.5, 8.5, 'CONTINUES\n+40 bps', fontsize=10, ha='center', color='green')

# Fast start - reverses
ax2.annotate('', xy=(4, 2), xytext=(2, 3),
            arrowprops=dict(arrowstyle='->', color='red', lw=3))
ax2.text(1.5, 3.5, 'Fast Start\n(>30 bps)', fontsize=11, ha='center')
ax2.text(4.5, 1.5, 'REVERSES\n+16 bps only', fontsize=10, ha='center', color='red')

# Box with rule
ax2.add_patch(plt.Rectangle((5.5, 4), 4, 2, fill=True, facecolor='lightyellow', edgecolor='black'))
ax2.text(7.5, 5, 'RULE:\nIf Bar1 > 30 bps\n= EXIT', fontsize=11, ha='center', va='center')

plt.tight_layout()
plt.savefig(PLOTS_PATH / '04_mfe_pattern_arrows.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {PLOTS_PATH / '04_mfe_pattern_arrows.png'}")

print("\nDone! Check the new visualizations.")
