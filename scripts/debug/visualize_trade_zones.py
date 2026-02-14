"""
Visualize Trade Zones - Winning vs Losing

Shows the forward price paths after entry:
- Green: Trades that hit TP (winning zone)
- Red: Trades that went into drawdown and never recovered (losing zone)

This helps identify patterns for optimization.

Run: python scripts/debug/visualize_trade_zones.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numba import njit
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("VISUALIZING TRADE ZONES - Winning vs Losing")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
TAKE_PROFIT_BPS = 12
MAX_HOLDING_BARS = 10
SAMPLE_SIZE = 1000  # Sample for visualization (too many trades to plot all)
SAMPLE_START = "2024-01-01"
SAMPLE_END = "2024-01-07"

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

# Filter sample period
data = ohlcv[(ohlcv.index >= SAMPLE_START) & (ohlcv.index < SAMPLE_END)].copy()
print(f"Sample period: {data.index[0]} to {data.index[-1]}")
print(f"Total candles: {len(data)}")

# =============================================================================
# CALCULATE INDICATORS
# =============================================================================
print("\n[2/4] Calculating indicators and finding entries...")
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

# Calculate forward MFE
@njit
def calculate_forward_mfe(highs, lows, close_prices, horizon):
    n = len(close_prices)
    mfe_long = np.full(n, np.nan)
    mfe_short = np.full(n, np.nan)

    for i in range(n - horizon):
        entry = close_prices[i]
        future_highs = highs[i+1:i+1+horizon]
        future_lows = lows[i+1:i+1+horizon]

        # LONG
        max_high = np.max(future_highs)
        mfe_long[i] = (max_high - entry) / entry * 10000

        # SHORT
        min_low = np.min(future_lows)
        mfe_short[i] = (entry - min_low) / entry * 10000

    return mfe_long, mfe_short

mfe_long, mfe_short = calculate_forward_mfe(
    data['high'].values,
    data['low'].values,
    data['close'].values,
    MAX_HOLDING_BARS
)

data['mfe_long'] = mfe_long
data['mfe_short'] = mfe_short

# Filter entries
long_entries = data[(data['entry_direction'] == 1)].dropna(subset=['mfe_long']).copy()
short_entries = data[(data['entry_direction'] == -1)].dropna(subset=['mfe_short']).copy()

# Determine success
long_entries['success'] = long_entries['mfe_long'] >= TAKE_PROFIT_BPS
short_entries['success'] = short_entries['mfe_short'] >= TAKE_PROFIT_BPS

print(f"\nLONG entries: {len(long_entries)} ({long_entries['success'].sum()} winners)")
print(f"SHORT entries: {len(short_entries)} ({short_entries['success'].sum()} winners)")

# Sample for visualization
if len(long_entries) > SAMPLE_SIZE // 2:
    long_sample = long_entries.sample(n=SAMPLE_SIZE // 2, random_state=42)
else:
    long_sample = long_entries

if len(short_entries) > SAMPLE_SIZE // 2:
    short_sample = short_entries.sample(n=SAMPLE_SIZE // 2, random_state=42)
else:
    short_sample = short_entries

# =============================================================================
# CREATE VISUALIZATION
# =============================================================================
print("\n[3/4] Creating visualizations...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

fig.suptitle('Trade Zones Analysis: Winning vs Losing Paths\nEMA7 Approach Direction Strategy',
             fontsize=16, fontweight='bold')

# -----------------------------------------------------------------------------
# CHART 1: LONG Trades - Forward Paths
# -----------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])

for idx, row in long_sample.iterrows():
    touch_idx = data.index.get_loc(idx)
    if touch_idx + MAX_HOLDING_BARS >= len(data):
        continue

    future_data = data.iloc[touch_idx : touch_idx + MAX_HOLDING_BARS + 1]

    # Normalize to entry = 0
    entry_price = row['close']
    normalized_prices = (future_data['close'] - entry_price) / entry_price * 10000

    color = 'green' if row['success'] else 'red'
    alpha = 0.3 if row['success'] else 0.2
    linewidth = 0.8 if row['success'] else 0.5

    ax1.plot(range(len(normalized_prices)), normalized_prices,
            color=color, alpha=alpha, linewidth=linewidth)

# TP line
ax1.axhline(y=TAKE_PROFIT_BPS, color='green', linestyle='--', linewidth=2, label=f'TP: {TAKE_PROFIT_BPS}bp', alpha=0.8)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

ax1.set_xlabel('Bars After Entry', fontsize=11)
ax1.set_ylabel('P&L (bps)', fontsize=11)
ax1.set_title(f'LONG Trades - Forward Price Paths (n={len(long_sample)})', fontsize=13, fontweight='bold', color='darkgreen')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

textstr = f"Win rate: {long_entries['success'].mean()*100:.1f}%\nWinners: {long_entries['success'].sum()}\nLosers: {(~long_entries['success']).sum()}"
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# -----------------------------------------------------------------------------
# CHART 2: SHORT Trades - Forward Paths
# -----------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])

for idx, row in short_sample.iterrows():
    touch_idx = data.index.get_loc(idx)
    if touch_idx + MAX_HOLDING_BARS >= len(data):
        continue

    future_data = data.iloc[touch_idx : touch_idx + MAX_HOLDING_BARS + 1]

    # Normalize to entry = 0 (for SHORT, inverse direction)
    entry_price = row['close']
    normalized_prices = (entry_price - future_data['close']) / entry_price * 10000

    color = 'green' if row['success'] else 'red'
    alpha = 0.3 if row['success'] else 0.2
    linewidth = 0.8 if row['success'] else 0.5

    ax2.plot(range(len(normalized_prices)), normalized_prices,
            color=color, alpha=alpha, linewidth=linewidth)

# TP line
ax2.axhline(y=TAKE_PROFIT_BPS, color='green', linestyle='--', linewidth=2, label=f'TP: {TAKE_PROFIT_BPS}bp', alpha=0.8)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

ax2.set_xlabel('Bars After Entry', fontsize=11)
ax2.set_ylabel('P&L (bps)', fontsize=11)
ax2.set_title(f'SHORT Trades - Forward Price Paths (n={len(short_sample)})', fontsize=13, fontweight='bold', color='darkred')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

textstr = f"Win rate: {short_entries['success'].mean()*100:.1f}%\nWinners: {short_entries['success'].sum()}\nLosers: {(~short_entries['success']).sum()}"
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# -----------------------------------------------------------------------------
# CHART 3: Distribution of Final P&L (LONG)
# -----------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 0])

# Calculate final P&L for all LONG trades
long_final_pnl = []
for idx, row in long_entries.iterrows():
    touch_idx = data.index.get_loc(idx)
    if touch_idx + MAX_HOLDING_BARS >= len(data):
        continue

    entry_price = row['close']
    exit_price = data.iloc[touch_idx + MAX_HOLDING_BARS]['close']
    pnl = (exit_price - entry_price) / entry_price * 10000
    long_final_pnl.append(pnl)

ax3.hist(long_final_pnl, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax3.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
ax3.axvline(x=TAKE_PROFIT_BPS, color='green', linestyle='--', linewidth=2, label=f'TP: {TAKE_PROFIT_BPS}bp')
ax3.axvline(x=np.mean(long_final_pnl), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(long_final_pnl):.1f}bp')

ax3.set_xlabel('Final P&L (bps)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('LONG - Distribution of Final P&L @ Exit', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# CHART 4: Distribution of Final P&L (SHORT)
# -----------------------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 1])

# Calculate final P&L for all SHORT trades
short_final_pnl = []
for idx, row in short_entries.iterrows():
    touch_idx = data.index.get_loc(idx)
    if touch_idx + MAX_HOLDING_BARS >= len(data):
        continue

    entry_price = row['close']
    exit_price = data.iloc[touch_idx + MAX_HOLDING_BARS]['close']
    pnl = (entry_price - exit_price) / entry_price * 10000
    short_final_pnl.append(pnl)

ax4.hist(short_final_pnl, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
ax4.axvline(x=TAKE_PROFIT_BPS, color='green', linestyle='--', linewidth=2, label=f'TP: {TAKE_PROFIT_BPS}bp')
ax4.axvline(x=np.mean(short_final_pnl), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(short_final_pnl):.1f}bp')

ax4.set_xlabel('Final P&L (bps)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('SHORT - Distribution of Final P&L @ Exit', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# CHART 5: MFE Distribution (LONG)
# -----------------------------------------------------------------------------
ax5 = fig.add_subplot(gs[2, 0])

ax5.hist(long_entries['mfe_long'], bins=50, color='green', alpha=0.6, edgecolor='black', label='Winners')
ax5.axvline(x=TAKE_PROFIT_BPS, color='darkgreen', linestyle='--', linewidth=2, label=f'TP Threshold: {TAKE_PROFIT_BPS}bp')
ax5.axvline(x=long_entries['mfe_long'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean MFE: {long_entries["mfe_long"].mean():.1f}bp')

ax5.set_xlabel('MFE (bps)', fontsize=11)
ax5.set_ylabel('Frequency', fontsize=11)
ax5.set_title('LONG - Maximum Favorable Excursion Distribution', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# CHART 6: MFE Distribution (SHORT)
# -----------------------------------------------------------------------------
ax6 = fig.add_subplot(gs[2, 1])

ax6.hist(short_entries['mfe_short'], bins=50, color='red', alpha=0.6, edgecolor='black', label='Winners')
ax6.axvline(x=TAKE_PROFIT_BPS, color='darkred', linestyle='--', linewidth=2, label=f'TP Threshold: {TAKE_PROFIT_BPS}bp')
ax6.axvline(x=short_entries['mfe_short'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean MFE: {short_entries["mfe_short"].mean():.1f}bp')

ax6.set_xlabel('MFE (bps)', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title('SHORT - Maximum Favorable Excursion Distribution', fontsize=13, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()

# Save
output_dir = Path("experiments/ema")
output_path = output_dir / "trade_zones_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] Saved visualization: {output_path}")

# =============================================================================
# DETAILED ANALYSIS
# =============================================================================
print("\n[4/4] Analyzing patterns...")

print("\n--- LONG TRADES ANALYSIS ---")
print(f"Mean MFE: {long_entries['mfe_long'].mean():.1f} bps")
print(f"Median MFE: {long_entries['mfe_long'].median():.1f} bps")
print(f"MFE >= TP: {(long_entries['mfe_long'] >= TAKE_PROFIT_BPS).sum()} ({(long_entries['mfe_long'] >= TAKE_PROFIT_BPS).mean()*100:.1f}%)")
print(f"Mean final P&L: {np.mean(long_final_pnl):.1f} bps")
print(f"Median final P&L: {np.median(long_final_pnl):.1f} bps")

print("\n--- SHORT TRADES ANALYSIS ---")
print(f"Mean MFE: {short_entries['mfe_short'].mean():.1f} bps")
print(f"Median MFE: {short_entries['mfe_short'].median():.1f} bps")
print(f"MFE >= TP: {(short_entries['mfe_short'] >= TAKE_PROFIT_BPS).sum()} ({(short_entries['mfe_short'] >= TAKE_PROFIT_BPS).mean()*100:.1f}%)")
print(f"Mean final P&L: {np.mean(short_final_pnl):.1f} bps")
print(f"Median final P&L: {np.median(short_final_pnl):.1f} bps")

print("\n--- KEY INSIGHTS ---")
print("1. Check forward paths: Do winning trades move quickly to TP?")
print("2. Check losing trades: Do they immediately reverse or struggle?")
print("3. MFE vs Final P&L: Large gap = trades reach profit then reverse")
print("4. Distribution shape: Symmetric? Skewed? Bimodal?")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
