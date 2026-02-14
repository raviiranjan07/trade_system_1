"""
SIMPLE EXPLANATION: How 200 SMA saved from the Account Killer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
PLOTS_PATH = Path("experiments/rsi/EXP-004/plots")

print("="*70)
print("HOW 200 SMA SAVED FROM ACCOUNT KILLER - SIMPLE EXPLANATION")
print("="*70)

# Load data
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)
df['sma_200'] = df['close'].rolling(200).mean()

# Account killer date
killer_date = pd.Timestamp('2024-02-26 15:00:00')

# Get data around the killer
zoom_start = pd.Timestamp('2024-02-01')
zoom_end = pd.Timestamp('2024-04-15')
zoom_data = df[zoom_start:zoom_end].copy()

# Get the killer signal data
killer_idx = df.index.get_loc(killer_date)
killer_price = df.iloc[killer_idx]['close']
killer_sma200 = df.iloc[killer_idx]['sma_200']

print(f"\nACCOUNT KILLER: 2024-02-26")
print(f"  Entry Price: ${killer_price:,.0f}")
print(f"  SMA 200: ${killer_sma200:,.0f}")
print(f"  Price vs SMA200: {(killer_price - killer_sma200) / killer_sma200 * 100:+.2f}%")
print(f"\n  Price > SMA200? {killer_price > killer_sma200}")
print(f"  Market: {'BULL' if killer_price > killer_sma200 else 'BEAR'}")

# Track what happened after
future_high = df.iloc[killer_idx:killer_idx+500]['high'].max()
loss_if_traded = (future_high - killer_price) / killer_price * 10000

print(f"\n  If we took this SHORT trade:")
print(f"    Price went to: ${future_high:,.0f}")
print(f"    Loss: {loss_if_traded:.0f} bps ({loss_if_traded/100:.1f}%)")

# =============================================================================
# CREATE THE EXPLANATION CHART
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# =============================================================================
# CHART 1: The situation at the killer signal
# =============================================================================
ax1 = axes[0]

ax1.plot(zoom_data.index, zoom_data['close'], 'b-', linewidth=2, label='BTC Price')
ax1.plot(zoom_data.index, zoom_data['sma_200'], 'orange', linewidth=3, label='SMA 200')

# Mark the killer signal
ax1.scatter(killer_date, killer_price, color='red', marker='v', s=400,
            edgecolors='yellow', linewidths=3, zorder=10, label='SHORT Signal')

# Draw horizontal lines at killer price and SMA
ax1.axhline(y=killer_price, color='red', linestyle='--', alpha=0.5, label=f'Entry: ${killer_price:,.0f}')
ax1.axhline(y=killer_sma200, color='orange', linestyle='--', alpha=0.5, label=f'SMA200: ${killer_sma200:,.0f}')

# Annotate
ax1.annotate(f'RSI said "OVERBOUGHT"\n→ SHORT signal\n\nPrice: ${killer_price:,.0f}\nSMA200: ${killer_sma200:,.0f}\n\nPrice > SMA200 = BULL MARKET',
            xy=(killer_date, killer_price),
            xytext=(killer_date - pd.Timedelta(days=15), killer_price * 1.05),
            fontsize=11, color='red',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='red'),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax1.set_title('STEP 1: RSI gives SHORT signal on Feb 26, 2024\nBUT price is ABOVE SMA200 (BULL market)', fontsize=14)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylabel('Price ($)')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# =============================================================================
# CHART 2: The rule check
# =============================================================================
ax2 = axes[1]

ax2.plot(zoom_data.index, zoom_data['close'], 'b-', linewidth=2, label='BTC Price')
ax2.plot(zoom_data.index, zoom_data['sma_200'], 'orange', linewidth=3, label='SMA 200')

# Shade above SMA (BULL zone)
ax2.fill_between(zoom_data.index, zoom_data['sma_200'], zoom_data['close'],
                 where=zoom_data['close'] > zoom_data['sma_200'],
                 alpha=0.3, color='green', label='BULL zone (price > SMA)')

# Mark killer
ax2.scatter(killer_date, killer_price, color='red', marker='v', s=400,
            edgecolors='yellow', linewidths=3, zorder=10)

# Big X to show SKIP
ax2.annotate('✗ SKIP!', xy=(killer_date, killer_price * 1.02),
            fontsize=20, color='green', fontweight='bold', ha='center')

# Rule box
ax2.text(0.5, 0.95,
         'RULE: SHORT only when price < SMA200 (BEAR market)\n'
         'CHECK: Price ($51,926) > SMA200 ($51,512)\n'
         'RESULT: ✗ SKIP THIS TRADE',
         transform=ax2.transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', alpha=0.9))

ax2.set_title('STEP 2: Apply the 200 SMA filter rule\nPrice is in BULL zone → SKIP the SHORT signal', fontsize=14)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_ylabel('Price ($)')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# =============================================================================
# CHART 3: What would have happened
# =============================================================================
ax3 = axes[2]

# Extended view to show the disaster
extended_end = pd.Timestamp('2024-04-30')
extended_data = df[zoom_start:extended_end].copy()

ax3.plot(extended_data.index, extended_data['close'], 'b-', linewidth=2, label='BTC Price')

# Mark entry and max loss
ax3.scatter(killer_date, killer_price, color='red', marker='v', s=400,
            edgecolors='yellow', linewidths=3, zorder=10, label='Entry if NOT filtered')

ax3.axhline(y=killer_price, color='red', linestyle='--', alpha=0.5)

# Find max price (worst case for SHORT)
max_price_idx = extended_data['high'].idxmax()
max_price = extended_data['high'].max()

ax3.scatter(max_price_idx, max_price, color='darkred', marker='x', s=300, zorder=10)
ax3.annotate(f'MAX LOSS: ${max_price:,.0f}\n-{loss_if_traded:.0f} bps ({loss_if_traded/100:.1f}%)',
            xy=(max_price_idx, max_price),
            xytext=(max_price_idx + pd.Timedelta(days=5), max_price * 0.98),
            fontsize=12, color='darkred', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=2))

# Shade the loss zone
ax3.fill_between(extended_data.index, killer_price, extended_data['close'],
                 where=extended_data['close'] > killer_price,
                 alpha=0.3, color='red', label='LOSS ZONE if traded')

# Add outcome text
ax3.text(0.5, 0.95,
         f'IF WE IGNORED THE FILTER AND TOOK THE SHORT:\n'
         f'Entry: ${killer_price:,.0f} → Price went to ${max_price:,.0f}\n'
         f'LOSS: {loss_if_traded:.0f} bps = {loss_if_traded/100:.1f}% = ACCOUNT KILLER\n\n'
         f'BUT WE SKIPPED IT → SAVED!',
         transform=ax3.transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='red', alpha=0.9))

ax3.set_title('STEP 3: What would have happened without the filter\nBTC ETF rally caused massive loss - but we SKIPPED it!', fontsize=14)
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)
ax3.set_ylabel('Price ($)')
ax3.set_xlabel('Date')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.tight_layout()
plt.savefig(PLOTS_PATH / '17_how_sma_saved_explained.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {PLOTS_PATH / '17_how_sma_saved_explained.png'}")

# =============================================================================
# SIMPLE TEXT SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SIMPLE EXPLANATION")
print("="*70)
print("""
THE ACCOUNT KILLER (Feb 26, 2024):

1. RSI crossed 80 (overbought) → Signal says "SHORT"

2. CHECK THE RULE:
   - Price: $51,926
   - SMA200: $51,512
   - Price > SMA200? YES → This is BULL market

3. THE RULE SAYS:
   "Only SHORT when price < SMA200 (BEAR market)"

4. DECISION: SKIP this SHORT signal (we're in BULL market)

5. WHAT HAPPENED NEXT:
   - BTC ETF rally started
   - Price went from $52k → $73k
   - If we had taken the SHORT: -4,000+ bps loss (40%!)

6. BECAUSE WE SKIPPED: $0 loss

That's how SMA200 saved from the account killer.
The signal looked normal, but the filter said "wrong market condition" → SKIP.
""")
