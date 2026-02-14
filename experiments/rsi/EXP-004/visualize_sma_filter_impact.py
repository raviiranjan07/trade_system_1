"""
EXP-004: Visualize 200 SMA Filter Impact
Shows how many signals are kept vs skipped
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
PLOTS_PATH = Path("experiments/rsi/EXP-004/plots")

RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
HORIZON = 10

print("="*70)
print("200 SMA FILTER IMPACT VISUALIZATION")
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

# Count signals by market condition
def count_signals_by_market(data, signal_col, signal_type):
    signals = data[data[signal_col] == True].copy()

    bull_success = 0
    bull_fail = 0
    bear_success = 0
    bear_fail = 0

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
        else:
            mfe = (entry_price - future_data['low'].min()) / entry_price * 10000

        is_bull = entry_price > sma_200
        is_success = mfe > 0

        if is_bull and is_success:
            bull_success += 1
        elif is_bull and not is_success:
            bull_fail += 1
        elif not is_bull and is_success:
            bear_success += 1
        else:
            bear_fail += 1

    return {
        'bull_success': bull_success,
        'bull_fail': bull_fail,
        'bear_success': bear_success,
        'bear_fail': bear_fail
    }

long_counts = count_signals_by_market(oos_data, 'rsi_oversold_cross', 'long')
short_counts = count_signals_by_market(oos_data, 'rsi_overbought_cross', 'short')

print("\nLONG signals breakdown:")
print(f"  BULL market: {long_counts['bull_success']} success + {long_counts['bull_fail']} fail = {long_counts['bull_success'] + long_counts['bull_fail']}")
print(f"  BEAR market: {long_counts['bear_success']} success + {long_counts['bear_fail']} fail = {long_counts['bear_success'] + long_counts['bear_fail']}")

print("\nSHORT signals breakdown:")
print(f"  BULL market: {short_counts['bull_success']} success + {short_counts['bull_fail']} fail = {short_counts['bull_success'] + short_counts['bull_fail']}")
print(f"  BEAR market: {short_counts['bear_success']} success + {short_counts['bear_fail']} fail = {short_counts['bear_success'] + short_counts['bear_fail']}")

# =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: LONG signals breakdown
ax1 = axes[0, 0]
categories = ['BULL\n(KEEP)', 'BEAR\n(SKIP)']
success = [long_counts['bull_success'], long_counts['bear_success']]
fail = [long_counts['bull_fail'], long_counts['bear_fail']]

x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, success, width, label='Success', color='green', alpha=0.7)
bars2 = ax1.bar(x + width/2, fail, width, label='Failure', color='red', alpha=0.7)

ax1.set_ylabel('Number of Signals')
ax1.set_title('LONG Signals by Market Condition\n(With 200 SMA Filter)')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)

# Plot 2: SHORT signals breakdown
ax2 = axes[0, 1]
categories = ['BULL\n(SKIP)', 'BEAR\n(KEEP)']
success = [short_counts['bull_success'], short_counts['bear_success']]
fail = [short_counts['bull_fail'], short_counts['bear_fail']]

bars1 = ax2.bar(x - width/2, success, width, label='Success', color='green', alpha=0.7)
bars2 = ax2.bar(x + width/2, fail, width, label='Failure', color='red', alpha=0.7)

ax2.set_ylabel('Number of Signals')
ax2.set_title('SHORT Signals by Market Condition\n(With 200 SMA Filter)')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    height = bar.get_height()
    ax2.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)

# Plot 3: Total signals - Before vs After filter
ax3 = axes[1, 0]

# Calculate totals
total_long = long_counts['bull_success'] + long_counts['bull_fail'] + long_counts['bear_success'] + long_counts['bear_fail']
total_short = short_counts['bull_success'] + short_counts['bull_fail'] + short_counts['bear_success'] + short_counts['bear_fail']

# After filter
long_kept = long_counts['bull_success'] + long_counts['bull_fail']
long_skipped = long_counts['bear_success'] + long_counts['bear_fail']
short_kept = short_counts['bear_success'] + short_counts['bear_fail']
short_skipped = short_counts['bull_success'] + short_counts['bull_fail']

categories = ['LONG', 'SHORT', 'TOTAL']
before = [total_long, total_short, total_long + total_short]
after = [long_kept, short_kept, long_kept + short_kept]
skipped = [long_skipped, short_skipped, long_skipped + short_skipped]

x = np.arange(len(categories))
width = 0.25

bars1 = ax3.bar(x - width, before, width, label='Before Filter', color='blue', alpha=0.7)
bars2 = ax3.bar(x, after, width, label='After Filter (KEPT)', color='green', alpha=0.7)
bars3 = ax3.bar(x + width, skipped, width, label='Skipped', color='gray', alpha=0.7)

ax3.set_ylabel('Number of Signals')
ax3.set_title('Signal Count: Before vs After 200 SMA Filter')
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    height = bar.get_height()
    ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for bar in bars3:
    height = bar.get_height()
    ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

# Plot 4: Failures - Before vs After filter
ax4 = axes[1, 1]

# Failures before
long_fail_before = long_counts['bull_fail'] + long_counts['bear_fail']
short_fail_before = short_counts['bull_fail'] + short_counts['bear_fail']

# Failures after (only in kept signals)
long_fail_after = long_counts['bull_fail']  # Only BULL LONG kept
short_fail_after = short_counts['bear_fail']  # Only BEAR SHORT kept

categories = ['LONG', 'SHORT', 'TOTAL']
fail_before = [long_fail_before, short_fail_before, long_fail_before + short_fail_before]
fail_after = [long_fail_after, short_fail_after, long_fail_after + short_fail_after]

x = np.arange(len(categories))
width = 0.35

bars1 = ax4.bar(x - width/2, fail_before, width, label='Failures Before', color='red', alpha=0.7)
bars2 = ax4.bar(x + width/2, fail_after, width, label='Failures After', color='orange', alpha=0.7)

ax4.set_ylabel('Number of Failures')
ax4.set_title('Failures: Before vs After 200 SMA Filter')
ax4.set_xticks(x)
ax4.set_xticklabels(categories)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    height = bar.get_height()
    ax4.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
for bar in bars2:
    height = bar.get_height()
    ax4.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(PLOTS_PATH / '13_sma_filter_impact.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {PLOTS_PATH / '13_sma_filter_impact.png'}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"""
                    BEFORE FILTER    AFTER FILTER    SKIPPED
LONG signals:       {total_long:>10}       {long_kept:>10}      {long_skipped:>10}
SHORT signals:      {total_short:>10}       {short_kept:>10}      {short_skipped:>10}
------------------------------------------------------------
TOTAL signals:      {total_long + total_short:>10}       {long_kept + short_kept:>10}      {long_skipped + short_skipped:>10}

FAILURES:
LONG failures:      {long_fail_before:>10}       {long_fail_after:>10}      {long_fail_before - long_fail_after:>10} avoided
SHORT failures:     {short_fail_before:>10}       {short_fail_after:>10}      {short_fail_before - short_fail_after:>10} avoided
------------------------------------------------------------
TOTAL failures:     {long_fail_before + short_fail_before:>10}       {long_fail_after + short_fail_after:>10}      {(long_fail_before + short_fail_before) - (long_fail_after + short_fail_after):>10} avoided
""")

# Calculate percentages
signals_kept_pct = (long_kept + short_kept) / (total_long + total_short) * 100
failures_avoided_pct = ((long_fail_before + short_fail_before) - (long_fail_after + short_fail_after)) / (long_fail_before + short_fail_before) * 100

print(f"Signals kept: {signals_kept_pct:.1f}%")
print(f"Failures avoided: {failures_avoided_pct:.1f}%")
