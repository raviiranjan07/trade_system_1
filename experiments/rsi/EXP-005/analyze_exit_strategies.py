"""
EXP-005: Exit Strategy Analysis
1. Trailing Stop Test (10, 15, 20, 25 bps)
2. MFE vs MAE Relationship (Bar 1 vs 3 vs 5 vs 10)
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

print("="*70)
print("EXP-005: EXIT STRATEGY ANALYSIS")
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

# =============================================================================
# PART 1: TRAILING STOP TEST
# =============================================================================

print("\n" + "="*70)
print("PART 1: TRAILING STOP TEST")
print("="*70)

def simulate_trailing_stop(data, signal_col, signal_type, trailing_stop_bps, filter_by_sma=True):
    """
    Simulate trading with a trailing stop.
    Returns list of exit profits for each trade.
    """
    signals = data[data[signal_col] == True].copy()

    results = []

    for idx in signals.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + HORIZON >= len(data):
            continue

        entry_price = data.iloc[idx_pos]['close']
        sma_200 = data.iloc[idx_pos]['sma_200']

        # Apply 200 SMA filter
        if filter_by_sma:
            if signal_type == 'long' and entry_price < sma_200:
                continue
            if signal_type == 'short' and entry_price > sma_200:
                continue

        # Simulate bar by bar with trailing stop
        highest_profit = 0
        exit_profit = None
        exit_bar = None

        for bar in range(1, HORIZON + 1):
            future_bar = data.iloc[idx_pos + bar]

            if signal_type == 'long':
                # For LONG: profit when price goes up
                current_high = (future_bar['high'] - entry_price) / entry_price * 10000
                current_close = (future_bar['close'] - entry_price) / entry_price * 10000
            else:
                # For SHORT: profit when price goes down
                current_high = (entry_price - future_bar['low']) / entry_price * 10000
                current_close = (entry_price - future_bar['close']) / entry_price * 10000

            # Update highest profit seen
            if current_high > highest_profit:
                highest_profit = current_high

            # Check if trailing stop hit (price dropped trailing_stop_bps from peak)
            drawdown_from_peak = highest_profit - current_close

            if drawdown_from_peak >= trailing_stop_bps and highest_profit > 0:
                # Trailing stop hit - exit at (peak - trailing_stop)
                exit_profit = highest_profit - trailing_stop_bps
                exit_bar = bar
                break

        # If no trailing stop hit, exit at Bar 10 close
        if exit_profit is None:
            if signal_type == 'long':
                exit_profit = (data.iloc[idx_pos + HORIZON]['close'] - entry_price) / entry_price * 10000
            else:
                exit_profit = (entry_price - data.iloc[idx_pos + HORIZON]['close']) / entry_price * 10000
            exit_bar = HORIZON

        results.append({
            'exit_profit': exit_profit,
            'exit_bar': exit_bar,
            'max_profit': highest_profit
        })

    return results

# Test different trailing stop sizes
trailing_stops = [10, 15, 20, 25, 30]

print("\nLONG signals (BULL only) - Trailing Stop Results:")
print("-" * 70)

long_ts_results = {}
for ts in trailing_stops:
    results = simulate_trailing_stop(oos_data, 'rsi_oversold_cross', 'long', ts)
    profits = [r['exit_profit'] for r in results]
    bars = [r['exit_bar'] for r in results]
    max_profits = [r['max_profit'] for r in results]

    avg_profit = np.mean(profits)
    avg_bar = np.mean(bars)
    avg_max = np.mean(max_profits)
    profit_captured = avg_profit / avg_max * 100 if avg_max > 0 else 0

    long_ts_results[ts] = {
        'avg_profit': avg_profit,
        'avg_bar': avg_bar,
        'avg_max': avg_max,
        'profit_captured': profit_captured,
        'count': len(results)
    }

    print(f"  TS={ts:2d} bps: Avg Exit={avg_profit:+6.1f} bps, Avg Bar={avg_bar:.1f}, Max Possible={avg_max:.1f} bps, Captured={profit_captured:.0f}%")

print("\nSHORT signals (BEAR only) - Trailing Stop Results:")
print("-" * 70)

short_ts_results = {}
for ts in trailing_stops:
    results = simulate_trailing_stop(oos_data, 'rsi_overbought_cross', 'short', ts)
    profits = [r['exit_profit'] for r in results]
    bars = [r['exit_bar'] for r in results]
    max_profits = [r['max_profit'] for r in results]

    avg_profit = np.mean(profits)
    avg_bar = np.mean(bars)
    avg_max = np.mean(max_profits)
    profit_captured = avg_profit / avg_max * 100 if avg_max > 0 else 0

    short_ts_results[ts] = {
        'avg_profit': avg_profit,
        'avg_bar': avg_bar,
        'avg_max': avg_max,
        'profit_captured': profit_captured,
        'count': len(results)
    }

    print(f"  TS={ts:2d} bps: Avg Exit={avg_profit:+6.1f} bps, Avg Bar={avg_bar:.1f}, Max Possible={avg_max:.1f} bps, Captured={profit_captured:.0f}%")

# Compare with NO trailing stop (just exit at Bar 10)
print("\nComparison - NO Trailing Stop (Exit at Bar 10):")
print("-" * 70)

# LONG no trailing stop
results_no_ts = simulate_trailing_stop(oos_data, 'rsi_oversold_cross', 'long', 9999)  # Very large = never triggers
profits_no_ts = [r['exit_profit'] for r in results_no_ts]
print(f"  LONG: Avg Exit={np.mean(profits_no_ts):+6.1f} bps (no trailing stop)")

# SHORT no trailing stop
results_no_ts = simulate_trailing_stop(oos_data, 'rsi_overbought_cross', 'short', 9999)
profits_no_ts = [r['exit_profit'] for r in results_no_ts]
print(f"  SHORT: Avg Exit={np.mean(profits_no_ts):+6.1f} bps (no trailing stop)")

# =============================================================================
# PART 2: MFE vs MAE RELATIONSHIP
# =============================================================================

print("\n" + "="*70)
print("PART 2: MFE vs MAE RELATIONSHIP (Bar 1 vs 3 vs 5 vs 10)")
print("="*70)

def analyze_early_profit_continuation(data, signal_col, signal_type, filter_by_sma=True):
    """
    Group trades by Bar 1 profit and see what happens at Bar 3, 5, 10.
    """
    signals = data[data[signal_col] == True].copy()

    results = []

    for idx in signals.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + HORIZON >= len(data):
            continue

        entry_price = data.iloc[idx_pos]['close']
        sma_200 = data.iloc[idx_pos]['sma_200']

        # Apply 200 SMA filter
        if filter_by_sma:
            if signal_type == 'long' and entry_price < sma_200:
                continue
            if signal_type == 'short' and entry_price > sma_200:
                continue

        # Get MFE at each bar
        bar_mfes = {}
        for bar in [1, 3, 5, 10]:
            if idx_pos + bar >= len(data):
                continue
            future_bar = data.iloc[idx_pos + bar]

            if signal_type == 'long':
                mfe = (future_bar['high'] - entry_price) / entry_price * 10000
            else:
                mfe = (entry_price - future_bar['low']) / entry_price * 10000

            bar_mfes[f'bar_{bar}'] = mfe

        if len(bar_mfes) == 4:
            results.append(bar_mfes)

    return pd.DataFrame(results)

# Analyze LONG signals
long_df = analyze_early_profit_continuation(oos_data, 'rsi_oversold_cross', 'long')

print(f"\nLONG signals (BULL only): {len(long_df)} trades")
print("-" * 70)

# Group by Bar 1 profit level
long_df['bar1_group'] = pd.cut(long_df['bar_1'],
                                bins=[-100, 0, 10, 20, 30, 100],
                                labels=['<0', '0-10', '10-20', '20-30', '>30'])

print("\nBar 1 Profit -> What happens at later bars?")
print("| Bar 1 Profit | Count | Bar 3 Avg | Bar 5 Avg | Bar 10 Avg | Pattern |")
print("|--------------|-------|-----------|-----------|------------|---------|")

for group in ['<0', '0-10', '10-20', '20-30', '>30']:
    subset = long_df[long_df['bar1_group'] == group]
    if len(subset) > 0:
        bar3_avg = subset['bar_3'].mean()
        bar5_avg = subset['bar_5'].mean()
        bar10_avg = subset['bar_10'].mean()
        bar1_avg = subset['bar_1'].mean()

        # Determine pattern
        if bar10_avg > bar1_avg * 1.2:
            pattern = "CONTINUES"
        elif bar10_avg < bar1_avg * 0.8:
            pattern = "REVERSES"
        else:
            pattern = "FLAT"

        print(f"| {group:12s} | {len(subset):5d} | {bar3_avg:+9.1f} | {bar5_avg:+9.1f} | {bar10_avg:+10.1f} | {pattern:7s} |")

# Analyze SHORT signals
short_df = analyze_early_profit_continuation(oos_data, 'rsi_overbought_cross', 'short')

print(f"\nSHORT signals (BEAR only): {len(short_df)} trades")
print("-" * 70)

short_df['bar1_group'] = pd.cut(short_df['bar_1'],
                                 bins=[-100, 0, 10, 20, 30, 100],
                                 labels=['<0', '0-10', '10-20', '20-30', '>30'])

print("\nBar 1 Profit -> What happens at later bars?")
print("| Bar 1 Profit | Count | Bar 3 Avg | Bar 5 Avg | Bar 10 Avg | Pattern |")
print("|--------------|-------|-----------|-----------|------------|---------|")

for group in ['<0', '0-10', '10-20', '20-30', '>30']:
    subset = short_df[short_df['bar1_group'] == group]
    if len(subset) > 0:
        bar3_avg = subset['bar_3'].mean()
        bar5_avg = subset['bar_5'].mean()
        bar10_avg = subset['bar_10'].mean()
        bar1_avg = subset['bar_1'].mean()

        # Determine pattern
        if bar10_avg > bar1_avg * 1.2:
            pattern = "CONTINUES"
        elif bar10_avg < bar1_avg * 0.8:
            pattern = "REVERSES"
        else:
            pattern = "FLAT"

        print(f"| {group:12s} | {len(subset):5d} | {bar3_avg:+9.1f} | {bar5_avg:+9.1f} | {bar10_avg:+10.1f} | {pattern:7s} |")

# =============================================================================
# VISUALIZATION 1: Trailing Stop Comparison
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# LONG trailing stop results
ax1 = axes[0]
ts_values = list(long_ts_results.keys())
avg_profits = [long_ts_results[ts]['avg_profit'] for ts in ts_values]
profit_captured = [long_ts_results[ts]['profit_captured'] for ts in ts_values]

x = np.arange(len(ts_values))
width = 0.35

bars1 = ax1.bar(x - width/2, avg_profits, width, label='Avg Exit Profit (bps)', color='green', alpha=0.7)
ax1_twin = ax1.twinx()
bars2 = ax1_twin.bar(x + width/2, profit_captured, width, label='% of Max Captured', color='blue', alpha=0.7)

ax1.set_xlabel('Trailing Stop Size (bps)')
ax1.set_ylabel('Avg Exit Profit (bps)', color='green')
ax1_twin.set_ylabel('% of Max Profit Captured', color='blue')
ax1.set_title('LONG (BULL): Trailing Stop Performance')
ax1.set_xticks(x)
ax1.set_xticklabels(ts_values)
ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# SHORT trailing stop results
ax2 = axes[1]
avg_profits = [short_ts_results[ts]['avg_profit'] for ts in ts_values]
profit_captured = [short_ts_results[ts]['profit_captured'] for ts in ts_values]

bars1 = ax2.bar(x - width/2, avg_profits, width, label='Avg Exit Profit (bps)', color='red', alpha=0.7)
ax2_twin = ax2.twinx()
bars2 = ax2_twin.bar(x + width/2, profit_captured, width, label='% of Max Captured', color='blue', alpha=0.7)

ax2.set_xlabel('Trailing Stop Size (bps)')
ax2.set_ylabel('Avg Exit Profit (bps)', color='red')
ax2_twin.set_ylabel('% of Max Profit Captured', color='blue')
ax2.set_title('SHORT (BEAR): Trailing Stop Performance')
ax2.set_xticks(x)
ax2.set_xticklabels(ts_values)
ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.suptitle('EXP-005: Trailing Stop Test - Which size captures most profit?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_PATH / '01_trailing_stop_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {PLOTS_PATH / '01_trailing_stop_comparison.png'}")

# =============================================================================
# VISUALIZATION 2: MFE Continuation/Reversal Pattern
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# LONG MFE pattern
ax1 = axes[0]
groups = ['<0', '0-10', '10-20', '20-30', '>30']
colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']

for i, group in enumerate(groups):
    subset = long_df[long_df['bar1_group'] == group]
    if len(subset) > 5:
        bars = [1, 3, 5, 10]
        values = [subset['bar_1'].mean(), subset['bar_3'].mean(),
                  subset['bar_5'].mean(), subset['bar_10'].mean()]
        ax1.plot(bars, values, 'o-', linewidth=2, markersize=8,
                label=f'Bar1: {group} bps (n={len(subset)})', color=colors[i])

ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax1.set_xlabel('Bar Number')
ax1.set_ylabel('Average MFE (bps)')
ax1.set_title('LONG: Does early profit continue or reverse?')
ax1.set_xticks([1, 3, 5, 10])
ax1.legend()
ax1.grid(True, alpha=0.3)

# SHORT MFE pattern
ax2 = axes[1]

for i, group in enumerate(groups):
    subset = short_df[short_df['bar1_group'] == group]
    if len(subset) > 5:
        bars = [1, 3, 5, 10]
        values = [subset['bar_1'].mean(), subset['bar_3'].mean(),
                  subset['bar_5'].mean(), subset['bar_10'].mean()]
        ax2.plot(bars, values, 'o-', linewidth=2, markersize=8,
                label=f'Bar1: {group} bps (n={len(subset)})', color=colors[i])

ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax2.set_xlabel('Bar Number')
ax2.set_ylabel('Average MFE (bps)')
ax2.set_title('SHORT: Does early profit continue or reverse?')
ax2.set_xticks([1, 3, 5, 10])
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('EXP-005: MFE vs MAE - Does early profit predict continuation?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_PATH / '02_mfe_continuation_pattern.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {PLOTS_PATH / '02_mfe_continuation_pattern.png'}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("EXP-005 SUMMARY")
print("="*70)

# Find best trailing stop
best_long_ts = max(long_ts_results.keys(), key=lambda x: long_ts_results[x]['avg_profit'])
best_short_ts = max(short_ts_results.keys(), key=lambda x: short_ts_results[x]['avg_profit'])

print(f"""
TRAILING STOP FINDINGS:
-----------------------
LONG (BULL):
  Best Trailing Stop: {best_long_ts} bps
  Avg Exit Profit: {long_ts_results[best_long_ts]['avg_profit']:+.1f} bps
  % of Max Captured: {long_ts_results[best_long_ts]['profit_captured']:.0f}%

SHORT (BEAR):
  Best Trailing Stop: {best_short_ts} bps
  Avg Exit Profit: {short_ts_results[best_short_ts]['avg_profit']:+.1f} bps
  % of Max Captured: {short_ts_results[best_short_ts]['profit_captured']:.0f}%

MFE CONTINUATION FINDINGS:
--------------------------
Check the tables above to see if:
- High early profit (>20 bps at Bar 1) continues or reverses
- Low early profit (<10 bps at Bar 1) eventually grows

This tells us whether to EXIT EARLY on quick profits or HOLD for more.
""")
