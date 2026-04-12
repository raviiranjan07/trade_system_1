"""
EXP-004: All Visualizations
Creates charts for every part of the experiment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
RESULTS_PATH = Path("experiments/rsi/EXP-004")
PLOTS_PATH = RESULTS_PATH / "plots"
PLOTS_PATH.mkdir(exist_ok=True)

RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
HORIZON = 10

print("="*70)
print("EXP-004: GENERATING ALL VISUALIZATIONS")
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
# PART 1: PATH ANALYSIS VISUALIZATION
# =============================================================================
print("\n[1/7] Creating Path Analysis visualization...")

def get_path_data(data, signal_col, signal_type):
    signals = data[data[signal_col] == True]
    paths_success = []
    paths_failure = []

    for idx in signals.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + HORIZON >= len(data):
            continue

        entry_price = data.iloc[idx_pos]['close']
        path = []

        for bar in range(1, HORIZON + 1):
            bar_data = data.iloc[idx_pos + bar]
            if signal_type == 'long':
                pnl = (bar_data['close'] - entry_price) / entry_price * 10000
            else:
                pnl = (entry_price - bar_data['close']) / entry_price * 10000
            path.append(pnl)

        # Check if success (MFE > 0)
        future_data = data.iloc[idx_pos + 1 : idx_pos + 1 + HORIZON]
        if signal_type == 'long':
            mfe = (future_data['high'].max() - entry_price) / entry_price * 10000
        else:
            mfe = (entry_price - future_data['low'].min()) / entry_price * 10000

        if mfe > 0:
            paths_success.append(path)
        else:
            paths_failure.append(path)

    return paths_success, paths_failure

# Get path data
long_success, long_failure = get_path_data(oos_data, 'rsi_oversold_cross', 'long')
short_success, short_failure = get_path_data(oos_data, 'rsi_overbought_cross', 'short')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# LONG paths
ax1 = axes[0]
if long_success:
    success_avg = np.mean(long_success, axis=0)
    ax1.plot(range(1, HORIZON + 1), success_avg, 'g-', linewidth=3, label=f'Success avg (n={len(long_success)})')
if long_failure:
    failure_avg = np.mean(long_failure, axis=0)
    ax1.plot(range(1, HORIZON + 1), failure_avg, 'r-', linewidth=3, label=f'Failure avg (n={len(long_failure)})')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.set_xlabel('Bar')
ax1.set_ylabel('PnL (bps)')
ax1.set_title('LONG (Oversold) - Path Analysis')
ax1.legend()
ax1.grid(True, alpha=0.3)

# SHORT paths
ax2 = axes[1]
if short_success:
    success_avg = np.mean(short_success, axis=0)
    ax2.plot(range(1, HORIZON + 1), success_avg, 'g-', linewidth=3, label=f'Success avg (n={len(short_success)})')
if short_failure:
    failure_avg = np.mean(short_failure, axis=0)
    ax2.plot(range(1, HORIZON + 1), failure_avg, 'r-', linewidth=3, label=f'Failure avg (n={len(short_failure)})')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.set_xlabel('Bar')
ax2.set_ylabel('PnL (bps)')
ax2.set_title('SHORT (Overbought) - Path Analysis')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_PATH / '01_path_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_PATH / '01_path_analysis.png'}")

# =============================================================================
# PART 2: EARLY EXIT RULES VISUALIZATION
# =============================================================================
print("\n[2/7] Creating Early Exit Rules visualization...")

exit_rules = [
    {'name': 'Exit if PnL < 0 at Bar 1', 'failures_caught': 11, 'winners_killed': 280, 'total_failures': 11, 'total_winners': 653},
    {'name': 'Exit if PnL < -10 at Bar 1', 'failures_caught': 10, 'winners_killed': 174, 'total_failures': 11, 'total_winners': 653},
    {'name': 'Exit if PnL < -20 at Bar 2', 'failures_caught': 9, 'winners_killed': 129, 'total_failures': 11, 'total_winners': 653},
]

fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(exit_rules))
width = 0.35

failures_pct = [r['failures_caught'] / r['total_failures'] * 100 for r in exit_rules]
winners_pct = [r['winners_killed'] / r['total_winners'] * 100 for r in exit_rules]

bars1 = ax.bar([i - width/2 for i in x], failures_pct, width, label='Failures Caught %', color='green', alpha=0.7)
bars2 = ax.bar([i + width/2 for i in x], winners_pct, width, label='Winners Killed %', color='red', alpha=0.7)

ax.set_ylabel('Percentage')
ax.set_title('Early Exit Rules - Failures Caught vs Winners Killed')
ax.set_xticks(x)
ax.set_xticklabels([r['name'] for r in exit_rules], rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_PATH / '02_early_exit_rules.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_PATH / '02_early_exit_rules.png'}")

# =============================================================================
# PART 3: RECOVERY ANALYSIS VISUALIZATION
# =============================================================================
print("\n[3/7] Creating Recovery Analysis visualization...")

recovery_data = [
    {'status': 'Recovered < 12h', 'count': 9, 'color': 'green'},
    {'status': 'Recovered 4.6 days', 'count': 1, 'color': 'orange'},
    {'status': 'NEVER recovered', 'count': 1, 'color': 'red'},
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
ax1 = axes[0]
colors = [r['color'] for r in recovery_data]
counts = [r['count'] for r in recovery_data]
labels = [f"{r['status']}\n({r['count']})" for r in recovery_data]
ax1.pie(counts, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90,
        explode=[0, 0, 0.1], textprops={'fontsize': 11})
ax1.set_title('SHORT Failure Recovery (11 failures)')

# Bar chart for recovery times
ax2 = axes[1]
recovery_times = {
    'Quick (<12h)': 6.9,
    'Slow (4.6 days)': 110.4,
    'Never': 480  # 20 days+
}
bars = ax2.bar(recovery_times.keys(), recovery_times.values(), color=['green', 'orange', 'red'], alpha=0.7)
ax2.set_ylabel('Hours')
ax2.set_title('Average Recovery Time by Category')
ax2.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    label = f'{height:.1f}h' if height < 200 else '20+ days'
    ax2.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(PLOTS_PATH / '03_recovery_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_PATH / '03_recovery_analysis.png'}")

# =============================================================================
# PART 4: CATASTROPHIC FAILURES VISUALIZATION
# =============================================================================
print("\n[4/7] Creating Catastrophic Failures visualization...")

# Get the account killer path
killer_date = pd.Timestamp('2024-02-26 15:00:00')
if killer_date in oos_data.index:
    idx_pos = oos_data.index.get_loc(killer_date)
    entry_price = oos_data.iloc[idx_pos]['close']

    # Track path for 500 bars (about 5 days)
    path_bars = min(500, len(oos_data) - idx_pos - 1)
    killer_path = []
    killer_prices = []

    for bar in range(1, path_bars + 1):
        bar_data = oos_data.iloc[idx_pos + bar]
        pnl = (entry_price - bar_data['close']) / entry_price * 10000
        killer_path.append(pnl)
        killer_prices.append(bar_data['close'])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # PnL path
    ax1 = axes[0]
    ax1.plot(range(1, path_bars + 1), killer_path, 'r-', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.fill_between(range(1, path_bars + 1), killer_path, 0,
                     where=[p < 0 for p in killer_path], color='red', alpha=0.3)
    ax1.set_xlabel('Bars after entry')
    ax1.set_ylabel('PnL (bps)')
    ax1.set_title(f'ACCOUNT KILLER (2024-02-26) - SHORT at ${entry_price:,.0f}')
    ax1.grid(True, alpha=0.3)

    # Add annotation for max drawdown
    min_pnl = min(killer_path)
    min_idx = killer_path.index(min_pnl)
    ax1.annotate(f'Max loss: {min_pnl:.0f} bps', xy=(min_idx, min_pnl),
                xytext=(min_idx + 50, min_pnl + 500),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, color='red')

    # Price path
    ax2 = axes[1]
    ax2.plot(range(1, path_bars + 1), killer_prices, 'b-', linewidth=2)
    ax2.axhline(y=entry_price, color='red', linestyle='--', label=f'Entry: ${entry_price:,.0f}')
    ax2.set_xlabel('Bars after entry')
    ax2.set_ylabel('Price ($)')
    ax2.set_title('BTC Price Path - ETF Rally')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / '04_catastrophic_failure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOTS_PATH / '04_catastrophic_failure.png'}")

# =============================================================================
# PART 5: ALL OPTIONS COMPARISON VISUALIZATION
# =============================================================================
print("\n[5/7] Creating All Options Comparison visualization...")

options = [
    {'name': 'LONG only', 'signals': 596, 'failures': 2, 'max_mae': 1004, 'recoverable': True},
    {'name': 'SHORT only', 'signals': 664, 'failures': 11, 'max_mae': 4208, 'recoverable': False},
    {'name': 'SHORT (bear only)', 'signals': 140, 'failures': 1, 'max_mae': 39, 'recoverable': True},
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Signals count
ax1 = axes[0]
colors = ['green', 'red', 'orange']
bars = ax1.bar([o['name'] for o in options], [o['signals'] for o in options], color=colors, alpha=0.7)
ax1.set_ylabel('Number of Signals')
ax1.set_title('Signal Count by Option')
ax1.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Failures count
ax2 = axes[1]
bars = ax2.bar([o['name'] for o in options], [o['failures'] for o in options], color=colors, alpha=0.7)
ax2.set_ylabel('Number of Failures')
ax2.set_title('Failure Count by Option')
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax2.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Max MAE
ax3 = axes[2]
bars = ax3.bar([o['name'] for o in options], [o['max_mae'] for o in options], color=colors, alpha=0.7)
ax3.set_ylabel('Max MAE (bps)')
ax3.set_title('Worst Case Loss by Option')
ax3.grid(True, alpha=0.3, axis='y')
for bar, opt in zip(bars, options):
    height = bar.get_height()
    label = f'{int(height)} bps'
    if not opt['recoverable']:
        label += '\n(FATAL)'
    ax3.annotate(label, xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(PLOTS_PATH / '05_options_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_PATH / '05_options_comparison.png'}")

# =============================================================================
# PART 6: LONG FAILURES VISUALIZATION
# =============================================================================
print("\n[6/7] Creating LONG Failures visualization...")

long_failures = [
    {'date': '2024-04-30', 'max_dd': -877, 'recovery_days': 3.2, 'market': 'BEAR'},
    {'date': '2025-03-09', 'max_dd': -1004, 'recovery_days': 5.2, 'market': 'BEAR'},
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Max drawdown comparison
ax1 = axes[0]
x = range(len(long_failures))
bars = ax1.bar(x, [abs(f['max_dd']) for f in long_failures], color='orange', alpha=0.7)
ax1.set_ylabel('Max Drawdown (bps)')
ax1.set_title('LONG Failures - Max Drawdown')
ax1.set_xticks(x)
ax1.set_xticklabels([f['date'] for f in long_failures])
ax1.grid(True, alpha=0.3, axis='y')
for bar, f in zip(bars, long_failures):
    height = bar.get_height()
    ax1.annotate(f'{int(height)} bps\n({f["market"]})', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Recovery time
ax2 = axes[1]
bars = ax2.bar(x, [f['recovery_days'] for f in long_failures], color='green', alpha=0.7)
ax2.set_ylabel('Recovery Time (days)')
ax2.set_title('LONG Failures - Recovery Time')
ax2.set_xticks(x)
ax2.set_xticklabels([f['date'] for f in long_failures])
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f} days', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Add note
fig.text(0.5, 0.01, 'Both LONG failures occurred in BEAR market (price < SMA200) and RECOVERED',
         ha='center', fontsize=12, style='italic', color='green')

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig(PLOTS_PATH / '06_long_failures.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_PATH / '06_long_failures.png'}")

# =============================================================================
# PART 7: SHORT FAILURES SMA FILTER VISUALIZATION
# =============================================================================
print("\n[7/7] Creating SHORT SMA Filter visualization...")

short_failures = [
    {'date': '2024-02-26', 'market': 'BULL', 'mae': 332, 'killer': True},
    {'date': '2024-05-24', 'market': 'BULL', 'mae': 101, 'killer': False},
    {'date': '2024-07-19', 'market': 'BULL', 'mae': 90, 'killer': False},
    {'date': '2024-07-25', 'market': 'BULL', 'mae': 134, 'killer': False},
    {'date': '2024-08-20', 'market': 'BULL', 'mae': 97, 'killer': False},
    {'date': '2025-01-20', 'market': 'BULL', 'mae': 468, 'killer': True},
    {'date': '2025-06-16', 'market': 'BULL', 'mae': 37, 'killer': False},
    {'date': '2025-07-06', 'market': 'BULL', 'mae': 61, 'killer': False},
    {'date': '2025-08-17', 'market': 'BEAR', 'mae': 39, 'killer': False},
    {'date': '2025-09-09', 'market': 'BULL', 'mae': 117, 'killer': False},
    {'date': '2025-12-07', 'market': 'BULL', 'mae': 90, 'killer': False},
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart of all failures by MAE
ax1 = axes[0]
colors = ['red' if f['killer'] else ('blue' if f['market'] == 'BEAR' else 'orange') for f in short_failures]
x = range(len(short_failures))
bars = ax1.bar(x, [f['mae'] for f in short_failures], color=colors, alpha=0.7)
ax1.set_ylabel('MAE (bps)')
ax1.set_title('All 11 SHORT Failures by MAE')
ax1.set_xticks(x)
ax1.set_xticklabels([f['date'].split('-')[1] + '/' + f['date'].split('-')[0][-2:] for f in short_failures],
                    rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.7, label='BULL - Account Killer'),
    Patch(facecolor='orange', alpha=0.7, label='BULL - Regular'),
    Patch(facecolor='blue', alpha=0.7, label='BEAR (not filtered)')
]
ax1.legend(handles=legend_elements, loc='upper right')

# Pie chart - BULL vs BEAR
ax2 = axes[1]
bull_count = sum(1 for f in short_failures if f['market'] == 'BULL')
bear_count = sum(1 for f in short_failures if f['market'] == 'BEAR')

colors_pie = ['orange', 'blue']
explode = (0.05, 0.1)
ax2.pie([bull_count, bear_count], labels=[f'BULL Market\n({bull_count} - SKIPPED)', f'BEAR Market\n({bear_count} - remains)'],
        colors=colors_pie, autopct='%1.0f%%', startangle=90, explode=explode,
        textprops={'fontsize': 11})
ax2.set_title('200 SMA Filter Effectiveness\n(91% failures avoided)')

plt.tight_layout()
plt.savefig(PLOTS_PATH / '07_short_sma_filter.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_PATH / '07_short_sma_filter.png'}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ALL VISUALIZATIONS COMPLETE")
print("="*70)
print(f"\nSaved to: {PLOTS_PATH}")
print("\nFiles created:")
print("  01_path_analysis.png       - How failures vs successes develop")
print("  02_early_exit_rules.png    - Impact of exit rules")
print("  03_recovery_analysis.png   - Recovery times for failures")
print("  04_catastrophic_failure.png - The account killer path")
print("  05_options_comparison.png  - LONG only vs SHORT options")
print("  06_long_failures.png       - The 2 LONG failures")
print("  07_short_sma_filter.png    - SMA filter effectiveness")
