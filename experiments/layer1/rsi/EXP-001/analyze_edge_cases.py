"""
EXP-001: RSI Edge Case Analysis
Analyzes failure cases and other edge cases to understand risk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Setup paths
DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
PLOTS_PATH = Path("experiments/rsi/EXP-001/plots")
RESULTS_PATH = Path("experiments/rsi/EXP-001")
PLOTS_PATH.mkdir(exist_ok=True)

# RSI Config to analyze
RSI_PERIOD = 14
OVERSOLD = 20
OVERBOUGHT = 80
HORIZON = 10  # bars to look forward

print("="*60)
print("RSI EDGE CASE ANALYSIS")
print(f"Config: RSI({RSI_PERIOD}) {OVERSOLD}/{OVERBOUGHT}, Horizon={HORIZON}")
print("="*60)

# Load data
print("\nLoading data...")
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate RSI and signals
print("Calculating RSI and detecting signals...")
df['rsi'] = calculate_rsi(df, RSI_PERIOD)
df['oversold'] = df['rsi'] < OVERSOLD
df['overbought'] = df['rsi'] > OVERBOUGHT
df['oversold_cross'] = (df['oversold'] == True) & (df['oversold'].shift(1) == False)
df['overbought_cross'] = (df['overbought'] == True) & (df['overbought'].shift(1) == False)

# Calculate volatility (ATR-like)
df['range'] = df['high'] - df['low']
df['volatility'] = df['range'].rolling(window=20).mean()
df['volatility_pct'] = df['volatility'] / df['close'] * 10000  # in bps

# Calculate trend (simple: price vs SMA)
df['sma50'] = df['close'].rolling(window=50).mean()
df['trend'] = np.where(df['close'] > df['sma50'], 'UP', 'DOWN')

# Split data
in_sample = df['2020-01-01':'2023-12-31'].copy()
out_sample = df['2024-01-01':'2025-12-31'].copy()

def analyze_signals(data, signal_type='oversold'):
    """
    Detailed analysis of each signal
    Returns DataFrame with all metrics for each signal
    """
    if signal_type == 'oversold':
        signal_col = 'oversold_cross'
        # For LONG: MFE = max gain, MAE = max loss
        direction = 1  # expecting price to go UP
    else:
        signal_col = 'overbought_cross'
        direction = -1  # expecting price to go DOWN

    signals = data[data[signal_col] == True].copy()
    results = []

    for idx in signals.index:
        idx_pos = data.index.get_loc(idx)

        # Skip if not enough future data
        if idx_pos + HORIZON >= len(data):
            continue

        entry_price = data.iloc[idx_pos]['close']
        future_data = data.iloc[idx_pos + 1 : idx_pos + 1 + HORIZON]

        if len(future_data) == 0:
            continue

        # Calculate metrics
        if direction == 1:  # LONG (oversold)
            # MFE = Maximum Favorable Excursion (best case)
            max_price = future_data['high'].max()
            mfe_bps = (max_price - entry_price) / entry_price * 10000

            # MAE = Maximum Adverse Excursion (worst drawdown)
            min_price = future_data['low'].min()
            mae_bps = (entry_price - min_price) / entry_price * 10000  # positive = bad

            # Final move at end of horizon
            final_price = future_data.iloc[-1]['close']
            final_move_bps = (final_price - entry_price) / entry_price * 10000

            # Time to best price (which bar?)
            best_bar = future_data['high'].idxmax()
            time_to_best = data.index.get_loc(best_bar) - idx_pos

            # Did it bounce? (any positive move)
            bounced = mfe_bps > 0

            # Success = moved at least 12 bps in right direction
            success = mfe_bps >= 12

        else:  # SHORT (overbought)
            min_price = future_data['low'].min()
            mfe_bps = (entry_price - min_price) / entry_price * 10000

            max_price = future_data['high'].max()
            mae_bps = (max_price - entry_price) / entry_price * 10000

            final_price = future_data.iloc[-1]['close']
            final_move_bps = (entry_price - final_price) / entry_price * 10000

            best_bar = future_data['low'].idxmin()
            time_to_best = data.index.get_loc(best_bar) - idx_pos

            bounced = mfe_bps > 0
            success = mfe_bps >= 12

        # Context at signal time
        volatility = data.iloc[idx_pos]['volatility_pct']
        trend = data.iloc[idx_pos]['trend']
        rsi_value = data.iloc[idx_pos]['rsi']

        # Check if consecutive signal (previous bar also had signal)
        prev_signal = data.iloc[idx_pos - 1][signal_col] if idx_pos > 0 else False
        consecutive = prev_signal

        results.append({
            'timestamp': idx,
            'entry_price': entry_price,
            'rsi': rsi_value,
            'mfe_bps': mfe_bps,  # Best case (max profit potential)
            'mae_bps': mae_bps,  # Worst case (max drawdown)
            'final_move_bps': final_move_bps,
            'time_to_best': time_to_best,
            'bounced': bounced,
            'success': success,  # Moved 12+ bps
            'volatility_bps': volatility,
            'trend': trend,
            'consecutive': consecutive,
            'signal_type': signal_type
        })

    return pd.DataFrame(results)

# Analyze both signal types for both periods
print("\nAnalyzing oversold signals (LONG)...")
oversold_in = analyze_signals(in_sample, 'oversold')
oversold_out = analyze_signals(out_sample, 'oversold')

print("Analyzing overbought signals (SHORT)...")
overbought_in = analyze_signals(in_sample, 'overbought')
overbought_out = analyze_signals(out_sample, 'overbought')

# Combine for overall analysis
all_in = pd.concat([oversold_in, overbought_in])
all_out = pd.concat([oversold_out, overbought_out])

# =============================================================================
# ANALYSIS 1: FAILURE CASES
# =============================================================================
print("\n" + "="*60)
print("ANALYSIS 1: FAILURE CASES")
print("="*60)

def analyze_failures(signals_df, period_name):
    failures = signals_df[signals_df['bounced'] == False]
    successes = signals_df[signals_df['bounced'] == True]

    print(f"\n{period_name}:")
    print(f"  Total signals: {len(signals_df)}")
    print(f"  Successes (bounced): {len(successes)} ({len(successes)/len(signals_df)*100:.1f}%)")
    print(f"  Failures (no bounce): {len(failures)} ({len(failures)/len(signals_df)*100:.1f}%)")

    if len(failures) > 0:
        print(f"\n  FAILURE DETAILS:")
        print(f"    Average loss (MAE): {failures['mae_bps'].mean():.1f} bps")
        print(f"    Max loss (MAE): {failures['mae_bps'].max():.1f} bps")
        print(f"    Failures in UP trend: {len(failures[failures['trend'] == 'UP'])} ({len(failures[failures['trend'] == 'UP'])/len(failures)*100:.1f}%)")
        print(f"    Failures in DOWN trend: {len(failures[failures['trend'] == 'DOWN'])} ({len(failures[failures['trend'] == 'DOWN'])/len(failures)*100:.1f}%)")
        print(f"    Avg volatility during failures: {failures['volatility_bps'].mean():.1f} bps")
        print(f"    Avg volatility during successes: {successes['volatility_bps'].mean():.1f} bps")

    return failures

failures_oversold_in = analyze_failures(oversold_in, "OVERSOLD In-Sample (2020-2023)")
failures_oversold_out = analyze_failures(oversold_out, "OVERSOLD Out-of-Sample (2024-2025)")
failures_overbought_in = analyze_failures(overbought_in, "OVERBOUGHT In-Sample (2020-2023)")
failures_overbought_out = analyze_failures(overbought_out, "OVERBOUGHT Out-of-Sample (2024-2025)")

# =============================================================================
# ANALYSIS 2: DRAWDOWN BEFORE BOUNCE (MAE)
# =============================================================================
print("\n" + "="*60)
print("ANALYSIS 2: DRAWDOWN BEFORE BOUNCE (MAE)")
print("="*60)

def analyze_mae(signals_df, period_name):
    # Only look at successful bounces
    successes = signals_df[signals_df['bounced'] == True]

    print(f"\n{period_name} (successful bounces only):")
    print(f"  Total successful signals: {len(successes)}")
    print(f"\n  MAE (Max Adverse Excursion = worst drawdown during trade):")
    print(f"    Mean MAE: {successes['mae_bps'].mean():.1f} bps")
    print(f"    Median MAE: {successes['mae_bps'].median():.1f} bps")
    print(f"    75th percentile MAE: {successes['mae_bps'].quantile(0.75):.1f} bps")
    print(f"    90th percentile MAE: {successes['mae_bps'].quantile(0.90):.1f} bps")
    print(f"    95th percentile MAE: {successes['mae_bps'].quantile(0.95):.1f} bps")
    print(f"    Max MAE: {successes['mae_bps'].max():.1f} bps")

    # How many would have been stopped out at different stop levels?
    for stop in [10, 15, 20, 25, 30, 50]:
        stopped = len(successes[successes['mae_bps'] > stop])
        pct = stopped / len(successes) * 100
        print(f"    Stopped out at -{stop}bp: {stopped} ({pct:.1f}%)")

    return successes

mae_oversold_in = analyze_mae(oversold_in, "OVERSOLD In-Sample")
mae_oversold_out = analyze_mae(oversold_out, "OVERSOLD Out-of-Sample")

# =============================================================================
# ANALYSIS 3: TIME TO REVERSAL
# =============================================================================
print("\n" + "="*60)
print("ANALYSIS 3: TIME TO REVERSAL")
print("="*60)

def analyze_time(signals_df, period_name):
    successes = signals_df[signals_df['bounced'] == True]

    print(f"\n{period_name}:")
    print(f"  Time to best price (in bars, 1 bar = 15 min):")
    print(f"    Mean: {successes['time_to_best'].mean():.1f} bars ({successes['time_to_best'].mean()*15:.0f} min)")
    print(f"    Median: {successes['time_to_best'].median():.1f} bars")

    # Distribution by bar
    for bar in [1, 2, 3, 5, 7, 10]:
        count = len(successes[successes['time_to_best'] <= bar])
        pct = count / len(successes) * 100
        print(f"    Best price within {bar} bars: {count} ({pct:.1f}%)")

analyze_time(oversold_in, "OVERSOLD In-Sample")
analyze_time(oversold_out, "OVERSOLD Out-of-Sample")

# =============================================================================
# ANALYSIS 4: CONSECUTIVE SIGNALS
# =============================================================================
print("\n" + "="*60)
print("ANALYSIS 4: CONSECUTIVE SIGNALS")
print("="*60)

def analyze_consecutive(signals_df, period_name):
    consecutive = signals_df[signals_df['consecutive'] == True]
    first_signal = signals_df[signals_df['consecutive'] == False]

    print(f"\n{period_name}:")
    print(f"  First signals (RSI just crossed): {len(first_signal)}")
    print(f"  Consecutive signals (RSI stayed extreme): {len(consecutive)}")

    if len(first_signal) > 0:
        print(f"\n  First signals performance:")
        print(f"    Bounce rate: {first_signal['bounced'].mean()*100:.1f}%")
        print(f"    Avg MFE: {first_signal['mfe_bps'].mean():.1f} bps")
        print(f"    Avg MAE: {first_signal['mae_bps'].mean():.1f} bps")

    if len(consecutive) > 0:
        print(f"\n  Consecutive signals performance:")
        print(f"    Bounce rate: {consecutive['bounced'].mean()*100:.1f}%")
        print(f"    Avg MFE: {consecutive['mfe_bps'].mean():.1f} bps")
        print(f"    Avg MAE: {consecutive['mae_bps'].mean():.1f} bps")

analyze_consecutive(oversold_in, "OVERSOLD In-Sample")
analyze_consecutive(oversold_out, "OVERSOLD Out-of-Sample")

# =============================================================================
# ANALYSIS 5: OUTCOME DISTRIBUTION
# =============================================================================
print("\n" + "="*60)
print("ANALYSIS 5: OUTCOME DISTRIBUTION (MFE)")
print("="*60)

def analyze_distribution(signals_df, period_name):
    print(f"\n{period_name}:")
    print(f"  MFE (Maximum Favorable Excursion = best profit potential):")
    print(f"    Min: {signals_df['mfe_bps'].min():.1f} bps")
    print(f"    10th percentile: {signals_df['mfe_bps'].quantile(0.10):.1f} bps")
    print(f"    25th percentile: {signals_df['mfe_bps'].quantile(0.25):.1f} bps")
    print(f"    Median: {signals_df['mfe_bps'].median():.1f} bps")
    print(f"    75th percentile: {signals_df['mfe_bps'].quantile(0.75):.1f} bps")
    print(f"    90th percentile: {signals_df['mfe_bps'].quantile(0.90):.1f} bps")
    print(f"    Max: {signals_df['mfe_bps'].max():.1f} bps")

    # How many reach different targets?
    for target in [12, 15, 20, 25, 30, 50, 100]:
        reached = len(signals_df[signals_df['mfe_bps'] >= target])
        pct = reached / len(signals_df) * 100
        print(f"    Reached +{target}bp: {reached} ({pct:.1f}%)")

analyze_distribution(oversold_in, "OVERSOLD In-Sample")
analyze_distribution(oversold_out, "OVERSOLD Out-of-Sample")

# =============================================================================
# ANALYSIS 6: TREND IMPACT
# =============================================================================
print("\n" + "="*60)
print("ANALYSIS 6: TREND IMPACT")
print("="*60)

def analyze_trend(signals_df, signal_type, period_name):
    print(f"\n{period_name} ({signal_type}):")

    up_trend = signals_df[signals_df['trend'] == 'UP']
    down_trend = signals_df[signals_df['trend'] == 'DOWN']

    if signal_type == 'oversold':  # LONG signal
        aligned = up_trend  # Going long in uptrend = aligned
        against = down_trend  # Going long in downtrend = against
        print(f"  WITH trend (long in uptrend): {len(aligned)} signals")
        print(f"  AGAINST trend (long in downtrend): {len(against)} signals")
    else:  # SHORT signal
        aligned = down_trend  # Going short in downtrend = aligned
        against = up_trend  # Going short in uptrend = against
        print(f"  WITH trend (short in downtrend): {len(aligned)} signals")
        print(f"  AGAINST trend (short in uptrend): {len(against)} signals")

    if len(aligned) > 0:
        print(f"\n  WITH trend performance:")
        print(f"    Bounce rate: {aligned['bounced'].mean()*100:.1f}%")
        print(f"    Avg MFE: {aligned['mfe_bps'].mean():.1f} bps")
        print(f"    Avg MAE: {aligned['mae_bps'].mean():.1f} bps")

    if len(against) > 0:
        print(f"\n  AGAINST trend performance:")
        print(f"    Bounce rate: {against['bounced'].mean()*100:.1f}%")
        print(f"    Avg MFE: {against['mfe_bps'].mean():.1f} bps")
        print(f"    Avg MAE: {against['mae_bps'].mean():.1f} bps")

analyze_trend(oversold_in, 'oversold', "In-Sample")
analyze_trend(oversold_out, 'oversold', "Out-of-Sample")

# =============================================================================
# SAVE DETAILED RESULTS
# =============================================================================
print("\n" + "="*60)
print("SAVING DETAILED RESULTS")
print("="*60)

# Save all signal details
all_signals = pd.concat([
    oversold_in.assign(period='in_sample'),
    oversold_out.assign(period='out_of_sample'),
    overbought_in.assign(period='in_sample'),
    overbought_out.assign(period='out_of_sample')
])
all_signals.to_csv(RESULTS_PATH / 'edge_case_details.csv', index=False)
print(f"Saved: edge_case_details.csv ({len(all_signals)} signals)")

# =============================================================================
# CREATE VISUALIZATIONS
# =============================================================================
print("\nCreating visualizations...")

# Chart 1: MAE Distribution (Drawdown before bounce)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(oversold_in[oversold_in['bounced']==True]['mae_bps'], bins=50,
             alpha=0.7, color='#2ecc71', edgecolor='white', label='In-Sample')
axes[0].hist(oversold_out[oversold_out['bounced']==True]['mae_bps'], bins=50,
             alpha=0.7, color='#3498db', edgecolor='white', label='Out-of-Sample')
axes[0].axvline(x=15, color='red', linestyle='--', linewidth=2, label='Stop Loss -15bp')
axes[0].axvline(x=25, color='orange', linestyle='--', linewidth=2, label='Stop Loss -25bp')
axes[0].set_xlabel('MAE (Max Drawdown in bps)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Drawdown Before Bounce (OVERSOLD signals)\nHow much price drops BEFORE bouncing up', fontsize=12)
axes[0].legend()

axes[1].hist(oversold_in['mfe_bps'], bins=50,
             alpha=0.7, color='#2ecc71', edgecolor='white', label='In-Sample')
axes[1].hist(oversold_out['mfe_bps'], bins=50,
             alpha=0.7, color='#3498db', edgecolor='white', label='Out-of-Sample')
axes[1].axvline(x=12, color='red', linestyle='--', linewidth=2, label='Min Profitable (12bp)')
axes[1].axvline(x=25, color='green', linestyle='--', linewidth=2, label='Target 25bp')
axes[1].set_xlabel('MFE (Max Favorable Excursion in bps)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Best Profit Potential (OVERSOLD signals)\nHow much price bounces up', fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.savefig(PLOTS_PATH / '12_mae_mfe_distribution.png', dpi=150)
plt.close()
print("Created: 12_mae_mfe_distribution.png")

# Chart 2: Time to Best Price
fig, ax = plt.subplots(figsize=(10, 6))
time_bins = range(1, 12)
in_counts = [len(oversold_in[oversold_in['time_to_best'] == t]) for t in time_bins]
out_counts = [len(oversold_out[oversold_out['time_to_best'] == t]) for t in time_bins]

x = np.arange(len(time_bins))
width = 0.35
ax.bar(x - width/2, in_counts, width, label='In-Sample', color='#2ecc71', alpha=0.8)
ax.bar(x + width/2, out_counts, width, label='Out-of-Sample', color='#3498db', alpha=0.8)
ax.set_xlabel('Bars After Signal (1 bar = 15 min)', fontsize=12)
ax.set_ylabel('Number of Signals', fontsize=12)
ax.set_title('When Does the Best Price Occur?\nHow long to wait for maximum profit', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f'{t}\n({t*15}m)' for t in time_bins])
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_PATH / '13_time_to_best.png', dpi=150)
plt.close()
print("Created: 13_time_to_best.png")

# Chart 3: Trend Impact
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# In-Sample
up_in = oversold_in[oversold_in['trend'] == 'UP']
down_in = oversold_in[oversold_in['trend'] == 'DOWN']
labels = ['Long in Uptrend\n(WITH trend)', 'Long in Downtrend\n(AGAINST trend)']
mfe_means = [up_in['mfe_bps'].mean(), down_in['mfe_bps'].mean()]
mae_means = [up_in['mae_bps'].mean(), down_in['mae_bps'].mean()]

x = np.arange(2)
width = 0.35
axes[0].bar(x - width/2, mfe_means, width, label='MFE (Profit)', color='#2ecc71')
axes[0].bar(x + width/2, mae_means, width, label='MAE (Drawdown)', color='#e74c3c')
axes[0].set_ylabel('Basis Points', fontsize=12)
axes[0].set_title('In-Sample: Trend Impact on OVERSOLD Signals', fontsize=12)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].legend()

# Out-of-Sample
up_out = oversold_out[oversold_out['trend'] == 'UP']
down_out = oversold_out[oversold_out['trend'] == 'DOWN']
mfe_means = [up_out['mfe_bps'].mean(), down_out['mfe_bps'].mean()]
mae_means = [up_out['mae_bps'].mean(), down_out['mae_bps'].mean()]

axes[1].bar(x - width/2, mfe_means, width, label='MFE (Profit)', color='#2ecc71')
axes[1].bar(x + width/2, mae_means, width, label='MAE (Drawdown)', color='#e74c3c')
axes[1].set_ylabel('Basis Points', fontsize=12)
axes[1].set_title('Out-of-Sample: Trend Impact on OVERSOLD Signals', fontsize=12)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].legend()

plt.tight_layout()
plt.savefig(PLOTS_PATH / '14_trend_impact.png', dpi=150)
plt.close()
print("Created: 14_trend_impact.png")

# Chart 4: Stop Loss Analysis - What stop level to use?
fig, ax = plt.subplots(figsize=(10, 6))

stop_levels = [5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100]
in_stopped = [len(oversold_in[oversold_in['mae_bps'] > s]) / len(oversold_in) * 100 for s in stop_levels]
out_stopped = [len(oversold_out[oversold_out['mae_bps'] > s]) / len(oversold_out) * 100 for s in stop_levels]

ax.plot(stop_levels, in_stopped, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='In-Sample')
ax.plot(stop_levels, out_stopped, 's-', color='#3498db', linewidth=2, markersize=8, label='Out-of-Sample')
ax.set_xlabel('Stop Loss Level (bps)', fontsize=12)
ax.set_ylabel('% of Trades Stopped Out', fontsize=12)
ax.set_title('Stop Loss Analysis: What % of winning trades would be stopped out?\nLower = better (fewer false stops)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Add annotations
for stop, pct_in, pct_out in zip(stop_levels, in_stopped, out_stopped):
    if stop in [15, 25, 50]:
        ax.annotate(f'{pct_in:.0f}%', (stop, pct_in), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9, color='#2ecc71')
        ax.annotate(f'{pct_out:.0f}%', (stop, pct_out), textcoords="offset points",
                   xytext=(0, -15), ha='center', fontsize=9, color='#3498db')

plt.tight_layout()
plt.savefig(PLOTS_PATH / '15_stop_loss_analysis.png', dpi=150)
plt.close()
print("Created: 15_stop_loss_analysis.png")

print("\n" + "="*60)
print("EDGE CASE ANALYSIS COMPLETE!")
print("="*60)
