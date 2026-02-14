"""
EXP-002: RSI + MA Trend Filter Visualizations

Creates visualizations comparing RSI alone vs RSI with MA filter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
RESULTS_PATH = Path("experiments/rsi/EXP-002")
PLOTS_PATH = RESULTS_PATH / "plots"
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# Configuration
RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
MA_PERIODS = [20, 50, 100]

print("Loading data...")
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

# Calculate indicators
def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sma(data, period):
    return data['close'].rolling(window=period).mean()

print("Calculating indicators...")
df['rsi'] = calculate_rsi(df, RSI_PERIOD)
for period in MA_PERIODS:
    df[f'sma_{period}'] = calculate_sma(df, period)

df['oversold'] = df['rsi'] < RSI_OVERSOLD
df['rsi_oversold_cross'] = (df['oversold'] == True) & (df['oversold'].shift(1) == False)

# =============================================================================
# PLOT 1: Signal Filtering Summary (Bar Chart)
# =============================================================================
print("\nCreating signal filtering summary chart...")

results_df = pd.read_csv(RESULTS_PATH / 'results.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, period_name in enumerate(['in_sample', 'out_of_sample']):
    ax = axes[idx]

    oversold_data = results_df[(results_df['period_name'] == period_name) &
                               (results_df['signal_type'] == 'oversold')]

    # Get RSI alone and WITH trend variants
    rsi_alone = oversold_data[oversold_data['filter'] == 'none']['total_signals'].values[0]

    filters = ['RSI Alone']
    signals = [rsi_alone]
    colors = ['#2196F3']

    for ma in MA_PERIODS:
        with_trend = oversold_data[(oversold_data['filter'] == 'with_trend') &
                                   (oversold_data['ma_period'] == ma)]
        if len(with_trend) > 0:
            filters.append(f'RSI + SMA({ma})')
            signals.append(with_trend['total_signals'].values[0])
            colors.append('#4CAF50')

    bars = ax.bar(filters, signals, color=colors, edgecolor='black')
    ax.set_title(f'{period_name.upper().replace("_", " ")}\nOversold (Long) Signals', fontsize=12)
    ax.set_ylabel('Number of Signals')
    ax.set_xlabel('Filter Type')

    # Add value labels on bars
    for bar, val in zip(bars, signals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(int(val)), ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, max(signals) * 1.15)
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(PLOTS_PATH / 'signal_filtering_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_PATH / 'signal_filtering_summary.png'}")

# =============================================================================
# PLOT 2: MFE/MAE Comparison
# =============================================================================
print("Creating MFE/MAE comparison chart...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, period_name in enumerate(['in_sample', 'out_of_sample']):
    oversold_data = results_df[(results_df['period_name'] == period_name) &
                               (results_df['signal_type'] == 'oversold')]

    # MFE comparison
    ax_mfe = axes[0, idx]

    labels = []
    mfes = []
    maes = []

    # RSI Alone
    rsi_alone = oversold_data[oversold_data['filter'] == 'none'].iloc[0]
    labels.append('RSI Alone')
    mfes.append(rsi_alone.get('avg_mfe', 0))
    maes.append(rsi_alone.get('avg_mae', 0))

    # WITH trend variants
    for ma in MA_PERIODS:
        with_trend = oversold_data[(oversold_data['filter'] == 'with_trend') &
                                   (oversold_data['ma_period'] == ma)]
        if len(with_trend) > 0 and with_trend.iloc[0]['total_signals'] > 0:
            labels.append(f'SMA({ma})\nWITH trend')
            mfes.append(with_trend.iloc[0].get('avg_mfe', 0))
            maes.append(with_trend.iloc[0].get('avg_mae', 0))

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax_mfe.bar(x - width/2, mfes, width, label='Avg MFE', color='#4CAF50', edgecolor='black')
    bars2 = ax_mfe.bar(x + width/2, maes, width, label='Avg MAE', color='#F44336', edgecolor='black')

    ax_mfe.set_title(f'{period_name.upper().replace("_", " ")}\nMFE vs MAE (bps)', fontsize=12)
    ax_mfe.set_ylabel('Basis Points')
    ax_mfe.set_xticks(x)
    ax_mfe.set_xticklabels(labels, fontsize=9)
    ax_mfe.legend()
    ax_mfe.axhline(y=12, color='orange', linestyle='--', linewidth=1, label='12bp threshold')

    # MFE/MAE Ratio
    ax_ratio = axes[1, idx]
    ratios = [m/a if a > 0 else 0 for m, a in zip(mfes, maes)]

    bars = ax_ratio.bar(labels, ratios, color='#9C27B0', edgecolor='black')
    ax_ratio.set_title(f'{period_name.upper().replace("_", " ")}\nMFE/MAE Ratio', fontsize=12)
    ax_ratio.set_ylabel('Ratio')
    ax_ratio.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Breakeven')
    ax_ratio.legend()

    for bar, val in zip(bars, ratios):
        ax_ratio.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                      f'{val:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_PATH / 'mfe_mae_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_PATH / 'mfe_mae_comparison.png'}")

# =============================================================================
# PLOT 3: Price Action with RSI and MA Filter
# =============================================================================
print("Creating price action visualization...")

# Find a period with RSI signals that shows both WITH and AGAINST trend
test_data = df['2024-04-01':'2024-04-07'].copy()

fig, axes = plt.subplots(3, 1, figsize=(20, 14), gridspec_kw={'height_ratios': [3, 1, 1]})

# Price with SMA(50) and SMA(100)
ax1 = axes[0]
ax1.plot(test_data.index, test_data['close'], label='Price', color='black', linewidth=1)
ax1.plot(test_data.index, test_data['sma_50'], label='SMA(50)', color='blue', linewidth=1.5, alpha=0.7)
ax1.plot(test_data.index, test_data['sma_100'], label='SMA(100)', color='purple', linewidth=1.5, alpha=0.7)

# Mark RSI oversold signals
oversold_signals = test_data[test_data['rsi_oversold_cross'] == True]
for idx in oversold_signals.index:
    price = test_data.loc[idx, 'close']
    sma50 = test_data.loc[idx, 'sma_50']

    if price > sma50:
        # WITH trend - green
        ax1.scatter(idx, price, color='green', marker='^', s=200, zorder=5,
                   label='RSI Oversold (WITH trend)' if idx == oversold_signals.index[0] else '')
    else:
        # AGAINST trend - orange
        ax1.scatter(idx, price, color='orange', marker='^', s=150, zorder=5,
                   label='RSI Oversold (AGAINST trend)' if idx == oversold_signals.index[0] else '')

ax1.set_title('Price Action with RSI Signals and MA Filter\n(Green = WITH trend, Orange = AGAINST trend)', fontsize=14)
ax1.set_ylabel('Price')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# RSI subplot
ax2 = axes[1]
ax2.plot(test_data.index, test_data['rsi'], label='RSI(14)', color='purple', linewidth=1)
ax2.axhline(y=RSI_OVERSOLD, color='green', linestyle='--', alpha=0.7, label=f'Oversold ({RSI_OVERSOLD})')
ax2.axhline(y=RSI_OVERBOUGHT, color='red', linestyle='--', alpha=0.7, label=f'Overbought ({RSI_OVERBOUGHT})')
ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
ax2.set_ylabel('RSI')
ax2.set_ylim(0, 100)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Price vs SMA difference
ax3 = axes[2]
price_vs_sma50 = (test_data['close'] - test_data['sma_50']) / test_data['sma_50'] * 100
ax3.fill_between(test_data.index, price_vs_sma50, 0,
                  where=price_vs_sma50 >= 0, color='green', alpha=0.3, label='Price > SMA(50)')
ax3.fill_between(test_data.index, price_vs_sma50, 0,
                  where=price_vs_sma50 < 0, color='red', alpha=0.3, label='Price < SMA(50)')
ax3.axhline(y=0, color='black', linewidth=1)
ax3.set_ylabel('Price vs SMA(50) %')
ax3.set_xlabel('Date')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_PATH / 'price_action_with_filter.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_PATH / 'price_action_with_filter.png'}")

# =============================================================================
# PLOT 4: Why MA Filter is Too Aggressive
# =============================================================================
print("Creating 'why filter is too aggressive' chart...")

fig, ax = plt.subplots(figsize=(12, 6))

# Data for the visualization
categories = ['RSI Alone\n(Baseline)', 'RSI + SMA(20)\nWITH trend', 'RSI + SMA(50)\nWITH trend', 'RSI + SMA(100)\nWITH trend']

# Out of sample data
signals = [596, 0, 21, 63]
failures = [2, 0, 0, 0]
success_rates = [99.7, 0, 100, 100]

x = np.arange(len(categories))
width = 0.6

colors = ['#2196F3', '#808080', '#4CAF50', '#4CAF50']
bars = ax.bar(x, signals, width, color=colors, edgecolor='black')

# Add labels
for bar, sig, fail, rate in zip(bars, signals, failures, success_rates):
    if sig > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f'{sig} signals\n{fail} failures\n{rate}% success',
                ha='center', va='bottom', fontsize=10)
    else:
        ax.text(bar.get_x() + bar.get_width()/2, 20,
                'Zero signals!\n(Too strict)',
                ha='center', va='bottom', fontsize=10, color='red')

ax.set_title('MA Filter is TOO Aggressive\n(Out-of-Sample: 2024-2025, Oversold Signals)', fontsize=14)
ax.set_ylabel('Number of Signals')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 700)

# Add annotation
ax.annotate('RSI already has 99.7% success!\nMA filter removes 97% of opportunities\nbut only improves success by 0.3%',
            xy=(0.7, 0.75), xycoords='axes fraction',
            fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig(PLOTS_PATH / 'filter_too_aggressive.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_PATH / 'filter_too_aggressive.png'}")

# =============================================================================
# PLOT 5: WITH vs AGAINST Trend Comparison
# =============================================================================
print("Creating WITH vs AGAINST trend comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, period_name in enumerate(['in_sample', 'out_of_sample']):
    ax = axes[idx]

    oversold_data = results_df[(results_df['period_name'] == period_name) &
                               (results_df['signal_type'] == 'oversold')]

    # For SMA(50)
    with_trend = oversold_data[(oversold_data['filter'] == 'with_trend') &
                               (oversold_data['ma_period'] == 50)]
    against_trend = oversold_data[(oversold_data['filter'] == 'against_trend') &
                                   (oversold_data['ma_period'] == 50)]

    if len(with_trend) > 0 and len(against_trend) > 0:
        categories = ['WITH Trend\n(Price > SMA50)', 'AGAINST Trend\n(Price < SMA50)']

        with_t = with_trend.iloc[0]
        against_t = against_trend.iloc[0]

        signals = [with_t['total_signals'], against_t['total_signals']]
        bounce_rates = [with_t.get('bounce_rate', 0), against_t.get('bounce_rate', 0)]

        x = np.arange(len(categories))

        bars = ax.bar(x, signals, width=0.6, color=['#4CAF50', '#FF9800'], edgecolor='black')

        for bar, sig, rate in zip(bars, signals, bounce_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{int(sig)} signals\n{rate:.1f}% bounce',
                    ha='center', va='bottom', fontsize=10)

        ax.set_title(f'{period_name.upper().replace("_", " ")}\nSMA(50) Filter: WITH vs AGAINST Trend', fontsize=12)
        ax.set_ylabel('Number of Signals')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, max(signals) * 1.25)

plt.tight_layout()
plt.savefig(PLOTS_PATH / 'with_vs_against_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_PATH / 'with_vs_against_trend.png'}")

print("\n" + "="*60)
print("ALL VISUALIZATIONS COMPLETE!")
print("="*60)
print(f"\nSaved to: {PLOTS_PATH}")
