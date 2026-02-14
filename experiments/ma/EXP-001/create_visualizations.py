"""
EXP-001 (MA): Visualizations for Moving Average Behavior Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup
RESULTS_PATH = Path("experiments/ma/EXP-001")
PLOTS_PATH = RESULTS_PATH / "plots"
PLOTS_PATH.mkdir(exist_ok=True)

# Load results
df = pd.read_csv(RESULTS_PATH / "results.csv")

# Split by period
in_sample = df[df['period_name'] == 'in_sample'].copy()
out_sample = df[df['period_name'] == 'out_of_sample'].copy()

plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# Chart 1: Continuation Rate Comparison by MA Type
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, period_name, data in zip(axes, ['In-Sample (2020-2023)', 'Out-of-Sample (2024-2025)'],
                                  [in_sample, out_sample]):
    periods = [7, 9, 20, 50, 100, 200]
    x = np.arange(len(periods))
    width = 0.25

    sma_rates = [data[(data['ma_type'] == 'SMA') & (data['period'] == p)]['cross_above_continuation_rate_h10'].values[0]
                 for p in periods]
    ema_rates = [data[(data['ma_type'] == 'EMA') & (data['period'] == p)]['cross_above_continuation_rate_h10'].values[0]
                 for p in periods]
    wma_rates = [data[(data['ma_type'] == 'WMA') & (data['period'] == p)]['cross_above_continuation_rate_h10'].values[0]
                 for p in periods]

    ax.bar(x - width, sma_rates, width, label='SMA', color='#3498db', alpha=0.8)
    ax.bar(x, ema_rates, width, label='EMA', color='#e74c3c', alpha=0.8)
    ax.bar(x + width, wma_rates, width, label='WMA', color='#2ecc71', alpha=0.8)

    ax.set_ylabel('Continuation Rate (%)', fontsize=12)
    ax.set_xlabel('MA Period', fontsize=12)
    ax.set_title(f'{period_name}\nCross Above → Expect UP', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.legend()
    ax.set_ylim(97, 100)

plt.suptitle('Trend Continuation Rate at H=10 by MA Type', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_PATH / '1_continuation_rate_by_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: 1_continuation_rate_by_type.png")

# =============================================================================
# Chart 2: Average Move Comparison by MA Type
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, period_name, data in zip(axes, ['In-Sample (2020-2023)', 'Out-of-Sample (2024-2025)'],
                                  [in_sample, out_sample]):
    periods = [7, 9, 20, 50, 100, 200]
    x = np.arange(len(periods))
    width = 0.25

    sma_moves = [data[(data['ma_type'] == 'SMA') & (data['period'] == p)]['cross_above_avg_move_h10'].values[0]
                 for p in periods]
    ema_moves = [data[(data['ma_type'] == 'EMA') & (data['period'] == p)]['cross_above_avg_move_h10'].values[0]
                 for p in periods]
    wma_moves = [data[(data['ma_type'] == 'WMA') & (data['period'] == p)]['cross_above_avg_move_h10'].values[0]
                 for p in periods]

    ax.bar(x - width, sma_moves, width, label='SMA', color='#3498db', alpha=0.8)
    ax.bar(x, ema_moves, width, label='EMA', color='#e74c3c', alpha=0.8)
    ax.bar(x + width, wma_moves, width, label='WMA', color='#2ecc71', alpha=0.8)

    ax.axhline(y=12, color='red', linestyle='--', linewidth=2, label='Min Profitable (12 bps)')

    ax.set_ylabel('Average Move (bps)', fontsize=12)
    ax.set_xlabel('MA Period', fontsize=12)
    ax.set_title(f'{period_name}\nCross Above → Expect UP', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.legend()

plt.suptitle('Average Move Size at H=10 by MA Type', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_PATH / '2_avg_move_by_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: 2_avg_move_by_type.png")

# =============================================================================
# Chart 3: Signal Frequency by MA Type
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, period_name, data in zip(axes, ['In-Sample (2020-2023)', 'Out-of-Sample (2024-2025)'],
                                  [in_sample, out_sample]):
    periods = [7, 9, 20, 50, 100, 200]
    x = np.arange(len(periods))
    width = 0.25

    sma_signals = [data[(data['ma_type'] == 'SMA') & (data['period'] == p)]['cross_above_count'].values[0]
                   for p in periods]
    ema_signals = [data[(data['ma_type'] == 'EMA') & (data['period'] == p)]['cross_above_count'].values[0]
                   for p in periods]
    wma_signals = [data[(data['ma_type'] == 'WMA') & (data['period'] == p)]['cross_above_count'].values[0]
                   for p in periods]

    ax.bar(x - width, sma_signals, width, label='SMA', color='#3498db', alpha=0.8)
    ax.bar(x, ema_signals, width, label='EMA', color='#e74c3c', alpha=0.8)
    ax.bar(x + width, wma_signals, width, label='WMA', color='#2ecc71', alpha=0.8)

    ax.set_ylabel('Number of Signals', fontsize=12)
    ax.set_xlabel('MA Period', fontsize=12)
    ax.set_title(f'{period_name}', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.legend()

plt.suptitle('Signal Frequency: Cross Above Events', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_PATH / '3_signal_frequency.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: 3_signal_frequency.png")

# =============================================================================
# Chart 4: In-Sample vs Out-of-Sample Comparison
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Group by MA type, show all periods
ma_types = ['SMA', 'EMA', 'WMA']
periods = [7, 9, 20, 50, 100, 200]

# Left plot: Continuation rates
ax = axes[0]
x = np.arange(len(ma_types) * len(periods))
labels = [f'{ma}({p})' for ma in ma_types for p in periods]

in_rates = []
out_rates = []
for ma in ma_types:
    for p in periods:
        in_rates.append(in_sample[(in_sample['ma_type'] == ma) & (in_sample['period'] == p)]['cross_above_continuation_rate_h10'].values[0])
        out_rates.append(out_sample[(out_sample['ma_type'] == ma) & (out_sample['period'] == p)]['cross_above_continuation_rate_h10'].values[0])

width = 0.35
ax.bar(x - width/2, in_rates, width, label='In-Sample', color='#2ecc71', alpha=0.8)
ax.bar(x + width/2, out_rates, width, label='Out-of-Sample', color='#3498db', alpha=0.8)
ax.set_ylabel('Continuation Rate (%)', fontsize=12)
ax.set_title('Continuation Rate: In-Sample vs Out-of-Sample', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.set_ylim(97, 100)

# Right plot: Average moves
ax = axes[1]
in_moves = []
out_moves = []
for ma in ma_types:
    for p in periods:
        in_moves.append(in_sample[(in_sample['ma_type'] == ma) & (in_sample['period'] == p)]['cross_above_avg_move_h10'].values[0])
        out_moves.append(out_sample[(out_sample['ma_type'] == ma) & (out_sample['period'] == p)]['cross_above_avg_move_h10'].values[0])

ax.bar(x - width/2, in_moves, width, label='In-Sample', color='#2ecc71', alpha=0.8)
ax.bar(x + width/2, out_moves, width, label='Out-of-Sample', color='#3498db', alpha=0.8)
ax.axhline(y=12, color='red', linestyle='--', linewidth=2, label='Min Profitable')
ax.set_ylabel('Average Move (bps)', fontsize=12)
ax.set_title('Average Move: In-Sample vs Out-of-Sample', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.legend()

plt.tight_layout()
plt.savefig(PLOTS_PATH / '4_in_vs_out_sample.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: 4_in_vs_out_sample.png")

# =============================================================================
# Chart 5: Summary Heatmap
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

for ax, period_name, data in zip(axes, ['In-Sample (2020-2023)', 'Out-of-Sample (2024-2025)'],
                                  [in_sample, out_sample]):
    # Create pivot table for continuation rates
    pivot_data = data.pivot(index='ma_type', columns='period', values='cross_above_continuation_rate_h10')
    pivot_data = pivot_data[periods]  # Ensure correct order

    im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=97, vmax=100)

    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods)
    ax.set_yticks(range(len(ma_types)))
    ax.set_yticklabels(pivot_data.index)
    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('MA Type', fontsize=12)
    ax.set_title(f'{period_name}', fontsize=12)

    # Add text annotations
    for i in range(len(ma_types)):
        for j in range(len(periods)):
            text = f'{pivot_data.iloc[i, j]:.1f}%'
            ax.text(j, i, text, ha='center', va='center', fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Continuation Rate (%)')

plt.suptitle('Trend Continuation Rate Heatmap at H=10', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_PATH / '5_summary_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: 5_summary_heatmap.png")

print("\n" + "="*50)
print("All visualizations saved to:", PLOTS_PATH)
print("="*50)
