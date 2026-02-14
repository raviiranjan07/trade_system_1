"""
EXP-001: RSI Behavior Analysis Visualizations
Creates charts to visualize RSI mean reversion patterns
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup
RESULTS_PATH = Path(__file__).parent / "results.csv"
PLOTS_PATH = Path(__file__).parent / "plots"
PLOTS_PATH.mkdir(exist_ok=True)

# Load results
df = pd.read_csv(RESULTS_PATH)

# Create config labels
df['config'] = df.apply(lambda x: f"RSI({int(x['period'])}) {int(x['oversold_threshold'])}/{int(x['overbought_threshold'])}", axis=1)

# Split by period
in_sample = df[df['period_name'] == 'in_sample'].copy()
out_sample = df[df['period_name'] == 'out_of_sample'].copy()

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'in_sample': '#2ecc71', 'out_sample': '#3498db'}

# =============================================================================
# Chart 1: Bounce Rate Comparison (In-Sample vs Out-of-Sample)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(in_sample))
width = 0.35

bars1 = ax.bar(x - width/2, in_sample['oversold_bounce_rate_h10'], width,
               label='In-Sample (2020-2023)', color=colors['in_sample'], alpha=0.8)
bars2 = ax.bar(x + width/2, out_sample['oversold_bounce_rate_h10'], width,
               label='Out-of-Sample (2024-2025)', color=colors['out_sample'], alpha=0.8)

ax.set_ylabel('Bounce Rate (%)', fontsize=12)
ax.set_xlabel('RSI Configuration', fontsize=12)
ax.set_title('RSI Oversold Bounce Rate at H=10 (10 bars = 2.5 hours)\nHow often price bounces up after RSI goes oversold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(in_sample['config'], rotation=45, ha='right')
ax.legend(loc='lower right')
ax.set_ylim(96, 100)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_PATH / '1_bounce_rate_comparison.png', dpi=150)
plt.close()
print("Created: 1_bounce_rate_comparison.png")

# =============================================================================
# Chart 2: Average Move Size Comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width/2, in_sample['oversold_avg_move_h10'], width,
               label='In-Sample (2020-2023)', color=colors['in_sample'], alpha=0.8)
bars2 = ax.bar(x + width/2, out_sample['oversold_avg_move_h10'], width,
               label='Out-of-Sample (2024-2025)', color=colors['out_sample'], alpha=0.8)

# Add minimum profitable line
ax.axhline(y=12, color='red', linestyle='--', linewidth=2, label='Min Profitable (12 bps)')

ax.set_ylabel('Average Bounce (bps)', fontsize=12)
ax.set_xlabel('RSI Configuration', fontsize=12)
ax.set_title('Average Bounce Size at H=10 (10 bars = 2.5 hours)\nHow much price typically moves up after oversold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(in_sample['config'], rotation=45, ha='right')
ax.legend(loc='upper right')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_PATH / '2_avg_move_comparison.png', dpi=150)
plt.close()
print("Created: 2_avg_move_comparison.png")

# =============================================================================
# Chart 3: Signal Frequency (Number of Trades)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width/2, in_sample['oversold_crosses'], width,
               label='In-Sample (2020-2023)', color=colors['in_sample'], alpha=0.8)
bars2 = ax.bar(x + width/2, out_sample['oversold_crosses'], width,
               label='Out-of-Sample (2024-2025)', color=colors['out_sample'], alpha=0.8)

ax.set_ylabel('Number of Oversold Signals', fontsize=12)
ax.set_xlabel('RSI Configuration', fontsize=12)
ax.set_title('Signal Frequency by RSI Configuration\nHow many trading opportunities each setting generates', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(in_sample['config'], rotation=45, ha='right')
ax.legend(loc='upper right')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{int(height):,}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{int(height):,}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_PATH / '3_signal_frequency.png', dpi=150)
plt.close()
print("Created: 3_signal_frequency.png")

# =============================================================================
# Chart 4: Bounce Rate by Horizon (Time Decay)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

horizons = [3, 5, 10, 15, 20]
horizon_labels = ['H=3\n(45min)', 'H=5\n(1.25hr)', 'H=10\n(2.5hr)', 'H=15\n(3.75hr)', 'H=20\n(5hr)']

# Select best config for visualization: RSI(14) 20/80
best_in = in_sample[in_sample['config'] == 'RSI(14) 20/80'].iloc[0]
best_out = out_sample[out_sample['config'] == 'RSI(14) 20/80'].iloc[0]

# Bounce rates by horizon
bounce_rates_in = [best_in[f'oversold_bounce_rate_h{h}'] for h in horizons]
bounce_rates_out = [best_out[f'oversold_bounce_rate_h{h}'] for h in horizons]

axes[0].plot(horizon_labels, bounce_rates_in, 'o-', color=colors['in_sample'],
             linewidth=2, markersize=10, label='In-Sample')
axes[0].plot(horizon_labels, bounce_rates_out, 's-', color=colors['out_sample'],
             linewidth=2, markersize=10, label='Out-of-Sample')
axes[0].set_ylabel('Bounce Rate (%)', fontsize=12)
axes[0].set_xlabel('Lookforward Horizon', fontsize=12)
axes[0].set_title('RSI(14) 20/80: Bounce Rate Over Time', fontsize=13)
axes[0].legend()
axes[0].set_ylim(98, 100)
axes[0].grid(True, alpha=0.3)

# Average moves by horizon
avg_moves_in = [best_in[f'oversold_avg_move_h{h}'] for h in horizons]
avg_moves_out = [best_out[f'oversold_avg_move_h{h}'] for h in horizons]

axes[1].plot(horizon_labels, avg_moves_in, 'o-', color=colors['in_sample'],
             linewidth=2, markersize=10, label='In-Sample')
axes[1].plot(horizon_labels, avg_moves_out, 's-', color=colors['out_sample'],
             linewidth=2, markersize=10, label='Out-of-Sample')
axes[1].axhline(y=12, color='red', linestyle='--', linewidth=2, label='Min Profitable (12 bps)')
axes[1].set_ylabel('Average Move (bps)', fontsize=12)
axes[1].set_xlabel('Lookforward Horizon', fontsize=12)
axes[1].set_title('RSI(14) 20/80: Average Move Over Time', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_PATH / '4_horizon_analysis.png', dpi=150)
plt.close()
print("Created: 4_horizon_analysis.png")

# =============================================================================
# Chart 5: Trade-off: Signal Frequency vs Move Size
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# In-sample scatter
scatter1 = ax.scatter(in_sample['oversold_crosses'], in_sample['oversold_avg_move_h10'],
                      s=200, c=colors['in_sample'], alpha=0.7, label='In-Sample', edgecolors='white', linewidth=2)

# Out-of-sample scatter
scatter2 = ax.scatter(out_sample['oversold_crosses'], out_sample['oversold_avg_move_h10'],
                      s=200, c=colors['out_sample'], alpha=0.7, label='Out-of-Sample', edgecolors='white', linewidth=2)

# Add labels for each point
for idx, row in in_sample.iterrows():
    ax.annotate(row['config'], (row['oversold_crosses'], row['oversold_avg_move_h10']),
                textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9)

# Add minimum profitable line
ax.axhline(y=12, color='red', linestyle='--', linewidth=2, label='Min Profitable (12 bps)')

ax.set_xlabel('Number of Signals (More Trades)', fontsize=12)
ax.set_ylabel('Average Move Size (bps)', fontsize=12)
ax.set_title('Trade-off: Signal Frequency vs Move Size\nTop-right = fewer but bigger moves, Bottom-left = more but smaller moves', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_PATH / '5_tradeoff_analysis.png', dpi=150)
plt.close()
print("Created: 5_tradeoff_analysis.png")

# =============================================================================
# Chart 6: Summary Heatmap - All Combinations
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Prepare data for heatmap
configs = in_sample['config'].tolist()
metrics_in = in_sample[['oversold_bounce_rate_h10', 'oversold_avg_move_h10', 'oversold_crosses']].values
metrics_out = out_sample[['oversold_bounce_rate_h10', 'oversold_avg_move_h10', 'oversold_crosses']].values

# Normalize for comparison
from matplotlib.colors import Normalize

# In-sample heatmap
data_in = np.array([
    in_sample['oversold_bounce_rate_h10'].values,
    in_sample['oversold_avg_move_h10'].values,
    in_sample['oversold_crosses'].values / 100  # Scale down for display
]).T

im1 = axes[0].imshow(data_in, cmap='RdYlGn', aspect='auto')
axes[0].set_yticks(range(len(configs)))
axes[0].set_yticklabels(configs)
axes[0].set_xticks([0, 1, 2])
axes[0].set_xticklabels(['Bounce Rate (%)', 'Avg Move (bps)', 'Signals (x100)'])
axes[0].set_title('In-Sample (2020-2023)', fontsize=13)

# Add text annotations
for i in range(len(configs)):
    axes[0].text(0, i, f'{in_sample.iloc[i]["oversold_bounce_rate_h10"]:.1f}%', ha='center', va='center', fontsize=10)
    axes[0].text(1, i, f'{in_sample.iloc[i]["oversold_avg_move_h10"]:.0f}', ha='center', va='center', fontsize=10)
    axes[0].text(2, i, f'{int(in_sample.iloc[i]["oversold_crosses"]):,}', ha='center', va='center', fontsize=10)

# Out-of-sample heatmap
data_out = np.array([
    out_sample['oversold_bounce_rate_h10'].values,
    out_sample['oversold_avg_move_h10'].values,
    out_sample['oversold_crosses'].values / 100
]).T

im2 = axes[1].imshow(data_out, cmap='RdYlGn', aspect='auto')
axes[1].set_yticks(range(len(configs)))
axes[1].set_yticklabels(configs)
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(['Bounce Rate (%)', 'Avg Move (bps)', 'Signals (x100)'])
axes[1].set_title('Out-of-Sample (2024-2025)', fontsize=13)

# Add text annotations
for i in range(len(configs)):
    axes[1].text(0, i, f'{out_sample.iloc[i]["oversold_bounce_rate_h10"]:.1f}%', ha='center', va='center', fontsize=10)
    axes[1].text(1, i, f'{out_sample.iloc[i]["oversold_avg_move_h10"]:.0f}', ha='center', va='center', fontsize=10)
    axes[1].text(2, i, f'{int(out_sample.iloc[i]["oversold_crosses"]):,}', ha='center', va='center', fontsize=10)

plt.suptitle('RSI Mean Reversion: Complete Summary at H=10', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_PATH / '6_summary_heatmap.png', dpi=150)
plt.close()
print("Created: 6_summary_heatmap.png")

print("\n" + "="*50)
print("All visualizations saved to:", PLOTS_PATH)
print("="*50)
