"""
Visualization of BTC 1-Minute Trading Analysis Findings

Run: .venv/Scripts/python.exe visualize_analysis.py

Generates comprehensive visualizations of all key findings.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Create output directory
output_dir = Path("visualizations")
output_dir.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# =============================================================================
# FIGURE 1: Direction Balance (50/50)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

directions = ['UP First', 'DOWN First', 'Neither']
percentages = [46.5, 47.1, 6.4]
colors = ['#2ecc71', '#e74c3c', '#95a5a6']

bars = ax.bar(directions, percentages, color=colors, edgecolor='black', linewidth=1.5)
ax.axhline(y=50, color='black', linestyle='--', alpha=0.7, label='50% line')

for bar, pct in zip(bars, percentages):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{pct}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Finding 2: Direction is Perfectly Balanced (50/50)\nNo Edge from Random Entry', fontsize=14, fontweight='bold')
ax.set_ylim(0, 60)
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / '1_direction_balance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 1_direction_balance.png")

# =============================================================================
# FIGURE 2: Clean Wins vs Any Wins by Target
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

targets = [8, 10, 15, 20, 25, 30, 40, 50]

# H=3 data
clean_h3 = [4.2, 3.5, 2.2, 1.4, 0.9, 0.6, 0.3, 0.2]
any_h3 = [38.4, 31.3, 19.3, 12.4, 8.4, 5.9, 3.2, 1.9]

# H=60 data
clean_h60 = [5.2, 4.5, 3.4, 2.5, 2.1, 1.6, 1.0, 0.6]  # Interpolated some values
any_h60 = [82.7, 76.0, 69.6, 60.0, 54.0, 45.0, 35.0, 29.9]

x = np.arange(len(targets))
width = 0.35

# H=3
ax = axes[0]
bars1 = ax.bar(x - width/2, clean_h3, width, label='Clean Win (no drawdown)', color='#27ae60')
bars2 = ax.bar(x + width/2, any_h3, width, label='Any Win', color='#3498db')
ax.set_xlabel('Target (bps)')
ax.set_ylabel('Win Rate (%)')
ax.set_title('H=3 bars: Clean vs Any Wins', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(targets)
ax.legend()
ax.set_ylim(0, 50)

# H=60
ax = axes[1]
bars1 = ax.bar(x - width/2, clean_h60, width, label='Clean Win (no drawdown)', color='#27ae60')
bars2 = ax.bar(x + width/2, any_h60, width, label='Any Win', color='#3498db')
ax.set_xlabel('Target (bps)')
ax.set_ylabel('Win Rate (%)')
ax.set_title('H=60 bars: Clean vs Any Wins', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(targets)
ax.legend()
ax.set_ylim(0, 100)

fig.suptitle('Finding 6: Most Trades See Drawdown Before Winning', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / '2_clean_vs_any_wins.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 2_clean_vs_any_wins.png")

# =============================================================================
# FIGURE 3: 3-Case Recovery Analysis
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

horizons = [3, 5, 10, 15, 30, 60]

# Target 8bp data
case1_8 = [5.3, 5.3, 5.4, 5.4, 5.4, 5.4]
case2_8 = [36.7, 45.9, 58.0, 64.6, 74.2, 81.8]
case3_8 = [58.0, 48.7, 36.6, 30.0, 20.4, 12.7]

# Target 15bp data
case1_15 = [9.6, 9.7, 9.8, 9.8, 9.8, 9.8]
case2_15 = [18.1, 25.7, 37.7, 45.3, 57.9, 68.7]
case3_15 = [72.2, 64.6, 52.5, 44.9, 32.3, 21.5]

# Target 25bp data
case1_25 = [16.2, 16.3, 16.3, 16.4, 16.4, 16.5]
case2_25 = [7.8, 12.4, 21.3, 27.8, 40.3, 53.2]
case3_25 = [76.1, 71.4, 62.4, 55.8, 43.3, 30.4]

colors = ['#e74c3c', '#27ae60', '#f39c12']

for ax, (c1, c2, c3, target) in zip(axes,
                                     [(case1_8, case2_8, case3_8, '8bp'),
                                      (case1_15, case2_15, case3_15, '15bp'),
                                      (case1_25, case2_25, case3_25, '25bp')]):
    x = np.arange(len(horizons))
    width = 0.25

    ax.bar(x - width, c1, width, label='Case 1: Wrong Dir', color=colors[0])
    ax.bar(x, c2, width, label='Case 2: Quick Rec', color=colors[1])
    ax.bar(x + width, c3, width, label='Case 3: Slow Rec', color=colors[2])

    ax.set_xlabel('Horizon (bars)')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Target = {target}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 100)

fig.suptitle('Finding 7: Recovery Analysis - Most Failures are Timing Issues, Not Direction',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / '3_recovery_cases.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 3_recovery_cases.png")

# =============================================================================
# FIGURE 4: MAE by Case
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

cases = ['Case 2\n(Quick Rec)', 'Case 3\n(Slow Rec)', 'Case 1\n(Wrong Dir)']
mae_ranges = [7.5, 34.5, 165]  # Midpoints of ranges
mae_errors = [[7.5-3, 34.5-16, 165-150], [12-7.5, 53-34.5, 180-165]]  # Error bars

colors = ['#27ae60', '#f39c12', '#e74c3c']
bars = ax.bar(cases, mae_ranges, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
labels = ['3-12bp', '16-53bp', '150-180bp']
for bar, label in zip(bars, labels):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            label, ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50bp threshold (wrong direction indicator)')
ax.set_ylabel('Maximum Adverse Excursion (bps)', fontsize=12)
ax.set_title('Finding 7: MAE by Recovery Case\nIf Drawdown > 50bp, Likely Wrong Direction',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 200)
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / '4_mae_by_case.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 4_mae_by_case.png")

# =============================================================================
# FIGURE 5: Recovery Time Distribution
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

targets_recovery = ['8bp', '15bp', '25bp']
horizons_labels = ['H=3', 'H=5', 'H=10', 'H=15', 'H=30', 'H=60']

# Median recovery time (bars after H)
recovery_8bp = [13, 17, 26, 34, 50, 75]
recovery_15bp = [21, 25, 34, 41, 59, 81]
recovery_25bp = [36, 40, 47, 54, 70, 90]

# 90th percentile total time from entry
total_90_8bp = [145, 171, 212, 244, 297, 355]
total_90_15bp = [196, 214, 248, 270, 319, 366]
total_90_25bp = [255, 266, 287, 304, 335, 374]

colors_gradient = plt.cm.Blues(np.linspace(0.3, 0.9, 6))

for ax, (med, p90, target) in zip(axes,
                                   [(recovery_8bp, total_90_8bp, '8bp'),
                                    (recovery_15bp, total_90_15bp, '15bp'),
                                    (recovery_25bp, total_90_25bp, '25bp')]):
    x = np.arange(len(horizons_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, med, width, label='Median Recovery Time', color='#3498db')
    bars2 = ax.bar(x + width/2, p90, width, label='90th Percentile Total', color='#e74c3c', alpha=0.7)

    ax.set_xlabel('Horizon')
    ax.set_ylabel('Bars')
    ax.set_title(f'Target = {target}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons_labels)
    ax.legend(loc='upper left', fontsize=9)

fig.suptitle('Recovery Time for Case 3 (Slow Recovery)\nMedian vs 90th Percentile',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / '5_recovery_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 5_recovery_time.png")

# =============================================================================
# FIGURE 6: Noise vs Real Moves
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

horizons = [3, 5, 10, 15, 30, 60]
noise = [45.0, 32.4, 18.1, 11.9, 5.1, 1.9]
real_up = [23.4, 27.0, 28.7, 28.0, 24.3, 19.2]
real_down = [23.7, 27.4, 29.2, 28.4, 24.6, 19.3]
real_both = [7.8, 13.3, 24.0, 31.7, 46.0, 59.6]

x = np.arange(len(horizons))
width = 0.2

ax.bar(x - 1.5*width, noise, width, label='Noise (<10bp)', color='#95a5a6')
ax.bar(x - 0.5*width, real_up, width, label='Real UP only', color='#27ae60')
ax.bar(x + 0.5*width, real_down, width, label='Real DOWN only', color='#e74c3c')
ax.bar(x + 1.5*width, real_both, width, label='Real BOTH', color='#9b59b6')

ax.set_xlabel('Horizon (bars)', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Finding 6: Noise vs Real Moves by Horizon\nAt H=3, 45% are noise. At H=60, 60% move BOTH ways.',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'H={h}' for h in horizons])
ax.legend(loc='upper right')
ax.set_ylim(0, 70)

plt.tight_layout()
plt.savefig(output_dir / '6_noise_vs_real.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 6_noise_vs_real.png")

# =============================================================================
# FIGURE 7: Adverse Excursion Distribution (H=60)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

outcomes = ['Hit 15bp\ntarget', 'Hit 25bp\ntarget', 'Hit 50bp\ntarget', 'Never hit\n15bp']
median_ae = [24.3, 22.7, 20.8, 39.8]
p75_ae = [52.1, 49.8, 47.2, 76.0]
p90_ae = [99.8, 97.2, 95.8, 132.2]

x = np.arange(len(outcomes))
width = 0.25

bars1 = ax.bar(x - width, median_ae, width, label='Median AE', color='#3498db')
bars2 = ax.bar(x, p75_ae, width, label='75th AE', color='#f39c12')
bars3 = ax.bar(x + width, p90_ae, width, label='90th AE', color='#e74c3c')

ax.set_xlabel('Trade Outcome', fontsize=12)
ax.set_ylabel('Adverse Excursion (bps)', fontsize=12)
ax.set_title('Finding 6: Adverse Excursion at H=60\nEven Winners Often Go 24+ bps Against You',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(outcomes)
ax.legend()
ax.set_ylim(0, 150)

plt.tight_layout()
plt.savefig(output_dir / '7_adverse_excursion.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 7_adverse_excursion.png")

# =============================================================================
# FIGURE 8: Summary Dashboard
# =============================================================================
fig = plt.figure(figsize=(16, 12))

# Title
fig.suptitle('BTC 1-Minute Scalping Analysis Summary\n3.15M Candles | 432 Parameter Combinations | 0 Profitable',
             fontsize=16, fontweight='bold', y=0.98)

# Subplot 1: Direction Balance
ax1 = fig.add_subplot(2, 3, 1)
directions = ['UP', 'DOWN']
vals = [46.5, 47.1]
colors = ['#2ecc71', '#e74c3c']
ax1.pie(vals, labels=directions, colors=colors, autopct='%1.1f%%', startangle=90)
ax1.set_title('Direction Balance\n(Perfect 50/50)', fontweight='bold')

# Subplot 2: Recovery Rate by Target
ax2 = fig.add_subplot(2, 3, 2)
targets = ['8bp', '15bp', '25bp']
wrong_dir = [5, 10, 16]
recover = [95, 90, 84]
x = np.arange(len(targets))
ax2.barh(x, recover, color='#27ae60', label='Eventually Recovers')
ax2.barh(x, wrong_dir, left=recover, color='#e74c3c', label='Wrong Direction')
ax2.set_yticks(x)
ax2.set_yticklabels(targets)
ax2.set_xlabel('Percentage (%)')
ax2.set_title('Recovery Rate\n(Most failures are timing)', fontweight='bold')
ax2.legend(loc='lower right', fontsize=8)
ax2.set_xlim(0, 100)

# Subplot 3: MAE Threshold
ax3 = fig.add_subplot(2, 3, 3)
cases = ['Quick\nRec', 'Slow\nRec', 'Wrong\nDir']
mae_vals = [7.5, 35, 165]
colors = ['#27ae60', '#f39c12', '#e74c3c']
ax3.bar(cases, mae_vals, color=colors)
ax3.axhline(y=50, color='black', linestyle='--', label='50bp threshold')
ax3.set_ylabel('MAE (bps)')
ax3.set_title('MAE by Case\n(>50bp = likely wrong)', fontweight='bold')
ax3.legend(fontsize=8)

# Subplot 4: Clean Win Rate
ax4 = fig.add_subplot(2, 3, 4)
targets = ['8bp', '15bp', '25bp', '50bp']
clean_rates = [4.2, 2.2, 0.9, 0.2]
dirty_rates = [34.2, 17.1, 7.5, 1.7]  # any_win - clean_win
ax4.bar(targets, clean_rates, label='Clean Win', color='#27ae60')
ax4.bar(targets, dirty_rates, bottom=clean_rates, label='Dirty Win', color='#3498db', alpha=0.7)
ax4.set_xlabel('Target')
ax4.set_ylabel('Win Rate (%)')
ax4.set_title('Clean vs Dirty Wins (H=3)\n(Most wins have drawdown)', fontweight='bold')
ax4.legend(fontsize=8)

# Subplot 5: Recovery Time
ax5 = fig.add_subplot(2, 3, 5)
horizons = ['H=3', 'H=10', 'H=30', 'H=60']
med_time = [24, 44, 89, 141]  # Target 15bp
ax5.bar(horizons, med_time, color='#3498db')
ax5.set_ylabel('Median Total Time (bars)')
ax5.set_title('Case 3 Recovery Time\n(Target=15bp)', fontweight='bold')

# Subplot 6: Key Numbers
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
key_stats = """
KEY FINDINGS:

Direction Balance:     50/50
Profitable Combos:     0 / 432
State Vector Edge:     0%
Best EV after fees:    -0.03 bps

THRESHOLDS:

Trading Fees:          8 bps
MAE Cutoff:            50 bps
Wrong Dir Recovery:    5-16%

RECOMMENDATION:

Random entry = NO EDGE
Need selective entry signal
"""
ax6.text(0.1, 0.9, key_stats, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(output_dir / '8_summary_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 8_summary_dashboard.png")

# =============================================================================
# FIGURE 9: Recovery Flow Diagram
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Draw flowchart style visualization
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Starting box
ax.add_patch(FancyBboxPatch((0.4, 0.85), 0.2, 0.1, boxstyle="round,pad=0.02",
                             facecolor='#3498db', edgecolor='black', linewidth=2))
ax.text(0.5, 0.9, 'ENTRY\n(Long Position)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Arrow down
ax.annotate('', xy=(0.5, 0.75), xytext=(0.5, 0.85),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Decision: Did price go below?
ax.add_patch(FancyBboxPatch((0.35, 0.6), 0.3, 0.15, boxstyle="round,pad=0.02",
                             facecolor='#f39c12', edgecolor='black', linewidth=2))
ax.text(0.5, 0.675, 'Price went\nbelow entry?', ha='center', va='center', fontsize=11, fontweight='bold')

# Clean Win branch (NO)
ax.annotate('', xy=(0.15, 0.675), xytext=(0.35, 0.675),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(0.25, 0.7, 'NO', ha='center', va='bottom', fontsize=10, color='green', fontweight='bold')

ax.add_patch(FancyBboxPatch((0.02, 0.6), 0.13, 0.15, boxstyle="round,pad=0.02",
                             facecolor='#27ae60', edgecolor='black', linewidth=2))
ax.text(0.085, 0.675, 'CLEAN\nWIN\n(2-5%)', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# YES branch
ax.annotate('', xy=(0.5, 0.5), xytext=(0.5, 0.6),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(0.52, 0.55, 'YES', ha='left', va='center', fontsize=10, color='red', fontweight='bold')

# How much drawdown?
ax.add_patch(FancyBboxPatch((0.35, 0.35), 0.3, 0.15, boxstyle="round,pad=0.02",
                             facecolor='#9b59b6', edgecolor='black', linewidth=2))
ax.text(0.5, 0.425, 'Drawdown\namount?', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# < 50bp branch (left)
ax.annotate('', xy=(0.15, 0.425), xytext=(0.35, 0.425),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(0.25, 0.45, '< 50bp', ha='center', va='bottom', fontsize=10, color='green', fontweight='bold')

# Case 2 & 3 box
ax.add_patch(FancyBboxPatch((0.02, 0.25), 0.13, 0.3, boxstyle="round,pad=0.02",
                             facecolor='#27ae60', edgecolor='black', linewidth=2))
ax.text(0.085, 0.4, 'CASE 2/3\nLikely\nRecovers\n\n84-95%\nwin rate\n\nHOLD',
        ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# > 50bp branch (right)
ax.annotate('', xy=(0.85, 0.425), xytext=(0.65, 0.425),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(0.75, 0.45, '> 50bp', ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')

# Case 1 box
ax.add_patch(FancyBboxPatch((0.85, 0.25), 0.13, 0.3, boxstyle="round,pad=0.02",
                             facecolor='#e74c3c', edgecolor='black', linewidth=2))
ax.text(0.915, 0.4, 'CASE 1\nLikely\nWrong Dir\n\n150-180bp\nMAE\n\nCUT',
        ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Title
ax.set_title('Decision Flow: Hold or Cut Based on Drawdown\n(Finding 7: Recovery Analysis)',
             fontsize=14, fontweight='bold', y=0.98)

# Legend box
legend_text = """
CASES:
Case 1: Wrong Direction (5-16%)  - MAE: 150-180bp
Case 2: Quick Recovery (within H) - MAE: 3-12bp
Case 3: Slow Recovery (after H)   - MAE: 16-53bp

RULE: If drawdown > 50bp, cut the trade
"""
ax.text(0.5, 0.1, legend_text, ha='center', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(output_dir / '9_decision_flow.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 9_decision_flow.png")

# =============================================================================
# COMPLETE
# =============================================================================
print("\n" + "="*60)
print("VISUALIZATION COMPLETE")
print("="*60)
print(f"\nAll visualizations saved to: {output_dir.absolute()}")
print("\nFiles created:")
for f in sorted(output_dir.glob("*.png")):
    print(f"  - {f.name}")
