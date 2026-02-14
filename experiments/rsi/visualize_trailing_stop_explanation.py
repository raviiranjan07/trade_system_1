"""
VISUALIZATION: How Trailing Stop Works
Clear visual explanation of why early exit at small loss is better
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

PLOTS_PATH = Path("experiments/rsi/plots")
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FIGURE 1: How Trailing Stop Works (Basic)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# LEFT: Trailing stop locks in PROFIT
ax1 = axes[0]
bars = np.arange(0, 11)
# Price path: goes up, then comes back down
price_profit = [0, 15, 30, 45, 50, 45, 40, 35, 30, 25, 20]

ax1.plot(bars, price_profit, 'b-o', linewidth=2, markersize=8, label='Price Path')
ax1.axhline(0, color='gray', linestyle='-', alpha=0.5)

# Show peak
ax1.scatter([4], [50], color='green', s=200, zorder=5, marker='^')
ax1.annotate('PEAK\n+50 bps', xy=(4, 50), xytext=(4.5, 55), fontsize=11, ha='left',
            arrowprops=dict(arrowstyle='->', color='green'))

# Show trailing stop trigger (20 bps below peak)
trailing_stop_level = 50 - 20  # 30 bps
ax1.axhline(trailing_stop_level, color='red', linestyle='--', linewidth=2, label='Trailing Stop Level (Peak - 20)')

# Mark exit point
ax1.scatter([8], [30], color='red', s=200, zorder=5, marker='X')
ax1.annotate('EXIT!\n+30 bps\n(+22 net)', xy=(8, 30), xytext=(8.5, 35), fontsize=11, ha='left',
            color='red', fontweight='bold')

ax1.set_xlabel('Bar', fontsize=12)
ax1.set_ylabel('Profit (bps)', fontsize=12)
ax1.set_title('WINNER: Trailing Stop Locks In Profit', fontsize=14, fontweight='bold', color='green')
ax1.set_xlim(-0.5, 10.5)
ax1.set_ylim(-10, 70)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Add explanation box
explanation1 = """
1. Price goes UP (+50 bps peak)
2. Trailing stop follows: 50 - 20 = 30 bps
3. Price drops back to 30 bps
4. STOP TRIGGERS! Exit at +30 bps
5. Net profit: +30 - 8 = +22 bps

GOOD: Locked in profit!
"""
ax1.text(0.02, 0.02, explanation1, transform=ax1.transAxes, fontsize=10,
         verticalalignment='bottom', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))


# RIGHT: Trailing stop triggers at LOSS (small peak)
ax2 = axes[1]
# Price path: small up, then down
price_loss = [0, 10, 18, 25, 20, 10, 5, 0, -5, -10, -15]

ax2.plot(bars, price_loss, 'b-o', linewidth=2, markersize=8, label='Price Path')
ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)

# Show peak (small)
ax2.scatter([3], [25], color='orange', s=200, zorder=5, marker='^')
ax2.annotate('PEAK\n+25 bps\n(small)', xy=(3, 25), xytext=(3.5, 32), fontsize=11, ha='left',
            arrowprops=dict(arrowstyle='->', color='orange'))

# Show trailing stop trigger (20 bps below peak = 5 bps)
trailing_stop_level = 25 - 20  # 5 bps
ax2.axhline(trailing_stop_level, color='red', linestyle='--', linewidth=2, label='Trailing Stop Level (Peak - 20)')

# Mark exit point
ax2.scatter([6], [5], color='red', s=200, zorder=5, marker='X')
ax2.annotate('EXIT!\n+5 bps\n(-3 net)', xy=(6, 5), xytext=(6.5, 12), fontsize=11, ha='left',
            color='red', fontweight='bold')

ax2.set_xlabel('Bar', fontsize=12)
ax2.set_ylabel('Profit (bps)', fontsize=12)
ax2.set_title('LOSER: Trailing Stop Triggers at Small Loss', fontsize=14, fontweight='bold', color='orange')
ax2.set_xlim(-0.5, 10.5)
ax2.set_ylim(-30, 50)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Add explanation box
explanation2 = """
1. Price goes UP (+25 bps peak)
2. Trailing stop follows: 25 - 20 = 5 bps
3. Price drops to 5 bps
4. STOP TRIGGERS! Exit at +5 bps
5. Net profit: +5 - 8 = -3 bps (LOSS)

Small peak - 20 bps = LOSS
"""
ax2.text(0.02, 0.02, explanation2, transform=ax2.transAxes, fontsize=10,
         verticalalignment='bottom', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('How Trailing Stop Works: Same Logic, Different Outcomes', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_PATH / 'trailing_stop_explanation_1.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_PATH / 'trailing_stop_explanation_1.png'}")


# =============================================================================
# FIGURE 2: Why Early Exit is BETTER
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Same price path for both
bars = np.arange(0, 11)
price_path = [0, 10, 18, 25, 20, 10, 5, 0, -30, -60, -100]

# LEFT: V1 - Exit early at small loss
ax1 = axes[0]
ax1.plot(bars, price_path, 'b-o', linewidth=2, markersize=8, label='Price Path')
ax1.axhline(0, color='gray', linestyle='-', alpha=0.5)

# Peak
ax1.scatter([3], [25], color='orange', s=150, zorder=5, marker='^')

# Trailing stop level
ax1.axhline(5, color='red', linestyle='--', linewidth=2, alpha=0.7)

# Exit point (Bar 6)
ax1.scatter([6], [5], color='red', s=200, zorder=5, marker='X')
ax1.annotate('V1 EXIT\nBar 6\n+5 bps gross\n-3 bps net', xy=(6, 5), xytext=(6.5, 20), fontsize=11,
            color='red', fontweight='bold', arrowprops=dict(arrowstyle='->', color='red'))

# Show what we avoided
ax1.fill_between(bars[6:], price_path[6:], 5, alpha=0.3, color='red', label='Avoided Loss')

ax1.set_xlabel('Bar', fontsize=12)
ax1.set_ylabel('Profit (bps)', fontsize=12)
ax1.set_title('V1: Exit Early at Small Loss (-3 bps)', fontsize=14, fontweight='bold', color='green')
ax1.set_xlim(-0.5, 10.5)
ax1.set_ylim(-120, 50)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

result1 = """
RESULT: -3 bps net

Exited at Bar 6
Avoided -100 bps disaster!
"""
ax1.text(0.02, 0.02, result1, transform=ax1.transAxes, fontsize=12,
         verticalalignment='bottom', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))


# RIGHT: Profit-Protected - Wait (don't exit at loss)
ax2 = axes[1]
ax2.plot(bars, price_path, 'b-o', linewidth=2, markersize=8, label='Price Path')
ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)

# Peak
ax2.scatter([3], [25], color='orange', s=150, zorder=5, marker='^')

# No exit at Bar 6 because exit would be loss
ax2.scatter([6], [5], color='gray', s=100, zorder=5, marker='o', alpha=0.5)
ax2.annotate('Would be -3 bps\nDON\'T EXIT\n(wait for better)', xy=(6, 5), xytext=(4, -20), fontsize=10,
            color='gray', arrowprops=dict(arrowstyle='->', color='gray'))

# Forced exit at Bar 10
ax2.scatter([10], [-100], color='darkred', s=200, zorder=5, marker='X')
ax2.annotate('FORCED EXIT\nBar 10\n-100 bps gross\n-108 bps net!', xy=(10, -100), xytext=(7, -80), fontsize=11,
            color='darkred', fontweight='bold', arrowprops=dict(arrowstyle='->', color='darkred'))

ax2.set_xlabel('Bar', fontsize=12)
ax2.set_ylabel('Profit (bps)', fontsize=12)
ax2.set_title('Profit-Protected: Wait... Then Disaster (-108 bps)', fontsize=14, fontweight='bold', color='red')
ax2.set_xlim(-0.5, 10.5)
ax2.set_ylim(-120, 50)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

result2 = """
RESULT: -108 bps net

Waited for "recovery"
Got DISASTER instead!
"""
ax2.text(0.02, 0.02, result2, transform=ax2.transAxes, fontsize=12,
         verticalalignment='bottom', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))


plt.suptitle('Why V1 (Early Exit) is BETTER: Same Trade, Different Outcome', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_PATH / 'trailing_stop_explanation_2.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_PATH / 'trailing_stop_explanation_2.png'}")


# =============================================================================
# FIGURE 3: Summary Comparison
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

summary_text = """
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                        TRAILING STOP: WHY EARLY EXIT WINS                            ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   SCENARIO: Price peaks at +25 bps, then starts falling                              ║
║                                                                                      ║
║   ┌─────────────────────────────────────────────────────────────────────────────┐    ║
║   │                                                                             │    ║
║   │   V1 (Original):                    Profit-Protected:                       │    ║
║   │   ─────────────                     ──────────────────                      │    ║
║   │                                                                             │    ║
║   │   Bar 3: Peak +25 bps               Bar 3: Peak +25 bps                     │    ║
║   │   Bar 6: Price at +5 bps            Bar 6: Price at +5 bps                  │    ║
║   │          Trailing stop triggers!           "Exit would be -3 bps loss"      │    ║
║   │          EXIT at +5 bps                    "Let's wait for recovery..."     │    ║
║   │                                                                             │    ║
║   │   Bar 7-10: We're OUT               Bar 7-10: Price keeps falling!          │    ║
║   │             (safe!)                          -30... -60... -100 bps         │    ║
║   │                                                                             │    ║
║   │   RESULT: -3 bps net                RESULT: -108 bps net                    │    ║
║   │           (small loss)                      (DISASTER!)                     │    ║
║   │                                                                             │    ║
║   └─────────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                      ║
║   ════════════════════════════════════════════════════════════════════════════════   ║
║                                                                                      ║
║   THE LESSON:                                                                        ║
║                                                                                      ║
║   • Trailing stop that exits at SMALL LOSS is actually PROTECTING you                ║
║   • "Wait for recovery" sounds good but often leads to BIGGER losses                 ║
║   • V1 total: +3024 bps  vs  Profit-Protected total: +2299 bps                       ║
║   • Taking small losses = BETTER overall result                                      ║
║                                                                                      ║
║   ════════════════════════════════════════════════════════════════════════════════   ║
║                                                                                      ║
║   "Cut your losses short, let your winners run"                                      ║
║   V1 does exactly this!                                                              ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))

plt.savefig(PLOTS_PATH / 'trailing_stop_explanation_3.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_PATH / 'trailing_stop_explanation_3.png'}")


# =============================================================================
# FIGURE 4: Visual Formula
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Scenario 1: Big peak = Profit
ax1 = axes[0, 0]
ax1.axis('off')
ax1.text(0.5, 0.7, 'BIG PEAK', fontsize=20, ha='center', fontweight='bold', color='green')
ax1.text(0.5, 0.5, '+50 bps', fontsize=24, ha='center', fontweight='bold')
ax1.text(0.5, 0.3, '- 20 bps (trailing stop)', fontsize=16, ha='center')
ax1.text(0.5, 0.15, '─────────────', fontsize=16, ha='center')
ax1.text(0.5, 0.0, '= +30 bps exit (+22 net)', fontsize=18, ha='center', color='green', fontweight='bold')
ax1.add_patch(plt.Rectangle((0.1, -0.1), 0.8, 1.0, fill=False, edgecolor='green', linewidth=3))

# Scenario 2: Small peak = Small loss
ax2 = axes[0, 1]
ax2.axis('off')
ax2.text(0.5, 0.7, 'SMALL PEAK', fontsize=20, ha='center', fontweight='bold', color='orange')
ax2.text(0.5, 0.5, '+25 bps', fontsize=24, ha='center', fontweight='bold')
ax2.text(0.5, 0.3, '- 20 bps (trailing stop)', fontsize=16, ha='center')
ax2.text(0.5, 0.15, '─────────────', fontsize=16, ha='center')
ax2.text(0.5, 0.0, '= +5 bps exit (-3 net)', fontsize=18, ha='center', color='orange', fontweight='bold')
ax2.add_patch(plt.Rectangle((0.1, -0.1), 0.8, 1.0, fill=False, edgecolor='orange', linewidth=3))

# The rule
ax3 = axes[1, 0]
ax3.axis('off')
ax3.text(0.5, 0.7, 'THE RULE:', fontsize=18, ha='center', fontweight='bold')
ax3.text(0.5, 0.45, 'If Peak > 28 bps', fontsize=16, ha='center')
ax3.text(0.5, 0.3, '→ Exit is PROFIT', fontsize=16, ha='center', color='green')
ax3.text(0.5, 0.1, 'If Peak < 28 bps', fontsize=16, ha='center')
ax3.text(0.5, -0.05, '→ Exit is LOSS', fontsize=16, ha='center', color='orange')
ax3.add_patch(plt.Rectangle((0.1, -0.15), 0.8, 1.0, fill=True, facecolor='lightyellow', edgecolor='black', linewidth=2))

# Why it works
ax4 = axes[1, 1]
ax4.axis('off')
ax4.text(0.5, 0.7, 'WHY V1 WORKS:', fontsize=18, ha='center', fontweight='bold')
ax4.text(0.5, 0.5, '• Takes small losses (-3 bps)', fontsize=14, ha='center')
ax4.text(0.5, 0.35, '• Avoids big disasters (-100 bps)', fontsize=14, ha='center')
ax4.text(0.5, 0.2, '• Locks in big wins (+50 bps)', fontsize=14, ha='center')
ax4.text(0.5, 0.0, 'NET RESULT: +3024 bps', fontsize=16, ha='center', color='green', fontweight='bold')
ax4.add_patch(plt.Rectangle((0.1, -0.15), 0.8, 1.0, fill=True, facecolor='lightgreen', edgecolor='green', linewidth=2))

plt.suptitle('Trailing Stop Formula: Peak - 20 bps = Exit Price', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_PATH / 'trailing_stop_explanation_4.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_PATH / 'trailing_stop_explanation_4.png'}")

print("\n" + "="*70)
print("All visualizations saved to experiments/rsi/plots/")
print("="*70)
print("""
Files created:
1. trailing_stop_explanation_1.png - How trailing stop works (winner vs loser)
2. trailing_stop_explanation_2.png - Why early exit is better (same trade comparison)
3. trailing_stop_explanation_3.png - Text summary
4. trailing_stop_explanation_4.png - Visual formula
""")
