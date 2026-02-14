"""
EXP-004: Analyze when MFE occurs (which bar?)
This helps determine optimal exit timing.
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
print("MFE TIMING ANALYSIS: At which BAR does maximum profit occur?")
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
# Analyze MFE timing for each signal type
# =============================================================================

def analyze_mfe_timing(data, signal_col, signal_type, filter_by_sma=False):
    """Find at which bar MFE occurs for each signal."""
    signals = data[data[signal_col] == True].copy()

    mfe_bars = []  # Which bar had MFE
    mfe_values = []  # What was the MFE value
    pnl_paths = []  # Full path for each signal

    for idx in signals.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + HORIZON >= len(data):
            continue

        entry_price = data.iloc[idx_pos]['close']
        sma_200 = data.iloc[idx_pos]['sma_200']

        # Apply 200 SMA filter if requested
        if filter_by_sma:
            if signal_type == 'long' and entry_price < sma_200:
                continue  # Skip LONG in BEAR
            if signal_type == 'short' and entry_price > sma_200:
                continue  # Skip SHORT in BULL

        # Calculate PnL at each bar
        bar_pnls = []
        for bar in range(1, HORIZON + 1):
            future_bar = data.iloc[idx_pos + bar]
            if signal_type == 'long':
                pnl = (future_bar['high'] - entry_price) / entry_price * 10000
            else:
                pnl = (entry_price - future_bar['low']) / entry_price * 10000
            bar_pnls.append(pnl)

        # Find which bar had MFE
        mfe = max(bar_pnls)
        mfe_bar = bar_pnls.index(mfe) + 1  # +1 because bars are 1-indexed

        mfe_bars.append(mfe_bar)
        mfe_values.append(mfe)
        pnl_paths.append(bar_pnls)

    return mfe_bars, mfe_values, pnl_paths

# Analyze WITHOUT filter (all signals)
print("\n" + "="*70)
print("WITHOUT 200 SMA FILTER (ALL SIGNALS)")
print("="*70)

long_mfe_bars, long_mfe_values, long_paths = analyze_mfe_timing(
    oos_data, 'rsi_oversold_cross', 'long', filter_by_sma=False
)
short_mfe_bars, short_mfe_values, short_paths = analyze_mfe_timing(
    oos_data, 'rsi_overbought_cross', 'short', filter_by_sma=False
)

print(f"\nLONG signals: {len(long_mfe_bars)}")
print(f"  MFE Bar distribution:")
for bar in range(1, HORIZON + 1):
    count = long_mfe_bars.count(bar)
    pct = count / len(long_mfe_bars) * 100 if long_mfe_bars else 0
    print(f"    Bar {bar:2d}: {count:3d} signals ({pct:5.1f}%)")

print(f"\nSHORT signals: {len(short_mfe_bars)}")
print(f"  MFE Bar distribution:")
for bar in range(1, HORIZON + 1):
    count = short_mfe_bars.count(bar)
    pct = count / len(short_mfe_bars) * 100 if short_mfe_bars else 0
    print(f"    Bar {bar:2d}: {count:3d} signals ({pct:5.1f}%)")

# Summary stats
print(f"\nLONG: Median MFE Bar = {np.median(long_mfe_bars):.0f}, Mean = {np.mean(long_mfe_bars):.1f}")
print(f"SHORT: Median MFE Bar = {np.median(short_mfe_bars):.0f}, Mean = {np.mean(short_mfe_bars):.1f}")

# Analyze WITH filter (filtered signals)
print("\n" + "="*70)
print("WITH 200 SMA FILTER (FILTERED SIGNALS)")
print("="*70)

long_mfe_bars_f, long_mfe_values_f, long_paths_f = analyze_mfe_timing(
    oos_data, 'rsi_oversold_cross', 'long', filter_by_sma=True
)
short_mfe_bars_f, short_mfe_values_f, short_paths_f = analyze_mfe_timing(
    oos_data, 'rsi_overbought_cross', 'short', filter_by_sma=True
)

print(f"\nLONG signals (BULL only): {len(long_mfe_bars_f)}")
print(f"  MFE Bar distribution:")
for bar in range(1, HORIZON + 1):
    count = long_mfe_bars_f.count(bar)
    pct = count / len(long_mfe_bars_f) * 100 if long_mfe_bars_f else 0
    print(f"    Bar {bar:2d}: {count:3d} signals ({pct:5.1f}%)")

print(f"\nSHORT signals (BEAR only): {len(short_mfe_bars_f)}")
print(f"  MFE Bar distribution:")
for bar in range(1, HORIZON + 1):
    count = short_mfe_bars_f.count(bar)
    pct = count / len(short_mfe_bars_f) * 100 if short_mfe_bars_f else 0
    print(f"    Bar {bar:2d}: {count:3d} signals ({pct:5.1f}%)")

print(f"\nLONG: Median MFE Bar = {np.median(long_mfe_bars_f):.0f}, Mean = {np.mean(long_mfe_bars_f):.1f}")
print(f"SHORT: Median MFE Bar = {np.median(short_mfe_bars_f):.0f}, Mean = {np.mean(short_mfe_bars_f):.1f}")

# =============================================================================
# VISUALIZATION 1: MFE Bar Distribution (Histogram)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: LONG without filter
ax1 = axes[0, 0]
ax1.hist(long_mfe_bars, bins=range(1, 12), align='left', edgecolor='black', color='green', alpha=0.7)
ax1.axvline(np.median(long_mfe_bars), color='red', linestyle='--', linewidth=2, label=f'Median: Bar {np.median(long_mfe_bars):.0f}')
ax1.set_xlabel('Bar Number')
ax1.set_ylabel('Count')
ax1.set_title(f'LONG (All): At which bar does MFE occur?\nn={len(long_mfe_bars)} signals')
ax1.set_xticks(range(1, 11))
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: SHORT without filter
ax2 = axes[0, 1]
ax2.hist(short_mfe_bars, bins=range(1, 12), align='left', edgecolor='black', color='red', alpha=0.7)
ax2.axvline(np.median(short_mfe_bars), color='blue', linestyle='--', linewidth=2, label=f'Median: Bar {np.median(short_mfe_bars):.0f}')
ax2.set_xlabel('Bar Number')
ax2.set_ylabel('Count')
ax2.set_title(f'SHORT (All): At which bar does MFE occur?\nn={len(short_mfe_bars)} signals')
ax2.set_xticks(range(1, 11))
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: LONG with filter
ax3 = axes[1, 0]
ax3.hist(long_mfe_bars_f, bins=range(1, 12), align='left', edgecolor='black', color='green', alpha=0.7)
ax3.axvline(np.median(long_mfe_bars_f), color='red', linestyle='--', linewidth=2, label=f'Median: Bar {np.median(long_mfe_bars_f):.0f}')
ax3.set_xlabel('Bar Number')
ax3.set_ylabel('Count')
ax3.set_title(f'LONG (BULL only): At which bar does MFE occur?\nn={len(long_mfe_bars_f)} signals')
ax3.set_xticks(range(1, 11))
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: SHORT with filter
ax4 = axes[1, 1]
ax4.hist(short_mfe_bars_f, bins=range(1, 12), align='left', edgecolor='black', color='red', alpha=0.7)
ax4.axvline(np.median(short_mfe_bars_f), color='blue', linestyle='--', linewidth=2, label=f'Median: Bar {np.median(short_mfe_bars_f):.0f}')
ax4.set_xlabel('Bar Number')
ax4.set_ylabel('Count')
ax4.set_title(f'SHORT (BEAR only): At which bar does MFE occur?\nn={len(short_mfe_bars_f)} signals')
ax4.set_xticks(range(1, 11))
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('MFE TIMING: At which bar does maximum profit occur?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_PATH / '18_mfe_bar_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {PLOTS_PATH / '18_mfe_bar_distribution.png'}")

# =============================================================================
# VISUALIZATION 2: Cumulative MFE by Bar
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Calculate what % of signals have reached MFE by each bar
def cumulative_mfe_pct(mfe_bars, horizon=10):
    cumulative = []
    for bar in range(1, horizon + 1):
        count = sum(1 for b in mfe_bars if b <= bar)
        pct = count / len(mfe_bars) * 100 if mfe_bars else 0
        cumulative.append(pct)
    return cumulative

# LONG cumulative
ax1 = axes[0]
long_cum = cumulative_mfe_pct(long_mfe_bars)
long_cum_f = cumulative_mfe_pct(long_mfe_bars_f)
bars = range(1, 11)

ax1.plot(bars, long_cum, 'g-o', linewidth=2, markersize=8, label='All LONG')
ax1.plot(bars, long_cum_f, 'g--s', linewidth=2, markersize=8, label='LONG (BULL only)')
ax1.axhline(50, color='gray', linestyle=':', alpha=0.5)
ax1.axhline(80, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('Bar Number')
ax1.set_ylabel('% of signals that have reached MFE')
ax1.set_title('LONG: Cumulative % of signals reaching MFE by bar')
ax1.set_xticks(bars)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add annotations
for i, (all_pct, filt_pct) in enumerate(zip(long_cum, long_cum_f)):
    ax1.annotate(f'{all_pct:.0f}%', (i+1, all_pct+2), ha='center', fontsize=8, color='darkgreen')

# SHORT cumulative
ax2 = axes[1]
short_cum = cumulative_mfe_pct(short_mfe_bars)
short_cum_f = cumulative_mfe_pct(short_mfe_bars_f)

ax2.plot(bars, short_cum, 'r-o', linewidth=2, markersize=8, label='All SHORT')
ax2.plot(bars, short_cum_f, 'r--s', linewidth=2, markersize=8, label='SHORT (BEAR only)')
ax2.axhline(50, color='gray', linestyle=':', alpha=0.5)
ax2.axhline(80, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Bar Number')
ax2.set_ylabel('% of signals that have reached MFE')
ax2.set_title('SHORT: Cumulative % of signals reaching MFE by bar')
ax2.set_xticks(bars)
ax2.legend()
ax2.grid(True, alpha=0.3)

for i, (all_pct, filt_pct) in enumerate(zip(short_cum, short_cum_f)):
    ax2.annotate(f'{all_pct:.0f}%', (i+1, all_pct+2), ha='center', fontsize=8, color='darkred')

plt.suptitle('When should we EXIT? Cumulative MFE timing', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_PATH / '19_mfe_cumulative.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {PLOTS_PATH / '19_mfe_cumulative.png'}")

# =============================================================================
# VISUALIZATION 3: Average PnL Path (all bars)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Calculate average PnL at each bar
def avg_pnl_path(paths):
    if not paths:
        return [0] * 10
    return [np.mean([p[i] for p in paths]) for i in range(10)]

# LONG paths
ax1 = axes[0]
long_avg = avg_pnl_path(long_paths)
long_avg_f = avg_pnl_path(long_paths_f)

ax1.plot(bars, long_avg, 'g-o', linewidth=2, markersize=8, label='All LONG')
ax1.plot(bars, long_avg_f, 'g--s', linewidth=2, markersize=8, label='LONG (BULL only)')
ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax1.set_xlabel('Bar Number')
ax1.set_ylabel('Average MFE (bps)')
ax1.set_title('LONG: Average MFE at each bar')
ax1.set_xticks(bars)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Mark the peak
max_bar = np.argmax(long_avg) + 1
ax1.scatter([max_bar], [long_avg[max_bar-1]], color='gold', s=200, zorder=10, edgecolors='black', linewidths=2)
ax1.annotate(f'Peak: Bar {max_bar}\n{long_avg[max_bar-1]:.1f} bps',
             (max_bar, long_avg[max_bar-1]), xytext=(max_bar+1, long_avg[max_bar-1]+10),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))

# SHORT paths
ax2 = axes[1]
short_avg = avg_pnl_path(short_paths)
short_avg_f = avg_pnl_path(short_paths_f)

ax2.plot(bars, short_avg, 'r-o', linewidth=2, markersize=8, label='All SHORT')
ax2.plot(bars, short_avg_f, 'r--s', linewidth=2, markersize=8, label='SHORT (BEAR only)')
ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax2.set_xlabel('Bar Number')
ax2.set_ylabel('Average MFE (bps)')
ax2.set_title('SHORT: Average MFE at each bar')
ax2.set_xticks(bars)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Mark the peak
max_bar = np.argmax(short_avg) + 1
ax2.scatter([max_bar], [short_avg[max_bar-1]], color='gold', s=200, zorder=10, edgecolors='black', linewidths=2)
ax2.annotate(f'Peak: Bar {max_bar}\n{short_avg[max_bar-1]:.1f} bps',
             (max_bar, short_avg[max_bar-1]), xytext=(max_bar+1, short_avg[max_bar-1]+10),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))

plt.suptitle('Average MFE Path: How does profit develop over time?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_PATH / '20_mfe_avg_path.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {PLOTS_PATH / '20_mfe_avg_path.png'}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("MFE TIMING SUMMARY")
print("="*70)

print(f"""
LONG SIGNALS:
  All signals ({len(long_mfe_bars)}):
    - Median MFE Bar: {np.median(long_mfe_bars):.0f}
    - Mean MFE Bar: {np.mean(long_mfe_bars):.1f}
    - By Bar 5: {cumulative_mfe_pct(long_mfe_bars)[4]:.0f}% have reached MFE

  BULL only ({len(long_mfe_bars_f)}):
    - Median MFE Bar: {np.median(long_mfe_bars_f):.0f}
    - Mean MFE Bar: {np.mean(long_mfe_bars_f):.1f}
    - By Bar 5: {cumulative_mfe_pct(long_mfe_bars_f)[4]:.0f}% have reached MFE

SHORT SIGNALS:
  All signals ({len(short_mfe_bars)}):
    - Median MFE Bar: {np.median(short_mfe_bars):.0f}
    - Mean MFE Bar: {np.mean(short_mfe_bars):.1f}
    - By Bar 5: {cumulative_mfe_pct(short_mfe_bars)[4]:.0f}% have reached MFE

  BEAR only ({len(short_mfe_bars_f)}):
    - Median MFE Bar: {np.median(short_mfe_bars_f):.0f}
    - Mean MFE Bar: {np.mean(short_mfe_bars_f):.1f}
    - By Bar 5: {cumulative_mfe_pct(short_mfe_bars_f)[4]:.0f}% have reached MFE

EXIT TIMING INSIGHT:
- Most MFEs occur at Bar 10 (end of horizon)
- This suggests price continues moving in our direction
- Holding longer may capture more profit
- But also exposes to reversal risk

NEXT QUESTION: Should we test longer horizons (H=15, H=20)?
""")
