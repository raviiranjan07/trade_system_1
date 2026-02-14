"""
ANALYSIS-4 Expansion:
1. Larger targets: 100bp, 150bp, 200bp
2. Time (bars) spent in drawdown before hitting target
3. Drawdown amount before hitting target (cross-check with ANALYSIS-5)

Run: .venv/Scripts/python.exe scripts/debug/test_analysis4_expanded.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [30, 60, 120, 240, 360, 480, 600]
TARGETS_BPS = [15, 25, 50, 100, 150, 200]  # Expanded targets
SAMPLE_SIZE = 50000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS-4 EXPANSION: Larger Targets + Time in Drawdown")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data: {len(train):,} candles")

close = train['close'].values
high = train['high'].values
low = train['low'].values
n = len(train)

np.random.seed(42)
max_h = max(HORIZONS)
valid_start = 100
sample_idx = np.random.choice(range(valid_start, n - max_h), size=min(SAMPLE_SIZE, n - max_h - valid_start), replace=False)
print(f"Sample size: {len(sample_idx):,}")


def analyze_detailed(indices, H, target_pct):
    """
    Detailed analysis including:
    - Clean/Dirty/Never Hit counts
    - Max drawdown before hitting target (MAE)
    - Time (bars) spent in drawdown before hitting target
    """
    clean_win = 0
    dirty_win = 0
    never_hit = 0

    mae_values = []  # Max Adverse Excursion for winning trades
    drawdown_bars = []  # Bars spent below entry before hitting target

    for i in indices:
        entry = close[i]
        target_price = entry * (1 + target_pct)

        hit_target_bar = None
        max_drawdown = 0
        bars_below_entry = 0
        ever_below = False

        for j in range(1, H + 1):
            if i + j >= n:
                break

            # Track drawdown
            bar_low = low[i + j]
            if bar_low < entry:
                ever_below = True
                bars_below_entry += 1
                drawdown = (entry - bar_low) / entry * 10000  # in bps
                max_drawdown = max(max_drawdown, drawdown)

            # Check if hit target
            if high[i + j] >= target_price:
                hit_target_bar = j
                break

        if hit_target_bar is not None:
            mae_values.append(max_drawdown)
            drawdown_bars.append(bars_below_entry)
            if ever_below:
                dirty_win += 1
            else:
                clean_win += 1
        else:
            never_hit += 1

    total = clean_win + dirty_win + never_hit
    win_count = clean_win + dirty_win

    # Stats
    if win_count > 0:
        mae_median = np.median(mae_values)
        mae_75th = np.percentile(mae_values, 75)
        mae_max = np.max(mae_values)
        bars_median = np.median(drawdown_bars)
        bars_75th = np.percentile(drawdown_bars, 75)
    else:
        mae_median = mae_75th = mae_max = 0
        bars_median = bars_75th = 0

    return {
        'clean_pct': 100 * clean_win / total,
        'dirty_pct': 100 * dirty_win / total,
        'never_pct': 100 * never_hit / total,
        'win_count': win_count,
        'mae_median': mae_median,
        'mae_75th': mae_75th,
        'mae_max': mae_max,
        'bars_median': bars_median,
        'bars_75th': bars_75th
    }


# =============================================================================
# TEST 1: EXPANDED TARGETS (Clean/Dirty/Never Hit)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: Clean/Dirty/Never Hit for EXPANDED TARGETS")
print("=" * 80)

results = []

for H in HORIZONS:
    print(f"\n--- H={H} bars ---")
    print(f"{'Target':<10} {'Clean':<10} {'Dirty':<10} {'Never Hit':<10} {'Win Rate':<10}")
    print("-" * 55)

    for target_bps in TARGETS_BPS:
        target_pct = target_bps / 10000
        r = analyze_detailed(sample_idx, H, target_pct)
        win_rate = r['clean_pct'] + r['dirty_pct']
        print(f"{target_bps}bp{'':<6} {r['clean_pct']:<10.1f} {r['dirty_pct']:<10.1f} {r['never_pct']:<10.1f} {win_rate:<10.1f}")

        results.append({
            'H': H,
            'target': target_bps,
            **r
        })


# =============================================================================
# TEST 2: MAE (Drawdown) BEFORE HITTING TARGET
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: Drawdown (MAE) BEFORE Hitting Target")
print("=" * 80)
print("(Only for trades that eventually hit target)")

for H in [60, 120, 240, 600]:
    print(f"\n--- H={H} bars ---")
    print(f"{'Target':<10} {'MAE Med':<12} {'MAE 75th':<12} {'MAE Max':<12}")
    print("-" * 50)

    for target_bps in TARGETS_BPS:
        target_pct = target_bps / 10000
        r = analyze_detailed(sample_idx, H, target_pct)
        print(f"{target_bps}bp{'':<6} {r['mae_median']:<12.1f} {r['mae_75th']:<12.1f} {r['mae_max']:<12.1f}")


# =============================================================================
# TEST 3: TIME (BARS) IN DRAWDOWN BEFORE HITTING TARGET
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: Bars Spent in Drawdown BEFORE Hitting Target")
print("=" * 80)
print("(How many bars was price below entry before winning)")

for H in [60, 120, 240, 600]:
    print(f"\n--- H={H} bars ---")
    print(f"{'Target':<10} {'Bars Med':<12} {'Bars 75th':<12}")
    print("-" * 35)

    for target_bps in TARGETS_BPS:
        target_pct = target_bps / 10000
        r = analyze_detailed(sample_idx, H, target_pct)
        print(f"{target_bps}bp{'':<6} {r['bars_median']:<12.1f} {r['bars_75th']:<12.1f}")


# =============================================================================
# MARKDOWN TABLES
# =============================================================================
print("\n" + "=" * 80)
print("MARKDOWN TABLES FOR ANALYSIS-4 UPDATE")
print("=" * 80)

print("\n### Extended Targets: Clean/Dirty/Never Hit")
print("\n**H=240 bars (4 hours):**")
print("| Target | Clean Win % | Dirty Win % | Never Hit | Win Rate |")
print("|--------|-------------|-------------|-----------|----------|")

for r in results:
    if r['H'] == 240:
        win_rate = r['clean_pct'] + r['dirty_pct']
        print(f"| {r['target']}bp | {r['clean_pct']:.1f}% | {r['dirty_pct']:.1f}% | {r['never_pct']:.1f}% | {win_rate:.1f}% |")

print("\n**H=600 bars (10 hours):**")
print("| Target | Clean Win % | Dirty Win % | Never Hit | Win Rate |")
print("|--------|-------------|-------------|-----------|----------|")

for r in results:
    if r['H'] == 600:
        win_rate = r['clean_pct'] + r['dirty_pct']
        print(f"| {r['target']}bp | {r['clean_pct']:.1f}% | {r['dirty_pct']:.1f}% | {r['never_pct']:.1f}% | {win_rate:.1f}% |")


print("\n### MAE (Drawdown) Before Hitting Target")
print("\n**H=240 bars:**")
print("| Target | MAE Median | MAE 75th | MAE Max |")
print("|--------|------------|----------|---------|")

for r in results:
    if r['H'] == 240:
        print(f"| {r['target']}bp | {r['mae_median']:.1f}bp | {r['mae_75th']:.1f}bp | {r['mae_max']:.0f}bp |")


print("\n### Time in Drawdown Before Hitting Target")
print("\n**H=240 bars:**")
print("| Target | Bars Median | Bars 75th |")
print("|--------|-------------|-----------|")

for r in results:
    if r['H'] == 240:
        print(f"| {r['target']}bp | {r['bars_median']:.0f} bars | {r['bars_75th']:.0f} bars |")


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("""
1. LARGER TARGETS (100bp, 150bp, 200bp):
   - Require longer horizons to have decent win rate
   - Even at H=600, 200bp target only has ~X% win rate

2. MAE (Drawdown) scales with target:
   - Larger targets = more drawdown before winning
   - Need to tolerate larger drawdowns for larger targets

3. TIME IN DRAWDOWN:
   - More bars spent below entry for larger targets
   - Important for position sizing and leverage
""")
