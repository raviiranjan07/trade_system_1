"""
CASE 3 TIME PATTERN ANALYSIS: Find patterns to define noise/rules

Questions to answer:
1. Recovery time distribution - when do most Case 3 trades recover?
2. MAE vs Recovery Time - does bigger drawdown mean longer recovery?
3. Point of no return - after what time does recovery become unlikely?
4. Can we define a rule: "Wait until X bars, if not hit, exit"?

Run: .venv/Scripts/python.exe scripts/debug/test_case3_patterns.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60, 120, 240, 360, 480, 600]
TARGETS_BPS = [12, 15, 25, 50, 100, 150, 200]
SAMPLE_SIZE = 50000
EXTENDED_H = 600

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("CASE 3 TIME PATTERN ANALYSIS")
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
max_h = EXTENDED_H + 100
valid_start = 100
sample_idx = np.random.choice(range(valid_start, n - max_h), size=min(SAMPLE_SIZE, n - max_h - valid_start), replace=False)
print(f"Sample size: {len(sample_idx):,}")


def analyze_case3_detailed(indices, H, target_pct, extended_H=600):
    """
    Detailed Case 3 analysis:
    - Recovery time distribution (10th, 25th, 50th, 75th, 90th percentiles)
    - MAE vs recovery time correlation
    - Time bins: how many recover in each time window?
    """
    case3_data = []

    for i in indices:
        entry = close[i]
        target_price = entry * (1 + target_pct)

        hit_within_H = False
        hit_extended = False
        ever_below = False
        max_drawdown = 0
        hit_bar = None

        # Check within H bars
        for j in range(1, H + 1):
            if i + j >= n:
                break

            bar_low = low[i + j]
            if bar_low < entry:
                ever_below = True
                drawdown = (entry - bar_low) / entry * 10000
                max_drawdown = max(max_drawdown, drawdown)

            if high[i + j] >= target_price:
                hit_within_H = True
                hit_bar = j
                break

        # If didn't hit within H, check extended
        if not hit_within_H:
            for j in range(H + 1, extended_H + 1):
                if i + j >= n:
                    break

                bar_low = low[i + j]
                if bar_low < entry:
                    ever_below = True
                    drawdown = (entry - bar_low) / entry * 10000
                    max_drawdown = max(max_drawdown, drawdown)

                if high[i + j] >= target_price:
                    hit_extended = True
                    hit_bar = j
                    break

        # If Case 3 (went below, didn't hit within H, but hit later)
        if hit_extended and ever_below:
            case3_data.append({
                'mae': max_drawdown,
                'total_time': hit_bar,
                'time_after_H': hit_bar - H
            })

    if not case3_data:
        return None

    df = pd.DataFrame(case3_data)

    # Time distribution
    time_percentiles = {
        'p10': np.percentile(df['total_time'], 10),
        'p25': np.percentile(df['total_time'], 25),
        'p50': np.percentile(df['total_time'], 50),
        'p75': np.percentile(df['total_time'], 75),
        'p90': np.percentile(df['total_time'], 90),
        'mean': df['total_time'].mean(),
        'count': len(df)
    }

    # MAE distribution
    mae_percentiles = {
        'mae_p10': np.percentile(df['mae'], 10),
        'mae_p25': np.percentile(df['mae'], 25),
        'mae_p50': np.percentile(df['mae'], 50),
        'mae_p75': np.percentile(df['mae'], 75),
        'mae_p90': np.percentile(df['mae'], 90),
    }

    # Time bins: when do they recover?
    time_bins = {
        'within_2H': (df['total_time'] <= 2*H).sum() / len(df) * 100,
        'within_3H': (df['total_time'] <= 3*H).sum() / len(df) * 100,
        'within_4H': (df['total_time'] <= 4*H).sum() / len(df) * 100,
        'within_5H': (df['total_time'] <= 5*H).sum() / len(df) * 100,
        'after_5H': (df['total_time'] > 5*H).sum() / len(df) * 100,
    }

    # MAE bins
    mae_bins = {
        'mae_under_30bp': (df['mae'] < 30).sum() / len(df) * 100,
        'mae_30_50bp': ((df['mae'] >= 30) & (df['mae'] < 50)).sum() / len(df) * 100,
        'mae_50_100bp': ((df['mae'] >= 50) & (df['mae'] < 100)).sum() / len(df) * 100,
        'mae_over_100bp': (df['mae'] >= 100).sum() / len(df) * 100,
    }

    return {**time_percentiles, **mae_percentiles, **time_bins, **mae_bins}


# =============================================================================
# ANALYZE PATTERNS
# =============================================================================
print("\n" + "=" * 80)
print("PATTERN 1: RECOVERY TIME DISTRIBUTION")
print("=" * 80)

for target_bps in TARGETS_BPS:
    print(f"\n--- Target = {target_bps}bp ---")
    print(f"{'H':<6} {'Count':<8} {'10th':<8} {'25th':<8} {'50th':<8} {'75th':<8} {'90th':<8} {'Mean':<8}")
    print("-" * 70)

    target_pct = target_bps / 10000
    for H in HORIZONS:
        result = analyze_case3_detailed(sample_idx, H, target_pct, EXTENDED_H)
        if result:
            print(f"{H:<6} {int(result['count']):<8} {result['p10']:<8.0f} {result['p25']:<8.0f} "
                  f"{result['p50']:<8.0f} {result['p75']:<8.0f} {result['p90']:<8.0f} {result['mean']:<8.0f}")


# =============================================================================
# PATTERN 2: TIME BINS (When do most Case 3 trades recover?)
# =============================================================================
print("\n" + "=" * 80)
print("PATTERN 2: WHEN DO CASE 3 TRADES RECOVER? (% within time windows)")
print("=" * 80)

for target_bps in TARGETS_BPS:
    print(f"\n--- Target = {target_bps}bp ---")
    print(f"{'H':<6} {'<=2H':<10} {'<=3H':<10} {'<=4H':<10} {'<=5H':<10} {'>5H':<10}")
    print("-" * 60)

    target_pct = target_bps / 10000
    for H in HORIZONS:
        result = analyze_case3_detailed(sample_idx, H, target_pct, EXTENDED_H)
        if result:
            print(f"{H:<6} {result['within_2H']:<10.1f} {result['within_3H']:<10.1f} "
                  f"{result['within_4H']:<10.1f} {result['within_5H']:<10.1f} {result['after_5H']:<10.1f}")


# =============================================================================
# PATTERN 3: MAE DISTRIBUTION FOR CASE 3
# =============================================================================
print("\n" + "=" * 80)
print("PATTERN 3: MAE DISTRIBUTION FOR CASE 3 TRADES")
print("=" * 80)

for target_bps in TARGETS_BPS:
    print(f"\n--- Target = {target_bps}bp ---")
    print(f"{'H':<6} {'<30bp':<12} {'30-50bp':<12} {'50-100bp':<12} {'>100bp':<12}")
    print("-" * 60)

    target_pct = target_bps / 10000
    for H in HORIZONS:
        result = analyze_case3_detailed(sample_idx, H, target_pct, EXTENDED_H)
        if result:
            print(f"{H:<6} {result['mae_under_30bp']:<12.1f} {result['mae_30_50bp']:<12.1f} "
                  f"{result['mae_50_100bp']:<12.1f} {result['mae_over_100bp']:<12.1f}")


# =============================================================================
# PATTERN 4: COMPLETE DATA BY HORIZON
# =============================================================================
print("\n" + "=" * 80)
print("PATTERN 4: RECOVERY WINDOWS BY HORIZON (ALL TARGETS)")
print("=" * 80)

for target_bps in TARGETS_BPS:
    print(f"\n### Target = {target_bps}bp - Recovery Time Windows")
    print(f"{'H':<6} {'<=2H %':<10} {'<=3H %':<10} {'<=4H %':<10} {'<=5H %':<10} {'>5H %':<10}")
    print("-" * 60)

    target_pct = target_bps / 10000
    for H in HORIZONS:
        result = analyze_case3_detailed(sample_idx, H, target_pct, EXTENDED_H)
        if result:
            print(f"{H:<6} {result['within_2H']:<10.1f} {result['within_3H']:<10.1f} "
                  f"{result['within_4H']:<10.1f} {result['within_5H']:<10.1f} {result['after_5H']:<10.1f}")


# =============================================================================
# PATTERN 5: MAE DISTRIBUTION BY HORIZON
# =============================================================================
print("\n" + "=" * 80)
print("PATTERN 5: MAE DISTRIBUTION BY HORIZON (ALL TARGETS)")
print("=" * 80)

for target_bps in TARGETS_BPS:
    print(f"\n### Target = {target_bps}bp - MAE Distribution")
    print(f"{'H':<6} {'<30bp %':<12} {'30-50bp %':<12} {'50-100bp %':<12} {'>100bp %':<12}")
    print("-" * 60)

    target_pct = target_bps / 10000
    for H in HORIZONS:
        result = analyze_case3_detailed(sample_idx, H, target_pct, EXTENDED_H)
        if result:
            print(f"{H:<6} {result['mae_under_30bp']:<12.1f} {result['mae_30_50bp']:<12.1f} "
                  f"{result['mae_50_100bp']:<12.1f} {result['mae_over_100bp']:<12.1f}")


# =============================================================================
# MARKDOWN TABLES: RECOVERY TIME WINDOWS
# =============================================================================
print("\n" + "=" * 80)
print("MARKDOWN TABLES: RECOVERY TIME WINDOWS")
print("=" * 80)

for target_bps in TARGETS_BPS:
    print(f"\n### Target {target_bps}bp: % of Case 3 trades recovered within time window")
    print("| H | <=2H | <=3H | <=4H | <=5H | >5H |")
    print("|---|------|------|------|------|-----|")

    target_pct = target_bps / 10000
    for H in HORIZONS:
        result = analyze_case3_detailed(sample_idx, H, target_pct, EXTENDED_H)
        if result:
            print(f"| {H} | {result['within_2H']:.1f}% | {result['within_3H']:.1f}% | "
                  f"{result['within_4H']:.1f}% | {result['within_5H']:.1f}% | {result['after_5H']:.1f}% |")


# =============================================================================
# MARKDOWN TABLES: MAE DISTRIBUTION
# =============================================================================
print("\n" + "=" * 80)
print("MARKDOWN TABLES: MAE DISTRIBUTION")
print("=" * 80)

for target_bps in TARGETS_BPS:
    print(f"\n### Target {target_bps}bp: % of Case 3 trades by MAE range")
    print("| H | <30bp | 30-50bp | 50-100bp | >100bp |")
    print("|---|-------|---------|----------|--------|")

    target_pct = target_bps / 10000
    for H in HORIZONS:
        result = analyze_case3_detailed(sample_idx, H, target_pct, EXTENDED_H)
        if result:
            print(f"| {H} | {result['mae_under_30bp']:.1f}% | {result['mae_30_50bp']:.1f}% | "
                  f"{result['mae_50_100bp']:.1f}% | {result['mae_over_100bp']:.1f}% |")


# =============================================================================
# MARKDOWN TABLES: TIME PERCENTILES
# =============================================================================
print("\n" + "=" * 80)
print("MARKDOWN TABLES: RECOVERY TIME PERCENTILES (bars from entry)")
print("=" * 80)

for target_bps in TARGETS_BPS:
    print(f"\n### Target {target_bps}bp: Recovery time percentiles")
    print("| H | 10th | 25th | 50th | 75th | 90th |")
    print("|---|------|------|------|------|------|")

    target_pct = target_bps / 10000
    for H in HORIZONS:
        result = analyze_case3_detailed(sample_idx, H, target_pct, EXTENDED_H)
        if result:
            print(f"| {H} | {result['p10']:.0f} | {result['p25']:.0f} | "
                  f"{result['p50']:.0f} | {result['p75']:.0f} | {result['p90']:.0f} |")


# =============================================================================
# MARKDOWN TABLES: MAE PERCENTILES
# =============================================================================
print("\n" + "=" * 80)
print("MARKDOWN TABLES: MAE PERCENTILES (bp drawdown)")
print("=" * 80)

for target_bps in TARGETS_BPS:
    print(f"\n### Target {target_bps}bp: MAE percentiles")
    print("| H | 10th | 25th | 50th | 75th | 90th |")
    print("|---|------|------|------|------|------|")

    target_pct = target_bps / 10000
    for H in HORIZONS:
        result = analyze_case3_detailed(sample_idx, H, target_pct, EXTENDED_H)
        if result:
            print(f"| {H} | {result['mae_p10']:.0f} | {result['mae_p25']:.0f} | "
                  f"{result['mae_p50']:.0f} | {result['mae_p75']:.0f} | {result['mae_p90']:.0f} |")
