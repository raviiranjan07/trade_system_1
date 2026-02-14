"""
Extended Horizons for ANALYSIS-1, 2, 3

Get data for H=60, 120, 240, 360, 480, 600

Run: .venv/Scripts/python.exe scripts/debug/test_extended_horizons_analysis123.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60, 120, 240, 360, 480, 600]
THRESHOLD_BPS = 12  # Rule #1
SAMPLE_SIZE = 200000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("EXTENDED HORIZONS FOR ANALYSIS-1, 2, 3")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# Use train data only
train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data: {len(train):,} candles")

close = train['close'].values
high = train['high'].values
low = train['low'].values
n = len(train)

# Sample
np.random.seed(42)
max_h = max(HORIZONS)
valid_start = 100
sample_idx = np.random.choice(range(valid_start, n - max_h), size=min(SAMPLE_SIZE, n - max_h - valid_start), replace=False)
print(f"Sample size: {len(sample_idx):,}")

threshold_pct = THRESHOLD_BPS / 10000

# =============================================================================
# ANALYSIS-1: Market Moves Enough
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS-1: Market Moves Enough (12bp threshold)")
print("=" * 80)

print(f"\n{'Horizon':<10} {'Noise (<12bp)':<15} {'Real UP only':<15} {'Real DOWN only':<17} {'Real BOTH':<12} {'Total Real':<12}")
print("-" * 85)

analysis1_results = []

for H in HORIZONS:
    noise = 0
    up_only = 0
    down_only = 0
    both = 0

    for i in sample_idx:
        entry = close[i]

        # Get max up and max down within horizon
        future_highs = high[i+1:i+H+1]
        future_lows = low[i+1:i+H+1]

        max_up = (future_highs.max() - entry) / entry if len(future_highs) > 0 else 0
        max_down = (entry - future_lows.min()) / entry if len(future_lows) > 0 else 0

        hit_up = max_up >= threshold_pct
        hit_down = max_down >= threshold_pct

        if hit_up and hit_down:
            both += 1
        elif hit_up:
            up_only += 1
        elif hit_down:
            down_only += 1
        else:
            noise += 1

    total = len(sample_idx)
    noise_pct = 100 * noise / total
    up_only_pct = 100 * up_only / total
    down_only_pct = 100 * down_only / total
    both_pct = 100 * both / total
    total_real_pct = 100 - noise_pct

    print(f"H={H:<7} {noise_pct:<15.1f} {up_only_pct:<15.1f} {down_only_pct:<17.1f} {both_pct:<12.1f} {total_real_pct:<12.1f}")

    analysis1_results.append({
        'H': H,
        'noise': noise_pct,
        'up_only': up_only_pct,
        'down_only': down_only_pct,
        'both': both_pct,
        'total_real': total_real_pct
    })

# =============================================================================
# ANALYSIS-2: Direction is 50/50
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS-2: Direction is 50/50 (which hits 12bp first)")
print("=" * 80)

print(f"\n{'Horizon':<10} {'UP First':<12} {'DOWN First':<14} {'Neither':<12} {'Ratio':<10}")
print("-" * 60)

analysis2_results = []

for H in HORIZONS:
    up_first = 0
    down_first = 0
    neither = 0

    for i in sample_idx:
        entry = close[i]
        up_target = entry * (1 + threshold_pct)
        down_target = entry * (1 - threshold_pct)

        hit_up_bar = None
        hit_down_bar = None

        for j in range(1, H + 1):
            if i + j >= n:
                break
            if hit_up_bar is None and high[i + j] >= up_target:
                hit_up_bar = j
            if hit_down_bar is None and low[i + j] <= down_target:
                hit_down_bar = j

        if hit_up_bar is None and hit_down_bar is None:
            neither += 1
        elif hit_up_bar is None:
            down_first += 1
        elif hit_down_bar is None:
            up_first += 1
        elif hit_up_bar < hit_down_bar:
            up_first += 1
        elif hit_down_bar < hit_up_bar:
            down_first += 1
        else:  # Same bar - use close direction
            if close[i + hit_up_bar] > entry:
                up_first += 1
            else:
                down_first += 1

    total = len(sample_idx)
    up_pct = 100 * up_first / total
    down_pct = 100 * down_first / total
    neither_pct = 100 * neither / total
    ratio = up_pct / down_pct if down_pct > 0 else 0

    print(f"H={H:<7} {up_pct:<12.1f} {down_pct:<14.1f} {neither_pct:<12.1f} {ratio:<10.2f}")

    analysis2_results.append({
        'H': H,
        'up_first': up_pct,
        'down_first': down_pct,
        'neither': neither_pct,
        'ratio': ratio
    })

# =============================================================================
# SUMMARY FOR COPY-PASTE
# =============================================================================
print("\n" + "=" * 80)
print("MARKDOWN TABLE FOR ANALYSIS-1")
print("=" * 80)

print("\n| Horizon | Noise (<12bp) | Real UP only | Real DOWN only | Real BOTH | Total Real |")
print("|---------|---------------|--------------|----------------|-----------|------------|")
for r in analysis1_results:
    print(f"| H={r['H']:<4} | {r['noise']:.1f}% | {r['up_only']:.1f}% | {r['down_only']:.1f}% | {r['both']:.1f}% | {r['total_real']:.1f}% |")

print("\n" + "=" * 80)
print("MARKDOWN TABLE FOR ANALYSIS-2")
print("=" * 80)

print("\n| Horizon | UP First | DOWN First | Neither (Noise) | Ratio |")
print("|---------|----------|------------|-----------------|-------|")
for r in analysis2_results:
    print(f"| H={r['H']:<4} | {r['up_first']:.1f}% | {r['down_first']:.1f}% | {r['neither']:.1f}% | {r['ratio']:.2f} |")
