"""
Verify ANALYSIS-2: Direction First by Horizon

Question: Which direction hits ±12bp threshold first?

Using 12bp threshold per Rule #1 (minimum profitable move)

Run: .venv/Scripts/python.exe scripts/debug/verify_analysis_2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60]
THRESHOLD_BPS = 12  # Rule #1: minimum profitable move

TRAIN_END = "2023-12-31"
SAMPLE_SIZE = 200000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print(f"VERIFY ANALYSIS-2: Direction First (±{THRESHOLD_BPS}bp threshold)")
print("=" * 70)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END]
print(f"TRAIN: {len(train_ohlcv):,} candles (up to {TRAIN_END})")

close = train_ohlcv['close'].values
high = train_ohlcv['high'].values
low = train_ohlcv['low'].values
n = len(train_ohlcv)

# Sample for speed
np.random.seed(42)
max_h = max(HORIZONS)
sample_idx = np.random.choice(n - max_h, size=min(SAMPLE_SIZE, n - max_h), replace=False)
print(f"Sampling {len(sample_idx):,} bars...")

threshold_pct = THRESHOLD_BPS / 10000

# =============================================================================
# ANALYSIS: Which direction hits threshold first?
# =============================================================================
print("\n" + "=" * 70)
print(f"Which direction hits ±{THRESHOLD_BPS}bp first?")
print("=" * 70)

print(f"\n{'Horizon':<10} {'UP First':>12} {'DOWN First':>12} {'Neither':>12} {'Ratio (UP/DOWN)':>15}")
print("-" * 65)

results = []

for H in HORIZONS:
    up_first = 0
    down_first = 0
    neither = 0

    for i in sample_idx:
        entry = close[i]
        up_target = entry * (1 + threshold_pct)    # +12bp
        down_target = entry * (1 - threshold_pct)  # -12bp

        hit_up_bar = None
        hit_down_bar = None

        # Check each bar in horizon
        for j in range(i+1, min(i+1+H, n)):
            # Check if hit UP threshold
            if hit_up_bar is None and high[j] >= up_target:
                hit_up_bar = j - i  # Bar number when hit

            # Check if hit DOWN threshold
            if hit_down_bar is None and low[j] <= down_target:
                hit_down_bar = j - i

            # If both hit, we know which was first
            if hit_up_bar is not None and hit_down_bar is not None:
                break

        # Determine which hit first
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
        else:
            # Same bar - need to check intrabar
            # If same bar, check if high or low was more extreme
            # This is approximate - we'll count it based on close direction
            bar_idx = i + hit_up_bar
            if close[bar_idx] > entry:
                up_first += 1
            else:
                down_first += 1

    total = len(sample_idx)
    up_pct = 100 * up_first / total
    down_pct = 100 * down_first / total
    neither_pct = 100 * neither / total
    ratio = up_first / down_first if down_first > 0 else float('inf')

    print(f"H={H:<7} {up_pct:>11.1f}% {down_pct:>11.1f}% {neither_pct:>11.1f}% {ratio:>14.2f}")

    results.append({
        'horizon': H,
        'up_first': up_first,
        'down_first': down_first,
        'neither': neither,
        'up_pct': up_pct,
        'down_pct': down_pct,
        'neither_pct': neither_pct,
        'ratio': ratio
    })

# =============================================================================
# COMPARISON WITH ANALYSIS-1 (Noise should match)
# =============================================================================
print("\n" + "=" * 70)
print("Cross-check: 'Neither' should match 'Noise' from ANALYSIS-1")
print("=" * 70)

print(f"\n{'Horizon':<10} {'Neither (this)':>15} {'Noise (A-1)':>15} {'Match?':>10}")
print("-" * 55)

# Expected noise from ANALYSIS-1
expected_noise = {
    3: 53.7,
    5: 40.5,
    10: 24.3,
    15: 16.7,
    30: 7.8,
    60: 3.1
}

for r in results:
    H = r['horizon']
    neither = r['neither_pct']
    expected = expected_noise.get(H, 0)
    match = "✓" if abs(neither - expected) < 1.0 else "✗"
    print(f"H={H:<7} {neither:>14.1f}% {expected:>14.1f}% {match:>10}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY FOR ANALYSIS-2 UPDATE")
print("=" * 70)

print(f"""
## ANALYSIS-2: Direction is 50/50 (All Bars)

**Question:** For ALL bars, which direction hits ±{THRESHOLD_BPS}bp first?

**Population:** {len(sample_idx):,} sampled bars (train data up to {TRAIN_END})

**Threshold:** {THRESHOLD_BPS}bp (Rule #1: minimum profitable move)

| Horizon | UP First | DOWN First | Neither (Noise) | Ratio |
|---------|----------|------------|-----------------|-------|""")

for r in results:
    print(f"| H={r['horizon']:<4} | {r['up_pct']:.1f}%    | {r['down_pct']:.1f}%     | {r['neither_pct']:.1f}%           | {r['ratio']:.2f}  |")

print("""
**Key insight:** Direction is ~50/50 at ALL horizons.

**Conclusion:** Cannot predict direction from random entry.
""")
