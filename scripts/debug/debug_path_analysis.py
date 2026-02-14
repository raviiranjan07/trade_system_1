"""
Path Analysis: How does price actually move?

Run: .venv/Scripts/python.exe debug_path_analysis.py

Questions to answer:
1. How often does price hit target BEFORE going below entry?
2. If it goes below entry, by how much (adverse excursion)?
3. Can we define noise as moves < 10 bps, real moves as >= 10 bps?
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60]
NOISE_THRESHOLD = 10  # bps - moves below this are "noise"

TRAIN_END = "2023-12-31"
SAMPLE_SIZE = 200000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("PATH ANALYSIS: How Does Price Actually Move?")
print("=" * 70)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END]
print(f"TRAIN: {len(train_ohlcv):,} candles")

close = train_ohlcv['close'].values
high = train_ohlcv['high'].values
low = train_ohlcv['low'].values
n = len(train_ohlcv)

# Sample for speed
np.random.seed(42)
max_h = max(HORIZONS)
sample_idx = np.random.choice(n - max_h, size=min(SAMPLE_SIZE, n - max_h), replace=False)
print(f"Sampling {len(sample_idx):,} bars...")

# =============================================================================
# QUESTION 1: How often does price hit target BEFORE going below entry?
# =============================================================================
print("\n" + "=" * 70)
print("Q1: How often does price hit target BEFORE going below entry?")
print("=" * 70)

print(f"\nFor various target levels, what % of trades hit target with ZERO drawdown?")
print(f"(i.e., price never goes below entry before hitting target)")

targets_to_test = [8, 10, 15, 20, 25, 30, 40, 50]

for H in HORIZONS:
    print(f"\n--- H = {H} bars ---")
    print(f"{'Target':>8} {'Clean Win %':>12} {'Any Win %':>12} {'Never Hit':>12}")
    print("-" * 50)

    for target_bps in targets_to_test:
        target_pct = target_bps / 10000

        clean_wins = 0  # Hit target without ever going below entry
        any_wins = 0    # Hit target at some point (even with drawdown)
        never_hit = 0   # Never hit target within horizon

        for i in sample_idx:
            entry = close[i]
            target_price = entry * (1 + target_pct)

            hit_target = False
            went_below = False
            hit_target_first = False

            for j in range(i+1, min(i+1+H, n)):
                # Check if went below entry
                if low[j] < entry and not hit_target:
                    went_below = True

                # Check if hit target
                if high[j] >= target_price:
                    hit_target = True
                    if not went_below:
                        hit_target_first = True
                    break

            if hit_target_first:
                clean_wins += 1
            if hit_target:
                any_wins += 1
            if not hit_target:
                never_hit += 1

        total = len(sample_idx)
        clean_pct = clean_wins / total * 100
        any_pct = any_wins / total * 100
        never_pct = never_hit / total * 100

        print(f"{target_bps:>7}bp {clean_pct:>11.1f}% {any_pct:>11.1f}% {never_pct:>11.1f}%")

# =============================================================================
# QUESTION 2: Adverse Excursion Analysis
# =============================================================================
print("\n" + "=" * 70)
print("Q2: If price goes below entry, by how much? (Adverse Excursion)")
print("=" * 70)

print(f"\nFor each horizon, what is the max adverse excursion (drawdown) before:")
print(f"  A) Price eventually hits various target levels")
print(f"  B) Price never hits target (timeout)")

for H in [10, 30, 60]:  # Focus on key horizons
    print(f"\n--- H = {H} bars ---")

    # Collect adverse excursions for different outcomes
    ae_hit_target = {t: [] for t in [15, 25, 50]}  # AE when target is hit
    ae_no_target = []  # AE when no target hit

    for i in sample_idx:
        entry = close[i]

        # Track max adverse excursion
        max_ae_bps = 0

        for j in range(i+1, min(i+1+H, n)):
            ae = (entry - low[j]) / entry * 10000
            if ae > max_ae_bps:
                max_ae_bps = ae

        # Check which targets were hit
        max_up = 0
        for j in range(i+1, min(i+1+H, n)):
            up = (high[j] - entry) / entry * 10000
            if up > max_up:
                max_up = up

        for target in ae_hit_target.keys():
            if max_up >= target:
                ae_hit_target[target].append(max_ae_bps)

        if max_up < 15:  # Didn't even hit 15 bps
            ae_no_target.append(max_ae_bps)

    print(f"\n{'Outcome':<25} {'Median AE':>10} {'75th AE':>10} {'90th AE':>10} {'Count':>10}")
    print("-" * 70)

    for target, ae_list in ae_hit_target.items():
        if len(ae_list) > 0:
            ae_arr = np.array(ae_list)
            print(f"Hit {target}bp target        {np.median(ae_arr):>9.1f}bp {np.percentile(ae_arr, 75):>9.1f}bp {np.percentile(ae_arr, 90):>9.1f}bp {len(ae_arr):>10,}")

    if len(ae_no_target) > 0:
        ae_arr = np.array(ae_no_target)
        print(f"Never hit 15bp target    {np.median(ae_arr):>9.1f}bp {np.percentile(ae_arr, 75):>9.1f}bp {np.percentile(ae_arr, 90):>9.1f}bp {len(ae_arr):>10,}")

# =============================================================================
# QUESTION 3: Noise vs Real Move Classification
# =============================================================================
print("\n" + "=" * 70)
print("Q3: Can we classify moves as NOISE (<10 bps) vs REAL (>=10 bps)?")
print("=" * 70)

print(f"\nFor each horizon, what % of bars are NOISE vs REAL?")
print(f"NOISE = max move in either direction < {NOISE_THRESHOLD} bps")
print(f"REAL = max move in either direction >= {NOISE_THRESHOLD} bps")

print(f"\n{'Horizon':>8} {'Noise %':>10} {'Real UP':>10} {'Real DOWN':>12} {'Real BOTH':>12}")
print("-" * 60)

for H in HORIZONS:
    noise_count = 0
    real_up_only = 0
    real_down_only = 0
    real_both = 0

    for i in sample_idx:
        entry = close[i]

        # Find max up and down moves
        max_up_bps = 0
        max_down_bps = 0

        for j in range(i+1, min(i+1+H, n)):
            up = (high[j] - entry) / entry * 10000
            down = (entry - low[j]) / entry * 10000
            if up > max_up_bps:
                max_up_bps = up
            if down > max_down_bps:
                max_down_bps = down

        is_real_up = max_up_bps >= NOISE_THRESHOLD
        is_real_down = max_down_bps >= NOISE_THRESHOLD

        if not is_real_up and not is_real_down:
            noise_count += 1
        elif is_real_up and not is_real_down:
            real_up_only += 1
        elif is_real_down and not is_real_up:
            real_down_only += 1
        else:
            real_both += 1

    total = len(sample_idx)
    print(f"H={H:<6} {noise_count/total*100:>9.1f}% {real_up_only/total*100:>9.1f}% {real_down_only/total*100:>9.1f}% {real_both/total*100:>11.1f}%")

# =============================================================================
# QUESTION 3b: If we only trade REAL moves, what's the distribution?
# =============================================================================
print("\n" + "-" * 70)
print("If we only trade when move is REAL (>= 10 bps), which direction wins?")
print("-" * 70)

print(f"\nFor REAL moves only: does UP or DOWN come first?")
print(f"(First to hit {NOISE_THRESHOLD} bps)")

print(f"\n{'Horizon':>8} {'UP First':>12} {'DOWN First':>12} {'Ratio':>10}")
print("-" * 50)

for H in HORIZONS:
    up_first = 0
    down_first = 0

    for i in sample_idx:
        entry = close[i]
        target_up = entry * (1 + NOISE_THRESHOLD/10000)
        target_down = entry * (1 - NOISE_THRESHOLD/10000)

        first_direction = None

        for j in range(i+1, min(i+1+H, n)):
            if high[j] >= target_up and first_direction is None:
                first_direction = 'UP'
                break
            if low[j] <= target_down and first_direction is None:
                first_direction = 'DOWN'
                break

        if first_direction == 'UP':
            up_first += 1
        elif first_direction == 'DOWN':
            down_first += 1

    total_real = up_first + down_first
    if total_real > 0:
        ratio = up_first / down_first if down_first > 0 else float('inf')
        print(f"H={H:<6} {up_first/total_real*100:>11.1f}% {down_first/total_real*100:>11.1f}% {ratio:>9.2f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
KEY INSIGHTS:

1. CLEAN WINS (hit target without drawdown):
   - The % of clean wins depends on target level and horizon
   - Higher targets = fewer clean wins
   - Longer horizons = more chance of drawdown before target

2. ADVERSE EXCURSION:
   - Even winning trades often see drawdown first
   - This tells us what stop level is realistic

3. NOISE vs REAL:
   - At H=3, about {100 - (real_up_only + real_down_only + real_both)/len(sample_idx)*100:.0f}% of bars are noise (< {NOISE_THRESHOLD} bps move)
   - REAL moves that go BOTH directions = opportunity exists, but direction is key

4. DIRECTION BALANCE:
   - Among REAL moves, UP first vs DOWN first is approximately 50/50
   - This confirms: need selective entry to predict direction
""")
