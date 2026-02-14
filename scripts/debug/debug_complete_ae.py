"""
Complete AE Analysis: Add Mean and Max to existing percentile stats.

Run: .venv/Scripts/python.exe debug_complete_ae.py

Adds:
- Mean AE (average)
- Max AE (worst case - critical for liquidation risk)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
TRAIN_END = "2023-12-31"
HORIZONS = [3, 5, 10, 15, 30, 60]
TARGETS = [15, 25, 50]  # Key targets
SAMPLE_SIZE = 200000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("COMPLETE AE ANALYSIS: Mean, Median, Percentiles, and MAX")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END]
print(f"Train data: {len(train_ohlcv):,} candles")

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
# COMPUTE COMPLETE AE STATS
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS: Complete AE Statistics")
print("=" * 80)

for H in HORIZONS:
    print(f"\n{'='*70}")
    print(f"H={H} bars")
    print(f"{'='*70}")
    print(f"{'Outcome':<20} {'Mean':<10} {'Median':<10} {'75th':<10} {'90th':<10} {'99th':<10} {'MAX':<12} {'Count':<10}")
    print("-" * 95)

    for target_bps in TARGETS:
        target_pct = target_bps / 10000

        # Collect AE for winners and losers
        winner_ae = []
        loser_ae = []

        for i in sample_idx:
            entry = close[i]
            target_price = entry * (1 + target_pct)

            max_adverse = 0
            hit_target = False

            for j in range(i + 1, min(i + 1 + H, n)):
                # Track adverse excursion
                adverse = (entry - low[j]) / entry * 10000  # in bps
                if adverse > max_adverse:
                    max_adverse = adverse

                # Check if hit target
                if high[j] >= target_price:
                    hit_target = True
                    winner_ae.append(max_adverse)
                    break

            if not hit_target:
                loser_ae.append(max_adverse)

        # Calculate stats for winners
        if len(winner_ae) > 0:
            w_arr = np.array(winner_ae)
            print(f"Hit {target_bps}bp target   "
                  f"{np.mean(w_arr):>8.1f}bp "
                  f"{np.median(w_arr):>8.1f}bp "
                  f"{np.percentile(w_arr, 75):>8.1f}bp "
                  f"{np.percentile(w_arr, 90):>8.1f}bp "
                  f"{np.percentile(w_arr, 99):>8.1f}bp "
                  f"{np.max(w_arr):>10.1f}bp "
                  f"{len(winner_ae):>10,}")

        # Calculate stats for losers (only for 15bp target)
        if target_bps == 15 and len(loser_ae) > 0:
            l_arr = np.array(loser_ae)
            print(f"Never hit {target_bps}bp   "
                  f"{np.mean(l_arr):>8.1f}bp "
                  f"{np.median(l_arr):>8.1f}bp "
                  f"{np.percentile(l_arr, 75):>8.1f}bp "
                  f"{np.percentile(l_arr, 90):>8.1f}bp "
                  f"{np.percentile(l_arr, 99):>8.1f}bp "
                  f"{np.max(l_arr):>10.1f}bp "
                  f"{len(loser_ae):>10,}")

# =============================================================================
# SUMMARY FOR LIQUIDATION RISK
# =============================================================================
print("\n" + "=" * 80)
print("LIQUIDATION RISK SUMMARY")
print("=" * 80)

print("""
KEY INSIGHT: Max AE tells you the WORST-CASE drawdown.

If Max AE = 500bp for winners at H=60:
- Even winning trades can see 5% drawdown before hitting target
- If using 20x leverage (5% liquidation), you'd be liquidated
- You need leverage < 100% / Max AE to survive all trades

SAFE LEVERAGE CALCULATION:
- Look at Max AE for your target/horizon
- Safe Leverage = 10000 / Max_AE_bps
- Example: Max AE = 200bp → Safe Leverage = 10000/200 = 50x
""")

# Calculate safe leverage for each H
print(f"\n{'H':<8} {'Target':<10} {'Max AE':<12} {'Safe Leverage':<15} {'99th AE':<12} {'99% Safe Lev':<15}")
print("-" * 75)

for H in [3, 15, 60]:
    for target_bps in [15, 25]:
        target_pct = target_bps / 10000
        winner_ae = []

        for i in sample_idx:
            entry = close[i]
            target_price = entry * (1 + target_pct)
            max_adverse = 0

            for j in range(i + 1, min(i + 1 + H, n)):
                adverse = (entry - low[j]) / entry * 10000
                if adverse > max_adverse:
                    max_adverse = adverse
                if high[j] >= target_price:
                    winner_ae.append(max_adverse)
                    break

        if len(winner_ae) > 0:
            w_arr = np.array(winner_ae)
            max_ae = np.max(w_arr)
            p99_ae = np.percentile(w_arr, 99)
            safe_lev = 10000 / max_ae if max_ae > 0 else float('inf')
            p99_lev = 10000 / p99_ae if p99_ae > 0 else float('inf')

            print(f"H={H:<5} {target_bps}bp{'':<6} {max_ae:>10.1f}bp {safe_lev:>13.0f}x {p99_ae:>10.1f}bp {p99_lev:>13.0f}x")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
