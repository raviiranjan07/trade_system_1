"""
Calculate 75th percentile AE for all horizons and targets.
"""

import pandas as pd
import numpy as np
from pathlib import Path

TRAIN_END = "2023-12-31"
HORIZONS = [3, 5, 10, 15, 30, 60]
TARGETS = [12, 15, 25, 50]
SAMPLE_SIZE = 200000

print("=" * 70)
print("AE ANALYSIS WITH 75th PERCENTILE")
print("=" * 70)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END]
print(f"Train data: {len(train_ohlcv):,} candles")

close = train_ohlcv['close'].values
high = train_ohlcv['high'].values
low = train_ohlcv['low'].values
n = len(train_ohlcv)

np.random.seed(42)
max_h = max(HORIZONS)
sample_idx = np.random.choice(n - max_h, size=min(SAMPLE_SIZE, n - max_h), replace=False)
print(f"Sampling {len(sample_idx):,} bars...")

for H in HORIZONS:
    print(f"\n{'='*70}")
    print(f"H={H} bars")
    print(f"{'='*70}")
    print(f"{'Outcome':<22} {'Mean':<10} {'Median':<10} {'75th':<10} {'MAX':<12} {'Count':<10}")
    print("-" * 80)

    for target_bps in TARGETS:
        target_pct = target_bps / 10000
        winner_ae = []

        for i in sample_idx:
            entry = close[i]
            target_price = entry * (1 + target_pct)

            max_adverse = 0
            hit_target = False

            for j in range(i + 1, min(i + 1 + H, n)):
                adverse = (entry - low[j]) / entry * 10000
                if adverse > max_adverse:
                    max_adverse = adverse

                if high[j] >= target_price:
                    hit_target = True
                    winner_ae.append(max_adverse)
                    break

        if len(winner_ae) > 0:
            w_arr = np.array(winner_ae)
            print(f"Hit {target_bps}bp target     "
                  f"{np.mean(w_arr):>8.1f}bp "
                  f"{np.median(w_arr):>8.1f}bp "
                  f"{np.percentile(w_arr, 75):>8.1f}bp "
                  f"{np.max(w_arr):>10.1f}bp "
                  f"{len(winner_ae):>10,}")

    # Also calculate for "Never hit 15bp"
    target_pct = 15 / 10000
    loser_ae = []

    for i in sample_idx:
        entry = close[i]
        target_price = entry * (1 + target_pct)

        max_adverse = 0
        hit_target = False

        for j in range(i + 1, min(i + 1 + H, n)):
            adverse = (entry - low[j]) / entry * 10000
            if adverse > max_adverse:
                max_adverse = adverse

            if high[j] >= target_price:
                hit_target = True
                break

        if not hit_target:
            loser_ae.append(max_adverse)

    if len(loser_ae) > 0:
        l_arr = np.array(loser_ae)
        print(f"Never hit 15bp        "
              f"{np.mean(l_arr):>8.1f}bp "
              f"{np.median(l_arr):>8.1f}bp "
              f"{np.percentile(l_arr, 75):>8.1f}bp "
              f"{np.max(l_arr):>10.1f}bp "
              f"{len(loser_ae):>10,}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
