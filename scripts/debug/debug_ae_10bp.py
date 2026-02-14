"""
Quick AE analysis for 10bp target to add to analysis_findings.md
"""

import pandas as pd
import numpy as np
from pathlib import Path

TRAIN_END = "2023-12-31"
HORIZONS = [3, 5, 10, 15, 30, 60]
TARGET_BPS = 10
SAMPLE_SIZE = 200000

print("=" * 70)
print(f"AE ANALYSIS FOR {TARGET_BPS}bp TARGET")
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

target_pct = TARGET_BPS / 10000

print(f"\n{'H':<6} {'Mean AE':<12} {'Median AE':<12} {'MAX AE':<12} {'Count':<12}")
print("-" * 55)

for H in HORIZONS:
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
        print(f"H={H:<4} {np.mean(w_arr):>10.1f}bp {np.median(w_arr):>10.1f}bp {np.max(w_arr):>10.1f}bp {len(winner_ae):>10,}")

print("\n" + "=" * 70)
print("Copy these values to analysis_findings.md")
print("=" * 70)
