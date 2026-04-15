"""L2-003: Deep data inspection — what does the training data actually look like?

Check specific conditions and see if direction is predictable there.

Run locally: python experiments/layer2/L2-003/L2_003_data_check.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("experiments/layer2/L2-003")
fc = pd.read_parquet(BASE / "feature_cache.parquet")
lb = pd.read_parquet(BASE / "labels.parquet")

common_idx = lb.index.intersection(fc.index)
lb = lb.loc[common_idx]
fc = fc.loc[common_idx]

direction = lb["direction"].values  # 0=LONG, 1=SHORT, 2=BOTH
valid = (direction == 0) | (direction == 1)
time_up = lb["time_to_first_up_hit"].values
time_down = lb["time_to_first_down_hit"].values

print(f"Total bars: {len(fc)}")
print(f"Valid (LONG/SHORT): {valid.sum()}")

# =============================================================================
# CHECK 1: Direction by RSI buckets
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 1: Direction by RSI7 buckets")
print(f"{'='*70}")

rsi = fc["rsi7"].values
print(f"\n  {'RSI range':<15s} | {'total':>6s} {'LONG':>6s} {'SHORT':>6s} {'BOTH':>6s} | {'LONG%':>6s} {'dir signal?'}")
print(f"  {'-'*15}-+-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*6}-{'-'*15}")

rsi_bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
for lo, hi in rsi_bins:
    mask = (rsi >= lo) & (rsi < hi)
    n = mask.sum()
    n_long = (mask & (direction == 0)).sum()
    n_short = (mask & (direction == 1)).sum()
    n_both = (mask & (direction == 2)).sum()
    n_valid = n_long + n_short
    pct = n_long / n_valid * 100 if n_valid > 0 else 0
    sig = "LONG bias" if pct > 55 else "SHORT bias" if pct < 45 else "50/50"
    print(f"  RSI {lo:2d}-{hi:2d}      | {n:6d} {n_long:6d} {n_short:6d} {n_both:6d} | {pct:5.1f}% {sig}")

# =============================================================================
# CHECK 2: Direction by ATR buckets
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 2: Direction by ATR percentile buckets")
print(f"{'='*70}")

atr = fc["atr_percentile"].values
print(f"\n  {'ATR range':<15s} | {'total':>6s} {'LONG':>6s} {'SHORT':>6s} | {'LONG%':>6s} {'dir signal?'}")
print(f"  {'-'*15}-+-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*6}-{'-'*15}")

atr_bins = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 100)]
for lo, hi in atr_bins:
    mask = (atr >= lo) & (atr < hi)
    n_long = (mask & (direction == 0)).sum()
    n_short = (mask & (direction == 1)).sum()
    n_valid = n_long + n_short
    pct = n_long / n_valid * 100 if n_valid > 0 else 0
    sig = "LONG bias" if pct > 55 else "SHORT bias" if pct < 45 else "50/50"
    print(f"  ATR {lo:2d}-{hi:2d}      | {mask.sum():6d} {n_long:6d} {n_short:6d} | {pct:5.1f}% {sig}")

# =============================================================================
# CHECK 3: Direction by RSI + ATR combinations
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 3: Direction by RSI + ATR combinations")
print(f"{'='*70}")

print(f"\n  {'Condition':<30s} | {'total':>6s} {'LONG':>6s} {'SHORT':>6s} | {'LONG%':>6s} {'race_gap':>8s}")
print(f"  {'-'*30}-+-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*6}-{'-'*8}")

combos = [
    ("RSI<10 + ATR>75", (rsi < 10) & (atr > 75)),
    ("RSI<10 + ATR>50", (rsi < 10) & (atr > 50)),
    ("RSI<10 + ATR<25", (rsi < 10) & (atr < 25)),
    ("RSI<20 + ATR>75", (rsi < 20) & (atr > 75)),
    ("RSI<20 + ATR>50", (rsi < 20) & (atr > 50)),
    ("RSI>80 + ATR>75", (rsi > 80) & (atr > 75)),
    ("RSI>80 + ATR>50", (rsi > 80) & (atr > 50)),
    ("RSI>80 + ATR<25", (rsi > 80) & (atr < 25)),
    ("RSI>90 + ATR>75", (rsi > 90) & (atr > 75)),
    ("RSI 40-60 + ATR>75", (rsi >= 40) & (rsi <= 60) & (atr > 75)),
    ("RSI 40-60 + ATR<25", (rsi >= 40) & (rsi <= 60) & (atr < 25)),
    ("RSI 40-60 + ATR 40-60", (rsi >= 40) & (rsi <= 60) & (atr >= 40) & (atr <= 60)),
]

for name, mask in combos:
    n_long = (mask & (direction == 0)).sum()
    n_short = (mask & (direction == 1)).sum()
    n_valid = n_long + n_short
    pct = n_long / n_valid * 100 if n_valid > 0 else 0

    # Average race gap for these bars
    valid_bars = mask & valid
    if valid_bars.sum() > 0:
        gaps = np.where(direction[valid_bars] == 0,
                       time_down[valid_bars] - time_up[valid_bars],
                       time_up[valid_bars] - time_down[valid_bars])
        # Handle cases where second threshold was never hit
        gaps = np.where(gaps < 0, 96, gaps)
        med_gap = np.median(np.abs(gaps))
    else:
        med_gap = 0

    print(f"  {name:<30s} | {mask.sum():6d} {n_long:6d} {n_short:6d} | {pct:5.1f}% {med_gap:7.0f}")

# =============================================================================
# CHECK 4: Direction by trend (price_above_sma200) + RSI
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 4: Direction by trend + RSI (V1 style conditions)")
print(f"{'='*70}")

above_sma = fc["price_above_sma200"].values
ema_sep = fc["ema_separation"].values

print(f"\n  {'Condition':<35s} | {'total':>6s} {'LONG':>6s} {'SHORT':>6s} | {'LONG%':>6s}")
print(f"  {'-'*35}-+-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*6}")

v1_combos = [
    ("RSI<10 + Bear (V1 SHORT)",      (rsi < 10) & (above_sma == 0)),
    ("RSI<10 + Bull",                  (rsi < 10) & (above_sma == 1)),
    ("RSI<20 + Bear",                  (rsi < 20) & (above_sma == 0)),
    ("RSI<20 + Bull (V1 LONG)",        (rsi < 20) & (above_sma == 1)),
    ("RSI>80 + Bear (V1 SHORT mirror)",(rsi > 80) & (above_sma == 0)),
    ("RSI>80 + Bull",                  (rsi > 80) & (above_sma == 1)),
    ("RSI>90 + Bear",                  (rsi > 90) & (above_sma == 0)),
    ("RSI>90 + Bull",                  (rsi > 90) & (above_sma == 1)),
    ("RSI<10 + EMA_sep>1%",            (rsi < 10) & (ema_sep > 1.0)),
    ("RSI>90 + EMA_sep>1%",            (rsi > 90) & (ema_sep > 1.0)),
    ("RSI<10 + EMA_sep<0.5%",          (rsi < 10) & (ema_sep < 0.5)),
    ("RSI>90 + EMA_sep<0.5%",          (rsi > 90) & (ema_sep < 0.5)),
]

for name, mask in v1_combos:
    n_long = (mask & (direction == 0)).sum()
    n_short = (mask & (direction == 1)).sum()
    n_valid = n_long + n_short
    pct = n_long / n_valid * 100 if n_valid > 0 else 0
    print(f"  {name:<35s} | {mask.sum():6d} {n_long:6d} {n_short:6d} | {pct:5.1f}%")

# =============================================================================
# CHECK 5: Race gap distribution — how close are the races?
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 5: Race gap distribution (all valid bars)")
print(f"{'='*70}")

valid_bars = valid
race_gaps = np.abs(np.where(direction[valid_bars] == 0,
                            time_down[valid_bars] - time_up[valid_bars],
                            time_up[valid_bars] - time_down[valid_bars]))
# Fix negative gaps (second threshold never hit)
race_gaps = np.where(time_up[valid_bars] == 0, 96, race_gaps)
race_gaps = np.where(time_down[valid_bars] == 0, 96, race_gaps)

print(f"\n  Total valid bars: {valid_bars.sum()}")
print(f"  Race gap distribution:")
print(f"    gap = 1 bar:   {(race_gaps == 1).sum():6d} ({(race_gaps == 1).mean()*100:.1f}%)")
print(f"    gap <= 2:      {(race_gaps <= 2).sum():6d} ({(race_gaps <= 2).mean()*100:.1f}%)")
print(f"    gap <= 3:      {(race_gaps <= 3).sum():6d} ({(race_gaps <= 3).mean()*100:.1f}%)")
print(f"    gap <= 5:      {(race_gaps <= 5).sum():6d} ({(race_gaps <= 5).mean()*100:.1f}%)")
print(f"    gap <= 10:     {(race_gaps <= 10).sum():6d} ({(race_gaps <= 10).mean()*100:.1f}%)")
print(f"    gap > 10:      {(race_gaps > 10).sum():6d} ({(race_gaps > 10).mean()*100:.1f}%)")
print(f"    gap > 20:      {(race_gaps > 20).sum():6d} ({(race_gaps > 20).mean()*100:.1f}%)")
print(f"    gap > 50:      {(race_gaps > 50).sum():6d} ({(race_gaps > 50).mean()*100:.1f}%)")
print(f"    median gap:    {np.median(race_gaps):.0f} bars")
print(f"    mean gap:      {race_gaps.mean():.1f} bars")

# =============================================================================
# CHECK 6: Direction accuracy by race gap (close races vs clear races)
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 6: If we only trade clear races (large gap), what's the LONG% ?")
print(f"{'='*70}")

print(f"\n  {'Gap filter':<20s} | {'n_bars':>6s} {'LONG':>6s} {'SHORT':>6s} | {'LONG%':>6s} | {'% of data'}")
print(f"  {'-'*20}-+-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*6}-+-{'-'*10}")

gap_filters = [
    ("All bars", race_gaps >= 0),
    ("gap > 3", race_gaps > 3),
    ("gap > 5", race_gaps > 5),
    ("gap > 10", race_gaps > 10),
    ("gap > 20", race_gaps > 20),
    ("gap > 30", race_gaps > 30),
    ("gap > 50", race_gaps > 50),
]

dir_valid = direction[valid_bars]
for name, gap_mask in gap_filters:
    n_long = (dir_valid[gap_mask] == 0).sum()
    n_short = (dir_valid[gap_mask] == 1).sum()
    n_valid = n_long + n_short
    pct = n_long / n_valid * 100 if n_valid > 0 else 0
    pct_data = gap_mask.sum() / len(race_gaps) * 100
    print(f"  {name:<20s} | {n_valid:6d} {n_long:6d} {n_short:6d} | {pct:5.1f}% | {pct_data:5.1f}%")

# =============================================================================
# CHECK 7: Sample bars — actual feature values for LONG vs SHORT
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 7: Sample bars — 5 LONG and 5 SHORT with RSI<20 + ATR>75")
print(f"{'='*70}")

sample_mask = (rsi < 20) & (atr > 75) & valid
sample_idx = np.where(sample_mask)[0]

if len(sample_idx) > 0:
    long_samples = sample_idx[direction[sample_idx] == 0][:5]
    short_samples = sample_idx[direction[sample_idx] == 1][:5]

    key_features = ["rsi7", "atr_percentile", "ema_separation", "range_bps",
                    "volume_ratio", "bb_position", "range_position", "momentum10",
                    "roc5", "price_above_sma200"]

    print(f"\n  LONG samples (RSI<20 + ATR>75):")
    print(f"  {'datetime':<22s}", end="")
    for f in key_features:
        print(f" {f:>12s}", end="")
    print(f" {'time_up':>8s} {'time_dn':>8s} {'gap':>5s}")

    for idx in long_samples:
        dt = common_idx[idx]
        print(f"  {str(dt):<22s}", end="")
        for f in key_features:
            print(f" {fc.iloc[idx][f]:12.2f}", end="")
        print(f" {time_up[idx]:8.0f} {time_down[idx]:8.0f} {abs(time_down[idx]-time_up[idx]):5.0f}")

    print(f"\n  SHORT samples (RSI<20 + ATR>75):")
    print(f"  {'datetime':<22s}", end="")
    for f in key_features:
        print(f" {f:>12s}", end="")
    print(f" {'time_up':>8s} {'time_dn':>8s} {'gap':>5s}")

    for idx in short_samples:
        dt = common_idx[idx]
        print(f"  {str(dt):<22s}", end="")
        for f in key_features:
            print(f" {fc.iloc[idx][f]:12.2f}", end="")
        print(f" {time_up[idx]:8.0f} {time_down[idx]:8.0f} {abs(time_down[idx]-time_up[idx]):5.0f}")
else:
    print("  No bars match RSI<20 + ATR>75")
