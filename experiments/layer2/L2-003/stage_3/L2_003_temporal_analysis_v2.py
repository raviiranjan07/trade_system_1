"""L2-003: Temporal analysis V2 — thorough ROC trajectory analysis

Tests if the PATTERN of how roc1-8 changes over bars predicts direction
better than single-bar values alone.

Run locally: python experiments/layer2/L2-003/L2_003_temporal_analysis_v2.py
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

close = fc["close"].values.astype(np.float64)
direction = lb["direction"].values
valid = (direction == 0) | (direction == 1)

# Compute roc1
roc1 = np.zeros(len(close))
roc1[1:] = (close[1:] - close[:-1]) / close[:-1] * 10000

rsi7 = fc["rsi7"].values
range_pos = fc["range_position"].values

print(f"Total bars: {len(fc)} | Valid: {valid.sum()}")

# =============================================================================
# CHECK 1: roc1 trajectory over 3, 5, 8 bars
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 1: Does roc1 TRAJECTORY predict better than single roc1?")
print(f"{'='*70}")

for lookback in [3, 5, 8]:
    print(f"\n  --- Lookback = {lookback} bars ---")

    # Compute trajectory features
    roc1_now = roc1.copy()
    roc1_mean = np.zeros(len(roc1))
    roc1_trend = np.zeros(len(roc1))  # slope of roc1 over lookback
    roc1_min = np.zeros(len(roc1))
    roc1_max = np.zeros(len(roc1))

    for i in range(lookback, len(roc1)):
        window = roc1[i-lookback:i+1]
        roc1_mean[i] = window.mean()
        roc1_min[i] = window.min()
        roc1_max[i] = window.max()
        # Trend: is roc1 getting more negative or more positive?
        roc1_trend[i] = roc1[i] - roc1[i-lookback]

    # Single bar: roc1 < -30 (bearish)
    mask_single = (roc1_now < -30) & valid
    n_s = mask_single.sum()
    pct_s = (direction[mask_single] == 0).mean() * 100 if n_s > 0 else 0

    # Trajectory: roc1 < -30 AND has been falling (trend < 0)
    mask_falling = (roc1_now < -30) & (roc1_trend < -20) & valid
    n_f = mask_falling.sum()
    pct_f = (direction[mask_falling] == 0).mean() * 100 if n_f > 0 else 0

    # Trajectory: roc1 < -30 AND was recovering (trend > 0)
    mask_recover = (roc1_now < -30) & (roc1_trend > 20) & valid
    n_r = mask_recover.sum()
    pct_r = (direction[mask_recover] == 0).mean() * 100 if n_r > 0 else 0

    # Trajectory: roc1 < -30 AND steady (trend near 0)
    mask_steady = (roc1_now < -30) & (roc1_trend >= -20) & (roc1_trend <= 20) & valid
    n_st = mask_steady.sum()
    pct_st = (direction[mask_steady] == 0).mean() * 100 if n_st > 0 else 0

    print(f"  roc1 < -30 (single bar):              {n_s:6d} bars, LONG={pct_s:.1f}%")
    print(f"  roc1 < -30 + falling (trend<-20):      {n_f:6d} bars, LONG={pct_f:.1f}%")
    print(f"  roc1 < -30 + recovering (trend>+20):   {n_r:6d} bars, LONG={pct_r:.1f}%")
    print(f"  roc1 < -30 + steady (trend -20 to +20):{n_st:6d} bars, LONG={pct_st:.1f}%")
    diff = abs(pct_f - pct_r)
    print(f"  Trajectory difference: {diff:.1f}% ({'SIGNIFICANT' if diff > 3 else 'minimal'})")

# =============================================================================
# CHECK 2: roc1 at different magnitudes — does trajectory matter more at extremes?
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 2: Trajectory effect at different roc1 magnitudes")
print(f"{'='*70}")

lookback = 5
roc1_trend_5 = np.zeros(len(roc1))
for i in range(lookback, len(roc1)):
    roc1_trend_5[i] = roc1[i] - roc1[i-lookback]

print(f"\n  {'roc1 range':<20s} | {'Single':>6s} | {'Falling':>7s} {'(n)':>6s} | {'Recover':>7s} {'(n)':>6s} | {'diff':>5s}")
print(f"  {'-'*20}-+-{'-'*6}-+-{'-'*7}-{'-'*6}-+-{'-'*7}-{'-'*6}-+-{'-'*5}")

for lo, hi, label in [(-200, -50, "roc1 < -50 (crash)"),
                       (-50, -20, "roc1 -50 to -20"),
                       (-20, -5, "roc1 -20 to -5"),
                       (5, 20, "roc1 +5 to +20"),
                       (20, 50, "roc1 +20 to +50"),
                       (50, 200, "roc1 > +50 (spike)")]:
    mask_base = (roc1 >= lo) & (roc1 < hi) & valid
    mask_fall = mask_base & (roc1_trend_5 < -20)
    mask_rec = mask_base & (roc1_trend_5 > 20)

    n_base = mask_base.sum()
    n_fall = mask_fall.sum()
    n_rec = mask_rec.sum()

    pct_base = (direction[mask_base] == 0).mean() * 100 if n_base > 0 else 0
    pct_fall = (direction[mask_fall] == 0).mean() * 100 if n_fall > 0 else 0
    pct_rec = (direction[mask_rec] == 0).mean() * 100 if n_rec > 0 else 0
    diff = abs(pct_fall - pct_rec)

    print(f"  {label:<20s} | {pct_base:5.1f}% | {pct_fall:6.1f}% {n_fall:5d} | {pct_rec:6.1f}% {n_rec:5d} | {diff:4.1f}%")

# =============================================================================
# CHECK 3: Multi-bar roc1 pattern — crash then bounce vs steady decline
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 3: Specific temporal patterns in roc1")
print(f"{'='*70}")

print(f"\n  {'Pattern':<50s} | {'N':>6s} {'LONG%':>6s}")
print(f"  {'-'*50}-+-{'-'*6}-{'-'*6}")

# Build roc1 at different lags
roc1_lag1 = np.zeros(len(roc1)); roc1_lag1[1:] = roc1[:-1]
roc1_lag2 = np.zeros(len(roc1)); roc1_lag2[2:] = roc1[:-2]
roc1_lag3 = np.zeros(len(roc1)); roc1_lag3[3:] = roc1[:-3]
roc1_lag4 = np.zeros(len(roc1)); roc1_lag4[4:] = roc1[:-4]

patterns = [
    # Crash then bounce
    ("Crash→bounce: lag2<-30, lag1<-30, now>0",
     (roc1_lag2 < -30) & (roc1_lag1 < -30) & (roc1 > 0)),

    ("Crash→bounce: lag3<-30, lag2<-20, lag1>0, now>0",
     (roc1_lag3 < -30) & (roc1_lag2 < -20) & (roc1_lag1 > 0) & (roc1 > 0)),

    # Steady decline
    ("Steady decline: all 4 lags negative",
     (roc1_lag4 < 0) & (roc1_lag3 < 0) & (roc1_lag2 < 0) & (roc1_lag1 < 0) & (roc1 < 0)),

    # Accelerating decline
    ("Accel decline: each more negative than prev",
     (roc1 < roc1_lag1) & (roc1_lag1 < roc1_lag2) & (roc1_lag2 < roc1_lag3) & (roc1 < -10)),

    # Decelerating decline (slowing down)
    ("Decel decline: negative but getting less negative",
     (roc1 < 0) & (roc1_lag1 < 0) & (roc1 > roc1_lag1) & (roc1_lag1 > roc1_lag2) & (roc1_lag2 < -20)),

    # Spike then reverse
    ("Spike up→reverse: lag2>30, lag1>0, now<0",
     (roc1_lag2 > 30) & (roc1_lag1 > 0) & (roc1 < 0)),

    ("Spike down→reverse: lag2<-30, lag1<0, now>0",
     (roc1_lag2 < -30) & (roc1_lag1 < 0) & (roc1 > 0)),

    # V-shape bottom
    ("V-bottom: lag4<-20, lag3<-30, lag2 most negative, lag1>lag2, now>lag1",
     (roc1_lag4 < -20) & (roc1_lag3 < -30) & (roc1_lag2 < roc1_lag3) &
     (roc1_lag1 > roc1_lag2) & (roc1 > roc1_lag1)),

    # Momentum building
    ("Momentum building: 4 bars increasing, all positive",
     (roc1 > 0) & (roc1 > roc1_lag1) & (roc1_lag1 > roc1_lag2) & (roc1_lag2 > 0)),

    # Momentum fading
    ("Momentum fading: positive but each smaller",
     (roc1 > 0) & (roc1 < roc1_lag1) & (roc1_lag1 < roc1_lag2) & (roc1_lag2 > 0)),
]

for name, mask in patterns:
    mask = mask & valid
    n = mask.sum()
    if n > 50:
        pct = (direction[mask] == 0).mean() * 100
        sig = "***" if pct > 55 or pct < 45 else ""
        print(f"  {name:<50s} | {n:6d} {pct:5.1f}% {sig}")
    else:
        print(f"  {name:<50s} | {n:6d} (too few)")

# =============================================================================
# CHECK 4: range_position trajectory
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 4: range_position trajectory over 5 bars")
print(f"{'='*70}")

rp_5ago = np.zeros(len(range_pos))
rp_5ago[5:] = range_pos[:-5]
rp_change = range_pos - rp_5ago

print(f"\n  {'Pattern':<40s} | {'N':>6s} {'LONG%':>6s} {'diff vs single':>14s}")
print(f"  {'-'*40}-+-{'-'*6}-{'-'*6}-{'-'*14}")

# Single bar
for rp_lo, rp_hi, label in [(0, 0.15, "rp < 0.15 (bottom)"),
                              (0.15, 0.30, "rp 0.15-0.30"),
                              (0.70, 0.85, "rp 0.70-0.85"),
                              (0.85, 1.0, "rp > 0.85 (top)")]:
    mask_single = (range_pos >= rp_lo) & (range_pos < rp_hi) & valid
    mask_falling = mask_single & (rp_change < -0.15)
    mask_rising = mask_single & (rp_change > 0.15)
    mask_stable = mask_single & (rp_change >= -0.15) & (rp_change <= 0.15)

    n_s = mask_single.sum()
    pct_s = (direction[mask_single] == 0).mean() * 100 if n_s > 0 else 0

    n_f = mask_falling.sum()
    pct_f = (direction[mask_falling] == 0).mean() * 100 if n_f > 0 else 0

    n_r = mask_rising.sum()
    pct_r = (direction[mask_rising] == 0).mean() * 100 if n_r > 0 else 0

    print(f"  {label:<40s} | {n_s:6d} {pct_s:5.1f}%")
    if n_f > 100:
        print(f"    + was falling (rp dropped >0.15)     | {n_f:6d} {pct_f:5.1f}% {pct_f-pct_s:+13.1f}%")
    if n_r > 100:
        print(f"    + was rising (rp rose >0.15)         | {n_r:6d} {pct_r:5.1f}% {pct_r-pct_s:+13.1f}%")

# =============================================================================
# CHECK 5: Combined roc1 + range_position trajectory
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 5: Combined roc1 trajectory + range_position trajectory")
print(f"{'='*70}")

print(f"\n  {'Pattern':<55s} | {'N':>6s} {'LONG%':>6s}")
print(f"  {'-'*55}-+-{'-'*6}-{'-'*6}")

combos = [
    ("roc1<-30 + rp<0.2 + rp was falling",
     (roc1 < -30) & (range_pos < 0.2) & (rp_change < -0.15)),
    ("roc1<-30 + rp<0.2 + rp was stable",
     (roc1 < -30) & (range_pos < 0.2) & (rp_change >= -0.15) & (rp_change <= 0.15)),
    ("roc1<-30 + rp<0.2 + rp was rising (bouncing)",
     (roc1 < -30) & (range_pos < 0.2) & (rp_change > 0.15)),
    ("roc1>+30 + rp>0.8 + rp was rising",
     (roc1 > 30) & (range_pos > 0.8) & (rp_change > 0.15)),
    ("roc1>+30 + rp>0.8 + rp was stable",
     (roc1 > 30) & (range_pos > 0.8) & (rp_change >= -0.15) & (rp_change <= 0.15)),
    ("roc1>+30 + rp>0.8 + rp was falling (reversing)",
     (roc1 > 30) & (range_pos > 0.8) & (rp_change < -0.15)),
    ("Crash→bounce: roc1_lag2<-30 + now>0 + rp<0.3",
     (roc1_lag2 < -30) & (roc1 > 0) & (range_pos < 0.3)),
    ("Spike→reverse: roc1_lag2>30 + now<0 + rp>0.7",
     (roc1_lag2 > 30) & (roc1 < 0) & (range_pos > 0.7)),
]

for name, mask in combos:
    mask = mask & valid
    n = mask.sum()
    if n > 50:
        pct = (direction[mask] == 0).mean() * 100
        sig = "***" if pct > 57 or pct < 43 else ""
        print(f"  {name:<55s} | {n:6d} {pct:5.1f}% {sig}")
    else:
        print(f"  {name:<55s} | {n:6d} (too few)")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*70}")
print("SUMMARY: Does temporal trajectory add predictive power?")
print(f"{'='*70}")
print(f"""
  If trajectory differences > 3%: temporal signal EXISTS, LSTM should learn it
  If trajectory differences < 3%: temporal signal TOO WEAK, MLP is sufficient
""")
