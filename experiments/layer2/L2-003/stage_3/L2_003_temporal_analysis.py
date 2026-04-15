"""L2-003: Temporal pattern analysis — do sequential patterns predict direction?

Checks if temporal patterns (e.g., "roc falling for 5 bars") consistently
predict direction, or if the same pattern gives 50/50 LONG/SHORT.

If patterns predict → model/architecture problem (LSTM should learn this)
If patterns don't predict → data problem (no temporal signal exists)

Run locally: python experiments/layer2/L2-003/L2_003_temporal_analysis.py
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
direction = lb["direction"].values  # 0=LONG, 1=SHORT, 2=BOTH
valid = (direction == 0) | (direction == 1)

# Compute roc1
roc1 = np.zeros(len(close))
roc1[1:] = (close[1:] - close[:-1]) / close[:-1] * 10000

rsi7 = fc["rsi7"].values
range_pos = fc["range_position"].values

print(f"Total bars: {len(fc)} | Valid: {valid.sum()}")

# =============================================================================
# CHECK 1: Consecutive falling roc1 (price dropping for N bars)
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 1: Consecutive falling bars — does streak predict direction?")
print(f"{'='*70}")

# Count consecutive negative roc1
consec_down = np.zeros(len(roc1), dtype=int)
for i in range(1, len(roc1)):
    if roc1[i] < 0:
        consec_down[i] = consec_down[i-1] + 1
    else:
        consec_down[i] = 0

consec_up = np.zeros(len(roc1), dtype=int)
for i in range(1, len(roc1)):
    if roc1[i] > 0:
        consec_up[i] = consec_up[i-1] + 1
    else:
        consec_up[i] = 0

print(f"\n  Consecutive DOWN bars → next direction:")
print(f"  {'Streak':<12s} | {'N bars':>6s} {'LONG':>6s} {'SHORT':>6s} | {'LONG%':>6s} {'signal?'}")
print(f"  {'-'*12}-+-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*6}-{'-'*10}")

for streak in [1, 2, 3, 4, 5, 6, 7, 8]:
    mask = (consec_down == streak) & valid
    n = mask.sum()
    if n > 0:
        n_long = (mask & (direction == 0)).sum()
        n_short = (mask & (direction == 1)).sum()
        pct = n_long / (n_long + n_short) * 100
        sig = "LONG bias" if pct > 55 else "SHORT bias" if pct < 45 else "50/50"
        print(f"  {streak} down bars | {n:6d} {n_long:6d} {n_short:6d} | {pct:5.1f}% {sig}")

print(f"\n  Consecutive UP bars → next direction:")
print(f"  {'Streak':<12s} | {'N bars':>6s} {'LONG':>6s} {'SHORT':>6s} | {'LONG%':>6s} {'signal?'}")
print(f"  {'-'*12}-+-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*6}-{'-'*10}")

for streak in [1, 2, 3, 4, 5, 6, 7, 8]:
    mask = (consec_up == streak) & valid
    n = mask.sum()
    if n > 0:
        n_long = (mask & (direction == 0)).sum()
        n_short = (mask & (direction == 1)).sum()
        pct = n_long / (n_long + n_short) * 100
        sig = "LONG bias" if pct > 55 else "SHORT bias" if pct < 45 else "50/50"
        print(f"  {streak} up bars   | {n:6d} {n_long:6d} {n_short:6d} | {pct:5.1f}% {sig}")

# =============================================================================
# CHECK 2: RSI trajectory — rising vs falling RSI over last 5 bars
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 2: RSI trajectory — direction after rising vs falling RSI")
print(f"{'='*70}")

rsi_change_5 = np.zeros(len(rsi7))
rsi_change_5[5:] = rsi7[5:] - rsi7[:-5]

print(f"\n  {'RSI change (5 bar)':<25s} | {'N bars':>6s} {'LONG':>6s} {'SHORT':>6s} | {'LONG%':>6s}")
print(f"  {'-'*25}-+-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*6}")

rsi_buckets = [
    ("RSI dropped >20", rsi_change_5 < -20),
    ("RSI dropped 10-20", (rsi_change_5 >= -20) & (rsi_change_5 < -10)),
    ("RSI dropped 5-10", (rsi_change_5 >= -10) & (rsi_change_5 < -5)),
    ("RSI stable (-5 to +5)", (rsi_change_5 >= -5) & (rsi_change_5 <= 5)),
    ("RSI rose 5-10", (rsi_change_5 > 5) & (rsi_change_5 <= 10)),
    ("RSI rose 10-20", (rsi_change_5 > 10) & (rsi_change_5 <= 20)),
    ("RSI rose >20", rsi_change_5 > 20),
]

for name, mask in rsi_buckets:
    mask = mask & valid
    n_long = (mask & (direction == 0)).sum()
    n_short = (mask & (direction == 1)).sum()
    n_valid = n_long + n_short
    pct = n_long / n_valid * 100 if n_valid > 0 else 0
    print(f"  {name:<25s} | {n_valid:6d} {n_long:6d} {n_short:6d} | {pct:5.1f}%")

# =============================================================================
# CHECK 3: ROC acceleration — is momentum speeding up or slowing down?
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 3: ROC acceleration — speeding up vs slowing down")
print(f"{'='*70}")

# roc1 acceleration = roc1(t) - roc1(t-1)
roc1_accel = np.zeros(len(roc1))
roc1_accel[1:] = roc1[1:] - roc1[:-1]

# Pattern: roc1 negative AND accelerating (getting more negative)
# vs roc1 negative AND decelerating (getting less negative)

print(f"\n  {'Pattern':<35s} | {'N':>6s} {'LONG':>6s} {'SHORT':>6s} | {'LONG%':>6s}")
print(f"  {'-'*35}-+-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*6}")

patterns = [
    ("Falling + accelerating down", (roc1 < 0) & (roc1_accel < -5)),
    ("Falling + steady", (roc1 < 0) & (roc1_accel >= -5) & (roc1_accel <= 5)),
    ("Falling + decelerating", (roc1 < 0) & (roc1_accel > 5)),
    ("Rising + accelerating up", (roc1 > 0) & (roc1_accel > 5)),
    ("Rising + steady", (roc1 > 0) & (roc1_accel >= -5) & (roc1_accel <= 5)),
    ("Rising + decelerating", (roc1 > 0) & (roc1_accel < -5)),
]

for name, mask in patterns:
    mask = mask & valid
    n_long = (mask & (direction == 0)).sum()
    n_short = (mask & (direction == 1)).sum()
    n_valid = n_long + n_short
    pct = n_long / n_valid * 100 if n_valid > 0 else 0
    print(f"  {name:<35s} | {n_valid:6d} {n_long:6d} {n_short:6d} | {pct:5.1f}%")

# =============================================================================
# CHECK 4: Range position trajectory — moving toward top or bottom
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 4: Range position trajectory")
print(f"{'='*70}")

rp_change_3 = np.zeros(len(range_pos))
rp_change_3[3:] = range_pos[3:] - range_pos[:-3]

print(f"\n  {'Pattern':<35s} | {'N':>6s} {'LONG':>6s} {'SHORT':>6s} | {'LONG%':>6s}")
print(f"  {'-'*35}-+-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*6}")

rp_patterns = [
    ("Near bottom + falling", (range_pos < 0.2) & (rp_change_3 < -0.1)),
    ("Near bottom + rising", (range_pos < 0.2) & (rp_change_3 > 0.1)),
    ("Near bottom + stable", (range_pos < 0.2) & (rp_change_3 >= -0.1) & (rp_change_3 <= 0.1)),
    ("Middle + falling", (range_pos >= 0.3) & (range_pos <= 0.7) & (rp_change_3 < -0.1)),
    ("Middle + rising", (range_pos >= 0.3) & (range_pos <= 0.7) & (rp_change_3 > 0.1)),
    ("Near top + falling", (range_pos > 0.8) & (rp_change_3 < -0.1)),
    ("Near top + rising", (range_pos > 0.8) & (rp_change_3 > 0.1)),
    ("Near top + stable", (range_pos > 0.8) & (rp_change_3 >= -0.1) & (rp_change_3 <= 0.1)),
]

for name, mask in rp_patterns:
    mask = mask & valid
    n_long = (mask & (direction == 0)).sum()
    n_short = (mask & (direction == 1)).sum()
    n_valid = n_long + n_short
    pct = n_long / n_valid * 100 if n_valid > 0 else 0
    print(f"  {name:<35s} | {n_valid:6d} {n_long:6d} {n_short:6d} | {pct:5.1f}%")

# =============================================================================
# CHECK 5: Combined temporal patterns
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 5: Combined temporal patterns — strongest combos")
print(f"{'='*70}")

print(f"\n  {'Pattern':<50s} | {'N':>6s} {'LONG%':>6s}")
print(f"  {'-'*50}-+-{'-'*6}-{'-'*6}")

combos = [
    ("5+ down bars + RSI<20", (consec_down >= 5) & (rsi7 < 20)),
    ("5+ down bars + RSI<10", (consec_down >= 5) & (rsi7 < 10)),
    ("3+ down bars + RSI dropping >10", (consec_down >= 3) & (rsi_change_5 < -10)),
    ("3+ down bars + range_pos<0.2", (consec_down >= 3) & (range_pos < 0.2)),
    ("5+ up bars + RSI>80", (consec_up >= 5) & (rsi7 > 80)),
    ("5+ up bars + RSI>90", (consec_up >= 5) & (rsi7 > 90)),
    ("3+ up bars + RSI rising >10", (consec_up >= 3) & (rsi_change_5 > 10)),
    ("3+ up bars + range_pos>0.8", (consec_up >= 3) & (range_pos > 0.8)),
    ("Falling + accel down + RSI<30", (roc1 < 0) & (roc1_accel < -5) & (rsi7 < 30)),
    ("Rising + accel up + RSI>70", (roc1 > 0) & (roc1_accel > 5) & (rsi7 > 70)),
    ("Bottom + bouncing (rp<0.2, rp rising)", (range_pos < 0.2) & (rp_change_3 > 0.1) & (rsi7 < 30)),
    ("Top + falling (rp>0.8, rp falling)", (range_pos > 0.8) & (rp_change_3 < -0.1) & (rsi7 > 70)),
]

for name, mask in combos:
    mask = mask & valid
    n_long = (mask & (direction == 0)).sum()
    n_short = (mask & (direction == 1)).sum()
    n_valid = n_long + n_short
    pct = n_long / n_valid * 100 if n_valid > 0 else 0
    sig = "***" if pct > 58 or pct < 42 else ""
    print(f"  {name:<50s} | {n_valid:6d} {pct:5.1f}% {sig}")

# =============================================================================
# CHECK 6: Temporal pattern vs single-bar — is temporal info additive?
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 6: Does temporal info add beyond single-bar features?")
print(f"{'='*70}")

# Compare: RSI<20 alone vs RSI<20 + falling for 3 bars
rsi_low = (rsi7 < 20) & valid
rsi_low_falling = (rsi7 < 20) & (rsi_change_5 < -10) & valid
rsi_low_stable = (rsi7 < 20) & (rsi_change_5 >= -5) & (rsi_change_5 <= 5) & valid

n1 = (rsi_low & (direction == 0)).sum()
t1 = rsi_low.sum()
n2 = (rsi_low_falling & (direction == 0)).sum()
t2 = rsi_low_falling.sum()
n3 = (rsi_low_stable & (direction == 0)).sum()
t3 = rsi_low_stable.sum()

print(f"\n  RSI<20 (any trajectory):        {t1:5d} bars, LONG={n1/t1*100:.1f}%")
print(f"  RSI<20 + RSI was falling fast:  {t2:5d} bars, LONG={n2/t2*100:.1f}%")
print(f"  RSI<20 + RSI was stable:        {t3:5d} bars, LONG={n3/t3*100:.1f}%")

if abs(n2/t2 - n3/t3) > 0.03:
    print(f"  → Trajectory MATTERS: {abs(n2/t2 - n3/t3)*100:.1f}% difference")
else:
    print(f"  → Trajectory does NOT matter: only {abs(n2/t2 - n3/t3)*100:.1f}% difference")

# Same for RSI>80
rsi_high = (rsi7 > 80) & valid
rsi_high_rising = (rsi7 > 80) & (rsi_change_5 > 10) & valid
rsi_high_stable = (rsi7 > 80) & (rsi_change_5 >= -5) & (rsi_change_5 <= 5) & valid

n1 = (rsi_high & (direction == 0)).sum()
t1 = rsi_high.sum()
n2 = (rsi_high_rising & (direction == 0)).sum()
t2 = rsi_high_rising.sum()
n3 = (rsi_high_stable & (direction == 0)).sum()
t3 = rsi_high_stable.sum()

print(f"\n  RSI>80 (any trajectory):        {t1:5d} bars, LONG={n1/t1*100:.1f}%")
print(f"  RSI>80 + RSI was rising fast:   {t2:5d} bars, LONG={n2/t2*100:.1f}%")
print(f"  RSI>80 + RSI was stable:        {t3:5d} bars, LONG={n3/t3*100:.1f}%")

if abs(n2/t2 - n3/t3) > 0.03:
    print(f"  → Trajectory MATTERS: {abs(n2/t2 - n3/t3)*100:.1f}% difference")
else:
    print(f"  → Trajectory does NOT matter: only {abs(n2/t2 - n3/t3)*100:.1f}% difference")
