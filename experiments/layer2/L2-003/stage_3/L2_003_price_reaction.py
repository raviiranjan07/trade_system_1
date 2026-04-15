"""L2-003: Price reaction analysis — what happens after big moves?

When price drops/bounces 1-2% in 1-2 hours, what happens next?
Does history tell us the direction?

Run locally: python experiments/layer2/L2-003/L2_003_price_reaction.py
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

close = fc["close"].values
direction = lb["direction"].values  # 0=LONG, 1=SHORT, 2=BOTH
time_up = lb["time_to_first_up_hit"].values
time_down = lb["time_to_first_down_hit"].values
mfe_up_96 = lb["mfe_up_96"].values
mfe_down_96 = lb["mfe_down_96"].values

print(f"Total bars: {len(fc)}")

# =============================================================================
# Calculate price change over different lookback windows
# =============================================================================
# Lookback in bars (15 min per bar)
lookbacks = {
    "30min (2 bars)": 2,
    "1hr (4 bars)": 4,
    "2hr (8 bars)": 8,
    "4hr (16 bars)": 16,
    "8hr (32 bars)": 32,
}

# Calculate price change for each lookback
for name, bars in lookbacks.items():
    col_name = f"pchg_{bars}"
    pchg = np.zeros(len(close))
    pchg[bars:] = (close[bars:] - close[:-bars]) / close[:-bars] * 100
    fc[col_name] = pchg

# =============================================================================
# CHECK 1: After big drops, what direction?
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 1: After price DROPS, what direction does ±15bps hit first?")
print(f"{'='*70}")

print(f"\n  {'Condition':<35s} | {'total':>6s} {'LONG':>6s} {'SHORT':>6s} {'BOTH':>6s} | {'LONG%':>6s} {'avg_gap':>7s}")
print(f"  {'-'*35}-+-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*6}-{'-'*7}")

drop_conditions = [
    # (name, lookback_bars, min_drop%, max_drop%)
    ("Drop >0.5% in 1hr", 4, -999, -0.5),
    ("Drop >1.0% in 1hr", 4, -999, -1.0),
    ("Drop >1.5% in 1hr", 4, -999, -1.5),
    ("Drop >2.0% in 1hr", 4, -999, -2.0),
    ("Drop >0.5% in 2hr", 8, -999, -0.5),
    ("Drop >1.0% in 2hr", 8, -999, -1.0),
    ("Drop >1.5% in 2hr", 8, -999, -1.5),
    ("Drop >2.0% in 2hr", 8, -999, -2.0),
    ("Drop >3.0% in 2hr", 8, -999, -3.0),
    ("Drop >0.5% in 4hr", 16, -999, -0.5),
    ("Drop >1.0% in 4hr", 16, -999, -1.0),
    ("Drop >2.0% in 4hr", 16, -999, -2.0),
    ("Drop >3.0% in 4hr", 16, -999, -3.0),
    ("Drop >5.0% in 4hr", 16, -999, -5.0),
]

for name, bars, _, min_drop in drop_conditions:
    col = f"pchg_{bars}"
    mask = fc[col].values <= min_drop
    n = mask.sum()
    if n == 0:
        print(f"  {name:<35s} | {0:6d} {'---':>6s} {'---':>6s} {'---':>6s} | {'---':>6s}")
        continue

    n_long = (mask & (direction == 0)).sum()
    n_short = (mask & (direction == 1)).sum()
    n_both = (mask & (direction == 2)).sum()
    n_valid = n_long + n_short
    pct = n_long / n_valid * 100 if n_valid > 0 else 0

    # Average race gap
    valid_bars = mask & ((direction == 0) | (direction == 1))
    if valid_bars.sum() > 0:
        gaps = np.where(direction[valid_bars] == 0,
                       np.abs(time_down[valid_bars] - time_up[valid_bars]),
                       np.abs(time_up[valid_bars] - time_down[valid_bars]))
        avg_gap = np.median(gaps)
    else:
        avg_gap = 0

    print(f"  {name:<35s} | {n:6d} {n_long:6d} {n_short:6d} {n_both:6d} | {pct:5.1f}% {avg_gap:6.0f}")

# =============================================================================
# CHECK 2: After big bounces (rallies), what direction?
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 2: After price RALLIES, what direction does ±15bps hit first?")
print(f"{'='*70}")

print(f"\n  {'Condition':<35s} | {'total':>6s} {'LONG':>6s} {'SHORT':>6s} {'BOTH':>6s} | {'LONG%':>6s} {'avg_gap':>7s}")
print(f"  {'-'*35}-+-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*6}-{'-'*7}")

rally_conditions = [
    ("Rally >0.5% in 1hr", 4, 0.5, 999),
    ("Rally >1.0% in 1hr", 4, 1.0, 999),
    ("Rally >1.5% in 1hr", 4, 1.5, 999),
    ("Rally >2.0% in 1hr", 4, 2.0, 999),
    ("Rally >0.5% in 2hr", 8, 0.5, 999),
    ("Rally >1.0% in 2hr", 8, 1.0, 999),
    ("Rally >1.5% in 2hr", 8, 1.5, 999),
    ("Rally >2.0% in 2hr", 8, 2.0, 999),
    ("Rally >3.0% in 2hr", 8, 3.0, 999),
    ("Rally >0.5% in 4hr", 16, 0.5, 999),
    ("Rally >1.0% in 4hr", 16, 1.0, 999),
    ("Rally >2.0% in 4hr", 16, 2.0, 999),
    ("Rally >3.0% in 4hr", 16, 3.0, 999),
    ("Rally >5.0% in 4hr", 16, 5.0, 999),
]

for name, bars, min_rally, _ in rally_conditions:
    col = f"pchg_{bars}"
    mask = fc[col].values >= min_rally
    n = mask.sum()
    if n == 0:
        print(f"  {name:<35s} | {0:6d} {'---':>6s} {'---':>6s} {'---':>6s} | {'---':>6s}")
        continue

    n_long = (mask & (direction == 0)).sum()
    n_short = (mask & (direction == 1)).sum()
    n_both = (mask & (direction == 2)).sum()
    n_valid = n_long + n_short
    pct = n_long / n_valid * 100 if n_valid > 0 else 0

    valid_bars = mask & ((direction == 0) | (direction == 1))
    if valid_bars.sum() > 0:
        gaps = np.where(direction[valid_bars] == 0,
                       np.abs(time_down[valid_bars] - time_up[valid_bars]),
                       np.abs(time_up[valid_bars] - time_down[valid_bars]))
        avg_gap = np.median(gaps)
    else:
        avg_gap = 0

    print(f"  {name:<35s} | {n:6d} {n_long:6d} {n_short:6d} {n_both:6d} | {pct:5.1f}% {avg_gap:6.0f}")

# =============================================================================
# CHECK 3: After big drop + RSI extreme, what happens?
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 3: After big drop + RSI extreme, what happens?")
print(f"{'='*70}")

rsi = fc["rsi7"].values
ema_sep = fc["ema_separation"].values
above_sma = fc["price_above_sma200"].values

print(f"\n  {'Condition':<45s} | {'total':>6s} {'LONG':>6s} {'SHORT':>6s} | {'LONG%':>6s}")
print(f"  {'-'*45}-+-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*6}")

combo_conditions = [
    # Drops + RSI
    ("Drop>1% 2hr + RSI<20", (fc["pchg_8"].values <= -1.0) & (rsi < 20)),
    ("Drop>1% 2hr + RSI<10", (fc["pchg_8"].values <= -1.0) & (rsi < 10)),
    ("Drop>2% 2hr + RSI<20", (fc["pchg_8"].values <= -2.0) & (rsi < 20)),
    ("Drop>2% 4hr + RSI<20", (fc["pchg_16"].values <= -2.0) & (rsi < 20)),
    ("Drop>1% 2hr + RSI<20 + Bull", (fc["pchg_8"].values <= -1.0) & (rsi < 20) & (above_sma == 1)),
    ("Drop>1% 2hr + RSI<20 + Bear", (fc["pchg_8"].values <= -1.0) & (rsi < 20) & (above_sma == 0)),
    ("Drop>1% 2hr + RSI<20 + EMA>1%", (fc["pchg_8"].values <= -1.0) & (rsi < 20) & (ema_sep > 1.0)),
    # Rallies + RSI
    ("Rally>1% 2hr + RSI>80", (fc["pchg_8"].values >= 1.0) & (rsi > 80)),
    ("Rally>1% 2hr + RSI>90", (fc["pchg_8"].values >= 1.0) & (rsi > 90)),
    ("Rally>2% 2hr + RSI>80", (fc["pchg_8"].values >= 2.0) & (rsi > 80)),
    ("Rally>2% 4hr + RSI>80", (fc["pchg_16"].values >= 2.0) & (rsi > 80)),
    ("Rally>1% 2hr + RSI>80 + Bear", (fc["pchg_8"].values >= 1.0) & (rsi > 80) & (above_sma == 0)),
    ("Rally>1% 2hr + RSI>80 + Bull", (fc["pchg_8"].values >= 1.0) & (rsi > 80) & (above_sma == 1)),
    ("Rally>1% 2hr + RSI>80 + EMA>1%", (fc["pchg_8"].values >= 1.0) & (rsi > 80) & (ema_sep > 1.0)),
]

for name, mask in combo_conditions:
    n = mask.sum()
    if n == 0:
        print(f"  {name:<45s} | {0:6d} {'---':>6s} {'---':>6s} | {'---':>6s}")
        continue
    n_long = (mask & (direction == 0)).sum()
    n_short = (mask & (direction == 1)).sum()
    n_valid = n_long + n_short
    pct = n_long / n_valid * 100 if n_valid > 0 else 0
    print(f"  {name:<45s} | {n:6d} {n_long:6d} {n_short:6d} | {pct:5.1f}%")

# =============================================================================
# CHECK 4: MFE after big drops — how far does bounce go?
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 4: After big drops, how far does price move? (MFE in 96 bars)")
print(f"{'='*70}")

print(f"\n  {'Condition':<35s} | {'n':>6s} | {'mfe_up':>8s} {'mfe_dn':>8s} {'diff':>8s}")
print(f"  {'-'*35}-+-{'-'*6}-+-{'-'*8}-{'-'*8}-{'-'*8}")

mfe_conditions = [
    ("All bars", np.ones(len(fc), dtype=bool)),
    ("Drop >1% in 2hr", fc["pchg_8"].values <= -1.0),
    ("Drop >2% in 2hr", fc["pchg_8"].values <= -2.0),
    ("Drop >3% in 4hr", fc["pchg_16"].values <= -3.0),
    ("Rally >1% in 2hr", fc["pchg_8"].values >= 1.0),
    ("Rally >2% in 2hr", fc["pchg_8"].values >= 2.0),
    ("Rally >3% in 4hr", fc["pchg_16"].values >= 3.0),
    ("Drop>1% 2hr + RSI<20", (fc["pchg_8"].values <= -1.0) & (rsi < 20)),
    ("Rally>1% 2hr + RSI>80", (fc["pchg_8"].values >= 1.0) & (rsi > 80)),
]

for name, mask in mfe_conditions:
    n = mask.sum()
    if n == 0:
        print(f"  {name:<35s} | {0:6d} | {'---':>8s} {'---':>8s} {'---':>8s}")
        continue
    up = mfe_up_96[mask].mean()
    dn = mfe_down_96[mask].mean()
    diff = up - dn
    print(f"  {name:<35s} | {n:6d} | {up:7.1f} {dn:7.1f} {diff:+7.1f}")

# =============================================================================
# CHECK 5: Does the drop CONTINUE or REVERSE?
# =============================================================================
print(f"\n{'='*70}")
print("CHECK 5: After big drops — continuation vs reversal")
print(f"{'='*70}")

print(f"\n  After Drop >1% in 2hr:")
mask = fc["pchg_8"].values <= -1.0
n = mask.sum()
if n > 0:
    # What happens in next 96 bars?
    up_bigger = mfe_up_96[mask] > mfe_down_96[mask]  # bounce more than continuation
    print(f"    Bars: {n}")
    print(f"    Bounce bigger (mfe_up > mfe_down): {up_bigger.sum()} ({up_bigger.mean()*100:.1f}%)")
    print(f"    Drop continues (mfe_down > mfe_up): {(~up_bigger).sum()} ({(~up_bigger).mean()*100:.1f}%)")

    # By how much?
    print(f"    When bounce wins: avg mfe_up={mfe_up_96[mask][up_bigger].mean():.1f}, avg mfe_down={mfe_down_96[mask][up_bigger].mean():.1f}")
    print(f"    When drop continues: avg mfe_up={mfe_up_96[mask][~up_bigger].mean():.1f}, avg mfe_down={mfe_down_96[mask][~up_bigger].mean():.1f}")

print(f"\n  After Rally >1% in 2hr:")
mask = fc["pchg_8"].values >= 1.0
n = mask.sum()
if n > 0:
    down_bigger = mfe_down_96[mask] > mfe_up_96[mask]
    print(f"    Bars: {n}")
    print(f"    Pullback bigger (mfe_down > mfe_up): {down_bigger.sum()} ({down_bigger.mean()*100:.1f}%)")
    print(f"    Rally continues (mfe_up > mfe_down): {(~down_bigger).sum()} ({(~down_bigger).mean()*100:.1f}%)")

    print(f"    When pullback wins: avg mfe_up={mfe_up_96[mask][down_bigger].mean():.1f}, avg mfe_down={mfe_down_96[mask][down_bigger].mean():.1f}")
    print(f"    When rally continues: avg mfe_up={mfe_up_96[mask][~down_bigger].mean():.1f}, avg mfe_down={mfe_down_96[mask][~down_bigger].mean():.1f}")
