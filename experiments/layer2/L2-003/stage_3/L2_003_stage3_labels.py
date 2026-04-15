"""L2-003 Stage 3: Compute Labels for Training

Loads feature_cache.parquet (has open, high, low, close, volume, atr_pct)
Computes lookahead labels for each valid bar
Saves labels.parquet

Labels:
  mfe_up_1,2,3,4,5,6,7,8,10,20,32,96   — max upward move from entry within H bars (bps)
  mfe_down_1,2,3,4,5,6,7,8,10,20,32,96 — max downward move from entry within H bars (bps)
  time_to_first_up_hit                  — first bar where high >= entry + 15bps (0 if never) [96 bar window]
  time_to_first_down_hit               — first bar where low  <= entry - 15bps (0 if never) [96 bar window]
  direction                            — LONG=0, SHORT=1, BOTH=2 [96 bar window]
  direction_h8                         — LONG=0, SHORT=1, BOTH=2, SKIP=3 [8 bar window]
  vol_expansion                        — 1 if ATR at peak bar > ATR at current bar
  volume_expansion                     — 1 if volume at peak bar > volume_ma20 at current bar

Entry = open[i+1] (next bar open)

Run: PYTHONPATH=src python experiments/layer2/L2-003/L2_003_stage3_labels.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent.parent.parent.parent
CACHE    = ROOT / "data/features/direction_prediction/feature_cache.parquet"
OUT_PATH = ROOT / "data/features/direction_prediction/labels.parquet"

HORIZONS      = [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 32, 96]
LOOKAHEAD_MAX = 96
MIN_LOOKBACK  = 96
THRESHOLD_BPS = 15.0   # Rule #2

# Direction labels
LONG  = 0
SHORT = 1
BOTH  = 2
SKIP  = 3

# =============================================================================
print("Loading feature cache...")
df = pd.read_parquet(CACHE)
print(f"  Shape: {df.shape}")
print(f"  Date range: {df.index.min()} to {df.index.max()}")

# Pre-compute volume_ma20
volume_ma20 = df["volume"].rolling(20).mean().values

# Extract arrays for speed
open_arr   = df["open"].values
high_arr   = df["high"].values
low_arr    = df["low"].values
atr_arr    = df["atr_pct"].values
volume_arr = df["volume"].values
dates      = df.index

N = len(df)

# Valid range: need MIN_LOOKBACK history and LOOKAHEAD_MAX future
valid_start = MIN_LOOKBACK
valid_end   = N - LOOKAHEAD_MAX  # exclusive

print(f"  Valid bars: {valid_start} to {valid_end-1} ({valid_end - valid_start} bars)")

# =============================================================================
# Pre-allocate output arrays
n_valid = valid_end - valid_start

mfe_up   = {H: np.zeros(n_valid, dtype=np.float32) for H in HORIZONS}
mfe_down = {H: np.zeros(n_valid, dtype=np.float32) for H in HORIZONS}
time_up   = np.zeros(n_valid, dtype=np.int16)
time_down = np.zeros(n_valid, dtype=np.int16)
direction = np.zeros(n_valid, dtype=np.int8)
dir_h8    = np.full(n_valid, SKIP, dtype=np.int8)
vol_exp   = np.zeros(n_valid, dtype=np.int8)
volm_exp  = np.zeros(n_valid, dtype=np.int8)

# =============================================================================
print("Computing labels...")

for idx, i in enumerate(range(valid_start, valid_end)):

    entry       = open_arr[i + 1]           # entry = next bar open
    thresh_up   = entry * (1 + THRESHOLD_BPS / 10000)
    thresh_down = entry * (1 - THRESHOLD_BPS / 10000)

    # --- mfe_up and mfe_down at each horizon ---
    for H in HORIZONS:
        window_high = high_arr[i+1 : i+H+1]
        window_low  = low_arr[i+1  : i+H+1]
        up   = (np.max(window_high) - entry) / entry * 10000
        down = (entry - np.min(window_low))  / entry * 10000
        mfe_up[H][idx]   = max(up,   0.0)
        mfe_down[H][idx] = max(down, 0.0)

    # --- time_to_first_up_hit (scan full 96 bars) ---
    first_up = 0
    for j in range(1, LOOKAHEAD_MAX + 1):
        if high_arr[i + j] >= thresh_up:
            first_up = j
            break
    time_up[idx] = first_up

    # --- time_to_first_down_hit (scan full 96 bars) ---
    first_down = 0
    for j in range(1, LOOKAHEAD_MAX + 1):
        if low_arr[i + j] <= thresh_down:
            first_down = j
            break
    time_down[idx] = first_down

    # --- direction (Rule #2) ---
    if first_up > 0 and first_down > 0:
        if first_up < first_down:
            direction[idx] = LONG
        elif first_down < first_up:
            direction[idx] = SHORT
        else:
            direction[idx] = BOTH
    elif first_up > 0:
        direction[idx] = LONG      # only up hit = LONG
    elif first_down > 0:
        direction[idx] = SHORT     # only down hit = SHORT
    else:
        direction[idx] = SKIP      # neither hit (remove later)

    # --- direction_h8 (which ±15bps hits first within 8 bars) ---
    first_up_h8 = 0
    first_down_h8 = 0
    for j in range(1, min(9, LOOKAHEAD_MAX + 1)):
        if i + j >= len(high_arr):
            break
        if first_up_h8 == 0 and high_arr[i + j] >= thresh_up:
            first_up_h8 = j
        if first_down_h8 == 0 and low_arr[i + j] <= thresh_down:
            first_down_h8 = j

    if first_up_h8 > 0 and first_down_h8 > 0:
        if first_up_h8 < first_down_h8:
            dir_h8[idx] = LONG
        elif first_down_h8 < first_up_h8:
            dir_h8[idx] = SHORT
        else:
            dir_h8[idx] = BOTH
    elif first_up_h8 > 0:
        dir_h8[idx] = LONG
    elif first_down_h8 > 0:
        dir_h8[idx] = SHORT
    else:
        dir_h8[idx] = SKIP

    # --- peak bar for vol/volume expansion ---
    if first_up > 0 and first_down > 0:
        peak_bar = i + first_up if first_up <= first_down else i + first_down
    elif first_up > 0:
        peak_bar = i + first_up
    elif first_down > 0:
        peak_bar = i + first_down
    else:
        peak_bar = i + LOOKAHEAD_MAX   # SKIP — use last bar as fallback

    vol_exp[idx]  = 1 if atr_arr[peak_bar]   > atr_arr[i]     else 0
    volm_exp[idx] = 1 if volume_arr[peak_bar] > volume_ma20[i] else 0

    if idx % 10000 == 0:
        print(f"  {idx}/{n_valid} bars done...")

# =============================================================================
print("Building output dataframe...")

out = pd.DataFrame(index=dates[valid_start:valid_end])

for H in HORIZONS:
    out[f"mfe_up_{H}"]   = mfe_up[H]
    out[f"mfe_down_{H}"] = mfe_down[H]

out["time_to_first_up_hit"]   = time_up
out["time_to_first_down_hit"] = time_down
out["direction"]              = direction
out["direction_h8"]           = dir_h8
out["vol_expansion"]          = vol_exp
out["volume_expansion"]       = volm_exp

print(f"\nLabel distribution (direction):")
counts = out["direction"].value_counts().sort_index()
labels_map = {0: "LONG", 1: "SHORT", 2: "BOTH", 3: "SKIP"}
for k, v in counts.items():
    print(f"  {labels_map[k]}: {v} ({v/len(out)*100:.1f}%)")

print(f"\nmfe_up_96  mean: {out['mfe_up_96'].mean():.1f} bps")
print(f"mfe_down_96 mean: {out['mfe_down_96'].mean():.1f} bps")
print(f"time_to_first_up_hit   hits: {(out['time_to_first_up_hit']>0).mean()*100:.1f}% of bars hit +15bps within 96")
print(f"time_to_first_down_hit hits: {(out['time_to_first_down_hit']>0).mean()*100:.1f}% of bars hit -15bps within 96")
print(f"vol_expansion=1:    {out['vol_expansion'].mean()*100:.1f}%")
print(f"volume_expansion=1: {out['volume_expansion'].mean()*100:.1f}%")

out.to_parquet(OUT_PATH)
print(f"\nSaved: {OUT_PATH}")
print(f"Shape: {out.shape}")
