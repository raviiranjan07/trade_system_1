"""Stage 6: End-to-end S/R + Base Features

Step 1-3: Load data, combine features, create H25 direction label.

Run: PYTHONPATH=src python experiments/brain/SR/stage6_endtoend.py
"""

import numpy as np
import pandas as pd

# === Step 1: Load 67K events ===
print("=" * 60)
print("STAGE 6: End-to-End S/R + Base Features")
print("=" * 60)

print("\nStep 1: Loading S/R datasets...")
train = np.load("experiments/brain/SR/datasets_every_bar/train.npz")
val = np.load("experiments/brain/SR/datasets_every_bar/val.npz")
test = np.load("experiments/brain/SR/datasets_every_bar/test.npz")

for name, d in [("Train", train), ("Val", val), ("Test", test)]:
    print(f"  {name}: {len(d['Y'])} events, bars range [{d['bars'].min()}-{d['bars'].max()}]")

# === Step 2: Load base features ===
print("\nStep 2: Loading base features from feature_cache...")
fc = pd.read_parquet("experiments/layer2/L2-003/feature_cache.parquet")
fc.index = fc.index.tz_localize(None) if fc.index.tz is not None else fc.index

# roc5 = single-bar rate of change (closest to roc in snapshot setup)
base_cols = ["roc5", "rsi7", "range_position"]
base_all = fc[base_cols].values.astype(np.float32)
print(f"  Base features shape: {base_all.shape}")
print(f"  Columns: {base_cols}")

# === Step 3: Create H25 direction label ===
print("\nStep 3: Creating H25 direction label...")
from brain.config import load_config
from brain.ingestion import load_ohlcv

base_cfg = load_config("configs/base.yaml")
df, _ = load_ohlcv(base_cfg)
high = df["high"].values
low = df["low"].values
open_arr = df["open"].values
N = len(df)

horizon = 25
threshold_bps = 15

LONG = 0
SHORT = 1
BOTH = 2
SKIP = 3

def create_h25_labels(bars):
    """Create H25 direction labels for given bar indices."""
    labels = np.full(len(bars), SKIP, dtype=np.int64)

    for idx in range(len(bars)):
        t = bars[idx]
        if t + 1 >= N or t + horizon + 1 > N:
            continue

        entry = open_arr[t + 1]
        if entry <= 0:
            continue

        thresh_up = entry * (1 + threshold_bps / 10000)
        thresh_down = entry * (1 - threshold_bps / 10000)

        first_up = 0
        first_down = 0

        for j in range(1, horizon + 1):
            if t + j >= N:
                break
            if first_up == 0 and high[t + j] >= thresh_up:
                first_up = j
            if first_down == 0 and low[t + j] <= thresh_down:
                first_down = j

        if first_up > 0 and first_down > 0:
            if first_up < first_down:
                labels[idx] = LONG
            elif first_down < first_up:
                labels[idx] = SHORT
            else:
                labels[idx] = BOTH
        elif first_up > 0:
            labels[idx] = LONG
        elif first_down > 0:
            labels[idx] = SHORT
        else:
            labels[idx] = SKIP

    return labels

# Create labels for each split
for name, d in [("Train", train), ("Val", val), ("Test", test)]:
    bars = d["bars"]
    labels = create_h25_labels(bars)

    # Get base features for these bars
    base = base_all[bars]
    base = np.nan_to_num(base, nan=0.0)

    # Get S/R features
    Xd = d["X_dynamic"][:, -1, :]  # last snapshot
    Xs = d["X_static"]

    # Save combined
    np.savez_compressed(
        f"experiments/brain/SR/datasets_every_bar/{name}_stage6.npz",
        X_sr_dynamic=Xd,
        X_sr_static=Xs,
        X_base=base,
        Y_h25=labels,
        Y_mfe=d["Y"],
        bars=bars,
    )

    # Print distribution
    print(f"\n  {name} H25 label distribution ({len(labels)} events):")
    for label, label_name in [(0, "LONG"), (1, "SHORT"), (2, "BOTH"), (3, "SKIP")]:
        count = (labels == label).sum()
        print(f"    {label_name}: {count} ({count/len(labels)*100:.1f}%)")

# === Summary ===
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Load back train to show full picture
train_s6 = np.load("experiments/brain/SR/datasets_every_bar/train_stage6.npz")
print(f"\nFeatures per event:")
print(f"  S/R dynamic: {train_s6['X_sr_dynamic'].shape[1]} features")
print(f"  S/R static:  {train_s6['X_sr_static'].shape[1]} features")
print(f"  Base:         {train_s6['X_base'].shape[1]} features")
print(f"  Total:        {train_s6['X_sr_dynamic'].shape[1] + train_s6['X_sr_static'].shape[1] + train_s6['X_base'].shape[1]} features")
print(f"\nLabels:")
print(f"  H25 direction: LONG/SHORT/BOTH/SKIP")
print(f"  MFE 3-class:   BOUNCE/BREAK/CHOP (kept for reference)")
print(f"\nSaved to: experiments/brain/SR/datasets_every_bar/*_stage6.npz")
