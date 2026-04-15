"""L2-003: Twin Bar Analysis — why do same features give opposite direction?

Finds bars with nearly identical features but opposite LONG/SHORT outcomes.
Studies what was different between them.

Run locally: PYTHONPATH=src python experiments/layer2/L2-003/L2_003_twin_analysis.py
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
BASE = Path("experiments/layer2/L2-003")
CACHE_PATH = BASE / "feature_cache.parquet"
LABELS_PATH = BASE / "labels.parquet"

HORIZONS = [1, 2, 3, 5, 10, 20, 32, 96]

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data...")
fc = pd.read_parquet(CACHE_PATH)
lb = pd.read_parquet(LABELS_PATH)

# Base features
MAG_CONTINUOUS = [
    "keltner_width", "std20", "donchian_width", "atr_percentile",
    "range_bps", "body_bps", "dist_from_high20_pct", "dist_from_low20_pct",
    "ema_separation", "volume_trend", "volume_ratio",
    "ll_count5", "down_bars5", "hh_count5",
]
MAG_BINARY = [
    "price_above_sma200", "rsi_extreme_oversold", "rsi_oversold_zone",
    "is_weekend", "session_asia_night", "session_europe", "session_us",
    "day_of_week", "session",
]
DIR_CONTINUOUS = [
    "rsi7", "roc5", "roc20", "momentum10",
    "bb_position", "range_position_50", "range_position",
    "ema20_slope", "ema50_dist_pct", "sma200_dist_pct", "sma200_slope",
]

fc["hour_ist"] = (fc.index.hour + 5) % 24
all_feat_cols = MAG_CONTINUOUS + MAG_BINARY + DIR_CONTINUOUS + ["hour_ist"]
all_feat_cols = [c for c in all_feat_cols if c in fc.columns]
F = len(all_feat_cols)

# Align
common_idx = lb.index.intersection(fc.index)
lb = lb.loc[common_idx]
fc_aligned = fc.loc[common_idx]

# Features
feat_arr = fc_aligned[all_feat_cols].values.astype(np.float32)

# Normalize
scaler_mean = feat_arr.mean(axis=0)
scaler_std = feat_arr.std(axis=0)
scaler_std[scaler_std < 1e-8] = 1.0
feat_norm = (feat_arr - scaler_mean) / scaler_std
feat_norm = np.nan_to_num(feat_norm, nan=0.0, posinf=0.0, neginf=0.0)

# Direction labels
direction = lb["direction"].values  # 0=LONG, 1=SHORT, 2=BOTH
valid_mask = (direction == 0) | (direction == 1)

# MFE and time data for studying price paths
mfe_up = lb[[f"mfe_up_{H}" for H in HORIZONS]].values.astype(np.float32)
mfe_down = lb[[f"mfe_down_{H}" for H in HORIZONS]].values.astype(np.float32)
time_up = lb["time_to_first_up_hit"].values
time_down = lb["time_to_first_down_hit"].values

# Raw OHLCV for price path
raw_close = fc["close"].values
raw_high = fc["high"].values
raw_low = fc["low"].values
raw_volume = fc["volume"].values
close_positions = np.array([fc.index.get_loc(dt) for dt in common_idx], dtype=np.int32)

print(f"  Features: {F}")
print(f"  Total bars: {len(common_idx)}")
print(f"  Valid (LONG/SHORT): {valid_mask.sum()}")

# =============================================================================
# STEP 1: How much of the data is "conflicting"?
# =============================================================================
print(f"\n{'='*70}")
print("STEP 1: How much data has conflicting feature-direction mappings?")
print(f"{'='*70}")

# Use nearest neighbors to find similar bars
valid_idx = np.where(valid_mask)[0]
feat_valid = feat_norm[valid_idx]
dir_valid = direction[valid_idx]

print(f"\n  Finding nearest neighbors for {len(valid_idx)} bars...")
nn = NearestNeighbors(n_neighbors=6, metric='euclidean', n_jobs=-1)
nn.fit(feat_valid)
distances, neighbors = nn.kneighbors(feat_valid)

# For each bar, check its 5 nearest neighbors — do they have the same direction?
same_dir_counts = []
for i in range(len(valid_idx)):
    my_dir = dir_valid[i]
    neighbor_dirs = dir_valid[neighbors[i, 1:]]  # exclude self (index 0)
    same_count = (neighbor_dirs == my_dir).sum()
    same_dir_counts.append(same_count)

same_dir_counts = np.array(same_dir_counts)

print(f"\n  For each bar, checking 5 nearest neighbors:")
print(f"    All 5 same direction:  {(same_dir_counts == 5).sum():6d} ({(same_dir_counts == 5).mean()*100:.1f}%)")
print(f"    4 of 5 same:           {(same_dir_counts == 4).sum():6d} ({(same_dir_counts == 4).mean()*100:.1f}%)")
print(f"    3 of 5 same:           {(same_dir_counts == 3).sum():6d} ({(same_dir_counts == 3).mean()*100:.1f}%)")
print(f"    2 of 5 same:           {(same_dir_counts == 2).sum():6d} ({(same_dir_counts == 2).mean()*100:.1f}%)")
print(f"    1 of 5 same:           {(same_dir_counts == 1).sum():6d} ({(same_dir_counts == 1).mean()*100:.1f}%)")
print(f"    0 of 5 same:           {(same_dir_counts == 0).sum():6d} ({(same_dir_counts == 0).mean()*100:.1f}%)")
print(f"\n    Mean same-direction neighbors: {same_dir_counts.mean():.2f} out of 5")
print(f"    If random: expect 2.5 out of 5")

if same_dir_counts.mean() > 3.0:
    print(f"    → Features DO cluster by direction — there IS some signal")
elif same_dir_counts.mean() > 2.7:
    print(f"    → Weak clustering — small signal exists")
else:
    print(f"    → No clustering — direction appears random given features")

# =============================================================================
# STEP 2: Find twin pairs (close features, opposite direction)
# =============================================================================
print(f"\n{'='*70}")
print("STEP 2: Finding twin pairs (similar features, opposite direction)")
print(f"{'='*70}")

twins = []
for i in range(len(valid_idx)):
    my_dir = dir_valid[i]
    for j_pos in range(1, 6):  # check 5 nearest neighbors
        j = neighbors[i, j_pos]
        if dir_valid[j] != my_dir:  # opposite direction
            dist = distances[i, j_pos]
            twins.append({
                'idx_a': valid_idx[i],
                'idx_b': valid_idx[j],
                'dir_a': my_dir,
                'dir_b': dir_valid[j],
                'distance': dist,
            })
            break  # take only closest opposite neighbor

print(f"  Found {len(twins)} twin pairs")

# Convert to dataframe for analysis
twins_df = pd.DataFrame(twins)
print(f"  Mean feature distance: {twins_df['distance'].mean():.4f}")
print(f"  Median feature distance: {twins_df['distance'].median():.4f}")

# =============================================================================
# STEP 3: For twin pairs, study what was DIFFERENT
# =============================================================================
print(f"\n{'='*70}")
print("STEP 3: What differs between twins? (same features, opposite direction)")
print(f"{'='*70}")

# Take closest 5000 twins for detailed analysis
closest_twins = twins_df.nsmallest(5000, 'distance')

# For each twin pair, compare price paths
twin_analysis = []
for _, twin in closest_twins.iterrows():
    a, b = int(twin['idx_a']), int(twin['idx_b'])
    pos_a, pos_b = close_positions[a], close_positions[b]

    entry_a = raw_close[pos_a]
    entry_b = raw_close[pos_b]

    # Time to hit for each twin
    t_up_a = time_up[a]
    t_down_a = time_down[a]
    t_up_b = time_up[b]
    t_down_b = time_down[b]

    # MFE at H10
    mfe_up_a = mfe_up[a, 4]   # H10 index
    mfe_down_a = mfe_down[a, 4]
    mfe_up_b = mfe_up[b, 4]
    mfe_down_b = mfe_down[b, 4]

    # Race gap: how many bars between first and second hit?
    if direction[a] == 0:  # LONG: up hit first
        race_gap_a = t_down_a - t_up_a if t_down_a > 0 else 96
    else:  # SHORT: down hit first
        race_gap_a = t_up_a - t_down_a if t_up_a > 0 else 96

    if direction[b] == 0:
        race_gap_b = t_down_b - t_up_b if t_down_b > 0 else 96
    else:
        race_gap_b = t_up_b - t_down_b if t_up_b > 0 else 96

    # Volume at entry and next few bars
    vol_a = raw_volume[pos_a:min(pos_a+5, len(raw_volume))].mean() if pos_a+5 < len(raw_volume) else 0
    vol_b = raw_volume[pos_b:min(pos_b+5, len(raw_volume))].mean() if pos_b+5 < len(raw_volume) else 0

    # Price movement in first 3 bars
    if pos_a + 3 < len(raw_close):
        move_3bar_a = (raw_close[pos_a+3] - raw_close[pos_a]) / raw_close[pos_a] * 10000
    else:
        move_3bar_a = 0
    if pos_b + 3 < len(raw_close):
        move_3bar_b = (raw_close[pos_b+3] - raw_close[pos_b]) / raw_close[pos_b] * 10000
    else:
        move_3bar_b = 0

    # Hour of day
    hour_a = common_idx[a].hour if hasattr(common_idx[a], 'hour') else 0
    hour_b = common_idx[b].hour if hasattr(common_idx[b], 'hour') else 0

    twin_analysis.append({
        'dist': twin['distance'],
        'dir_a': int(twin['dir_a']),  # 0=LONG, 1=SHORT
        'dir_b': int(twin['dir_b']),
        't_up_a': t_up_a, 't_down_a': t_down_a,
        't_up_b': t_up_b, 't_down_b': t_down_b,
        'race_gap_a': race_gap_a, 'race_gap_b': race_gap_b,
        'mfe_up_a': mfe_up_a, 'mfe_down_a': mfe_down_a,
        'mfe_up_b': mfe_up_b, 'mfe_down_b': mfe_down_b,
        'vol_a': vol_a, 'vol_b': vol_b,
        'move_3bar_a': move_3bar_a, 'move_3bar_b': move_3bar_b,
        'hour_a': hour_a, 'hour_b': hour_b,
    })

ta = pd.DataFrame(twin_analysis)

# --- Race gap analysis ---
print(f"\n  --- How close was the race? (bars between 1st and 2nd threshold hit) ---")
print(f"    Twin A (winner) race gap:")
print(f"      gap = 1 bar:   {(ta['race_gap_a'] == 1).sum():5d} ({(ta['race_gap_a'] == 1).mean()*100:.1f}%)")
print(f"      gap <= 3 bars: {(ta['race_gap_a'] <= 3).sum():5d} ({(ta['race_gap_a'] <= 3).mean()*100:.1f}%)")
print(f"      gap <= 5 bars: {(ta['race_gap_a'] <= 5).sum():5d} ({(ta['race_gap_a'] <= 5).mean()*100:.1f}%)")
print(f"      gap > 10 bars: {(ta['race_gap_a'] > 10).sum():5d} ({(ta['race_gap_a'] > 10).mean()*100:.1f}%)")
print(f"      median gap:    {ta['race_gap_a'].median():.0f} bars")
print(f"    Twin B (opposite) race gap:")
print(f"      median gap:    {ta['race_gap_b'].median():.0f} bars")

# --- MFE comparison ---
print(f"\n  --- MFE at H10 (bps) ---")
print(f"    LONG twins:")
long_a = ta[ta['dir_a'] == 0]
short_a = ta[ta['dir_a'] == 1]
if len(long_a) > 0:
    print(f"      mean mfe_up:   {long_a['mfe_up_a'].mean():.1f} bps")
    print(f"      mean mfe_down: {long_a['mfe_down_a'].mean():.1f} bps")
    print(f"      Their SHORT twin:")
    print(f"      mean mfe_up:   {long_a['mfe_up_b'].mean():.1f} bps")
    print(f"      mean mfe_down: {long_a['mfe_down_b'].mean():.1f} bps")

if len(short_a) > 0:
    print(f"    SHORT twins:")
    print(f"      mean mfe_up:   {short_a['mfe_up_a'].mean():.1f} bps")
    print(f"      mean mfe_down: {short_a['mfe_down_a'].mean():.1f} bps")
    print(f"      Their LONG twin:")
    print(f"      mean mfe_up:   {short_a['mfe_up_b'].mean():.1f} bps")
    print(f"      mean mfe_down: {short_a['mfe_down_b'].mean():.1f} bps")

# --- Volume comparison ---
print(f"\n  --- Volume at entry (next 5 bars avg) ---")
vol_ratio = ta['vol_a'] / ta['vol_b'].clip(1)
print(f"    Vol ratio (A/B): mean={vol_ratio.mean():.3f}  std={vol_ratio.std():.3f}")
print(f"    → {'Volume differs significantly' if vol_ratio.std() > 0.5 else 'Volume is similar between twins'}")

# --- First 3 bar movement ---
print(f"\n  --- Price move in first 3 bars (bps) ---")
print(f"    LONG twins:  mean 3-bar move = {long_a['move_3bar_a'].mean():.1f} bps")
print(f"    SHORT twins: mean 3-bar move = {short_a['move_3bar_a'].mean():.1f} bps")
print(f"    Their opposite twin 3-bar move:")
print(f"    LONG's SHORT twin: {long_a['move_3bar_b'].mean():.1f} bps")
print(f"    SHORT's LONG twin: {short_a['move_3bar_b'].mean():.1f} bps")

# --- Time of day ---
print(f"\n  --- Hour distribution of twins ---")
print(f"    Mean hour (A): {ta['hour_a'].mean():.1f}")
print(f"    Mean hour (B): {ta['hour_b'].mean():.1f}")
print(f"    Same hour: {(ta['hour_a'] == ta['hour_b']).sum()} ({(ta['hour_a'] == ta['hour_b']).mean()*100:.1f}%)")

# =============================================================================
# STEP 4: Feature-by-feature difference between twins
# =============================================================================
print(f"\n{'='*70}")
print("STEP 4: Which features differ MOST between twins?")
print(f"{'='*70}")

# For closest 5000 twins, compute feature difference
feat_diffs = []
for _, twin in closest_twins.iterrows():
    a, b = int(twin['idx_a']), int(twin['idx_b'])
    diff = np.abs(feat_norm[a] - feat_norm[b])
    feat_diffs.append(diff)

feat_diffs = np.array(feat_diffs)  # (5000, 35)
mean_diffs = feat_diffs.mean(axis=0)

# Sort by difference
sorted_idx = np.argsort(-mean_diffs)
print(f"\n  Features ranked by difference between twins:")
print(f"  {'Rank':<5s} {'Feature':<25s} | {'mean diff':>10s} | {'interpretation'}")
print(f"  {'-'*5}-{'-'*25}-+-{'-'*10}-+-{'-'*30}")

for rank, idx in enumerate(sorted_idx):
    diff = mean_diffs[idx]
    if diff > 0.3:
        interp = "BIG difference — potential hidden signal"
    elif diff > 0.1:
        interp = "Moderate difference"
    else:
        interp = "Small — twins are truly similar here"
    print(f"  {rank+1:<5d} {all_feat_cols[idx]:<25s} | {diff:10.4f} | {interp}")

# =============================================================================
# STEP 5: Consistency check — do features predict direction at all?
# =============================================================================
print(f"\n{'='*70}")
print("STEP 5: For each feature, does direction correlate?")
print(f"{'='*70}")

print(f"\n  {'Feature':<25s} | {'LONG mean':>10s} {'SHORT mean':>10s} {'diff':>8s} | {'signal?'}")
print(f"  {'-'*25}-+-{'-'*10}-{'-'*10}-{'-'*8}-+-{'-'*10}")

for f_idx, fname in enumerate(all_feat_cols):
    long_vals = feat_norm[valid_idx[dir_valid == 0], f_idx]
    short_vals = feat_norm[valid_idx[dir_valid == 1], f_idx]
    long_mean = long_vals.mean()
    short_mean = short_vals.mean()
    diff = abs(long_mean - short_mean)
    sig = "YES" if diff > 0.02 else "weak" if diff > 0.01 else "no"
    if diff > 0.01:
        print(f"  {fname:<25s} | {long_mean:10.4f} {short_mean:10.4f} {diff:8.4f} | {sig}")

print(f"\n  (Only showing features with diff > 0.01)")
