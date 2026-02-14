"""
Test: Can similarity search identify noise vs signal states?

Run: .venv/Scripts/python.exe debug_similarity_noise.py

HYPOTHESIS:
- If similar historical states have CONSISTENT outcomes → signal (trade it)
- If similar historical states have RANDOM outcomes → noise (skip it)

TEST:
1. Build state vectors
2. For each bar, find K similar historical states
3. Compute "consistency" = how skewed outcomes are among neighbors
4. Check: Do high-consistency states actually perform better?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import faiss

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZON = 60
INVALIDATION_RATIO = 0.5
K_NEIGHBORS = 50  # Number of similar states to find

TRAIN_END = "2023-12-31"

# Use subset for speed
SAMPLE_SIZE = 200000  # Sample this many bars for analysis

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
print("=" * 70)
print("SIMILARITY-BASED NOISE DETECTION TEST")
print("=" * 70)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# =============================================================================
# COMPUTE FEATURES (State Vector)
# =============================================================================
print("\nComputing state vector features...")

df = ohlcv.copy()

# EMAs
df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

# EMA slopes (z-scored)
df['ema50_slope'] = df['ema50'].pct_change(5)
df['ema200_slope'] = df['ema200'].pct_change(5)
df['ema50_slope_z'] = (df['ema50_slope'] - df['ema50_slope'].rolling(100).mean()) / df['ema50_slope'].rolling(100).std()
df['ema200_slope_z'] = (df['ema200_slope'] - df['ema200_slope'].rolling(100).mean()) / df['ema200_slope'].rolling(100).std()

# Trend alignment
df['trend_alignment'] = 0
df.loc[(df['close'] > df['ema50']) & (df['ema50'] > df['ema200']), 'trend_alignment'] = 1
df.loc[(df['close'] < df['ema50']) & (df['ema50'] < df['ema200']), 'trend_alignment'] = -1

# Returns
df['return_5m'] = df['close'].pct_change(5)
df['return_15m'] = df['close'].pct_change(15)
df['return_5m_z'] = (df['return_5m'] - df['return_5m'].rolling(100).mean()) / df['return_5m'].rolling(100).std()
df['return_15m_z'] = (df['return_15m'] - df['return_15m'].rolling(100).mean()) / df['return_15m'].rolling(100).std()

# RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df['rsi'] = 100 - (100 / (1 + rs))
df['rsi_z'] = (df['rsi'] - 50) / df['rsi'].rolling(100).std()

# ATR percentile
df['tr'] = np.maximum(df['high'] - df['low'],
                      np.maximum(abs(df['high'] - df['close'].shift(1)),
                                abs(df['low'] - df['close'].shift(1))))
df['atr'] = df['tr'].rolling(14).mean()
df['atr_pct'] = df['atr'].rolling(100).rank(pct=True) * 100

# Volume
df['volume_z'] = (df['volume'] - df['volume'].rolling(100).mean()) / df['volume'].rolling(100).std()

# VWAP distance
df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
df['vwap_distance_z'] = (df['close'] - df['vwap']) / df['atr']

# Range position
df['range_high'] = df['high'].rolling(20).max()
df['range_low'] = df['low'].rolling(20).min()
df['range_position'] = (df['close'] - df['range_low']) / (df['range_high'] - df['range_low']).replace(0, np.nan)

# Drop NaN
df = df.dropna()

# State vector columns
state_cols = [
    'ema50_slope_z', 'ema200_slope_z', 'trend_alignment',
    'return_5m_z', 'return_15m_z', 'rsi_z',
    'atr_pct', 'volume_z', 'vwap_distance_z', 'range_position'
]

print(f"After feature computation: {len(df):,} candles")
print(f"State vector: {len(state_cols)} dimensions")

# Filter to TRAIN only
train_df = df[df.index <= TRAIN_END].copy()
print(f"TRAIN data: {len(train_df):,} candles")

# =============================================================================
# COMPUTE OUTCOMES
# =============================================================================
print("\nComputing outcomes...")

# Compute thresholds
close = train_df['close'].values
high = train_df['high'].values
low = train_df['low'].values

# Sample for threshold computation
sample_idx = np.random.choice(len(train_df) - HORIZON, size=min(50000, len(train_df) - HORIZON), replace=False)
max_moves = []
for i in sample_idx:
    entry = close[i]
    future_high = np.max(high[i+1:i+1+HORIZON])
    future_low = np.min(low[i+1:i+1+HORIZON])
    max_up = (future_high - entry) / entry
    max_down = (entry - future_low) / entry
    max_moves.append(max(max_up, max_down))

median_move = np.percentile(max_moves, 50)
target_pct = median_move
stop_pct = target_pct * INVALIDATION_RATIO

print(f"Target: {target_pct * 10000:.1f} bps, Stop: {stop_pct * 10000:.1f} bps")

# Compute outcomes for all bars
n = len(train_df)
outcomes = np.full(n, 'TIMEOUT', dtype=object)

print("Computing outcomes for all bars...")
for i in range(n - HORIZON):
    if i % 100000 == 0:
        print(f"  Progress: {i:,}/{n-HORIZON:,}")

    entry = close[i]
    target = entry * (1 + target_pct)
    stop = entry * (1 - stop_pct)

    for j in range(i+1, i+1+HORIZON):
        if low[j] <= stop:
            outcomes[i] = 'LOSS'
            break
        if high[j] >= target:
            outcomes[i] = 'WIN'
            break

train_df = train_df.iloc[:-HORIZON].copy()
train_df['outcome'] = outcomes[:-HORIZON]

# Filter to WIN/LOSS only (exclude TIMEOUT for cleaner analysis)
analysis_df = train_df[train_df['outcome'].isin(['WIN', 'LOSS'])].copy()
print(f"\nWIN/LOSS bars: {len(analysis_df):,}")
print(f"  WIN: {(analysis_df['outcome'] == 'WIN').sum():,} ({(analysis_df['outcome'] == 'WIN').mean()*100:.1f}%)")
print(f"  LOSS: {(analysis_df['outcome'] == 'LOSS').sum():,} ({(analysis_df['outcome'] == 'LOSS').mean()*100:.1f}%)")

# =============================================================================
# SAMPLE FOR SPEED
# =============================================================================
if len(analysis_df) > SAMPLE_SIZE:
    print(f"\nSampling {SAMPLE_SIZE:,} bars for analysis...")
    analysis_df = analysis_df.sample(n=SAMPLE_SIZE, random_state=42)

# =============================================================================
# BUILD KNN INDEX (using FAISS)
# =============================================================================
print("\nBuilding KNN index with FAISS...")

X = analysis_df[state_cols].values.astype(np.float32)
y = (analysis_df['outcome'] == 'WIN').astype(int).values

# Standardize manually (no sklearn)
X_mean = np.nanmean(X, axis=0)
X_std = np.nanstd(X, axis=0)
X_std[X_std == 0] = 1  # Avoid division by zero
X_scaled = (X - X_mean) / X_std

# Handle NaN/Inf
X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0).astype(np.float32)

# Build FAISS index
d = X_scaled.shape[1]  # Dimension
index = faiss.IndexFlatL2(d)
index.add(X_scaled)

print(f"Index built with {index.ntotal:,} vectors")

# =============================================================================
# COMPUTE CONSISTENCY FOR EACH BAR
# =============================================================================
print(f"\nComputing neighbor consistency (K={K_NEIGHBORS})...")

# Search for K+1 neighbors (includes self)
distances, indices = index.search(X_scaled, K_NEIGHBORS + 1)

# For each bar, compute the WIN rate among its neighbors (excluding self)
neighbor_win_rates = []
for i in range(len(X_scaled)):
    neighbor_idx = indices[i, 1:]  # Exclude self (index 0)
    neighbor_outcomes = y[neighbor_idx]
    win_rate = np.mean(neighbor_outcomes)
    neighbor_win_rates.append(win_rate)

analysis_df = analysis_df.copy()
analysis_df['neighbor_win_rate'] = neighbor_win_rates

# Consistency = how far from 50% (random)
# 0% or 100% = highly consistent
# 50% = random (noise)
analysis_df['consistency'] = np.abs(analysis_df['neighbor_win_rate'] - 0.5) * 2  # Scale to 0-1

# =============================================================================
# ANALYZE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS: Does neighbor consistency predict actual outcome?")
print("=" * 70)

# Bin by consistency
consistency_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
analysis_df['consistency_bin'] = pd.cut(analysis_df['consistency'], bins=consistency_bins)

print(f"\n{'Consistency Range':<20} {'Actual WR':>12} {'Samples':>12} {'Interpretation'}")
print("-" * 60)

for bin_range in analysis_df['consistency_bin'].cat.categories:
    bin_df = analysis_df[analysis_df['consistency_bin'] == bin_range]
    if len(bin_df) > 0:
        actual_wr = bin_df['outcome'].apply(lambda x: 1 if x == 'WIN' else 0).mean() * 100
        n_samples = len(bin_df)

        if bin_range.right <= 0.2:
            interp = "LOW consistency (noisy)"
        elif bin_range.right <= 0.4:
            interp = "MEDIUM consistency"
        else:
            interp = "HIGH consistency (signal)"

        print(f"{str(bin_range):<20} {actual_wr:>11.1f}% {n_samples:>12,} {interp}")

# =============================================================================
# KEY TEST: Filter by consistency
# =============================================================================
print("\n" + "-" * 70)
print("KEY TEST: Does filtering by consistency improve win rate?")
print("-" * 70)

base_wr = (analysis_df['outcome'] == 'WIN').mean() * 100
breakeven = stop_pct / (target_pct + stop_pct) * 100

print(f"\nBreak-even WR: {breakeven:.1f}%")
print(f"Base WR (all bars): {base_wr:.1f}%")

# Filter by high consistency
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
print(f"\n{'Min Consistency':>15} {'Filtered WR':>12} {'Samples':>12} {'% Remaining':>12} {'vs Base':>10}")
print("-" * 65)

for thresh in thresholds:
    filtered = analysis_df[analysis_df['consistency'] >= thresh]
    if len(filtered) > 100:  # Need enough samples
        filtered_wr = (filtered['outcome'] == 'WIN').mean() * 100
        n_samples = len(filtered)
        pct_remaining = len(filtered) / len(analysis_df) * 100
        diff = filtered_wr - base_wr
        print(f"{thresh:>15.1f} {filtered_wr:>11.1f}% {n_samples:>12,} {pct_remaining:>11.1f}% {diff:>+9.1f}pp")

# =============================================================================
# ADDITIONAL TEST: Filter by neighbor WIN rate direction
# =============================================================================
print("\n" + "-" * 70)
print("ADDITIONAL TEST: Follow neighbor signal direction")
print("-" * 70)

# If neighbors say WIN (>50% win rate), does actual outcome tend to be WIN?
# If neighbors say LOSS (<50% win rate), does actual outcome tend to be LOSS?

high_signal = analysis_df[analysis_df['neighbor_win_rate'] >= 0.5]
low_signal = analysis_df[analysis_df['neighbor_win_rate'] < 0.5]

high_actual_wr = (high_signal['outcome'] == 'WIN').mean() * 100
low_actual_wr = (low_signal['outcome'] == 'WIN').mean() * 100

print(f"\nNeighbors predict WIN (win_rate >= 50%):")
print(f"  Samples: {len(high_signal):,}")
print(f"  Actual WIN rate: {high_actual_wr:.1f}%")

print(f"\nNeighbors predict LOSS (win_rate < 50%):")
print(f"  Samples: {len(low_signal):,}")
print(f"  Actual WIN rate: {low_actual_wr:.1f}%")

print(f"\nDifference: {high_actual_wr - low_actual_wr:.1f}pp")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Check if consistency helps
best_filtered_wr = max(
    (analysis_df[analysis_df['consistency'] >= t]['outcome'] == 'WIN').mean() * 100
    for t in thresholds if len(analysis_df[analysis_df['consistency'] >= t]) > 100
)

print(f"""
BASE METRICS:
- Base win rate: {base_wr:.1f}%
- Break-even needed: {breakeven:.1f}%
- Gap: {base_wr - breakeven:.1f}pp

CONSISTENCY FILTERING:
- Best filtered WR: {best_filtered_wr:.1f}%
- Improvement: {best_filtered_wr - base_wr:+.1f}pp

NEIGHBOR DIRECTION:
- Neighbors predict WIN: Actual WR = {high_actual_wr:.1f}%
- Neighbors predict LOSS: Actual WR = {low_actual_wr:.1f}%
- Predictive power: {high_actual_wr - low_actual_wr:.1f}pp difference

CONCLUSION:
""")

if best_filtered_wr >= breakeven:
    print("  CONSISTENCY FILTERING WORKS! High consistency states beat break-even.")
elif best_filtered_wr - base_wr > 3:
    print("  Consistency filtering helps moderately, but not enough for profitability.")
else:
    print("  Consistency filtering does NOT significantly improve outcomes.")
    print("  The state vector may not capture meaningful patterns.")
