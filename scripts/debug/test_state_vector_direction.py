"""
Test: Can similar state vectors predict direction?

Features used (all OOS validated):
1. atr21_pct - Volatility
2. bb_width - Volatility
3. ema_separation - Trend
4. dist_from_high50_pct - Location
5. dist_from_low50_pct - Location
6. body_bps - Candle size

Method:
1. For each test candle, find K most similar historical states
2. Check what direction those historical states went
3. If majority went UP -> predict UP
4. Measure prediction accuracy vs 50% baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
K_NEIGHBORS = 100  # Number of similar states to find
TARGET_BPS = 12    # Direction threshold (hit +12bp or -12bp first)
HORIZON = 10       # Bars to look ahead for direction

STATE_FEATURES = [
    'atr21_pct',
    'bb_width',
    'ema_separation',
    'dist_from_high50_pct',
    'dist_from_low50_pct',
    'body_bps',
]

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
print("=" * 80)
print("STATE VECTOR DIRECTION PREDICTION TEST")
print("=" * 80)
print(f"\nFeatures: {STATE_FEATURES}")
print(f"K neighbors: {K_NEIGHBORS}")
print(f"Target: {TARGET_BPS}bp")
print(f"Horizon: {HORIZON} bars")

# Load 1-min data
data_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
df_1m = pd.read_parquet(data_path)
df_1m = df_1m.sort_index()
print(f"\nLoaded {len(df_1m):,} 1-minute candles")

# Resample to 15-min
df = df_1m.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()
print(f"Resampled to {len(df):,} 15-minute candles")

# =============================================================================
# CALCULATE FEATURES
# =============================================================================
print("\nCalculating features...")

# ATR21 as percentage of price
high_low = df['high'] - df['low']
high_close = abs(df['high'] - df['close'].shift(1))
low_close = abs(df['low'] - df['close'].shift(1))
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr21_pct'] = (tr.rolling(21).mean() / df['close']) * 10000  # in bps

# Bollinger Width
sma20 = df['close'].rolling(20).mean()
std20 = df['close'].rolling(20).std()
bb_upper = sma20 + 2 * std20
bb_lower = sma20 - 2 * std20
df['bb_width'] = ((bb_upper - bb_lower) / sma20) * 10000  # in bps

# EMA Separation (9 vs 50)
ema9 = df['close'].ewm(span=9).mean()
ema50 = df['close'].ewm(span=50).mean()
df['ema_separation'] = ((ema9 - ema50) / df['close']) * 10000  # in bps

# Distance from 50-bar high/low
high50 = df['high'].rolling(50).max()
low50 = df['low'].rolling(50).min()
range50 = high50 - low50
df['dist_from_high50_pct'] = ((high50 - df['close']) / (range50 + 1e-9)) * 100
df['dist_from_low50_pct'] = ((df['close'] - low50) / (range50 + 1e-9)) * 100

# Body size in bps
df['body_bps'] = abs(df['close'] - df['open']) / df['open'] * 10000

# Drop NaN
df = df.dropna()

# =============================================================================
# CALCULATE DIRECTION (which hits first: +target or -target)
# =============================================================================
print("Calculating direction labels...")

@njit
def calc_direction(opens, highs, lows, target_bps, horizon):
    """
    For each bar, determine if price hits +target or -target first.
    Returns: 1 = UP first, -1 = DOWN first, 0 = neither
    """
    n = len(opens)
    directions = np.zeros(n, dtype=np.int32)

    for i in range(n - horizon):
        entry = opens[i + 1]  # Enter at next bar open
        target_up = entry * (1 + target_bps / 10000)
        target_down = entry * (1 - target_bps / 10000)

        hit_up = -1
        hit_down = -1

        for j in range(1, horizon + 1):
            if i + 1 + j >= n:
                break
            if hit_up < 0 and highs[i + 1 + j] >= target_up:
                hit_up = j
            if hit_down < 0 and lows[i + 1 + j] <= target_down:
                hit_down = j
            if hit_up > 0 and hit_down > 0:
                break

        if hit_up > 0 and hit_down > 0:
            if hit_up < hit_down:
                directions[i] = 1
            elif hit_down < hit_up:
                directions[i] = -1
            else:
                directions[i] = 0  # Same bar - ambiguous
        elif hit_up > 0:
            directions[i] = 1
        elif hit_down > 0:
            directions[i] = -1
        else:
            directions[i] = 0

    return directions

df['direction'] = calc_direction(
    df['open'].values,
    df['high'].values,
    df['low'].values,
    TARGET_BPS,
    HORIZON
)

# =============================================================================
# SPLIT DATA
# =============================================================================
train_end = '2023-12-31'
df_train = df[df.index <= train_end].copy()
df_test = df[df.index > train_end].copy()

print(f"\nTrain data (2020-2023): {len(df_train):,} candles")
print(f"Test data (2024-2025): {len(df_test):,} candles")

# Check direction distribution
train_up = (df_train['direction'] == 1).sum()
train_down = (df_train['direction'] == -1).sum()
train_neither = (df_train['direction'] == 0).sum()
print(f"\nTrain direction: UP={train_up:,} ({train_up/len(df_train)*100:.1f}%), DOWN={train_down:,} ({train_down/len(df_train)*100:.1f}%), Neither={train_neither:,}")

test_up = (df_test['direction'] == 1).sum()
test_down = (df_test['direction'] == -1).sum()
test_neither = (df_test['direction'] == 0).sum()
print(f"Test direction: UP={test_up:,} ({test_up/len(df_test)*100:.1f}%), DOWN={test_down:,} ({test_down/len(df_test)*100:.1f}%), Neither={test_neither:,}")

# =============================================================================
# NORMALIZE FEATURES
# =============================================================================
print("\nNormalizing features...")

# Z-score normalization using train statistics
train_means = df_train[STATE_FEATURES].mean()
train_stds = df_train[STATE_FEATURES].std()

for col in STATE_FEATURES:
    df_train[f'{col}_z'] = (df_train[col] - train_means[col]) / (train_stds[col] + 1e-9)
    df_test[f'{col}_z'] = (df_test[col] - train_means[col]) / (train_stds[col] + 1e-9)

z_features = [f'{col}_z' for col in STATE_FEATURES]

# =============================================================================
# FIND SIMILAR STATES AND PREDICT DIRECTION
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING SIMILARITY-BASED PREDICTION...")
print("=" * 80)

# Prepare train matrix
train_matrix = df_train[z_features].values
train_directions = df_train['direction'].values

# Only use train samples with clear direction (not 0)
valid_train_mask = train_directions != 0
train_matrix_valid = train_matrix[valid_train_mask]
train_directions_valid = train_directions[valid_train_mask]
print(f"\nValid train samples (with direction): {len(train_directions_valid):,}")

# Test matrix
test_matrix = df_test[z_features].values
test_directions = df_test['direction'].values

# Only test on samples with clear direction
valid_test_mask = test_directions != 0
test_indices = np.where(valid_test_mask)[0]
print(f"Valid test samples: {len(test_indices):,}")

# Store results
predictions = []
actuals = []
up_ratios = []

print("\nProgress: ", end="", flush=True)
for idx, i in enumerate(test_indices):
    if idx % 5000 == 0:
        print(f"{idx//1000}k..", end="", flush=True)

    # Current state
    current_state = test_matrix[i]

    # Calculate distances to all train states
    distances = np.sqrt(np.sum((train_matrix_valid - current_state) ** 2, axis=1))

    # Find K nearest neighbors
    nearest_idx = np.argsort(distances)[:K_NEIGHBORS]
    neighbor_directions = train_directions_valid[nearest_idx]

    # Calculate up ratio
    up_count = np.sum(neighbor_directions == 1)
    down_count = np.sum(neighbor_directions == -1)
    up_ratio = up_count / (up_count + down_count)

    up_ratios.append(up_ratio)
    actuals.append(test_directions[i])

    # Make prediction
    if up_ratio > 0.5:
        predictions.append(1)
    else:
        predictions.append(-1)

print("Done!")

predictions = np.array(predictions)
actuals = np.array(actuals)
up_ratios = np.array(up_ratios)

# =============================================================================
# ANALYZE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

# Overall accuracy
correct = (predictions == actuals).sum()
total = len(predictions)
accuracy = correct / total * 100

print(f"\nOverall Results:")
print(f"  Total predictions: {total:,}")
print(f"  Correct: {correct:,}")
print(f"  Accuracy: {accuracy:.2f}%")
print(f"  Edge vs random (50%): {accuracy - 50:.2f}%")

# Breakdown by confidence (up_ratio)
print(f"\nAccuracy by Confidence Level:")
print("-" * 60)

confidence_bins = [
    (0.50, 0.55, "50-55%"),
    (0.55, 0.60, "55-60%"),
    (0.60, 0.65, "60-65%"),
    (0.65, 0.70, "65-70%"),
    (0.70, 1.00, "70%+"),
]

for low, high, label in confidence_bins:
    # UP predictions in this confidence range
    mask_up = (up_ratios >= low) & (up_ratios < high)
    if mask_up.sum() > 0:
        acc_up = (predictions[mask_up] == actuals[mask_up]).mean() * 100
        print(f"  UP confidence {label}: {mask_up.sum():,} samples, accuracy = {acc_up:.1f}%")

    # DOWN predictions (mirror)
    mask_down = (up_ratios <= (1-low)) & (up_ratios > (1-high))
    if mask_down.sum() > 0:
        acc_down = (predictions[mask_down] == actuals[mask_down]).mean() * 100
        print(f"  DOWN confidence {label}: {mask_down.sum():,} samples, accuracy = {acc_down:.1f}%")

# High confidence only (>60%)
high_conf_mask = (up_ratios >= 0.60) | (up_ratios <= 0.40)
if high_conf_mask.sum() > 0:
    high_conf_acc = (predictions[high_conf_mask] == actuals[high_conf_mask]).mean() * 100
    print(f"\nHigh Confidence (>60% bias):")
    print(f"  Samples: {high_conf_mask.sum():,}")
    print(f"  Accuracy: {high_conf_acc:.1f}%")

# Very high confidence (>70%)
very_high_mask = (up_ratios >= 0.70) | (up_ratios <= 0.30)
if very_high_mask.sum() > 0:
    very_high_acc = (predictions[very_high_mask] == actuals[very_high_mask]).mean() * 100
    print(f"\nVery High Confidence (>70% bias):")
    print(f"  Samples: {very_high_mask.sum():,}")
    print(f"  Accuracy: {very_high_acc:.1f}%")

# =============================================================================
# CONCLUSION
# =============================================================================
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if accuracy > 52:
    print(f"\nState vector has SOME predictive power: {accuracy:.1f}% > 50%")
    print("Edge is small but may be exploitable with proper risk management.")
elif accuracy > 50:
    print(f"\nState vector has MARGINAL edge: {accuracy:.1f}%")
    print("Too small to be profitable after fees (need ~55%+).")
else:
    print(f"\nState vector has NO predictive power: {accuracy:.1f}%")
    print("Similar states do NOT predict direction better than random.")

print("\nDONE!")
