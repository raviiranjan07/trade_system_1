"""
Test: EMA9/EMA50 State Vector for Direction Prediction

Features (derived from EMA9 and EMA50 only):
1. ema_separation - (EMA9 - EMA50) / price (trend direction/strength)
2. ema9_slope - EMA9 slope (short-term momentum)
3. ema50_slope - EMA50 slope (longer-term trend)
4. price_to_ema9 - (close - EMA9) / price (price position vs EMA9)
5. price_to_ema50 - (close - EMA50) / price (price position vs EMA50)

Settings:
- Timeframe: 15-min
- Horizon: H=1 (15 minutes ahead)
- Target: 12bp
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
K_NEIGHBORS = 100
TARGET_BPS = 12
HORIZON = 1  # H=1 on 15-min = 15 minutes

STATE_FEATURES = [
    'ema_separation',
    'ema9_slope',
    'ema50_slope',
    'price_to_ema9',
    'price_to_ema50',
]

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
print("=" * 80)
print("EMA9/EMA50 STATE VECTOR DIRECTION TEST")
print("=" * 80)
print(f"\nFeatures: {STATE_FEATURES}")
print(f"K neighbors: {K_NEIGHBORS}")
print(f"Target: {TARGET_BPS}bp")
print(f"Horizon: {HORIZON} bars (= {HORIZON * 15} minutes)")

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
# CALCULATE EMA FEATURES
# =============================================================================
print("\nCalculating EMA features...")

# EMAs
df['ema9'] = df['close'].ewm(span=9).mean()
df['ema50'] = df['close'].ewm(span=50).mean()

# Feature 1: EMA Separation (EMA9 - EMA50) / price - in bps
df['ema_separation'] = ((df['ema9'] - df['ema50']) / df['close']) * 10000

# Feature 2: EMA9 slope (change over last 3 bars) - in bps
df['ema9_slope'] = ((df['ema9'] - df['ema9'].shift(3)) / df['close']) * 10000

# Feature 3: EMA50 slope (change over last 3 bars) - in bps
df['ema50_slope'] = ((df['ema50'] - df['ema50'].shift(3)) / df['close']) * 10000

# Feature 4: Price to EMA9 - in bps
df['price_to_ema9'] = ((df['close'] - df['ema9']) / df['close']) * 10000

# Feature 5: Price to EMA50 - in bps
df['price_to_ema50'] = ((df['close'] - df['ema50']) / df['close']) * 10000

# Drop NaN
df = df.dropna()

# =============================================================================
# CALCULATE DIRECTION
# =============================================================================
print("Calculating direction labels...")

@njit
def calc_direction(opens, highs, lows, target_bps, horizon):
    n = len(opens)
    directions = np.zeros(n, dtype=np.int32)

    for i in range(n - horizon):
        entry = opens[i + 1]
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
                directions[i] = 0
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

# Direction distribution
train_up = (df_train['direction'] == 1).sum()
train_down = (df_train['direction'] == -1).sum()
train_neither = (df_train['direction'] == 0).sum()
print(f"\nTrain direction: UP={train_up:,} ({train_up/len(df_train)*100:.1f}%), DOWN={train_down:,} ({train_down/len(df_train)*100:.1f}%), Neither={train_neither:,} ({train_neither/len(df_train)*100:.1f}%)")

test_up = (df_test['direction'] == 1).sum()
test_down = (df_test['direction'] == -1).sum()
test_neither = (df_test['direction'] == 0).sum()
print(f"Test direction: UP={test_up:,} ({test_up/len(df_test)*100:.1f}%), DOWN={test_down:,} ({test_down/len(df_test)*100:.1f}%), Neither={test_neither:,} ({test_neither/len(df_test)*100:.1f}%)")

# =============================================================================
# NORMALIZE FEATURES
# =============================================================================
print("\nNormalizing features...")

train_means = df_train[STATE_FEATURES].mean()
train_stds = df_train[STATE_FEATURES].std()

for col in STATE_FEATURES:
    df_train[f'{col}_z'] = (df_train[col] - train_means[col]) / (train_stds[col] + 1e-9)
    df_test[f'{col}_z'] = (df_test[col] - train_means[col]) / (train_stds[col] + 1e-9)

z_features = [f'{col}_z' for col in STATE_FEATURES]

# =============================================================================
# SIMPLE RULE-BASED TEST (EMA Cross)
# =============================================================================
print("\n" + "=" * 80)
print("SIMPLE EMA CROSS RULE TEST")
print("=" * 80)

# Rule: If EMA9 > EMA50, predict UP. If EMA9 < EMA50, predict DOWN.
df_test['ema_prediction'] = np.where(df_test['ema9'] > df_test['ema50'], 1, -1)

# Only test on samples with direction
valid_mask = df_test['direction'] != 0
ema_correct = (df_test.loc[valid_mask, 'ema_prediction'] == df_test.loc[valid_mask, 'direction']).sum()
ema_total = valid_mask.sum()
ema_accuracy = ema_correct / ema_total * 100

print(f"\nSimple EMA Cross Rule (EMA9 > EMA50 -> UP):")
print(f"  Accuracy: {ema_accuracy:.2f}%")
print(f"  Edge vs random: {ema_accuracy - 50:.2f}%")

# =============================================================================
# SIMILARITY-BASED TEST
# =============================================================================
print("\n" + "=" * 80)
print("SIMILARITY-BASED PREDICTION (KNN)")
print("=" * 80)

# Prepare matrices
train_matrix = df_train[z_features].values
train_directions = df_train['direction'].values

valid_train_mask = train_directions != 0
train_matrix_valid = train_matrix[valid_train_mask]
train_directions_valid = train_directions[valid_train_mask]
print(f"\nValid train samples: {len(train_directions_valid):,}")

test_matrix = df_test[z_features].values
test_directions = df_test['direction'].values

valid_test_mask = test_directions != 0
test_indices = np.where(valid_test_mask)[0]
print(f"Valid test samples: {len(test_indices):,}")

# Run similarity search
predictions = []
actuals = []
up_ratios = []

print("\nProgress: ", end="", flush=True)
for idx, i in enumerate(test_indices):
    if idx % 5000 == 0:
        print(f"{idx//1000}k..", end="", flush=True)

    current_state = test_matrix[i]
    distances = np.sqrt(np.sum((train_matrix_valid - current_state) ** 2, axis=1))
    nearest_idx = np.argsort(distances)[:K_NEIGHBORS]
    neighbor_directions = train_directions_valid[nearest_idx]

    up_count = np.sum(neighbor_directions == 1)
    down_count = np.sum(neighbor_directions == -1)
    up_ratio = up_count / (up_count + down_count)

    up_ratios.append(up_ratio)
    actuals.append(test_directions[i])

    if up_ratio > 0.5:
        predictions.append(1)
    else:
        predictions.append(-1)

print("Done!")

predictions = np.array(predictions)
actuals = np.array(actuals)
up_ratios = np.array(up_ratios)

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

correct = (predictions == actuals).sum()
total = len(predictions)
accuracy = correct / total * 100

print(f"\nKNN Similarity Results:")
print(f"  Total predictions: {total:,}")
print(f"  Correct: {correct:,}")
print(f"  Accuracy: {accuracy:.2f}%")
print(f"  Edge vs random: {accuracy - 50:.2f}%")

# By confidence
print(f"\nAccuracy by Confidence Level:")
print("-" * 60)

for low, high, label in [(0.50, 0.55, "50-55%"), (0.55, 0.60, "55-60%"),
                          (0.60, 0.65, "60-65%"), (0.65, 0.70, "65-70%"), (0.70, 1.00, "70%+")]:
    mask_up = (up_ratios >= low) & (up_ratios < high)
    if mask_up.sum() > 0:
        acc = (predictions[mask_up] == actuals[mask_up]).mean() * 100
        print(f"  UP conf {label}: {mask_up.sum():,} samples, accuracy = {acc:.1f}%")

    mask_down = (up_ratios <= (1-low)) & (up_ratios > (1-high))
    if mask_down.sum() > 0:
        acc = (predictions[mask_down] == actuals[mask_down]).mean() * 100
        print(f"  DOWN conf {label}: {mask_down.sum():,} samples, accuracy = {acc:.1f}%")

# High confidence
high_conf_mask = (up_ratios >= 0.60) | (up_ratios <= 0.40)
if high_conf_mask.sum() > 0:
    high_acc = (predictions[high_conf_mask] == actuals[high_conf_mask]).mean() * 100
    print(f"\nHigh Confidence (>60% bias): {high_conf_mask.sum():,} samples, accuracy = {high_acc:.1f}%")

# =============================================================================
# COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"\n{'Method':<30} {'Accuracy':>10} {'Edge':>10}")
print("-" * 50)
print(f"{'Random (baseline)':<30} {'50.00%':>10} {'0.00%':>10}")
print(f"{'Simple EMA Cross':<30} {f'{ema_accuracy:.2f}%':>10} {f'{ema_accuracy-50:.2f}%':>10}")
print(f"{'KNN Similarity':<30} {f'{accuracy:.2f}%':>10} {f'{accuracy-50:.2f}%':>10}")
if high_conf_mask.sum() > 0:
    print(f"{'KNN High Conf (60%+)':<30} {f'{high_acc:.2f}%':>10} {f'{high_acc-50:.2f}%':>10}")

print("\nDONE!")
