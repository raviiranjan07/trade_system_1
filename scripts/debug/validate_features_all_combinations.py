"""
Validate Features Across ALL 36 Target/Horizon Combinations

Find features that work UNIVERSALLY, not just for T=25bp H=30.

Run: .venv/Scripts/python.exe scripts/debug/validate_features_all_combinations.py
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
TARGETS = [12, 15, 20, 25, 30, 50]  # in bps
HORIZONS = [5, 10, 15, 30, 60, 120]  # in bars (minutes)
EXTENDED_H = 500

# Features to test (same as validation script)
FEATURES_TO_TEST = [
    'range_bps', 'body_bps', 'upper_wick_pct', 'lower_wick_pct',
    'range_position', 'gap_bps',
    'ema9_dist_pct', 'ema20_dist_pct', 'ema50_dist_pct', 'ema100_dist_pct', 'ema200_dist_pct',
    'ema20_slope', 'ema50_slope', 'ema200_slope',
    'ema_separation',
    'rsi', 'rsi7', 'rsi21',
    'roc5', 'roc10', 'roc20',
    'momentum5', 'momentum10',
    'atr_pct', 'atr_percentile', 'atr7_pct', 'atr21_pct',
    'bb_position', 'std20',
    'volume_ratio', 'volume_trend',
    'hour', 'day_of_week', 'is_weekend', 'session',
    'hh_count5', 'll_count5', 'up_bars5', 'down_bars5',
    'dist_from_high20_pct', 'dist_from_low20_pct'
]


# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("VALIDATE FEATURES ACROSS ALL 36 COMBINATIONS")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
test = ohlcv[ohlcv.index > "2023-12-31"].copy()
print(f"Train data (2020-2023): {len(train):,} candles")
print(f"Test data (2024-2025):  {len(test):,} candles")


# =============================================================================
# CALCULATE FEATURES
# =============================================================================
def calculate_all_features(df):
    """Calculate all features for a dataframe."""

    # --- Price-based features ---
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
    df['body_bps'] = abs(df['close'] - df['open']) / df['close'] * 10000
    df['upper_wick_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 0.0001)
    df['lower_wick_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 0.0001)
    df['range_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
    df['gap_bps'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 10000

    # --- Moving Averages ---
    for period in [9, 20, 50, 100, 200]:
        df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'ema{period}_dist_pct'] = (df['close'] - df[f'ema{period}']) / df[f'ema{period}'] * 100

    for period in [20, 50, 200]:
        df[f'ema{period}_slope'] = (df[f'ema{period}'] - df[f'ema{period}'].shift(5)) / df[f'ema{period}'].shift(5) * 100

    df['ema_separation'] = abs(df['ema50'] - df['ema200']) / df['close'] * 100

    # --- Momentum ---
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / (loss + 0.0001)
    df['rsi'] = 100 - (100 / (1 + rs))

    for period in [7, 21]:
        gain_p = delta.where(delta > 0, 0).ewm(span=period, adjust=False).mean()
        loss_p = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
        rs_p = gain_p / (loss_p + 0.0001)
        df[f'rsi{period}'] = 100 - (100 / (1 + rs_p))

    for period in [5, 10, 20]:
        df[f'roc{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

    df['momentum5'] = df['close'] - df['close'].shift(5)
    df['momentum10'] = df['close'] - df['close'].shift(10)

    # --- Volatility ---
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

    df['atr_percentile'] = df['atr_pct'].rolling(window=500, min_periods=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    for period in [7, 21]:
        atr_p = tr.ewm(span=period, adjust=False).mean()
        df[f'atr{period}_pct'] = atr_p / df['close'] * 100

    bb_std = df['close'].rolling(20).std()
    bb_upper = df['ema20'] + 2 * bb_std
    bb_lower = df['ema20'] - 2 * bb_std
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 0.0001)

    df['std20'] = df['close'].rolling(20).std() / df['close'] * 100

    # --- Volume ---
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()

    # --- Time ---
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['session'] = pd.cut(df['hour'], bins=[-1, 4, 8, 12, 16, 20, 24], labels=[0, 1, 2, 3, 4, 5]).astype(float)

    # --- Structure ---
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['hh_count5'] = df['higher_high'].rolling(5).sum()
    df['ll_count5'] = df['lower_low'].rolling(5).sum()
    df['up_bars5'] = (df['close'] > df['open']).rolling(5).sum()
    df['down_bars5'] = (df['close'] < df['open']).rolling(5).sum()
    df['high20'] = df['high'].rolling(20).max()
    df['low20'] = df['low'].rolling(20).min()
    df['dist_from_high20_pct'] = (df['high20'] - df['close']) / df['close'] * 100
    df['dist_from_low20_pct'] = (df['close'] - df['low20']) / df['close'] * 100

    return df


# =============================================================================
# CASE LABELING
# =============================================================================
@njit
def label_case(entry, highs, lows, target_pct, H, extended_H):
    n = len(highs)
    if n == 0:
        return -1
    target_price = entry * (1 + target_pct)
    went_below = False
    hit_within_H = False
    hit_extended = False

    for j in range(min(H, n)):
        if lows[j] < entry:
            went_below = True
        if highs[j] >= target_price:
            hit_within_H = True
            break

    if not hit_within_H:
        for j in range(H, min(extended_H, n)):
            if lows[j] < entry:
                went_below = True
            if highs[j] >= target_price:
                hit_extended = True
                break

    if not went_below and hit_within_H:
        return 0
    elif not hit_within_H and not hit_extended:
        return 1
    elif went_below and hit_within_H:
        return 2
    elif went_below and hit_extended:
        return 3
    else:
        return -1


# =============================================================================
# ANALYZE FUNCTION
# =============================================================================
def analyze_feature_effect(df, target_bps, horizon, feature_name):
    """Calculate feature effect for one target/horizon combination."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    start_idx = 500
    end_idx = len(df) - EXTENDED_H
    valid_indices = np.arange(start_idx, end_idx)

    target_pct = target_bps / 10000

    # Pre-calculate cases
    cases = np.zeros(len(valid_indices), dtype=np.int8)
    for idx, i in enumerate(valid_indices):
        entry = close[i]
        future_highs = high[i+1:i+1+EXTENDED_H]
        future_lows = low[i+1:i+1+EXTENDED_H]
        cases[idx] = label_case(entry, future_highs, future_lows, target_pct, horizon, EXTENDED_H)

    # Get feature values
    if feature_name not in df.columns:
        return None

    feature_values = df.iloc[valid_indices][feature_name].values

    # Handle NaN
    valid_mask = ~np.isnan(feature_values) & (cases >= 0)
    feat = feature_values[valid_mask]
    case = cases[valid_mask]

    if len(feat) < 1000:
        return None

    try:
        q25, q75 = np.percentile(feat, [25, 75])
        if q25 == q75:
            return None

        q1_mask = feat < q25
        q4_mask = feat >= q75

        q1_cases = case[q1_mask]
        q4_cases = case[q4_mask]

        if len(q1_cases) < 100 or len(q4_cases) < 100:
            return None

        q1_case1 = (q1_cases == 1).sum() / len(q1_cases) * 100
        q4_case1 = (q4_cases == 1).sum() / len(q4_cases) * 100
        effect = q1_case1 - q4_case1

        return effect
    except:
        return None


# =============================================================================
# RUN ANALYSIS
# =============================================================================
print("\nCalculating features...")
train = calculate_all_features(train)
test = calculate_all_features(test)
print("Features calculated.")

# Store results: feature -> list of (target, horizon, train_effect, test_effect, valid)
all_results = []

total_combinations = len(TARGETS) * len(HORIZONS)
combo_count = 0

print(f"\nTesting {len(FEATURES_TO_TEST)} features across {total_combinations} combinations...")

for target in TARGETS:
    for horizon in HORIZONS:
        combo_count += 1
        print(f"\n[{combo_count}/{total_combinations}] Target={target}bp, Horizon={horizon}...")

        for feature in FEATURES_TO_TEST:
            train_effect = analyze_feature_effect(train, target, horizon, feature)
            test_effect = analyze_feature_effect(test, target, horizon, feature)

            if train_effect is not None and test_effect is not None:
                same_direction = (train_effect * test_effect) > 0
                valid = same_direction and abs(test_effect) >= 3

                all_results.append({
                    'feature': feature,
                    'target': target,
                    'horizon': horizon,
                    'train_effect': train_effect,
                    'test_effect': test_effect,
                    'valid': valid
                })


# =============================================================================
# AGGREGATE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("AGGREGATING RESULTS")
print("=" * 80)

results_df = pd.DataFrame(all_results)

# Count valid combinations per feature
feature_summary = []
for feature in FEATURES_TO_TEST:
    feat_data = results_df[results_df['feature'] == feature]
    if len(feat_data) == 0:
        continue

    valid_count = feat_data['valid'].sum()
    total_count = len(feat_data)
    valid_pct = valid_count / total_count * 100

    avg_train_effect = feat_data['train_effect'].mean()
    avg_test_effect = feat_data['test_effect'].mean()

    feature_summary.append({
        'feature': feature,
        'valid_combinations': valid_count,
        'total_combinations': total_count,
        'valid_pct': valid_pct,
        'avg_train_effect': avg_train_effect,
        'avg_test_effect': avg_test_effect
    })

summary_df = pd.DataFrame(feature_summary)
summary_df = summary_df.sort_values('valid_pct', ascending=False)


# =============================================================================
# DISPLAY RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("FEATURE VALIDITY ACROSS ALL 36 COMBINATIONS")
print("=" * 80)

print("\n| Rank | Feature | Valid/Total | Valid % | Avg Train | Avg Test | Status |")
print("|------|---------|-------------|---------|-----------|----------|--------|")

for i, (_, row) in enumerate(summary_df.iterrows()):
    if row['valid_pct'] >= 80:
        status = "ROBUST"
    elif row['valid_pct'] >= 50:
        status = "STRONG"
    elif row['valid_pct'] >= 25:
        status = "PARTIAL"
    else:
        status = "WEAK"

    print(f"| {i+1} | {row['feature']} | {int(row['valid_combinations'])}/{int(row['total_combinations'])} | {row['valid_pct']:.0f}% | {row['avg_train_effect']:+.1f}pp | {row['avg_test_effect']:+.1f}pp | {status} |")


# =============================================================================
# SUMMARY BY CATEGORY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY BY STATUS")
print("=" * 80)

robust = summary_df[summary_df['valid_pct'] >= 80]
strong = summary_df[(summary_df['valid_pct'] >= 50) & (summary_df['valid_pct'] < 80)]
partial = summary_df[(summary_df['valid_pct'] >= 25) & (summary_df['valid_pct'] < 50)]
weak = summary_df[summary_df['valid_pct'] < 25]

print(f"\n**ROBUST (>=80%):** {len(robust)} features")
for _, row in robust.iterrows():
    print(f"  - {row['feature']}: {row['valid_pct']:.0f}% valid, avg test effect {row['avg_test_effect']:+.1f}pp")

print(f"\n**STRONG (50-80%):** {len(strong)} features")
for _, row in strong.iterrows():
    print(f"  - {row['feature']}: {row['valid_pct']:.0f}% valid, avg test effect {row['avg_test_effect']:+.1f}pp")

print(f"\n**PARTIAL (25-50%):** {len(partial)} features")
for _, row in partial.iterrows():
    print(f"  - {row['feature']}: {row['valid_pct']:.0f}% valid")

print(f"\n**WEAK (<25%):** {len(weak)} features")
for _, row in weak.iterrows():
    print(f"  - {row['feature']}: {row['valid_pct']:.0f}% valid")


# =============================================================================
# SAVE RESULTS
# =============================================================================
output_path = Path("experiments/feature_validation_all_combinations.csv")
results_df.to_csv(output_path, index=False)

summary_path = Path("experiments/feature_validation_summary.csv")
summary_df.to_csv(summary_path, index=False)

print(f"\n\nDetailed results saved to: {output_path}")
print(f"Summary saved to: {summary_path}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
