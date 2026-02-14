"""
Validate ALL 44 Features on OOS (2024-2025 data)

Compare train vs test effect for each feature.
Keep only features that hold on BOTH datasets.

Run: .venv/Scripts/python.exe scripts/debug/validate_all_features_oos.py
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
TARGETS = [12, 15, 20, 25, 30, 50]
HORIZONS = [5, 10, 15, 30, 60, 120]
EXTENDED_H = 500

ATR_PERIOD = 14
RSI_PERIOD = 14
EMA_PERIODS = [9, 20, 50, 100, 200]
ROC_PERIODS = [5, 10, 20]
VOLUME_AVG_PERIOD = 20


# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("VALIDATE ALL 44 FEATURES ON OOS (2024-2025)")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
test = ohlcv[ohlcv.index > "2023-12-31"].copy()
print(f"Train data (2020-2023): {len(train):,} candles")
print(f"Test data (2024-2025):  {len(test):,} candles")


# =============================================================================
# CALCULATE FEATURES FUNCTION
# =============================================================================
def calculate_all_features(df):
    """Calculate all 44 features for a dataframe."""

    # --- Price-based features ---
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
    df['body_bps'] = abs(df['close'] - df['open']) / df['close'] * 10000
    df['upper_wick_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 0.0001)
    df['lower_wick_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 0.0001)
    df['range_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
    df['gap_bps'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 10000
    df['candle_direction'] = np.where(df['close'] > df['open'], 1, np.where(df['close'] < df['open'], -1, 0))

    # --- Moving Averages ---
    for period in EMA_PERIODS:
        df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'ema{period}_dist_pct'] = (df['close'] - df[f'ema{period}']) / df[f'ema{period}'] * 100

    for period in [20, 50, 200]:
        df[f'ema{period}_slope'] = (df[f'ema{period}'] - df[f'ema{period}'].shift(5)) / df[f'ema{period}'].shift(5) * 100

    df['ema_separation'] = abs(df['ema50'] - df['ema200']) / df['close'] * 100
    df['trend_direction'] = np.where(df['ema50'] > df['ema200'], 1, -1)

    # --- Momentum ---
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(span=RSI_PERIOD, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=RSI_PERIOD, adjust=False).mean()
    rs = gain / (loss + 0.0001)
    df['rsi'] = 100 - (100 / (1 + rs))

    for period in [7, 21]:
        gain_p = delta.where(delta > 0, 0).ewm(span=period, adjust=False).mean()
        loss_p = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
        rs_p = gain_p / (loss_p + 0.0001)
        df[f'rsi{period}'] = 100 - (100 / (1 + rs_p))

    for period in ROC_PERIODS:
        df[f'roc{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

    df['momentum5'] = df['close'] - df['close'].shift(5)
    df['momentum10'] = df['close'] - df['close'].shift(10)

    # --- Volatility ---
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
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
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(VOLUME_AVG_PERIOD).mean()
    df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
    df['volume_price_trend'] = df['volume_ratio'] * df['candle_direction']

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
# CASE LABELING FUNCTION
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
# ANALYSIS FUNCTION
# =============================================================================
def analyze_dataset(df, dataset_name):
    """Analyze all features for a dataset."""
    print(f"\n{'='*80}")
    print(f"Analyzing {dataset_name}...")
    print(f"{'='*80}")

    # Calculate features
    print("  Calculating features...")
    df = calculate_all_features(df)

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    start_idx = 500
    end_idx = len(df) - EXTENDED_H
    valid_indices = np.arange(start_idx, end_idx)

    # Pre-calculate cases for T=25bp, H=30 (representative)
    print("  Pre-calculating cases (T=25bp, H=30)...")
    target_pct = 25 / 10000
    horizon = 30
    cases = np.zeros(len(valid_indices), dtype=np.int8)

    for idx, i in enumerate(valid_indices):
        entry = close[i]
        future_highs = high[i+1:i+1+EXTENDED_H]
        future_lows = low[i+1:i+1+EXTENDED_H]
        cases[idx] = label_case(entry, future_highs, future_lows, target_pct, horizon, EXTENDED_H)

    # Features to test
    FEATURES_TO_TEST = [
        'range_bps', 'body_bps', 'upper_wick_pct', 'lower_wick_pct',
        'range_position', 'gap_bps', 'candle_direction',
        'ema9_dist_pct', 'ema20_dist_pct', 'ema50_dist_pct', 'ema100_dist_pct', 'ema200_dist_pct',
        'ema20_slope', 'ema50_slope', 'ema200_slope',
        'ema_separation', 'trend_direction',
        'rsi', 'rsi7', 'rsi21',
        'roc5', 'roc10', 'roc20',
        'momentum5', 'momentum10',
        'atr_pct', 'atr_percentile', 'atr7_pct', 'atr21_pct',
        'bb_position', 'std20',
        'volume_ratio', 'volume_trend', 'volume_price_trend',
        'hour', 'day_of_week', 'is_weekend', 'session',
        'hh_count5', 'll_count5', 'up_bars5', 'down_bars5',
        'dist_from_high20_pct', 'dist_from_low20_pct'
    ]

    # Test each feature
    print("  Testing features...")
    results = []
    feature_df = df.iloc[valid_indices].copy()

    for feature in FEATURES_TO_TEST:
        if feature not in feature_df.columns:
            continue

        feature_values = feature_df[feature].values

        # Handle NaN and calculate effect
        valid_mask = ~np.isnan(feature_values) & (cases >= 0)
        feat = feature_values[valid_mask]
        case = cases[valid_mask]

        if len(feat) < 1000:
            continue

        try:
            q25, q75 = np.percentile(feat, [25, 75])
            if q25 == q75:
                continue

            q1_mask = feat < q25
            q4_mask = feat >= q75

            q1_cases = case[q1_mask]
            q4_cases = case[q4_mask]

            if len(q1_cases) < 100 or len(q4_cases) < 100:
                continue

            q1_case1 = (q1_cases == 1).sum() / len(q1_cases) * 100
            q4_case1 = (q4_cases == 1).sum() / len(q4_cases) * 100
            effect = q1_case1 - q4_case1

            results.append({
                'feature': feature,
                'q1_case1': q1_case1,
                'q4_case1': q4_case1,
                'effect_pp': effect,
                'count': len(feat)
            })
        except:
            continue

    return pd.DataFrame(results)


# =============================================================================
# RUN ANALYSIS
# =============================================================================
train_results = analyze_dataset(train, "TRAIN (2020-2023)")
test_results = analyze_dataset(test, "TEST (2024-2025)")


# =============================================================================
# COMPARE TRAIN vs TEST
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON: TRAIN vs TEST")
print("=" * 80)

# Merge results
comparison = train_results.merge(
    test_results,
    on='feature',
    suffixes=('_train', '_test')
)

# Calculate if pattern holds
comparison['train_effect'] = comparison['effect_pp_train']
comparison['test_effect'] = comparison['effect_pp_test']
comparison['effect_diff'] = comparison['test_effect'] - comparison['train_effect']
comparison['same_direction'] = (comparison['train_effect'] * comparison['test_effect']) > 0
comparison['valid'] = comparison['same_direction'] & (abs(comparison['test_effect']) >= 3)

# Sort by train effect
comparison = comparison.sort_values('train_effect', key=abs, ascending=False)

print("\n| Rank | Feature | Train Effect | Test Effect | Diff | Valid? |")
print("|------|---------|--------------|-------------|------|--------|")

for i, row in comparison.iterrows():
    rank = comparison.index.get_loc(i) + 1
    valid_str = "YES" if row['valid'] else "NO"
    print(f"| {rank} | {row['feature']} | {row['train_effect']:+.1f}pp | {row['test_effect']:+.1f}pp | {row['effect_diff']:+.1f}pp | {valid_str} |")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

valid_features = comparison[comparison['valid']].sort_values('test_effect', key=abs, ascending=False)
invalid_features = comparison[~comparison['valid']]

print(f"\nValid features (hold on both train and test): {len(valid_features)}")
print(f"Invalid features (don't hold on test): {len(invalid_features)}")

print("\n### VALID FEATURES (Ranked by Test Effect)")
print("\n| Rank | Feature | Train Effect | Test Effect | Category |")
print("|------|---------|--------------|-------------|----------|")

categories = {
    'range_bps': 'Price', 'body_bps': 'Price', 'upper_wick_pct': 'Price',
    'lower_wick_pct': 'Price', 'range_position': 'Price', 'gap_bps': 'Price',
    'candle_direction': 'Price',
    'ema9_dist_pct': 'MA', 'ema20_dist_pct': 'MA', 'ema50_dist_pct': 'MA',
    'ema100_dist_pct': 'MA', 'ema200_dist_pct': 'MA', 'ema20_slope': 'MA',
    'ema50_slope': 'MA', 'ema200_slope': 'MA', 'ema_separation': 'MA',
    'trend_direction': 'MA',
    'rsi': 'Momentum', 'rsi7': 'Momentum', 'rsi21': 'Momentum',
    'roc5': 'Momentum', 'roc10': 'Momentum', 'roc20': 'Momentum',
    'momentum5': 'Momentum', 'momentum10': 'Momentum',
    'atr_pct': 'Volatility', 'atr_percentile': 'Volatility', 'atr7_pct': 'Volatility',
    'atr21_pct': 'Volatility', 'bb_position': 'Volatility', 'std20': 'Volatility',
    'volume_ratio': 'Volume', 'volume_trend': 'Volume', 'volume_price_trend': 'Volume',
    'hour': 'Time', 'day_of_week': 'Time', 'is_weekend': 'Time', 'session': 'Time',
    'hh_count5': 'Structure', 'll_count5': 'Structure', 'up_bars5': 'Structure',
    'down_bars5': 'Structure', 'dist_from_high20_pct': 'Structure',
    'dist_from_low20_pct': 'Structure'
}

for i, (_, row) in enumerate(valid_features.iterrows()):
    cat = categories.get(row['feature'], 'Unknown')
    print(f"| {i+1} | {row['feature']} | {row['train_effect']:+.1f}pp | {row['test_effect']:+.1f}pp | {cat} |")

print("\n### INVALID FEATURES (Don't hold on test)")
for _, row in invalid_features.iterrows():
    print(f"  - {row['feature']}: Train {row['train_effect']:+.1f}pp, Test {row['test_effect']:+.1f}pp")


# =============================================================================
# SAVE RESULTS
# =============================================================================
output_path = Path("experiments/feature_validation_oos.csv")
comparison.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
