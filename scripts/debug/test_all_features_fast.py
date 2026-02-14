"""
Test 50+ Features Data-Driven (Fast Approach)

1. Pre-calculate cases for all target/horizon combinations (once)
2. Calculate all features (once)
3. Test each feature against case outcomes (fast lookups)

Run: .venv/Scripts/python.exe scripts/debug/test_all_features_fast.py
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
TARGETS = [12, 15, 20, 25, 30, 50]  # basis points
HORIZONS = [5, 10, 15, 30, 60, 120]  # bars
EXTENDED_H = 500

# Feature calculation parameters
ATR_PERIOD = 14
RSI_PERIOD = 14
EMA_PERIODS = [9, 20, 50, 100, 200]
ROC_PERIODS = [5, 10, 20]
VOLUME_AVG_PERIOD = 20


# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("TEST 50+ FEATURES - DATA DRIVEN (FAST)")
print(f"Targets: {TARGETS}")
print(f"Horizons: {HORIZONS}")
print(f"Combinations: {len(TARGETS) * len(HORIZONS)}")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data (2020-2023): {len(train):,} candles")


# =============================================================================
# STEP 1: CALCULATE ALL FEATURES (ONCE)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: Calculating all features...")
print("=" * 80)

df = train.copy()

# --- Price-based features ---
print("  Price-based features...")
df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
df['body_bps'] = abs(df['close'] - df['open']) / df['close'] * 10000
df['upper_wick_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 0.0001)
df['lower_wick_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 0.0001)
df['range_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
df['gap_bps'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 10000
df['candle_direction'] = np.where(df['close'] > df['open'], 1, np.where(df['close'] < df['open'], -1, 0))

# --- Moving Averages ---
print("  Moving average features...")
for period in EMA_PERIODS:
    df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    df[f'ema{period}_dist_pct'] = (df['close'] - df[f'ema{period}']) / df[f'ema{period}'] * 100

# EMA slopes
for period in [20, 50, 200]:
    df[f'ema{period}_slope'] = (df[f'ema{period}'] - df[f'ema{period}'].shift(5)) / df[f'ema{period}'].shift(5) * 100

# Trend strength (EMA separation)
df['ema_separation'] = abs(df['ema50'] - df['ema200']) / df['close'] * 100
df['trend_direction'] = np.where(df['ema50'] > df['ema200'], 1, -1)

# --- Momentum ---
print("  Momentum features...")
# RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).ewm(span=RSI_PERIOD, adjust=False).mean()
loss = (-delta.where(delta < 0, 0)).ewm(span=RSI_PERIOD, adjust=False).mean()
rs = gain / (loss + 0.0001)
df['rsi'] = 100 - (100 / (1 + rs))

# RSI with different periods
for period in [7, 21]:
    gain_p = delta.where(delta > 0, 0).ewm(span=period, adjust=False).mean()
    loss_p = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
    rs_p = gain_p / (loss_p + 0.0001)
    df[f'rsi{period}'] = 100 - (100 / (1 + rs_p))

# Rate of Change
for period in ROC_PERIODS:
    df[f'roc{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

# Momentum (price change)
df['momentum5'] = df['close'] - df['close'].shift(5)
df['momentum10'] = df['close'] - df['close'].shift(10)

# --- Volatility ---
print("  Volatility features...")
# ATR
tr1 = df['high'] - df['low']
tr2 = abs(df['high'] - df['close'].shift(1))
tr3 = abs(df['low'] - df['close'].shift(1))
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
df['atr_pct'] = df['atr'] / df['close'] * 100

# ATR percentile (rolling rank)
df['atr_percentile'] = df['atr_pct'].rolling(window=500, min_periods=100).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
)

# ATR with different periods
for period in [7, 21]:
    atr_p = tr.ewm(span=period, adjust=False).mean()
    df[f'atr{period}_pct'] = atr_p / df['close'] * 100

# Bollinger Band position
bb_std = df['close'].rolling(20).std()
bb_upper = df['ema20'] + 2 * bb_std
bb_lower = df['ema20'] - 2 * bb_std
df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 0.0001)

# Standard deviation
df['std20'] = df['close'].rolling(20).std() / df['close'] * 100

# --- Volume ---
print("  Volume features...")
df['volume_ratio'] = df['volume'] / df['volume'].rolling(VOLUME_AVG_PERIOD).mean()
df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
df['volume_price_trend'] = df['volume_ratio'] * df['candle_direction']

# --- Time ---
print("  Time features...")
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Session (0=Asia night, 1=Asia, 2=Europe, 3=US open, 4=US, 5=US close)
df['session'] = pd.cut(df['hour'], bins=[-1, 4, 8, 12, 16, 20, 24], labels=[0, 1, 2, 3, 4, 5]).astype(float)

# --- Structure ---
print("  Structure features...")
# Higher highs / Lower lows count
df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
df['hh_count5'] = df['higher_high'].rolling(5).sum()
df['ll_count5'] = df['lower_low'].rolling(5).sum()

# Trend persistence
df['up_bars5'] = (df['close'] > df['open']).rolling(5).sum()
df['down_bars5'] = (df['close'] < df['open']).rolling(5).sum()

# Distance from recent high/low
df['high20'] = df['high'].rolling(20).max()
df['low20'] = df['low'].rolling(20).min()
df['dist_from_high20_pct'] = (df['high20'] - df['close']) / df['close'] * 100
df['dist_from_low20_pct'] = (df['close'] - df['low20']) / df['close'] * 100

print(f"  Total features calculated: {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])}")


# =============================================================================
# STEP 2: PRE-CALCULATE CASES (ONCE PER TARGET/HORIZON)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: Pre-calculating cases for all target/horizon combinations...")
print("=" * 80)

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

close = df['close'].values
high = df['high'].values
low = df['low'].values

start_idx = 500  # Skip warmup period
end_idx = len(df) - EXTENDED_H
valid_indices = np.arange(start_idx, end_idx)

# Pre-calculate cases for all combinations
case_data = {}
combo_count = 0
total_combos = len(TARGETS) * len(HORIZONS)

for target in TARGETS:
    for horizon in HORIZONS:
        combo_count += 1
        print(f"\r  Processing {combo_count}/{total_combos}: T={target}bp, H={horizon}...", end="", flush=True)

        target_pct = target / 10000
        cases = np.zeros(len(valid_indices), dtype=np.int8)

        for idx, i in enumerate(valid_indices):
            entry = close[i]
            future_highs = high[i+1:i+1+EXTENDED_H]
            future_lows = low[i+1:i+1+EXTENDED_H]
            cases[idx] = label_case(entry, future_highs, future_lows, target_pct, horizon, EXTENDED_H)

        case_data[(target, horizon)] = cases

print(f"\n  Cases pre-calculated for {total_combos} combinations")


# =============================================================================
# STEP 3: DEFINE FEATURE LIST TO TEST
# =============================================================================
FEATURES_TO_TEST = [
    # Price-based
    'range_bps', 'body_bps', 'upper_wick_pct', 'lower_wick_pct',
    'range_position', 'gap_bps', 'candle_direction',

    # Moving Averages
    'ema9_dist_pct', 'ema20_dist_pct', 'ema50_dist_pct', 'ema100_dist_pct', 'ema200_dist_pct',
    'ema20_slope', 'ema50_slope', 'ema200_slope',
    'ema_separation', 'trend_direction',

    # Momentum
    'rsi', 'rsi7', 'rsi21',
    'roc5', 'roc10', 'roc20',
    'momentum5', 'momentum10',

    # Volatility
    'atr_pct', 'atr_percentile', 'atr7_pct', 'atr21_pct',
    'bb_position', 'std20',

    # Volume
    'volume_ratio', 'volume_trend', 'volume_price_trend',

    # Time
    'hour', 'day_of_week', 'is_weekend', 'session',

    # Structure
    'hh_count5', 'll_count5', 'up_bars5', 'down_bars5',
    'dist_from_high20_pct', 'dist_from_low20_pct'
]

print(f"\n  Features to test: {len(FEATURES_TO_TEST)}")


# =============================================================================
# STEP 4: TEST EACH FEATURE
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: Testing each feature against case outcomes...")
print("=" * 80)

def calculate_effect_size(feature_values, cases, n_bins=4):
    """Calculate P(Case1) by quartile bins and return effect size."""
    valid_mask = ~np.isnan(feature_values) & (cases >= 0)
    feat = feature_values[valid_mask]
    case = cases[valid_mask]

    if len(feat) < 1000:
        return None, None, None, None

    try:
        # Use quartiles for binning
        q25, q50, q75 = np.percentile(feat, [25, 50, 75])

        # Handle constant or near-constant features
        if q25 == q75:
            return None, None, None, None

        bins = [('Q1', feat < q25), ('Q4', feat >= q75)]

        q1_cases = case[bins[0][1]]
        q4_cases = case[bins[1][1]]

        if len(q1_cases) < 100 or len(q4_cases) < 100:
            return None, None, None, None

        q1_case1 = (q1_cases == 1).sum() / len(q1_cases) * 100
        q4_case1 = (q4_cases == 1).sum() / len(q4_cases) * 100

        effect = q1_case1 - q4_case1  # Positive = Q1 has more Case1 (low values are bad)

        return q1_case1, q4_case1, effect, len(feat)
    except:
        return None, None, None, None

# Store all results
all_results = []

# Get feature values for valid indices
feature_df = df.iloc[valid_indices].copy()

for feature in FEATURES_TO_TEST:
    if feature not in feature_df.columns:
        print(f"  Skipping {feature} - not found")
        continue

    feature_values = feature_df[feature].values

    for target in TARGETS:
        for horizon in HORIZONS:
            cases = case_data[(target, horizon)]

            q1, q4, effect, count = calculate_effect_size(feature_values, cases)

            if effect is not None:
                all_results.append({
                    'feature': feature,
                    'target': target,
                    'horizon': horizon,
                    'q1_case1': q1,
                    'q4_case1': q4,
                    'effect_pp': effect,
                    'count': count
                })

print(f"  Tested {len(all_results)} feature/target/horizon combinations")


# =============================================================================
# STEP 5: ANALYZE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS: Top Features by Effect Size")
print("=" * 80)

results_df = pd.DataFrame(all_results)

# Average effect across all target/horizon combinations
avg_effect = results_df.groupby('feature').agg({
    'effect_pp': 'mean',
    'q1_case1': 'mean',
    'q4_case1': 'mean',
    'count': 'mean'
}).reset_index()

avg_effect = avg_effect.sort_values('effect_pp', key=abs, ascending=False)

print("\n### Average Effect Across All Targets/Horizons")
print("\n| Rank | Feature | Q1 P(Case1) | Q4 P(Case1) | Effect | Interpretation |")
print("|------|---------|-------------|-------------|--------|----------------|")

for i, row in avg_effect.head(20).iterrows():
    rank = avg_effect.index.get_loc(i) + 1
    effect = row['effect_pp']
    if effect > 0:
        interp = "Low values = bad"
    else:
        interp = "High values = bad"

    if abs(effect) >= 10:
        strength = "STRONG"
    elif abs(effect) >= 5:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    print(f"| {rank} | {row['feature']} | {row['q1_case1']:.1f}% | {row['q4_case1']:.1f}% | {effect:+.1f}pp | {strength}: {interp} |")


# =============================================================================
# STEP 6: FEATURES BY CATEGORY
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS: Best Feature per Category")
print("=" * 80)

categories = {
    'Price': ['range_bps', 'body_bps', 'upper_wick_pct', 'lower_wick_pct', 'range_position', 'gap_bps', 'candle_direction'],
    'Moving Avg': ['ema9_dist_pct', 'ema20_dist_pct', 'ema50_dist_pct', 'ema100_dist_pct', 'ema200_dist_pct', 'ema20_slope', 'ema50_slope', 'ema200_slope', 'ema_separation', 'trend_direction'],
    'Momentum': ['rsi', 'rsi7', 'rsi21', 'roc5', 'roc10', 'roc20', 'momentum5', 'momentum10'],
    'Volatility': ['atr_pct', 'atr_percentile', 'atr7_pct', 'atr21_pct', 'bb_position', 'std20'],
    'Volume': ['volume_ratio', 'volume_trend', 'volume_price_trend'],
    'Time': ['hour', 'day_of_week', 'is_weekend', 'session'],
    'Structure': ['hh_count5', 'll_count5', 'up_bars5', 'down_bars5', 'dist_from_high20_pct', 'dist_from_low20_pct']
}

print("\n| Category | Best Feature | Effect | Q1 Case1 | Q4 Case1 |")
print("|----------|--------------|--------|----------|----------|")

for cat_name, cat_features in categories.items():
    cat_df = avg_effect[avg_effect['feature'].isin(cat_features)]
    if len(cat_df) > 0:
        best = cat_df.loc[cat_df['effect_pp'].abs().idxmax()]
        print(f"| {cat_name} | {best['feature']} | {best['effect_pp']:+.1f}pp | {best['q1_case1']:.1f}% | {best['q4_case1']:.1f}% |")


# =============================================================================
# STEP 7: THRESHOLD ANALYSIS (Data-Driven)
# =============================================================================
print("\n" + "=" * 80)
print("DATA-DRIVEN THRESHOLDS: Top 5 Features")
print("=" * 80)

top_features = avg_effect.head(5)['feature'].tolist()

for feature in top_features:
    print(f"\n### {feature}")

    feat_vals = feature_df[feature].values
    valid_mask = ~np.isnan(feat_vals)
    feat_valid = feat_vals[valid_mask]

    # Find natural breakpoints using percentiles
    percentiles = [10, 25, 50, 75, 90]
    print(f"Percentile boundaries: ", end="")
    for p in percentiles:
        val = np.percentile(feat_valid, p)
        print(f"P{p}={val:.2f}, ", end="")
    print()

    # Show P(Case1) at T=25bp, H=30 for different ranges
    cases_25_30 = case_data[(25, 30)]
    cases_valid = cases_25_30[valid_mask[valid_indices]]
    feat_for_cases = feat_valid[:len(cases_valid)]

    # Define bins based on percentiles from data
    p10 = np.percentile(feat_for_cases, 10)
    p25 = np.percentile(feat_for_cases, 25)
    p50 = np.percentile(feat_for_cases, 50)
    p75 = np.percentile(feat_for_cases, 75)
    p90 = np.percentile(feat_for_cases, 90)

    bins = [
        (f'0-10%', feat_for_cases < p10),
        (f'10-25%', (feat_for_cases >= p10) & (feat_for_cases < p25)),
        (f'25-50%', (feat_for_cases >= p25) & (feat_for_cases < p50)),
        (f'50-75%', (feat_for_cases >= p50) & (feat_for_cases < p75)),
        (f'75-90%', (feat_for_cases >= p75) & (feat_for_cases < p90)),
        (f'90-100%', feat_for_cases >= p90)
    ]

    print(f"| Percentile | P(Case1) | Count |")
    print(f"|------------|----------|-------|")
    for bin_name, bin_mask in bins:
        bin_cases = cases_valid[bin_mask]
        if len(bin_cases) > 0:
            case1_pct = (bin_cases == 1).sum() / len(bin_cases) * 100
            print(f"| {bin_name} | {case1_pct:.1f}% | {len(bin_cases):,} |")


# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("Saving results...")
print("=" * 80)

output_path = Path("experiments/feature_test_results.csv")
results_df.to_csv(output_path, index=False)
print(f"Detailed results saved to: {output_path}")

output_path2 = Path("experiments/feature_test_summary.csv")
avg_effect.to_csv(output_path2, index=False)
print(f"Summary saved to: {output_path2}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
