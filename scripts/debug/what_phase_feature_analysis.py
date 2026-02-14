"""
WHAT Phase Feature Analysis - Historical Taxonomy

Identify which market conditions were associated with Noise vs Real moves in historical data.

Question: "When feature X was in Q1/Q2/Q3/Q4, what outcomes did we observe?"

Outcomes:
- Noise: Hit neither +12bp nor -12bp within H bars
- Real UP: Hit +12bp but NOT -12bp
- Real DOWN: Hit -12bp but NOT +12bp
- Real BOTH: Hit both directions

Run: .venv/Scripts/python.exe scripts/debug/what_phase_feature_analysis.py
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
THRESHOLD_BP = 12  # Noise threshold (±12bp)
HORIZON = 30  # bars (minutes)
EXTENDED_H = 500  # For outcome labeling

# All 56 features to test
FEATURES_TO_TEST = [
    # Price (6)
    'range_bps', 'body_bps', 'upper_wick_pct', 'lower_wick_pct', 'range_position', 'gap_bps',

    # MA Distance (5)
    'ema9_dist_pct', 'ema20_dist_pct', 'ema50_dist_pct', 'ema100_dist_pct', 'ema200_dist_pct',

    # MA Slope (3)
    'ema20_slope', 'ema50_slope', 'ema200_slope',

    # Trend (1)
    'ema_separation',

    # Momentum - RSI (3)
    'rsi', 'rsi7', 'rsi21',

    # Momentum - ROC (3)
    'roc5', 'roc10', 'roc20',

    # Momentum - Other (2)
    'momentum5', 'momentum10',

    # Volatility (6)
    'atr_pct', 'atr7_pct', 'atr21_pct', 'atr_percentile', 'bb_position', 'std20',

    # Volume (2)
    'volume_ratio', 'volume_trend',

    # Time (4)
    'hour', 'day_of_week', 'is_weekend', 'session',

    # Structure - Swing (6)
    'hh_count5', 'll_count5', 'up_bars5', 'down_bars5',
    'dist_from_high20_pct', 'dist_from_low20_pct',

    # === NEW FEATURES (15) ===

    # Pattern (4)
    'consecutive_up_bars', 'consecutive_down_bars', 'inside_bar', 'outside_bar',

    # Structure Extended (4)
    'hh_count20', 'll_count20', 'bars_since_swing_high', 'bars_since_swing_low',

    # Volatility Extended (3)
    'atr14_pct', 'true_range_pct', 'volatility_ratio',

    # Price Action (2)
    'tail_body_ratio', 'wick_imbalance',

    # Time Extended (2)
    'minute_of_hour', 'bars_since_hour_open',
]

print("=" * 80)
print("WHAT PHASE FEATURE ANALYSIS - HISTORICAL TAXONOMY")
print("=" * 80)
print(f"\nThreshold: ±{THRESHOLD_BP}bp")
print(f"Horizon: {HORIZON} bars")
print(f"Features to test: {len(FEATURES_TO_TEST)}")

# =============================================================================
# LOAD DATA
# =============================================================================
ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data (2020-2023): {len(train):,} candles")


# =============================================================================
# CALCULATE FEATURES
# =============================================================================
print("\nCalculating features...")

def calculate_all_features(df):
    """Calculate all 56 features for a dataframe."""

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
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high', 'prev_close']].max(axis=1) - df[['low', 'prev_close']].min(axis=1)

    for period in [7, 14, 21]:
        df[f'atr{period}'] = df['tr'].rolling(period).mean()
        df[f'atr{period}_pct'] = df[f'atr{period}'] / df['close'] * 100

    df['atr_pct'] = df['atr14_pct']  # Alias

    df['std20'] = df['close'].rolling(20).std() / df['close'] * 100
    df['atr_percentile'] = df['atr14'].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100)

    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.0001)

    # --- Volume ---
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 0.0001)
    df['volume_trend'] = (df['volume_sma'] - df['volume_sma'].shift(10)) / (df['volume_sma'].shift(10) + 0.0001) * 100

    # --- Time ---
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['minute_of_hour'] = df.index.minute

    # Session
    def get_session(hour):
        if 0 <= hour < 8: return 0  # Asia
        elif 8 <= hour < 16: return 1  # Europe
        elif 16 <= hour < 24: return 2  # US
        return 0
    df['session'] = df['hour'].apply(get_session)

    # Bars since hour open
    df['bars_since_hour_open'] = df.groupby(df.index.hour).cumcount()

    # --- Structure ---
    df['high_5'] = df['high'].rolling(5).max()
    df['low_5'] = df['low'].rolling(5).min()
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()

    df['hh_count5'] = (df['high'] == df['high_5']).rolling(5).sum()
    df['ll_count5'] = (df['low'] == df['low_5']).rolling(5).sum()
    df['hh_count20'] = (df['high'] == df['high_20']).rolling(20).sum()
    df['ll_count20'] = (df['low'] == df['low_20']).rolling(20).sum()

    df['up_bars5'] = (df['close'] > df['open']).rolling(5).sum()
    df['down_bars5'] = (df['close'] < df['open']).rolling(5).sum()

    df['dist_from_high20_pct'] = (df['high_20'] - df['close']) / df['close'] * 100
    df['dist_from_low20_pct'] = (df['close'] - df['low_20']) / df['close'] * 100

    # Bars since swing high/low (simplified: since last 5-bar high/low)
    df['is_swing_high'] = df['high'] == df['high_5']
    df['is_swing_low'] = df['low'] == df['low_5']
    df['bars_since_swing_high'] = (~df['is_swing_high']).groupby(df['is_swing_high'].cumsum()).cumcount()
    df['bars_since_swing_low'] = (~df['is_swing_low']).groupby(df['is_swing_low'].cumsum()).cumcount()

    # --- Pattern features ---
    df['is_up'] = (df['close'] > df['open']).astype(int)
    df['is_down'] = (df['close'] < df['open']).astype(int)

    # Consecutive up/down bars
    df['consecutive_up_bars'] = df['is_up'].groupby((df['is_up'] != df['is_up'].shift()).cumsum()).cumsum()
    df['consecutive_down_bars'] = df['is_down'].groupby((df['is_down'] != df['is_down'].shift()).cumsum()).cumsum()
    df.loc[df['is_up'] == 0, 'consecutive_up_bars'] = 0
    df.loc[df['is_down'] == 0, 'consecutive_down_bars'] = 0

    # Inside/Outside bar
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    df['inside_bar'] = ((df['high'] <= prev_high) & (df['low'] >= prev_low)).astype(int)
    df['outside_bar'] = ((df['high'] > prev_high) & (df['low'] < prev_low)).astype(int)

    # --- Volatility Extended ---
    df['true_range_pct'] = df['tr'] / df['close'] * 100
    df['volatility_ratio'] = df['atr7'] / (df['atr21'] + 0.0001)

    # --- Price Action ---
    body = df['body_bps']
    total_wick = (df['upper_wick_pct'] + df['lower_wick_pct']) * df['range_bps']
    df['tail_body_ratio'] = total_wick / (body + 0.001)
    df['wick_imbalance'] = (df['upper_wick_pct'] - df['lower_wick_pct']) * 100

    return df

train = calculate_all_features(train)
print("Features calculated.")


# =============================================================================
# LABEL WHAT OUTCOMES
# =============================================================================
print("\nLabeling WHAT outcomes...")

@njit
def label_what_outcomes(close, high, low, threshold_bp, horizon, extended_h):
    """
    Label each bar with WHAT outcome.

    Returns:
        outcomes: array of integers
            0 = Noise (hit neither +threshold nor -threshold)
            1 = Real UP (hit +threshold, not -threshold)
            2 = Real DOWN (hit -threshold, not +threshold)
            3 = Real BOTH (hit both directions)
    """
    n = len(close)
    outcomes = np.full(n, -1, dtype=np.int32)

    for i in range(n - extended_h):
        entry_price = close[i]
        threshold_abs = entry_price * threshold_bp / 10000

        hit_up = False
        hit_down = False

        for j in range(i + 1, min(i + horizon + 1, n)):
            if high[j] >= entry_price + threshold_abs:
                hit_up = True
            if low[j] <= entry_price - threshold_abs:
                hit_down = True

            if hit_up and hit_down:
                break

        if hit_up and hit_down:
            outcomes[i] = 3  # Real BOTH
        elif hit_up:
            outcomes[i] = 1  # Real UP
        elif hit_down:
            outcomes[i] = 2  # Real DOWN
        else:
            outcomes[i] = 0  # Noise

    return outcomes

outcomes = label_what_outcomes(
    train['close'].values,
    train['high'].values,
    train['low'].values,
    THRESHOLD_BP,
    HORIZON,
    EXTENDED_H
)

train['what_outcome'] = outcomes

# Filter valid labels
train_labeled = train[train['what_outcome'] >= 0].copy()
print(f"Labeled bars: {len(train_labeled):,}")

# Outcome distribution
outcome_counts = train_labeled['what_outcome'].value_counts().sort_index()
print("\nOutcome distribution:")
print(f"  Noise (0):      {outcome_counts.get(0, 0):,} ({outcome_counts.get(0, 0)/len(train_labeled)*100:.1f}%)")
print(f"  Real UP (1):    {outcome_counts.get(1, 0):,} ({outcome_counts.get(1, 0)/len(train_labeled)*100:.1f}%)")
print(f"  Real DOWN (2):  {outcome_counts.get(2, 0):,} ({outcome_counts.get(2, 0)/len(train_labeled)*100:.1f}%)")
print(f"  Real BOTH (3):  {outcome_counts.get(3, 0):,} ({outcome_counts.get(3, 0)/len(train_labeled)*100:.1f}%)")


# =============================================================================
# BINNING ANALYSIS
# =============================================================================
print("\nRunning binning analysis...")

results = []

for feature in FEATURES_TO_TEST:
    if feature not in train_labeled.columns:
        print(f"  SKIP: {feature} (not in dataframe)")
        continue

    # Drop NaN
    valid = train_labeled[[feature, 'what_outcome']].dropna()
    if len(valid) < 1000:
        print(f"  SKIP: {feature} (insufficient data: {len(valid)})")
        continue

    # Bin into quartiles
    try:
        valid['quartile'] = pd.qcut(valid[feature], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    except ValueError:
        # Handle case with too many duplicates
        print(f"  SKIP: {feature} (cannot bin - too many duplicates)")
        continue

    # Count outcomes per quartile
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_data = valid[valid['quartile'] == q]
        if len(q_data) == 0:
            continue

        total = len(q_data)
        noise_count = (q_data['what_outcome'] == 0).sum()
        up_count = (q_data['what_outcome'] == 1).sum()
        down_count = (q_data['what_outcome'] == 2).sum()
        both_count = (q_data['what_outcome'] == 3).sum()

        results.append({
            'feature': feature,
            'quartile': q,
            'noise_pct': noise_count / total * 100,
            'real_up_pct': up_count / total * 100,
            'real_down_pct': down_count / total * 100,
            'real_both_pct': both_count / total * 100,
            'sample_size': total,
            'noise_count': noise_count,
            'real_up_count': up_count,
            'real_down_count': down_count,
            'real_both_count': both_count,
        })

    print(f"  OK: {feature}")

results_df = pd.DataFrame(results)


# =============================================================================
# SAVE RESULTS
# =============================================================================
output_path = Path("experiments/what_phase_feature_taxonomy.csv")
output_path.parent.mkdir(exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nTotal features analyzed: {results_df['feature'].nunique()}")
print(f"Output: {output_path}")
