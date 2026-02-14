"""
Validate ALL Features on 15-MINUTE Timeframe

Tests 100+ features to find which have predictive power on 15min BTC data.
Uses Q1 vs Q4 quartile analysis to measure Case 1 (wrong direction) rate difference.

Run: .venv/Scripts/python.exe scripts/debug/validate_features_15min_all.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION (Same as existing validation)
# =============================================================================
TARGETS = [12, 15, 20, 25, 30, 50]  # in bps
HORIZONS = [1,3, 5, 10, 15, 30, 60, 120]  # in 15-min bars
EXTENDED_H = 200  # Extended horizon for recovery detection

# =============================================================================
# LOAD AND RESAMPLE DATA
# =============================================================================
print("=" * 80)
print("VALIDATE ALL FEATURES ON 15-MINUTE TIMEFRAME")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv_1m):,} 1-minute candles")

# Resample to 15-minute
print("Resampling to 15-minute candles...")
ohlcv = ohlcv_1m.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()
print(f"Created {len(ohlcv):,} 15-minute candles")

# Split train/test
train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
test = ohlcv[ohlcv.index > "2023-12-31"].copy()
print(f"Train data (2020-2023): {len(train):,} candles")
print(f"Test data (2024-2025):  {len(test):,} candles")


# =============================================================================
# FEATURE CALCULATION - ALL FEATURES (100+)
# =============================================================================
def calculate_all_features(df):
    """Calculate ALL 100+ features for 15-minute data."""
    df = df.copy()

    # =========================================================================
    # PRICE-BASED FEATURES
    # =========================================================================
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
    df['body_bps'] = abs(df['close'] - df['open']) / df['close'] * 10000
    df['upper_wick_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 0.0001)
    df['lower_wick_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 0.0001)
    df['range_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
    df['gap_bps'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 10000
    df['true_range_pct'] = df['range_bps']

    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']
    df['tail_body_ratio'] = (total_range - body) / (body + 0.0001)
    df['wick_imbalance'] = df['upper_wick_pct'] - df['lower_wick_pct']

    # =========================================================================
    # MOVING AVERAGES - EMA
    # =========================================================================
    for period in [9, 20, 50, 100, 200]:
        df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'ema{period}_dist_pct'] = (df['close'] - df[f'ema{period}']) / df[f'ema{period}'] * 100

    for period in [9, 20, 50, 100, 200]:
        df[f'ema{period}_slope'] = (df[f'ema{period}'] - df[f'ema{period}'].shift(3)) / df[f'ema{period}'].shift(3) * 100

    df['ema_separation'] = abs(df['ema50'] - df['ema200']) / df['close'] * 100
    df['ema_separation_9_20'] = abs(df['ema9'] - df['ema20']) / df['close'] * 100
    df['ema_separation_20_50'] = abs(df['ema20'] - df['ema50']) / df['close'] * 100

    # =========================================================================
    # MOVING AVERAGES - SMA
    # =========================================================================
    for period in [10, 20, 50, 100, 200]:
        df[f'sma{period}'] = df['close'].rolling(period).mean()
        df[f'sma{period}_dist_pct'] = (df['close'] - df[f'sma{period}']) / df[f'sma{period}'] * 100

    # =========================================================================
    # RSI
    # =========================================================================
    delta = df['close'].diff()
    for period in [7, 14, 21]:
        gain = delta.where(delta > 0, 0).ewm(span=period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
        rs = gain / (loss + 0.0001)
        df[f'rsi{period}'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi14']

    # =========================================================================
    # ROC (Rate of Change)
    # =========================================================================
    for period in [3, 5, 10, 20, 50]:
        df[f'roc{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

    # =========================================================================
    # MOMENTUM
    # =========================================================================
    for period in [3, 5, 10, 20]:
        df[f'momentum{period}'] = df['close'] - df['close'].shift(period)

    # =========================================================================
    # ATR / VOLATILITY
    # =========================================================================
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    for period in [7, 14, 21]:
        atr = tr.ewm(span=period, adjust=False).mean()
        df[f'atr{period}'] = atr
        df[f'atr{period}_pct'] = atr / df['close'] * 100

    df['atr'] = df['atr14']
    df['atr_pct'] = df['atr14_pct']

    df['atr_percentile'] = df['atr_pct'].rolling(window=100, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    for period in [10, 20, 50]:
        df[f'std{period}'] = df['close'].rolling(period).std() / df['close'] * 100

    df['volatility_ratio'] = df['std10'] / (df['std50'] + 0.0001)

    # =========================================================================
    # BOLLINGER BANDS
    # =========================================================================
    bb_std = df['close'].rolling(20).std()
    bb_upper = df['sma20'] + 2 * bb_std
    bb_lower = df['sma20'] - 2 * bb_std
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 0.0001)
    df['bb_width'] = (bb_upper - bb_lower) / df['sma20'] * 100

    # =========================================================================
    # VOLUME
    # =========================================================================
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
    df['volume_std'] = df['volume'].rolling(20).std() / (df['volume'].rolling(20).mean() + 0.0001)

    # =========================================================================
    # TIME
    # =========================================================================
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['session'] = pd.cut(df['hour'], bins=[-1, 4, 8, 12, 16, 20, 24], labels=[0, 1, 2, 3, 4, 5]).astype(float)

    # =========================================================================
    # STRUCTURE
    # =========================================================================
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['higher_close'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['lower_close'] = (df['close'] < df['close'].shift(1)).astype(int)

    for window in [3, 5, 10]:
        df[f'hh_count{window}'] = df['higher_high'].rolling(window).sum()
        df[f'll_count{window}'] = df['lower_low'].rolling(window).sum()
        df[f'up_bars{window}'] = (df['close'] > df['open']).rolling(window).sum()
        df[f'down_bars{window}'] = (df['close'] < df['open']).rolling(window).sum()

    for period in [10, 20, 50]:
        df[f'high{period}'] = df['high'].rolling(period).max()
        df[f'low{period}'] = df['low'].rolling(period).min()
        df[f'dist_from_high{period}_pct'] = (df[f'high{period}'] - df['close']) / df['close'] * 100
        df[f'dist_from_low{period}_pct'] = (df['close'] - df[f'low{period}']) / df['close'] * 100
        df[f'range_position_{period}'] = (df['close'] - df[f'low{period}']) / (df[f'high{period}'] - df[f'low{period}'] + 0.0001)

    # =========================================================================
    # MACD
    # =========================================================================
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['macd_pct'] = df['macd'] / df['close'] * 100

    # =========================================================================
    # STOCHASTIC
    # =========================================================================
    for period in [14, 21]:
        low_n = df['low'].rolling(period).min()
        high_n = df['high'].rolling(period).max()
        df[f'stoch_k{period}'] = 100 * (df['close'] - low_n) / (high_n - low_n + 0.0001)
        df[f'stoch_d{period}'] = df[f'stoch_k{period}'].rolling(3).mean()
    df['stoch_k'] = df['stoch_k14']
    df['stoch_d'] = df['stoch_d14']

    # =========================================================================
    # CCI (Commodity Channel Index)
    # =========================================================================
    for period in [14, 20]:
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df[f'cci{period}'] = (tp - sma_tp) / (0.015 * mad + 0.0001)
    df['cci'] = df['cci14']

    # =========================================================================
    # WILLIAMS %R
    # =========================================================================
    for period in [14, 21]:
        high_n = df['high'].rolling(period).max()
        low_n = df['low'].rolling(period).min()
        df[f'williams_r{period}'] = -100 * (high_n - df['close']) / (high_n - low_n + 0.0001)
    df['williams_r'] = df['williams_r14']

    # =========================================================================
    # ADX (Average Directional Index)
    # =========================================================================
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr14 = df['atr14']
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / (atr14 + 0.0001))
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / (atr14 + 0.0001))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    df['adx'] = dx.ewm(span=14, adjust=False).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    df['di_diff'] = plus_di - minus_di

    # =========================================================================
    # AROON
    # =========================================================================
    for period in [14, 25]:
        df[f'aroon_up{period}'] = 100 * df['high'].rolling(period + 1).apply(lambda x: x.argmax(), raw=True) / period
        df[f'aroon_down{period}'] = 100 * df['low'].rolling(period + 1).apply(lambda x: x.argmin(), raw=True) / period
        df[f'aroon_osc{period}'] = df[f'aroon_up{period}'] - df[f'aroon_down{period}']
    df['aroon_up'] = df['aroon_up14']
    df['aroon_down'] = df['aroon_down14']
    df['aroon_osc'] = df['aroon_osc14']

    # =========================================================================
    # MFI (Money Flow Index)
    # =========================================================================
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df['mfi'] = 100 - (100 / (1 + pos_mf / (neg_mf + 0.0001)))

    # =========================================================================
    # OBV (On-Balance Volume)
    # =========================================================================
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv
    df['obv_slope'] = (obv - obv.shift(5)) / (obv.shift(5).abs() + 0.0001) * 100

    # =========================================================================
    # CMF (Chaikin Money Flow)
    # =========================================================================
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 0.0001) * df['volume']
    df['cmf'] = mfv.rolling(20).sum() / (df['volume'].rolling(20).sum() + 0.0001)

    # =========================================================================
    # VWAP
    # =========================================================================
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (tp * df['volume']).cumsum() / (df['volume'].cumsum() + 0.0001)
    df['vwap_dist_pct'] = (df['close'] - df['vwap']) / df['vwap'] * 100

    # =========================================================================
    # KELTNER CHANNEL
    # =========================================================================
    kc_middle = df['ema20']
    kc_range = df['atr14'] * 2
    df['keltner_upper'] = kc_middle + kc_range
    df['keltner_lower'] = kc_middle - kc_range
    df['keltner_position'] = (df['close'] - df['keltner_lower']) / (df['keltner_upper'] - df['keltner_lower'] + 0.0001)
    df['keltner_width'] = kc_range / kc_middle * 100

    # =========================================================================
    # DONCHIAN CHANNEL
    # =========================================================================
    for period in [20, 50]:
        df[f'donchian_upper{period}'] = df['high'].rolling(period).max()
        df[f'donchian_lower{period}'] = df['low'].rolling(period).min()
        df[f'donchian_position{period}'] = (df['close'] - df[f'donchian_lower{period}']) / (df[f'donchian_upper{period}'] - df[f'donchian_lower{period}'] + 0.0001)
        df[f'donchian_width{period}'] = (df[f'donchian_upper{period}'] - df[f'donchian_lower{period}']) / df['close'] * 100

    # =========================================================================
    # Z-SCORE
    # =========================================================================
    for period in [20, 50, 100]:
        mean = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        df[f'zscore{period}'] = (df['close'] - mean) / (std + 0.0001)

    # =========================================================================
    # CONSECUTIVE BARS
    # =========================================================================
    df['consecutive_up'] = df['higher_close'].groupby((df['higher_close'] != df['higher_close'].shift()).cumsum()).cumsum()
    df['consecutive_down'] = df['lower_close'].groupby((df['lower_close'] != df['lower_close'].shift()).cumsum()).cumsum()

    # =========================================================================
    # RETURN STATISTICS
    # =========================================================================
    returns = df['close'].pct_change()
    df['return_1'] = returns * 100
    df['return_mean5'] = returns.rolling(5).mean() * 100
    df['return_std5'] = returns.rolling(5).std() * 100
    df['return_skew20'] = returns.rolling(20).skew()
    df['return_kurt20'] = returns.rolling(20).kurt()

    # =========================================================================
    # TREND FEATURES
    # =========================================================================
    df['trend_strength'] = abs(df['ema9'] - df['ema50']) / (df['atr14'] + 0.0001)
    df['trend_direction'] = np.sign(df['ema9'] - df['ema50'])

    # =========================================================================
    # CANDLE PATTERNS
    # =========================================================================
    df['doji'] = (df['body_bps'] < df['range_bps'] * 0.1).astype(int)
    df['hammer'] = ((df['lower_wick_pct'] > 0.6) & (df['upper_wick_pct'] < 0.1)).astype(int)
    df['shooting_star'] = ((df['upper_wick_pct'] > 0.6) & (df['lower_wick_pct'] < 0.1)).astype(int)

    return df


# =============================================================================
# FEATURE LIST
# =============================================================================
def get_all_feature_names():
    """Return list of all feature names to test."""
    features = [
        # Price-based (9)
        'range_bps', 'body_bps', 'upper_wick_pct', 'lower_wick_pct', 'range_position',
        'gap_bps', 'true_range_pct', 'tail_body_ratio', 'wick_imbalance',

        # EMA distance (5)
        'ema9_dist_pct', 'ema20_dist_pct', 'ema50_dist_pct', 'ema100_dist_pct', 'ema200_dist_pct',

        # EMA slope (5)
        'ema9_slope', 'ema20_slope', 'ema50_slope', 'ema100_slope', 'ema200_slope',

        # EMA separation (3)
        'ema_separation', 'ema_separation_9_20', 'ema_separation_20_50',

        # SMA distance (5)
        'sma10_dist_pct', 'sma20_dist_pct', 'sma50_dist_pct', 'sma100_dist_pct', 'sma200_dist_pct',

        # RSI (4)
        'rsi', 'rsi7', 'rsi14', 'rsi21',

        # ROC (5)
        'roc3', 'roc5', 'roc10', 'roc20', 'roc50',

        # Momentum (4)
        'momentum3', 'momentum5', 'momentum10', 'momentum20',

        # ATR/Volatility (9)
        'atr_pct', 'atr7_pct', 'atr14_pct', 'atr21_pct', 'atr_percentile',
        'std10', 'std20', 'std50', 'volatility_ratio',

        # Bollinger Bands (2)
        'bb_position', 'bb_width',

        # Volume (3)
        'volume_ratio', 'volume_trend', 'volume_std',

        # Time (4)
        'hour', 'day_of_week', 'is_weekend', 'session',

        # Structure - counts (12)
        'hh_count3', 'hh_count5', 'hh_count10',
        'll_count3', 'll_count5', 'll_count10',
        'up_bars3', 'up_bars5', 'up_bars10',
        'down_bars3', 'down_bars5', 'down_bars10',

        # Distance from high/low (9)
        'dist_from_high10_pct', 'dist_from_high20_pct', 'dist_from_high50_pct',
        'dist_from_low10_pct', 'dist_from_low20_pct', 'dist_from_low50_pct',
        'range_position_10', 'range_position_20', 'range_position_50',

        # MACD (4)
        'macd', 'macd_signal', 'macd_histogram', 'macd_pct',

        # Stochastic (6)
        'stoch_k', 'stoch_d', 'stoch_k14', 'stoch_d14', 'stoch_k21', 'stoch_d21',

        # CCI (3)
        'cci', 'cci14', 'cci20',

        # Williams %R (3)
        'williams_r', 'williams_r14', 'williams_r21',

        # ADX (4)
        'adx', 'plus_di', 'minus_di', 'di_diff',

        # Aroon (9)
        'aroon_up', 'aroon_down', 'aroon_osc',
        'aroon_up14', 'aroon_down14', 'aroon_osc14',
        'aroon_up25', 'aroon_down25', 'aroon_osc25',

        # MFI (1)
        'mfi',

        # OBV (1)
        'obv_slope',

        # CMF (1)
        'cmf',

        # VWAP (1)
        'vwap_dist_pct',

        # Keltner Channel (2)
        'keltner_position', 'keltner_width',

        # Donchian Channel (4)
        'donchian_position20', 'donchian_width20', 'donchian_position50', 'donchian_width50',

        # Z-score (3)
        'zscore20', 'zscore50', 'zscore100',

        # Consecutive (2)
        'consecutive_up', 'consecutive_down',

        # Return stats (5)
        'return_1', 'return_mean5', 'return_std5', 'return_skew20', 'return_kurt20',

        # Trend (2)
        'trend_strength', 'trend_direction',

        # Candle patterns (3)
        'doji', 'hammer', 'shooting_star',
    ]
    return features


# =============================================================================
# CASE LABELING
# =============================================================================
@njit
def label_case(entry, highs, lows, target_pct, H, extended_H):
    """
    Label trade outcome:
    0 = Clean Win (hit target without drawdown)
    1 = Wrong Direction (never hit target)
    2 = Quick Recovery (hit target within H with drawdown)
    3 = Slow Recovery (hit target after H)
    """
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
def analyze_feature(df, target_bps, horizon, feature_name):
    """Analyze Q1 vs Q4 Case 1 rate for a feature."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    start_idx = 200
    end_idx = len(df) - EXTENDED_H

    if end_idx <= start_idx:
        return None

    valid_indices = np.arange(start_idx, end_idx)
    target_pct = target_bps / 10000

    cases = np.zeros(len(valid_indices), dtype=np.int8)
    for idx, i in enumerate(valid_indices):
        entry = close[i]
        future_highs = high[i+1:i+1+EXTENDED_H]
        future_lows = low[i+1:i+1+EXTENDED_H]
        cases[idx] = label_case(entry, future_highs, future_lows, target_pct, horizon, EXTENDED_H)

    if feature_name not in df.columns:
        return None

    feature_values = df.iloc[valid_indices][feature_name].values
    valid_mask = ~np.isnan(feature_values) & (cases >= 0)
    feat = feature_values[valid_mask]
    case = cases[valid_mask]

    if len(feat) < 500:
        return None

    try:
        q25, q75 = np.percentile(feat, [25, 75])
        if q25 == q75:
            return None

        q1_mask = feat < q25
        q4_mask = feat >= q75

        q1_cases = case[q1_mask]
        q4_cases = case[q4_mask]

        if len(q1_cases) < 50 or len(q4_cases) < 50:
            return None

        q1_case1 = (q1_cases == 1).sum() / len(q1_cases) * 100
        q4_case1 = (q4_cases == 1).sum() / len(q4_cases) * 100
        effect = q1_case1 - q4_case1

        return {
            'q1_case1_pct': q1_case1,
            'q4_case1_pct': q4_case1,
            'effect': effect,
            'q1_count': len(q1_cases),
            'q4_count': len(q4_cases)
        }
    except:
        return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
print("\n" + "=" * 80)
print("CALCULATING FEATURES...")
print("=" * 80)

train_feat = calculate_all_features(train)
test_feat = calculate_all_features(test)

features = get_all_feature_names()
print(f"\nTotal features to test: {len(features)}")

results = []
total = len(features) * len(TARGETS) * len(HORIZONS)
count = 0

print("\n" + "=" * 80)
print("RUNNING QUARTILE ANALYSIS...")
print("=" * 80)

for feature in features:
    for target in TARGETS:
        for horizon in HORIZONS:
            count += 1
            if count % 500 == 0:
                print(f"Progress: {count}/{total} ({count/total*100:.1f}%)")

            train_result = analyze_feature(train_feat, target, horizon, feature)
            if train_result is None:
                continue

            test_result = analyze_feature(test_feat, target, horizon, feature)
            if test_result is None:
                continue

            results.append({
                'feature': feature,
                'target_bps': target,
                'horizon': horizon,
                'train_q1_case1': train_result['q1_case1_pct'],
                'train_q4_case1': train_result['q4_case1_pct'],
                'train_effect': train_result['effect'],
                'test_q1_case1': test_result['q1_case1_pct'],
                'test_q4_case1': test_result['q4_case1_pct'],
                'test_effect': test_result['effect'],
            })

results_df = pd.DataFrame(results)

output_path = Path("experiments/feature_validation_15min_all.csv")
results_df.to_csv(output_path, index=False)
print(f"\nSaved detailed results to: {output_path}")
print(f"Total combinations tested: {len(results_df)}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FEATURE SUMMARY")
print("=" * 80)

summary = results_df.groupby('feature').agg({
    'train_effect': 'mean',
    'test_effect': 'mean',
}).reset_index()

summary['valid_combinations'] = results_df.groupby('feature').size().values
summary['oos_valid'] = (summary['train_effect'] * summary['test_effect'] > 0) & (abs(summary['test_effect']) > 2)
summary = summary.sort_values('test_effect', ascending=False)

summary_path = Path("experiments/feature_validation_15min_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"Saved summary to: {summary_path}")

print("\n" + "-" * 80)
print("TOP 20 FEATURES (by Test Effect)")
print("-" * 80)
print(f"{'Feature':<30} {'Train':>10} {'Test':>10} {'Valid':>8}")
print("-" * 80)

for _, row in summary.head(20).iterrows():
    v = "YES" if row['oos_valid'] else ""
    print(f"{row['feature']:<30} {row['train_effect']:>10.2f}% {row['test_effect']:>10.2f}% {v:>8}")

valid_count = summary['oos_valid'].sum()
print(f"\n{'='*80}")
print(f"FEATURES WITH EDGE (OOS Validated): {valid_count} / {len(summary)}")
print("=" * 80)

print("\nDONE!")
