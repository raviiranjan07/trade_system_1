"""
RE-ANALYSIS: All Indicator Features on Multiple Timeframes

This script redoes ANALYSIS-4, 10, 11, 12, 13, 14, 15 on proper timeframes.
Timeframes: 3, 5, 10, 15, 30, 60, 120, 240, 480 minutes

For each timeframe:
- Resample data to that timeframe
- Calculate indicators (RSI, EMA, ATR, Volume, Range Position) on resampled data
- Measure predictive power: Does indicator value predict next-bar direction?

Run: python scripts/debug/analysis_indicators_multi_timeframe.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
TIMEFRAMES_MINUTES = [3, 5, 10, 15, 30, 60, 120, 240, 480]
HORIZONS_BARS = [1, 3, 5, 10]  # How many bars ahead to check
TARGET_BPS = 15  # Minimum move to count as "real" direction

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("RE-ANALYSIS: INDICATOR FEATURES ON MULTIPLE TIMEFRAMES")
print("Timeframes: 3, 5, 10, 15, 30, 60, 120, 240, 480 minutes")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv_1m):,} 1-minute candles")

train_1m = ohlcv_1m[ohlcv_1m.index <= "2023-12-31"].copy()
print(f"Train data: {len(train_1m):,} 1-minute candles")


# =============================================================================
# RESAMPLE TO DIFFERENT TIMEFRAMES
# =============================================================================
def resample_ohlcv(df, minutes):
    """Resample 1-minute OHLCV to higher timeframe."""
    tf_str = f'{minutes}min'
    resampled = df.resample(tf_str).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return resampled


print("\nResampling to different timeframes...")
timeframe_data = {}
for tf_min in TIMEFRAMES_MINUTES:
    timeframe_data[tf_min] = resample_ohlcv(train_1m, tf_min)
    print(f"  {tf_min}-minute: {len(timeframe_data[tf_min]):,} candles")


# =============================================================================
# CALCULATE INDICATORS FOR A TIMEFRAME
# =============================================================================
def calculate_indicators(df):
    """Calculate all indicators on the given dataframe."""
    result = df.copy()

    # RSI (14 periods)
    delta = result['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    result['rsi'] = 100 - (100 / (1 + rs))

    # EMAs
    for period in [9, 20, 50, 100, 200]:
        result[f'ema{period}'] = result['close'].ewm(span=period, adjust=False).mean()
        result[f'ema{period}_dist'] = (result['close'] - result[f'ema{period}']) / result[f'ema{period}'] * 10000  # in bps

    # ATR (14 periods)
    high_low = result['high'] - result['low']
    high_close = abs(result['high'] - result['close'].shift())
    low_close = abs(result['low'] - result['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    result['atr'] = tr.rolling(14).mean()
    result['atr_pct'] = result['atr'] / result['close'] * 10000  # in bps

    # ATR Percentile (rolling 100 bars)
    result['atr_percentile'] = result['atr'].rolling(100).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) == 100 else np.nan
    )

    # Volume
    result['volume_ma'] = result['volume'].rolling(20).mean()
    result['volume_ratio'] = result['volume'] / result['volume_ma']

    # Range Position
    result['range_high'] = result['high'].rolling(20).max()
    result['range_low'] = result['low'].rolling(20).min()
    result['range_position'] = (result['close'] - result['range_low']) / (result['range_high'] - result['range_low'])

    # Future returns (for analysis)
    for h in HORIZONS_BARS:
        result[f'future_return_{h}'] = (result['close'].shift(-h) - result['close']) / result['close'] * 10000
        result[f'future_high_{h}'] = result['high'].shift(-1).rolling(h).max().shift(-h+1)
        result[f'future_low_{h}'] = result['low'].shift(-1).rolling(h).min().shift(-h+1)
        result[f'future_max_up_{h}'] = (result[f'future_high_{h}'] - result['close']) / result['close'] * 10000
        result[f'future_max_down_{h}'] = (result['close'] - result[f'future_low_{h}']) / result['close'] * 10000

    return result


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def analyze_rsi_bins(df, h=5):
    """Analyze RSI vs future direction."""
    df = df.dropna(subset=['rsi', f'future_return_{h}'])

    bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
    df['rsi_bin'] = pd.cut(df['rsi'], bins=bins)

    results = []
    for bin_range in df['rsi_bin'].unique():
        if pd.isna(bin_range):
            continue
        subset = df[df['rsi_bin'] == bin_range]
        if len(subset) < 100:
            continue

        up_pct = (subset[f'future_return_{h}'] > 0).mean() * 100
        avg_return = subset[f'future_return_{h}'].mean()

        results.append({
            'rsi_bin': str(bin_range),
            'count': len(subset),
            'up_pct': up_pct,
            'avg_return': avg_return
        })

    return pd.DataFrame(results)


def analyze_ema_distance(df, ema_period, h=5):
    """Analyze EMA distance vs future direction."""
    col = f'ema{ema_period}_dist'
    df = df.dropna(subset=[col, f'future_return_{h}'])

    # Bin by EMA distance
    bins = [-np.inf, -50, -20, -10, 0, 10, 20, 50, np.inf]
    labels = ['<-50', '-50to-20', '-20to-10', '-10to0', '0to10', '10to20', '20to50', '>50']
    df['ema_bin'] = pd.cut(df[col], bins=bins, labels=labels)

    results = []
    for bin_label in labels:
        subset = df[df['ema_bin'] == bin_label]
        if len(subset) < 100:
            continue

        up_pct = (subset[f'future_return_{h}'] > 0).mean() * 100
        avg_return = subset[f'future_return_{h}'].mean()

        results.append({
            'ema_bin': bin_label,
            'count': len(subset),
            'up_pct': up_pct,
            'avg_return': avg_return
        })

    return pd.DataFrame(results)


def analyze_atr_percentile(df, h=5):
    """Analyze ATR percentile vs future direction and magnitude."""
    df = df.dropna(subset=['atr_percentile', f'future_return_{h}', f'future_max_up_{h}', f'future_max_down_{h}'])

    bins = [0, 20, 40, 60, 80, 100]
    labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
    df['atr_bin'] = pd.cut(df['atr_percentile'], bins=bins, labels=labels)

    results = []
    for bin_label in labels:
        subset = df[df['atr_bin'] == bin_label]
        if len(subset) < 100:
            continue

        up_pct = (subset[f'future_return_{h}'] > 0).mean() * 100
        avg_return = subset[f'future_return_{h}'].mean()
        avg_max_up = subset[f'future_max_up_{h}'].mean()
        avg_max_down = subset[f'future_max_down_{h}'].mean()

        results.append({
            'atr_bin': bin_label,
            'count': len(subset),
            'up_pct': up_pct,
            'avg_return': avg_return,
            'avg_max_up': avg_max_up,
            'avg_max_down': avg_max_down
        })

    return pd.DataFrame(results)


def analyze_volume_ratio(df, h=5):
    """Analyze volume ratio vs future direction."""
    df = df.dropna(subset=['volume_ratio', f'future_return_{h}'])

    bins = [0, 0.5, 0.8, 1.0, 1.5, 2.0, np.inf]
    labels = ['<0.5', '0.5-0.8', '0.8-1.0', '1.0-1.5', '1.5-2.0', '>2.0']
    df['vol_bin'] = pd.cut(df['volume_ratio'], bins=bins, labels=labels)

    results = []
    for bin_label in labels:
        subset = df[df['vol_bin'] == bin_label]
        if len(subset) < 100:
            continue

        up_pct = (subset[f'future_return_{h}'] > 0).mean() * 100
        avg_return = subset[f'future_return_{h}'].mean()

        results.append({
            'vol_bin': bin_label,
            'count': len(subset),
            'up_pct': up_pct,
            'avg_return': avg_return
        })

    return pd.DataFrame(results)


def analyze_range_position(df, h=5):
    """Analyze range position vs future direction."""
    df = df.dropna(subset=['range_position', f'future_return_{h}'])

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    df['range_bin'] = pd.cut(df['range_position'], bins=bins, labels=labels)

    results = []
    for bin_label in labels:
        subset = df[df['range_bin'] == bin_label]
        if len(subset) < 100:
            continue

        up_pct = (subset[f'future_return_{h}'] > 0).mean() * 100
        avg_return = subset[f'future_return_{h}'].mean()

        results.append({
            'range_bin': bin_label,
            'count': len(subset),
            'up_pct': up_pct,
            'avg_return': avg_return
        })

    return pd.DataFrame(results)


def calculate_predictive_power(df, feature_col, h=5):
    """Calculate Cohen's d for a feature's predictive power."""
    df = df.dropna(subset=[feature_col, f'future_return_{h}'])

    up_bars = df[df[f'future_return_{h}'] > 0][feature_col]
    down_bars = df[df[f'future_return_{h}'] < 0][feature_col]

    if len(up_bars) < 100 or len(down_bars) < 100:
        return np.nan

    # Cohen's d
    pooled_std = np.sqrt((up_bars.std()**2 + down_bars.std()**2) / 2)
    if pooled_std == 0:
        return np.nan

    cohens_d = (up_bars.mean() - down_bars.mean()) / pooled_std
    return cohens_d


# =============================================================================
# RUN ANALYSIS ON ALL TIMEFRAMES
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING ANALYSIS ON ALL TIMEFRAMES...")
print("=" * 80)

all_results = {}

for tf_min in TIMEFRAMES_MINUTES:
    print(f"\n{'='*60}")
    print(f"TIMEFRAME: {tf_min} MINUTES")
    print(f"{'='*60}")

    df = timeframe_data[tf_min].copy()
    df = calculate_indicators(df)
    df = df.dropna()

    print(f"Candles with indicators: {len(df):,}")

    tf_results = {'tf': tf_min}

    # =========================================================================
    # ANALYSIS-11: RSI Feature
    # =========================================================================
    print(f"\n### RSI Analysis (H=5 bars = {5*tf_min} minutes)")
    rsi_results = analyze_rsi_bins(df, h=5)
    if len(rsi_results) > 0:
        print(rsi_results.to_string(index=False))
        tf_results['rsi'] = rsi_results

        # Check if RSI extremes predict direction
        extreme_low = rsi_results[rsi_results['rsi_bin'].str.contains('20')]
        extreme_high = rsi_results[rsi_results['rsi_bin'].str.contains('80')]

        if len(extreme_low) > 0 and len(extreme_high) > 0:
            low_up = extreme_low['up_pct'].values[0] if len(extreme_low) > 0 else 50
            high_up = extreme_high['up_pct'].values[0] if len(extreme_high) > 0 else 50
            print(f"\nRSI < 20: {low_up:.1f}% go UP (expect >50% for mean reversion)")
            print(f"RSI > 80: {high_up:.1f}% go UP (expect <50% for mean reversion)")

    # Cohen's d for RSI
    rsi_d = calculate_predictive_power(df, 'rsi', h=5)
    print(f"RSI Cohen's d: {rsi_d:.4f}" if not np.isnan(rsi_d) else "RSI Cohen's d: N/A")
    tf_results['rsi_cohens_d'] = rsi_d

    # =========================================================================
    # ANALYSIS-10: EMA Distance Feature
    # =========================================================================
    print(f"\n### EMA20 Distance Analysis")
    ema_results = analyze_ema_distance(df, 20, h=5)
    if len(ema_results) > 0:
        print(ema_results.to_string(index=False))
        tf_results['ema20'] = ema_results

    ema_d = calculate_predictive_power(df, 'ema20_dist', h=5)
    print(f"EMA20 Distance Cohen's d: {ema_d:.4f}" if not np.isnan(ema_d) else "EMA20 Distance Cohen's d: N/A")
    tf_results['ema20_cohens_d'] = ema_d

    # =========================================================================
    # ANALYSIS-12: ATR Feature
    # =========================================================================
    print(f"\n### ATR Percentile Analysis")
    atr_results = analyze_atr_percentile(df, h=5)
    if len(atr_results) > 0:
        print(atr_results.to_string(index=False))
        tf_results['atr'] = atr_results

    atr_d = calculate_predictive_power(df, 'atr_percentile', h=5)
    print(f"ATR Percentile Cohen's d: {atr_d:.4f}" if not np.isnan(atr_d) else "ATR Percentile Cohen's d: N/A")
    tf_results['atr_cohens_d'] = atr_d

    # =========================================================================
    # ANALYSIS-13: Volume Feature
    # =========================================================================
    print(f"\n### Volume Ratio Analysis")
    vol_results = analyze_volume_ratio(df, h=5)
    if len(vol_results) > 0:
        print(vol_results.to_string(index=False))
        tf_results['volume'] = vol_results

    vol_d = calculate_predictive_power(df, 'volume_ratio', h=5)
    print(f"Volume Ratio Cohen's d: {vol_d:.4f}" if not np.isnan(vol_d) else "Volume Ratio Cohen's d: N/A")
    tf_results['volume_cohens_d'] = vol_d

    # =========================================================================
    # ANALYSIS-14: Range Position Feature
    # =========================================================================
    print(f"\n### Range Position Analysis")
    range_results = analyze_range_position(df, h=5)
    if len(range_results) > 0:
        print(range_results.to_string(index=False))
        tf_results['range'] = range_results

    range_d = calculate_predictive_power(df, 'range_position', h=5)
    print(f"Range Position Cohen's d: {range_d:.4f}" if not np.isnan(range_d) else "Range Position Cohen's d: N/A")
    tf_results['range_cohens_d'] = range_d

    all_results[tf_min] = tf_results


# =============================================================================
# SUMMARY: COHEN'S D BY TIMEFRAME
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: PREDICTIVE POWER (COHEN'S D) BY TIMEFRAME")
print("=" * 80)
print("\nCohen's d interpretation: |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large")

print("\n| Timeframe | RSI | EMA20 Dist | ATR Pctl | Volume | Range Pos |")
print("|-----------|-----|------------|----------|--------|-----------|")

for tf_min in TIMEFRAMES_MINUTES:
    r = all_results[tf_min]
    rsi = f"{r.get('rsi_cohens_d', np.nan):.4f}" if not np.isnan(r.get('rsi_cohens_d', np.nan)) else "N/A"
    ema = f"{r.get('ema20_cohens_d', np.nan):.4f}" if not np.isnan(r.get('ema20_cohens_d', np.nan)) else "N/A"
    atr = f"{r.get('atr_cohens_d', np.nan):.4f}" if not np.isnan(r.get('atr_cohens_d', np.nan)) else "N/A"
    vol = f"{r.get('volume_cohens_d', np.nan):.4f}" if not np.isnan(r.get('volume_cohens_d', np.nan)) else "N/A"
    rng = f"{r.get('range_cohens_d', np.nan):.4f}" if not np.isnan(r.get('range_cohens_d', np.nan)) else "N/A"

    print(f"| {tf_min}min | {rsi} | {ema} | {atr} | {vol} | {rng} |")


# =============================================================================
# KEY FINDINGS
# =============================================================================
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print("""
ANALYSIS-4, 10-15 RE-DONE ON MULTIPLE TIMEFRAMES:

1. RSI (ANALYSIS-11):
   - Does RSI predict direction on any timeframe?
   - Check if RSI<20 has higher UP% than RSI>80

2. EMA Distance (ANALYSIS-10):
   - Does distance from EMA predict direction?
   - Positive distance = above EMA, negative = below

3. ATR Percentile (ANALYSIS-12):
   - Does volatility regime predict direction or magnitude?
   - High ATR = more volatile, low ATR = less volatile

4. Volume (ANALYSIS-13):
   - Does volume predict direction?
   - High volume ratio = above average volume

5. Range Position (ANALYSIS-14):
   - Does position in recent range predict direction?
   - High range position = near recent highs

INTERPRETATION:
- Cohen's d < 0.1: No predictive power
- Cohen's d 0.1-0.2: Negligible
- Cohen's d > 0.2: Some predictive power (worth investigating)
""")
