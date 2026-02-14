"""
EMA Bounce Analysis with OOS Validation - 15min Timeframe

Tests which EMAs act as the best dynamic support/resistance.
Validates results on 2024-2025 OOS data.

Tests:
- In UPTREND: When price touches EMA from above → Does it bounce UP?
- In DOWNTREND: When price touches EMA from below → Does it bounce DOWN?

EMAs tested: 7, 9, 20, 25, 50, 100, 200
Touch threshold: 0-0.2% from EMA
Horizons: H=3, 5, 10 (45min, 75min, 2.5hrs)
Train: 2020-2023 | Test: 2024-2025

Run: python scripts/debug/ema_bounce_validation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EMA BOUNCE ANALYSIS WITH OOS VALIDATION - 15min Timeframe")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_PERIODS = [7, 9, 20, 25, 50, 100, 200]
TOUCH_THRESHOLD_MIN = 0.0  # 0%
TOUCH_THRESHOLD_MAX = 0.2  # 0.2%
HORIZONS = [3, 5, 10]  # 15-min bars
TRAIN_END = "2023-12-31"

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/7] Loading data...")
ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)
print(f"Loaded {len(ohlcv_1m):,} 1-minute candles")

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
print(f"Date range: {ohlcv.index[0]} to {ohlcv.index[-1]}")

# Split train/test
train = ohlcv[ohlcv.index <= TRAIN_END].copy()
test = ohlcv[ohlcv.index > TRAIN_END].copy()
print(f"\nTrain data (2020-2023): {len(train):,} candles")
print(f"Test data (2024-2025):  {len(test):,} candles")

# =============================================================================
# CALCULATE FORWARD MFE/MAE FUNCTION
# =============================================================================
@njit
def calculate_forward_mfe_mae(highs, lows, close_prices, horizon):
    """Calculate forward MFE and MAE for LONG and SHORT."""
    n = len(close_prices)
    mfe_long = np.full(n, np.nan)
    mae_long = np.full(n, np.nan)
    mfe_short = np.full(n, np.nan)
    mae_short = np.full(n, np.nan)

    for i in range(n - horizon):
        entry = close_prices[i]
        future_highs = highs[i+1:i+1+horizon]
        future_lows = lows[i+1:i+1+horizon]

        # LONG
        max_high = np.max(future_highs)
        min_low = np.min(future_lows)
        mfe_long[i] = (max_high - entry) / entry * 10000  # bps
        mae_long[i] = (min_low - entry) / entry * 10000   # bps

        # SHORT
        mfe_short[i] = (entry - min_low) / entry * 10000  # bps
        mae_short[i] = (max_high - entry) / entry * 10000 # bps

    return mfe_long, mae_long, mfe_short, mae_short

# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================
def analyze_ema_bounces(df, dataset_name):
    """Run EMA bounce analysis on given dataset."""
    print(f"\n{'='*80}")
    print(f"ANALYZING {dataset_name}")
    print(f"{'='*80}")

    df = df.copy()

    # Calculate EMAs
    print(f"\n[2/7] Calculating EMAs for {dataset_name}...")
    for period in EMA_PERIODS:
        df[f'ema{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # Detect trends
    print(f"[3/7] Detecting trends for {dataset_name}...")
    df['trend'] = 0
    df.loc[df['ema50'] > df['ema200'], 'trend'] = 1  # Uptrend
    df.loc[df['ema50'] < df['ema200'], 'trend'] = -1  # Downtrend

    uptrend_bars = (df['trend'] == 1).sum()
    downtrend_bars = (df['trend'] == -1).sum()
    print(f"Uptrend: {uptrend_bars:,} ({uptrend_bars/len(df)*100:.1f}%)")
    print(f"Downtrend: {downtrend_bars:,} ({downtrend_bars/len(df)*100:.1f}%)")

    # Calculate forward MFE/MAE
    print(f"[4/7] Calculating forward MFE/MAE for {dataset_name}...")
    for h in HORIZONS:
        mfe_long, mae_long, mfe_short, mae_short = calculate_forward_mfe_mae(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            h
        )
        df[f'mfe_long_h{h}'] = mfe_long
        df[f'mae_long_h{h}'] = mae_long
        df[f'mfe_short_h{h}'] = mfe_short
        df[f'mae_short_h{h}'] = mae_short

    # Detect touch events
    print(f"[5/7] Detecting EMA touch events for {dataset_name}...")
    results = []

    for ema_period in EMA_PERIODS:
        ema_col = f'ema{ema_period}'

        # Calculate distance from EMA
        df['dist_from_ema'] = abs(df['close'] - df[ema_col]) / df[ema_col] * 100

        # Touch = within 0-0.2% of EMA
        df['touch'] = (df['dist_from_ema'] >= TOUCH_THRESHOLD_MIN) & \
                      (df['dist_from_ema'] <= TOUCH_THRESHOLD_MAX)

        # Split by trend
        uptrend_touches = df[(df['trend'] == 1) & (df['touch'])].copy()
        downtrend_touches = df[(df['trend'] == -1) & (df['touch'])].copy()

        # Analyze for each horizon
        for h in HORIZONS:
            # UPTREND (Support test)
            if len(uptrend_touches) > 0:
                uptrend_valid = uptrend_touches.dropna(subset=[f'mfe_long_h{h}', f'mae_long_h{h}'])

                if len(uptrend_valid) > 0:
                    uptrend_valid['success'] = uptrend_valid[f'mfe_long_h{h}'] >= 12

                    success_rate = uptrend_valid['success'].mean() * 100
                    avg_bounce_mfe = uptrend_valid.loc[uptrend_valid['success'], f'mfe_long_h{h}'].mean()
                    avg_fail_mae = uptrend_valid.loc[~uptrend_valid['success'], f'mae_long_h{h}'].mean()
                    median_bounce = uptrend_valid.loc[uptrend_valid['success'], f'mfe_long_h{h}'].median()

                    results.append({
                        'dataset': dataset_name,
                        'ema_period': ema_period,
                        'trend_direction': 'UPTREND',
                        'horizon': h,
                        'total_touches': len(uptrend_valid),
                        'success_rate': success_rate,
                        'avg_bounce_mfe': avg_bounce_mfe,
                        'avg_fail_mae': avg_fail_mae,
                        'median_bounce': median_bounce
                    })

            # DOWNTREND (Resistance test)
            if len(downtrend_touches) > 0:
                downtrend_valid = downtrend_touches.dropna(subset=[f'mfe_short_h{h}', f'mae_short_h{h}'])

                if len(downtrend_valid) > 0:
                    downtrend_valid['success'] = downtrend_valid[f'mfe_short_h{h}'] >= 12

                    success_rate = downtrend_valid['success'].mean() * 100
                    avg_bounce_mfe = downtrend_valid.loc[downtrend_valid['success'], f'mfe_short_h{h}'].mean()
                    avg_fail_mae = downtrend_valid.loc[~downtrend_valid['success'], f'mae_short_h{h}'].mean()
                    median_bounce = downtrend_valid.loc[downtrend_valid['success'], f'mfe_short_h{h}'].median()

                    results.append({
                        'dataset': dataset_name,
                        'ema_period': ema_period,
                        'trend_direction': 'DOWNTREND',
                        'horizon': h,
                        'total_touches': len(downtrend_valid),
                        'success_rate': success_rate,
                        'avg_bounce_mfe': avg_bounce_mfe,
                        'avg_fail_mae': avg_fail_mae,
                        'median_bounce': median_bounce
                    })

    return pd.DataFrame(results)

# =============================================================================
# RUN ANALYSIS ON BOTH DATASETS
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING ANALYSIS ON TRAIN AND TEST DATASETS")
print("=" * 80)

train_results = analyze_ema_bounces(train, "TRAIN (2020-2023)")
test_results = analyze_ema_bounces(test, "TEST (2024-2025)")

# Combine results
all_results = pd.concat([train_results, test_results], ignore_index=True)

# =============================================================================
# SAVE DETAILED RESULTS
# =============================================================================
print(f"\n[6/7] Saving results...")

output_dir = Path("experiments/ema")
output_dir.mkdir(parents=True, exist_ok=True)

detailed_path = output_dir / "ema_bounce_validation_detailed.csv"
all_results.to_csv(detailed_path, index=False)
print(f"[OK] Saved detailed results: {detailed_path}")
print(f"  Total rows: {len(all_results)}")

# =============================================================================
# CREATE VALIDATION COMPARISON
# =============================================================================
print(f"[7/7] Creating validation comparison...")

validation_rows = []

for ema in EMA_PERIODS:
    for h in HORIZONS:
        # Get train and test data for this EMA+horizon
        train_data = train_results[
            (train_results['ema_period'] == ema) &
            (train_results['horizon'] == h)
        ]
        test_data = test_results[
            (test_results['ema_period'] == ema) &
            (test_results['horizon'] == h)
        ]

        if len(train_data) > 0 and len(test_data) > 0:
            # Overall metrics
            train_success = train_data['success_rate'].mean()
            test_success = test_data['success_rate'].mean()
            train_bounce = train_data['avg_bounce_mfe'].mean()
            test_bounce = test_data['avg_bounce_mfe'].mean()

            # Difference
            success_diff = test_success - train_success
            bounce_diff = test_bounce - train_bounce

            # Validation status (within 5% is considered validated)
            validated = abs(success_diff) <= 5.0

            # Split by trend
            train_up = train_data[train_data['trend_direction'] == 'UPTREND']
            test_up = test_data[test_data['trend_direction'] == 'UPTREND']
            train_down = train_data[train_data['trend_direction'] == 'DOWNTREND']
            test_down = test_data[test_data['trend_direction'] == 'DOWNTREND']

            validation_rows.append({
                'ema_period': ema,
                'horizon': h,
                'train_success': train_success,
                'test_success': test_success,
                'success_diff': success_diff,
                'train_bounce': train_bounce,
                'test_bounce': test_bounce,
                'bounce_diff': bounce_diff,
                'validated': validated,
                'train_uptrend_success': train_up['success_rate'].mean() if len(train_up) > 0 else 0,
                'test_uptrend_success': test_up['success_rate'].mean() if len(test_up) > 0 else 0,
                'train_downtrend_success': train_down['success_rate'].mean() if len(train_down) > 0 else 0,
                'test_downtrend_success': test_down['success_rate'].mean() if len(test_down) > 0 else 0,
                'train_total_touches': train_data['total_touches'].sum(),
                'test_total_touches': test_data['total_touches'].sum()
            })

validation_df = pd.DataFrame(validation_rows).sort_values(['ema_period', 'horizon'])

validation_path = output_dir / "ema_bounce_validation_comparison.csv"
validation_df.to_csv(validation_path, index=False)
print(f"[OK] Saved validation comparison: {validation_path}")

# =============================================================================
# SUMMARY - BEST VALIDATED PERFORMERS
# =============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY - OOS PERFORMANCE")
print("=" * 80)

# Filter only validated and H=10 (best horizon)
validated_h10 = validation_df[
    (validation_df['validated'] == True) &
    (validation_df['horizon'] == 10)
].sort_values('test_success', ascending=False)

print("\nBEST VALIDATED EMAs @ H=10 (2.5 hours):")
print("-" * 80)
for idx, row in validated_h10.iterrows():
    print(f"\nEMA{int(row['ema_period'])}:")
    print(f"  Train success: {row['train_success']:.1f}%")
    print(f"  Test success:  {row['test_success']:.1f}%")
    print(f"  Difference:    {row['success_diff']:+.1f}%")
    print(f"  Train bounce:  {row['train_bounce']:.1f} bps")
    print(f"  Test bounce:   {row['test_bounce']:.1f} bps")
    print(f"  Train touches: {int(row['train_total_touches']):,}")
    print(f"  Test touches:  {int(row['test_total_touches']):,}")
    print(f"  Status: {'VALIDATED' if row['validated'] else 'FAILED'}")

# Count validated
total_combos = len(validation_df)
validated_combos = validation_df['validated'].sum()
validation_rate = validated_combos / total_combos * 100

print("\n" + "=" * 80)
print(f"OVERALL VALIDATION RATE: {validated_combos}/{total_combos} ({validation_rate:.1f}%)")
print("=" * 80)

# Top 3 by test success @ H=10
print("\nTOP 3 PERFORMERS (Test data, H=10):")
print("-" * 80)
top3 = validated_h10.head(3)
for i, (idx, row) in enumerate(top3.iterrows(), 1):
    print(f"\n{i}. EMA{int(row['ema_period'])}: {row['test_success']:.1f}% success, {row['test_bounce']:.1f} bps bounce")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
