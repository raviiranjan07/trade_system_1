"""
EMA Bounce Analysis - 15min Timeframe

Tests which EMAs act as the best dynamic support/resistance.

Tests:
- In UPTREND: When price touches EMA from above → Does it bounce UP?
- In DOWNTREND: When price touches EMA from below → Does it bounce DOWN?

EMAs tested: 7, 9, 20, 25, 50, 100, 200
Touch threshold: 0-0.2% from EMA
Horizons: H=3, 5, 10 (45min, 75min, 2.5hrs)

Run: python scripts/debug/ema_bounce_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EMA BOUNCE ANALYSIS - 15min Timeframe")
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
print("\n[1/6] Loading data...")
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

# =============================================================================
# CALCULATE EMAs
# =============================================================================
print("\n[2/6] Calculating EMAs...")
for period in EMA_PERIODS:
    ohlcv[f'ema{period}'] = ohlcv['close'].ewm(span=period, adjust=False).mean()
print(f"Calculated {len(EMA_PERIODS)} EMAs: {EMA_PERIODS}")

# =============================================================================
# DEFINE TREND
# =============================================================================
print("\n[3/6] Detecting trends...")
# Use EMA50 vs EMA200 for trend direction
ohlcv['trend'] = 0
ohlcv.loc[ohlcv['ema50'] > ohlcv['ema200'], 'trend'] = 1  # Uptrend
ohlcv.loc[ohlcv['ema50'] < ohlcv['ema200'], 'trend'] = -1  # Downtrend

uptrend_bars = (ohlcv['trend'] == 1).sum()
downtrend_bars = (ohlcv['trend'] == -1).sum()
neutral_bars = (ohlcv['trend'] == 0).sum()
print(f"Uptrend bars: {uptrend_bars:,} ({uptrend_bars/len(ohlcv)*100:.1f}%)")
print(f"Downtrend bars: {downtrend_bars:,} ({downtrend_bars/len(ohlcv)*100:.1f}%)")
print(f"Neutral bars: {neutral_bars:,} ({neutral_bars/len(ohlcv)*100:.1f}%)")

# =============================================================================
# CALCULATE FORWARD MFE/MAE
# =============================================================================
print("\n[4/6] Calculating forward MFE/MAE...")

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

# Calculate for each horizon
for h in HORIZONS:
    print(f"  H={h}...")
    mfe_long, mae_long, mfe_short, mae_short = calculate_forward_mfe_mae(
        ohlcv['high'].values,
        ohlcv['low'].values,
        ohlcv['close'].values,
        h
    )
    ohlcv[f'mfe_long_h{h}'] = mfe_long
    ohlcv[f'mae_long_h{h}'] = mae_long
    ohlcv[f'mfe_short_h{h}'] = mfe_short
    ohlcv[f'mae_short_h{h}'] = mae_short

# =============================================================================
# DETECT TOUCH EVENTS
# =============================================================================
print("\n[5/6] Detecting EMA touch events...")

results = []

for ema_period in EMA_PERIODS:
    ema_col = f'ema{ema_period}'
    print(f"\n  EMA{ema_period}:")

    # Calculate distance from EMA
    ohlcv['dist_from_ema'] = abs(ohlcv['close'] - ohlcv[ema_col]) / ohlcv[ema_col] * 100

    # Touch = within 0-0.2% of EMA
    ohlcv['touch'] = (ohlcv['dist_from_ema'] >= TOUCH_THRESHOLD_MIN) & \
                     (ohlcv['dist_from_ema'] <= TOUCH_THRESHOLD_MAX)

    # Split by trend
    uptrend_touches = ohlcv[(ohlcv['trend'] == 1) & (ohlcv['touch'])].copy()
    downtrend_touches = ohlcv[(ohlcv['trend'] == -1) & (ohlcv['touch'])].copy()

    print(f"    Uptrend touches: {len(uptrend_touches):,}")
    print(f"    Downtrend touches: {len(downtrend_touches):,}")

    # Analyze for each horizon
    for h in HORIZONS:
        # UPTREND (Support test - expect bounce UP)
        if len(uptrend_touches) > 0:
            uptrend_valid = uptrend_touches.dropna(subset=[f'mfe_long_h{h}', f'mae_long_h{h}'])

            if len(uptrend_valid) > 0:
                # Success = MFE > threshold (12 bps minimum profit)
                uptrend_valid['success'] = uptrend_valid[f'mfe_long_h{h}'] >= 12

                success_rate = uptrend_valid['success'].mean() * 100
                avg_bounce_mfe = uptrend_valid.loc[uptrend_valid['success'], f'mfe_long_h{h}'].mean()
                avg_fail_mae = uptrend_valid.loc[~uptrend_valid['success'], f'mae_long_h{h}'].mean()
                median_bounce = uptrend_valid.loc[uptrend_valid['success'], f'mfe_long_h{h}'].median()
                median_fail = uptrend_valid.loc[~uptrend_valid['success'], f'mae_long_h{h}'].median()

                results.append({
                    'ema_period': ema_period,
                    'trend_direction': 'UPTREND',
                    'horizon': h,
                    'total_touches': len(uptrend_valid),
                    'success_rate': success_rate,
                    'avg_bounce_mfe': avg_bounce_mfe,
                    'avg_fail_mae': avg_fail_mae,
                    'median_bounce': median_bounce,
                    'median_fail': median_fail
                })

        # DOWNTREND (Resistance test - expect bounce DOWN)
        if len(downtrend_touches) > 0:
            downtrend_valid = downtrend_touches.dropna(subset=[f'mfe_short_h{h}', f'mae_short_h{h}'])

            if len(downtrend_valid) > 0:
                # Success = MFE > threshold (12 bps minimum profit)
                downtrend_valid['success'] = downtrend_valid[f'mfe_short_h{h}'] >= 12

                success_rate = downtrend_valid['success'].mean() * 100
                avg_bounce_mfe = downtrend_valid.loc[downtrend_valid['success'], f'mfe_short_h{h}'].mean()
                avg_fail_mae = downtrend_valid.loc[~downtrend_valid['success'], f'mae_short_h{h}'].mean()
                median_bounce = downtrend_valid.loc[downtrend_valid['success'], f'mfe_short_h{h}'].median()
                median_fail = downtrend_valid.loc[~downtrend_valid['success'], f'mae_short_h{h}'].median()

                results.append({
                    'ema_period': ema_period,
                    'trend_direction': 'DOWNTREND',
                    'horizon': h,
                    'total_touches': len(downtrend_valid),
                    'success_rate': success_rate,
                    'avg_bounce_mfe': avg_bounce_mfe,
                    'avg_fail_mae': avg_fail_mae,
                    'median_bounce': median_bounce,
                    'median_fail': median_fail
                })

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n[6/6] Saving results...")

results_df = pd.DataFrame(results)

# Detailed results
output_dir = Path("experiments/ema")
output_dir.mkdir(parents=True, exist_ok=True)

detailed_path = output_dir / "ema_bounce_analysis_15min_detailed.csv"
results_df.to_csv(detailed_path, index=False)
print(f"\n[OK] Saved detailed results: {detailed_path}")
print(f"  Total rows: {len(results_df)}")

# Summary - Best performers per EMA
summary_rows = []
for ema in EMA_PERIODS:
    ema_data = results_df[results_df['ema_period'] == ema]

    if len(ema_data) > 0:
        # Find best horizon (highest success rate)
        best_row = ema_data.loc[ema_data['success_rate'].idxmax()]

        # Overall stats
        avg_success = ema_data['success_rate'].mean()
        total_touches = ema_data['total_touches'].sum()
        avg_magnitude = ema_data['avg_bounce_mfe'].mean()

        # Split by trend
        uptrend_data = ema_data[ema_data['trend_direction'] == 'UPTREND']
        downtrend_data = ema_data[ema_data['trend_direction'] == 'DOWNTREND']

        summary_rows.append({
            'ema_period': ema,
            'best_horizon': int(best_row['horizon']),
            'best_success_rate': best_row['success_rate'],
            'avg_success_rate': avg_success,
            'avg_bounce_magnitude': avg_magnitude,
            'total_opportunities': total_touches,
            'uptrend_avg_success': uptrend_data['success_rate'].mean() if len(uptrend_data) > 0 else 0,
            'downtrend_avg_success': downtrend_data['success_rate'].mean() if len(downtrend_data) > 0 else 0
        })

summary_df = pd.DataFrame(summary_rows).sort_values('best_success_rate', ascending=False)

summary_path = output_dir / "ema_bounce_analysis_15min_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"[OK] Saved summary: {summary_path}")

# =============================================================================
# DISPLAY SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY - EMA BOUNCE PERFORMANCE")
print("=" * 80)
print(summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("TOP 3 PERFORMERS (by success rate):")
print("=" * 80)
top3 = summary_df.head(3)
for idx, row in top3.iterrows():
    print(f"\n{int(row['ema_period'])}. EMA{int(row['ema_period'])}:")
    print(f"   Best horizon: H={int(row['best_horizon'])} ({int(row['best_horizon'])*15} minutes)")
    print(f"   Best success rate: {row['best_success_rate']:.1f}%")
    print(f"   Avg success rate: {row['avg_success_rate']:.1f}%")
    print(f"   Avg bounce magnitude: {row['avg_bounce_magnitude']:.1f} bps")
    print(f"   Total opportunities: {int(row['total_opportunities']):,}")
    print(f"   Uptrend success: {row['uptrend_avg_success']:.1f}%")
    print(f"   Downtrend success: {row['downtrend_avg_success']:.1f}%")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
