"""
EMA Approach Direction Analysis - CORRECT LOGIC

Tests which EMA works best as dynamic support/resistance based on APPROACH DIRECTION:
- LONG: Price was ABOVE EMA, pulled back to touch it (support test)
- SHORT: Price was BELOW EMA, rallied to touch it (resistance test)

EMAs tested: 7, 9, 20, 25, 50, 100, 200
Timeframe: 15-min
Horizons: H=3, 5, 10 (45min, 75min, 2.5 hours)
Data split: 2020-2023 train, 2024-2025 test

Run: python scripts/debug/ema_approach_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EMA APPROACH DIRECTION ANALYSIS - CORRECT LOGIC")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_PERIODS = [7, 9, 20, 25, 50, 100, 200]
HORIZONS = [3, 5, 10]
LOOKBACK = 1  # How many bars back to check approach direction
TOUCH_THRESHOLD_MIN = 0.0
TOUCH_THRESHOLD_MAX = 0.2
TARGET_BPS = 12

TRAIN_START = "2020-01-01"
TRAIN_END = "2024-01-01"
TEST_START = "2024-01-01"

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/4] Loading data...")
ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)

# Resample to 15-minute
ohlcv = ohlcv_1m.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Split train/test
train = ohlcv[(ohlcv.index >= TRAIN_START) & (ohlcv.index < TRAIN_END)].copy()
test = ohlcv[ohlcv.index >= TEST_START].copy()

print(f"Train data: {len(train):,} candles ({train.index[0]} to {train.index[-1]})")
print(f"Test data: {len(test):,} candles ({test.index[0]} to {test.index[-1]})")

# =============================================================================
# CALCULATE FORWARD MFE
# =============================================================================
@njit
def calculate_forward_mfe(highs, lows, close_prices, horizon):
    n = len(close_prices)
    mfe_long = np.full(n, np.nan)
    mfe_short = np.full(n, np.nan)

    for i in range(n - horizon):
        entry = close_prices[i]
        future_highs = highs[i+1:i+1+horizon]
        future_lows = lows[i+1:i+1+horizon]

        # LONG
        max_high = np.max(future_highs)
        mfe_long[i] = (max_high - entry) / entry * 10000

        # SHORT
        min_low = np.min(future_lows)
        mfe_short[i] = (entry - min_low) / entry * 10000

    return mfe_long, mfe_short

# =============================================================================
# ANALYZE EACH EMA
# =============================================================================
print("\n[2/4] Analyzing EMAs with approach direction logic...")

results = []

for ema_period in EMA_PERIODS:
    print(f"\n  Analyzing EMA{ema_period}...")

    for horizon in HORIZONS:
        # --- TRAIN DATA ---
        train_temp = train.copy()
        train_temp[f'ema{ema_period}'] = train_temp['close'].ewm(span=ema_period, adjust=False).mean()

        # Touch detection
        train_temp['dist_from_ema'] = abs(train_temp['close'] - train_temp[f'ema{ema_period}']) / train_temp[f'ema{ema_period}'] * 100
        train_temp['touch'] = (train_temp['dist_from_ema'] >= TOUCH_THRESHOLD_MIN) & (train_temp['dist_from_ema'] <= TOUCH_THRESHOLD_MAX)

        # Check approach direction (CORRECT LOGIC)
        train_temp['was_above_ema'] = train_temp['close'].shift(LOOKBACK) > train_temp[f'ema{ema_period}'].shift(LOOKBACK)
        train_temp['was_below_ema'] = train_temp['close'].shift(LOOKBACK) < train_temp[f'ema{ema_period}'].shift(LOOKBACK)

        # Calculate MFE
        mfe_long, mfe_short = calculate_forward_mfe(
            train_temp['high'].values,
            train_temp['low'].values,
            train_temp['close'].values,
            horizon
        )
        train_temp['mfe_long'] = mfe_long
        train_temp['mfe_short'] = mfe_short

        # Filter: touch + approach direction
        long_entries = train_temp[train_temp['touch'] & train_temp['was_above_ema']].dropna(subset=['mfe_long'])
        short_entries = train_temp[train_temp['touch'] & train_temp['was_below_ema']].dropna(subset=['mfe_short'])

        # Success rate
        long_success = (long_entries['mfe_long'] >= TARGET_BPS).mean() * 100 if len(long_entries) > 0 else 0
        short_success = (short_entries['mfe_short'] >= TARGET_BPS).mean() * 100 if len(short_entries) > 0 else 0
        overall_train = ((long_entries['mfe_long'] >= TARGET_BPS).sum() + (short_entries['mfe_short'] >= TARGET_BPS).sum()) / (len(long_entries) + len(short_entries)) * 100 if (len(long_entries) + len(short_entries)) > 0 else 0

        # --- TEST DATA ---
        test_temp = test.copy()
        test_temp[f'ema{ema_period}'] = test_temp['close'].ewm(span=ema_period, adjust=False).mean()

        # Touch detection
        test_temp['dist_from_ema'] = abs(test_temp['close'] - test_temp[f'ema{ema_period}']) / test_temp[f'ema{ema_period}'] * 100
        test_temp['touch'] = (test_temp['dist_from_ema'] >= TOUCH_THRESHOLD_MIN) & (test_temp['dist_from_ema'] <= TOUCH_THRESHOLD_MAX)

        # Check approach direction (CORRECT LOGIC)
        test_temp['was_above_ema'] = test_temp['close'].shift(LOOKBACK) > test_temp[f'ema{ema_period}'].shift(LOOKBACK)
        test_temp['was_below_ema'] = test_temp['close'].shift(LOOKBACK) < test_temp[f'ema{ema_period}'].shift(LOOKBACK)

        # Calculate MFE
        mfe_long, mfe_short = calculate_forward_mfe(
            test_temp['high'].values,
            test_temp['low'].values,
            test_temp['close'].values,
            horizon
        )
        test_temp['mfe_long'] = mfe_long
        test_temp['mfe_short'] = mfe_short

        # Filter: touch + approach direction
        long_entries_test = test_temp[test_temp['touch'] & test_temp['was_above_ema']].dropna(subset=['mfe_long'])
        short_entries_test = test_temp[test_temp['touch'] & test_temp['was_below_ema']].dropna(subset=['mfe_short'])

        # Success rate
        long_success_test = (long_entries_test['mfe_long'] >= TARGET_BPS).mean() * 100 if len(long_entries_test) > 0 else 0
        short_success_test = (short_entries_test['mfe_short'] >= TARGET_BPS).mean() * 100 if len(short_entries_test) > 0 else 0
        overall_test = ((long_entries_test['mfe_long'] >= TARGET_BPS).sum() + (short_entries_test['mfe_short'] >= TARGET_BPS).sum()) / (len(long_entries_test) + len(short_entries_test)) * 100 if (len(long_entries_test) + len(short_entries_test)) > 0 else 0

        # Average MFE
        avg_mfe_long_train = long_entries['mfe_long'].mean() if len(long_entries) > 0 else 0
        avg_mfe_short_train = short_entries['mfe_short'].mean() if len(short_entries) > 0 else 0
        avg_mfe_long_test = long_entries_test['mfe_long'].mean() if len(long_entries_test) > 0 else 0
        avg_mfe_short_test = short_entries_test['mfe_short'].mean() if len(short_entries_test) > 0 else 0

        # Save results
        results.append({
            'ema_period': ema_period,
            'horizon': horizon,
            'dataset': 'train',
            'total_long': len(long_entries),
            'total_short': len(short_entries),
            'long_success_rate': long_success,
            'short_success_rate': short_success,
            'overall_success_rate': overall_train,
            'avg_mfe_long': avg_mfe_long_train,
            'avg_mfe_short': avg_mfe_short_train
        })

        results.append({
            'ema_period': ema_period,
            'horizon': horizon,
            'dataset': 'test',
            'total_long': len(long_entries_test),
            'total_short': len(short_entries_test),
            'long_success_rate': long_success_test,
            'short_success_rate': short_success_test,
            'overall_success_rate': overall_test,
            'avg_mfe_long': avg_mfe_long_test,
            'avg_mfe_short': avg_mfe_short_test
        })

print("\n[OK] Analysis complete!")

# =============================================================================
# SAVE DETAILED RESULTS
# =============================================================================
print("\n[3/4] Saving results...")

results_df = pd.DataFrame(results)

output_dir = Path("experiments/ema")
output_dir.mkdir(parents=True, exist_ok=True)

detailed_path = output_dir / "ema_approach_direction_analysis_detailed.csv"
results_df.to_csv(detailed_path, index=False)
print(f"[OK] Saved detailed results: {detailed_path}")

# =============================================================================
# CREATE SUMMARY
# =============================================================================
print("\n[4/4] Creating summary...")

summary_rows = []

for ema in EMA_PERIODS:
    for horizon in HORIZONS:
        train_row = results_df[(results_df['ema_period'] == ema) & (results_df['horizon'] == horizon) & (results_df['dataset'] == 'train')].iloc[0]
        test_row = results_df[(results_df['ema_period'] == ema) & (results_df['horizon'] == horizon) & (results_df['dataset'] == 'test')].iloc[0]

        diff = test_row['overall_success_rate'] - train_row['overall_success_rate']

        summary_rows.append({
            'ema_period': ema,
            'horizon': horizon,
            'train_success': train_row['overall_success_rate'],
            'test_success': test_row['overall_success_rate'],
            'difference': diff,
            'train_long': train_row['total_long'],
            'train_short': train_row['total_short'],
            'test_long': test_row['total_long'],
            'test_short': test_row['total_short'],
            'validated': abs(diff) <= 5.0  # Within 5pp = validated
        })

summary_df = pd.DataFrame(summary_rows).sort_values('test_success', ascending=False)

summary_path = output_dir / "ema_approach_direction_analysis_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"[OK] Saved summary: {summary_path}")

# =============================================================================
# DISPLAY RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS - EMA APPROACH DIRECTION ANALYSIS")
print("=" * 80)
print("\nLogic: LONG when price from above, SHORT when price from below")
print(f"Target: {TARGET_BPS} bps")
print(f"Touch threshold: {TOUCH_THRESHOLD_MIN}-{TOUCH_THRESHOLD_MAX}%")
print(f"Lookback: {LOOKBACK} bar")

print("\n--- TOP 5 PERFORMERS (TEST DATA) ---")
top5 = summary_df.head(5)
print(top5[['ema_period', 'horizon', 'train_success', 'test_success', 'difference', 'validated']].to_string(index=False))

print("\n--- VALIDATION SUMMARY ---")
validated = summary_df[summary_df['validated']]
print(f"Validated combinations: {len(validated)} / {len(summary_df)} ({len(validated)/len(summary_df)*100:.1f}%)")

best = summary_df.iloc[0]
print(f"\n--- BEST PERFORMER ---")
print(f"EMA{int(best['ema_period'])} @ H={int(best['horizon'])}")
print(f"  Train: {best['train_success']:.1f}%")
print(f"  Test: {best['test_success']:.1f}%")
print(f"  Difference: {best['difference']:+.1f} pp")
print(f"  Validated: {'YES' if best['validated'] else 'NO'}")
print(f"  Test entries: {int(best['test_long'])} LONG, {int(best['test_short'])} SHORT")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
