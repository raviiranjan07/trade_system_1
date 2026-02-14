"""
ANALYSIS-12 TRAIN/TEST VALIDATION: EMA Bounce Magnitude

Train: 2020-2023
Test: 2024

Tests if EMA bounce findings hold out-of-sample.

Run: python scripts/debug/analysis12_train_test.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
TIMEFRAMES_MINUTES = [5, 15, 60]  # Focus on key timeframes
EMA_PERIODS = [9, 50, 200]  # Focus on key EMAs
TOUCH_THRESHOLD_BPS = 15
APPROACH_BARS = 5
BOUNCE_CONFIRM_BPS = 10
H_BARS = 10  # Fixed horizon for comparison

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS-12 TRAIN/TEST: EMA BOUNCE MAGNITUDE")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv_1m):,} 1-minute candles")

# Split train/test
train_1m = ohlcv_1m[ohlcv_1m.index <= "2023-12-31"].copy()
test_1m = ohlcv_1m[(ohlcv_1m.index >= "2024-01-01") & (ohlcv_1m.index <= "2024-12-31")].copy()

print(f"Train data (2020-2023): {len(train_1m):,} candles")
print(f"Test data (2024): {len(test_1m):,} candles")


# =============================================================================
# RESAMPLE
# =============================================================================
def resample_ohlcv(df, minutes):
    tf_str = f'{minutes}min'
    return df.resample(tf_str).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================
def analyze_ema_bounce(df, ema_period, H_bars):
    """Analyze EMA bounce for given parameters."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    ema = df['close'].ewm(span=ema_period, adjust=False).mean().values
    n = len(df)

    touch_pct = TOUCH_THRESHOLD_BPS / 10000
    confirm_pct = BOUNCE_CONFIRM_BPS / 10000

    # Support test
    support_touches = 0
    support_bounces = 0
    bounce_magnitudes = []
    bounce_corrections = []

    # Resistance test
    resist_touches = 0
    resist_rejects = 0
    reject_magnitudes = []
    reject_corrections = []

    valid_start = max(ema_period, APPROACH_BARS) + 10
    max_end = n - H_bars - 5

    for i in range(valid_start, max_end):
        current_close = close[i]
        current_ema = ema[i]

        distance_pct = abs(current_close - current_ema) / current_ema
        if distance_pct > touch_pct:
            continue

        above_count = sum(1 for k in range(1, APPROACH_BARS + 1) if i-k >= 0 and close[i-k] > ema[i-k])
        below_count = sum(1 for k in range(1, APPROACH_BARS + 1) if i-k >= 0 and close[i-k] < ema[i-k])

        # SUPPORT TEST
        if above_count >= APPROACH_BARS * 0.6:
            support_touches += 1
            max_up = 0
            max_down = 0
            for j in range(1, H_bars + 1):
                if i + j >= n:
                    break
                up_move = (high[i + j] - current_close) / current_close * 10000
                down_move = (current_close - low[i + j]) / current_close * 10000
                max_up = max(max_up, up_move)
                max_down = max(max_down, down_move)

            if max_up >= confirm_pct * 10000:
                support_bounces += 1
                bounce_magnitudes.append(max_up)
                bounce_corrections.append(max_down)

        # RESISTANCE TEST
        elif below_count >= APPROACH_BARS * 0.6:
            resist_touches += 1
            max_down = 0
            max_up = 0
            for j in range(1, H_bars + 1):
                if i + j >= n:
                    break
                down_move = (current_close - low[i + j]) / current_close * 10000
                up_move = (high[i + j] - current_close) / current_close * 10000
                max_down = max(max_down, down_move)
                max_up = max(max_up, up_move)

            if max_down >= confirm_pct * 10000:
                resist_rejects += 1
                reject_magnitudes.append(max_down)
                reject_corrections.append(max_up)

    return {
        'support_touches': support_touches,
        'support_bounces': support_bounces,
        'support_rate': support_bounces / support_touches * 100 if support_touches > 0 else 0,
        'bounce_median': np.median(bounce_magnitudes) if bounce_magnitudes else 0,
        'bounce_corr': np.median(bounce_corrections) if bounce_corrections else 0,
        'resist_touches': resist_touches,
        'resist_rejects': resist_rejects,
        'resist_rate': resist_rejects / resist_touches * 100 if resist_touches > 0 else 0,
        'reject_median': np.median(reject_magnitudes) if reject_magnitudes else 0,
        'reject_corr': np.median(reject_corrections) if reject_corrections else 0,
    }


# =============================================================================
# RUN TRAIN/TEST
# =============================================================================
print("\n" + "=" * 80)
print("TRAIN vs TEST COMPARISON")
print("=" * 80)

results = []

for tf_min in TIMEFRAMES_MINUTES:
    print(f"\n### {tf_min}-MINUTE TIMEFRAME")

    train_df = resample_ohlcv(train_1m, tf_min)
    test_df = resample_ohlcv(test_1m, tf_min)

    print(f"Train: {len(train_df):,} candles | Test: {len(test_df):,} candles")

    print("\n**SUPPORT (Bounce UP from EMA)**")
    print("| EMA | Train Rate | Test Rate | Diff | Train Bounce | Test Bounce |")
    print("|-----|------------|-----------|------|--------------|-------------|")

    for ema in EMA_PERIODS:
        train_r = analyze_ema_bounce(train_df, ema, H_BARS)
        test_r = analyze_ema_bounce(test_df, ema, H_BARS)

        if train_r['support_touches'] >= 50 and test_r['support_touches'] >= 20:
            diff = test_r['support_rate'] - train_r['support_rate']
            sign = "+" if diff > 0 else ""
            print(f"| EMA{ema} | {train_r['support_rate']:.1f}% | {test_r['support_rate']:.1f}% | "
                  f"{sign}{diff:.1f}% | {train_r['bounce_median']:.1f}bp | {test_r['bounce_median']:.1f}bp |")

            results.append({
                'tf': tf_min,
                'ema': ema,
                'type': 'support',
                'train_rate': train_r['support_rate'],
                'test_rate': test_r['support_rate'],
                'train_bounce': train_r['bounce_median'],
                'test_bounce': test_r['bounce_median'],
            })

    print("\n**RESISTANCE (Reject DOWN from EMA)**")
    print("| EMA | Train Rate | Test Rate | Diff | Train Drop | Test Drop |")
    print("|-----|------------|-----------|------|------------|-----------|")

    for ema in EMA_PERIODS:
        train_r = analyze_ema_bounce(train_df, ema, H_BARS)
        test_r = analyze_ema_bounce(test_df, ema, H_BARS)

        if train_r['resist_touches'] >= 50 and test_r['resist_touches'] >= 20:
            diff = test_r['resist_rate'] - train_r['resist_rate']
            sign = "+" if diff > 0 else ""
            print(f"| EMA{ema} | {train_r['resist_rate']:.1f}% | {test_r['resist_rate']:.1f}% | "
                  f"{sign}{diff:.1f}% | {train_r['reject_median']:.1f}bp | {test_r['reject_median']:.1f}bp |")

            results.append({
                'tf': tf_min,
                'ema': ema,
                'type': 'resistance',
                'train_rate': train_r['resist_rate'],
                'test_rate': test_r['resist_rate'],
                'train_bounce': train_r['reject_median'],
                'test_bounce': test_r['reject_median'],
            })


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: TRAIN vs TEST")
print("=" * 80)

if results:
    avg_train_rate = np.mean([r['train_rate'] for r in results])
    avg_test_rate = np.mean([r['test_rate'] for r in results])
    avg_train_bounce = np.mean([r['train_bounce'] for r in results])
    avg_test_bounce = np.mean([r['test_bounce'] for r in results])

    print(f"""
OVERALL COMPARISON:
                    TRAIN (2020-2023)    TEST (2024)       CHANGE
Success Rate:       {avg_train_rate:.1f}%               {avg_test_rate:.1f}%             {avg_test_rate - avg_train_rate:+.1f}%
Bounce/Drop:        {avg_train_bounce:.1f}bp              {avg_test_bounce:.1f}bp            {avg_test_bounce - avg_train_bounce:+.1f}bp

INTERPRETATION:
- If Test Rate close to Train Rate: Pattern is STABLE
- If Test Rate < Train Rate by >5%: Pattern may be DECAYING
- If Test Bounce < Train Bounce: Market moves LESS than expected
""")

    # Check stability
    stable_count = sum(1 for r in results if abs(r['test_rate'] - r['train_rate']) < 5)
    decay_count = sum(1 for r in results if r['test_rate'] < r['train_rate'] - 5)

    print(f"Stability Check:")
    print(f"  - Stable patterns (diff < 5%): {stable_count}/{len(results)}")
    print(f"  - Decaying patterns (test < train by >5%): {decay_count}/{len(results)}")

    if stable_count >= len(results) * 0.7:
        print("\nVERDICT: EMA bounce pattern is STABLE in 2024 out-of-sample test")
    elif decay_count >= len(results) * 0.5:
        print("\nVERDICT: EMA bounce pattern is DECAYING in 2024")
    else:
        print("\nVERDICT: Mixed results - some patterns stable, some decaying")
