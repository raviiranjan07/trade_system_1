"""
EXP-003: RSI Failure Deep Dive

Objective: Find the ACTUAL cause of RSI overbought failures.
We have 11 failures in OOS (2024-2025). Let's examine each one.

What we're looking for:
- Volume patterns
- RSI intensity (how high?)
- Time patterns (hour, day of week)
- Volatility
- Price structure
- Momentum before signal
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
RESULTS_PATH = Path("experiments/rsi/EXP-003")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 80
HORIZON = 10

print("="*70)
print("RSI FAILURE DEEP DIVE - OVERBOUGHT SIGNALS")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)
print(f"Total bars: {len(df):,}")

# =============================================================================
# CALCULATE INDICATORS
# =============================================================================

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

print("Calculating indicators...")

# RSI
df['rsi'] = calculate_rsi(df, RSI_PERIOD)

# Moving averages
df['sma_20'] = df['close'].rolling(20).mean()
df['sma_50'] = df['close'].rolling(50).mean()

# Volume metrics
df['volume_sma_20'] = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume'] / df['volume_sma_20']

# Volatility (ATR-like)
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
df['atr_14'] = df['tr'].rolling(14).mean()
df['volatility_pct'] = df['atr_14'] / df['close'] * 100

# Momentum (price change over last N bars)
df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 10000  # in bps
df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 10000

# Distance from MAs
df['dist_sma20_pct'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
df['dist_sma50_pct'] = (df['close'] - df['sma_50']) / df['sma_50'] * 100

# RSI overbought cross
df['overbought'] = df['rsi'] > RSI_OVERBOUGHT
df['rsi_overbought_cross'] = (df['overbought'] == True) & (df['overbought'].shift(1) == False)

# Time features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek  # 0=Monday, 6=Sunday

# =============================================================================
# FIND FAILURES IN OOS (2024-2025)
# =============================================================================

print("\nFinding overbought failures in Out-of-Sample (2024-2025)...")

oos_data = df['2024-01-01':'2025-12-30'].copy()
print(f"OOS bars: {len(oos_data):,}")

# Find all overbought signals
overbought_signals = oos_data[oos_data['rsi_overbought_cross'] == True].copy()
print(f"Total overbought signals: {len(overbought_signals)}")

# Analyze each signal
failures = []
successes = []

for idx in overbought_signals.index:
    idx_pos = oos_data.index.get_loc(idx)

    if idx_pos + HORIZON >= len(oos_data):
        continue

    entry_price = oos_data.iloc[idx_pos]['close']
    future_data = oos_data.iloc[idx_pos + 1 : idx_pos + 1 + HORIZON]

    if len(future_data) == 0:
        continue

    # For SHORT: we want price to go DOWN
    min_price = future_data['low'].min()
    max_price = future_data['high'].max()

    mfe = (entry_price - min_price) / entry_price * 10000  # profit if it dropped
    mae = (max_price - entry_price) / entry_price * 10000  # loss if it went up

    dropped = mfe > 0  # success if price dropped at all

    signal_data = {
        'timestamp': idx,
        'entry_price': entry_price,
        'rsi_value': oos_data.loc[idx, 'rsi'],
        'mfe_bps': mfe,
        'mae_bps': mae,
        'volume_ratio': oos_data.loc[idx, 'volume_ratio'],
        'volatility_pct': oos_data.loc[idx, 'volatility_pct'],
        'momentum_5_bps': oos_data.loc[idx, 'momentum_5'],
        'momentum_10_bps': oos_data.loc[idx, 'momentum_10'],
        'dist_sma20_pct': oos_data.loc[idx, 'dist_sma20_pct'],
        'dist_sma50_pct': oos_data.loc[idx, 'dist_sma50_pct'],
        'hour': oos_data.loc[idx, 'hour'],
        'dayofweek': oos_data.loc[idx, 'dayofweek'],
        'success': dropped
    }

    if dropped:
        successes.append(signal_data)
    else:
        failures.append(signal_data)

print(f"\nSuccesses: {len(successes)}")
print(f"Failures: {len(failures)}")

# =============================================================================
# ANALYZE FAILURES
# =============================================================================

print("\n" + "="*70)
print("FAILURE CASES - DETAILED ANALYSIS")
print("="*70)

if failures:
    failures_df = pd.DataFrame(failures)
    successes_df = pd.DataFrame(successes)

    print("\n--- EACH FAILURE CASE ---")
    for i, row in failures_df.iterrows():
        print(f"\nFailure #{i+1}: {row['timestamp']}")
        print(f"  Price: ${row['entry_price']:,.0f}")
        print(f"  RSI: {row['rsi_value']:.1f}")
        print(f"  MAE (went against): {row['mae_bps']:.1f} bps")
        print(f"  Volume ratio: {row['volume_ratio']:.2f}x avg")
        print(f"  Volatility: {row['volatility_pct']:.3f}%")
        print(f"  5-bar momentum: {row['momentum_5_bps']:.0f} bps")
        print(f"  10-bar momentum: {row['momentum_10_bps']:.0f} bps")
        print(f"  Distance from SMA20: {row['dist_sma20_pct']:.2f}%")
        print(f"  Distance from SMA50: {row['dist_sma50_pct']:.2f}%")
        print(f"  Hour: {int(row['hour'])}:00 UTC")
        print(f"  Day: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][int(row['dayofweek'])]}")

    # =============================================================================
    # COMPARE FAILURES vs SUCCESSES
    # =============================================================================

    print("\n" + "="*70)
    print("FAILURES vs SUCCESSES - STATISTICAL COMPARISON")
    print("="*70)

    comparison_metrics = [
        ('rsi_value', 'RSI Value'),
        ('volume_ratio', 'Volume Ratio'),
        ('volatility_pct', 'Volatility %'),
        ('momentum_5_bps', '5-bar Momentum (bps)'),
        ('momentum_10_bps', '10-bar Momentum (bps)'),
        ('dist_sma20_pct', 'Dist from SMA20 %'),
        ('dist_sma50_pct', 'Dist from SMA50 %'),
    ]

    print(f"\n{'Metric':<25} {'Failures Mean':<15} {'Success Mean':<15} {'Difference':<15}")
    print("-"*70)

    insights = []

    for col, name in comparison_metrics:
        fail_mean = failures_df[col].mean()
        succ_mean = successes_df[col].mean()
        diff = fail_mean - succ_mean
        diff_pct = (diff / abs(succ_mean) * 100) if succ_mean != 0 else 0

        print(f"{name:<25} {fail_mean:<15.2f} {succ_mean:<15.2f} {diff:+.2f} ({diff_pct:+.1f}%)")

        if abs(diff_pct) > 20:  # Notable difference
            insights.append((name, diff_pct, fail_mean, succ_mean))

    # =============================================================================
    # TIME PATTERNS
    # =============================================================================

    print("\n" + "="*70)
    print("TIME PATTERNS")
    print("="*70)

    print("\n--- Hour Distribution ---")
    print(f"{'Hour':<10} {'Failures':<12} {'Successes':<12} {'Failure Rate':<15}")
    print("-"*50)

    for hour in range(24):
        fail_count = len(failures_df[failures_df['hour'] == hour])
        succ_count = len(successes_df[successes_df['hour'] == hour])
        total = fail_count + succ_count
        fail_rate = (fail_count / total * 100) if total > 0 else 0

        if total > 0:
            marker = " <-- HIGH" if fail_rate > 5 else ""
            print(f"{hour:02d}:00{'':<5} {fail_count:<12} {succ_count:<12} {fail_rate:.1f}%{marker}")

    print("\n--- Day of Week Distribution ---")
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print(f"{'Day':<12} {'Failures':<12} {'Successes':<12} {'Failure Rate':<15}")
    print("-"*50)

    for dow in range(7):
        fail_count = len(failures_df[failures_df['dayofweek'] == dow])
        succ_count = len(successes_df[successes_df['dayofweek'] == dow])
        total = fail_count + succ_count
        fail_rate = (fail_count / total * 100) if total > 0 else 0

        if total > 0:
            marker = " <-- HIGH" if fail_rate > 5 else ""
            print(f"{days[dow]:<12} {fail_count:<12} {succ_count:<12} {fail_rate:.1f}%{marker}")

    # =============================================================================
    # KEY INSIGHTS
    # =============================================================================

    print("\n" + "="*70)
    print("KEY INSIGHTS FROM FAILURE ANALYSIS")
    print("="*70)

    if insights:
        print("\nMetrics with >20% difference between failures and successes:")
        for name, diff_pct, fail_val, succ_val in insights:
            direction = "HIGHER" if diff_pct > 0 else "LOWER"
            print(f"  - {name}: Failures have {abs(diff_pct):.0f}% {direction} values")
            print(f"    (Failures: {fail_val:.2f}, Successes: {succ_val:.2f})")

    # RSI intensity check
    print(f"\n--- RSI Intensity Check ---")
    print(f"Failures - RSI range: {failures_df['rsi_value'].min():.1f} to {failures_df['rsi_value'].max():.1f}")
    print(f"Successes - RSI range: {successes_df['rsi_value'].min():.1f} to {successes_df['rsi_value'].max():.1f}")

    high_rsi_failures = len(failures_df[failures_df['rsi_value'] > 85])
    high_rsi_successes = len(successes_df[successes_df['rsi_value'] > 85])
    total_high_rsi = high_rsi_failures + high_rsi_successes
    if total_high_rsi > 0:
        print(f"\nRSI > 85: {high_rsi_failures}/{total_high_rsi} failures ({high_rsi_failures/total_high_rsi*100:.1f}% failure rate)")

    # Momentum check
    print(f"\n--- Momentum Check ---")
    strong_momentum_threshold = 100  # 100 bps = 1%
    strong_mom_failures = len(failures_df[failures_df['momentum_10_bps'] > strong_momentum_threshold])
    strong_mom_successes = len(successes_df[successes_df['momentum_10_bps'] > strong_momentum_threshold])
    total_strong = strong_mom_failures + strong_mom_successes
    if total_strong > 0:
        print(f"10-bar momentum > {strong_momentum_threshold} bps: {strong_mom_failures}/{total_strong} failures ({strong_mom_failures/total_strong*100:.1f}% failure rate)")

    # Volume check
    print(f"\n--- Volume Check ---")
    high_vol_failures = len(failures_df[failures_df['volume_ratio'] > 1.5])
    high_vol_successes = len(successes_df[successes_df['volume_ratio'] > 1.5])
    total_high_vol = high_vol_failures + high_vol_successes
    if total_high_vol > 0:
        print(f"Volume > 1.5x avg: {high_vol_failures}/{total_high_vol} failures ({high_vol_failures/total_high_vol*100:.1f}% failure rate)")

    # Save results
    failures_df.to_csv(RESULTS_PATH / 'failures_detailed.csv', index=False)
    successes_df.to_csv(RESULTS_PATH / 'successes_sample.csv', index=False)
    print(f"\nSaved detailed data to: {RESULTS_PATH}")

else:
    print("No failures found to analyze.")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
