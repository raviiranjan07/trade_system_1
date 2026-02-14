"""
EXP-002: RSI + MA Trend Filter
Tests if adding MA trend filter improves RSI signal quality

Hypothesis: RSI failures happen when trading AGAINST trend.
Adding MA filter should reduce failures and improve MFE/MAE ratio.

ENTRY RULES:
- RSI alone: RSI(14) < 20 (oversold) → GO LONG
- RSI + MA:  RSI(14) < 20 AND price > SMA(50) → GO LONG (with trend only)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
RESULTS_PATH = Path("experiments/rsi/EXP-002")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Configuration
RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
MA_PERIODS = [20, 50, 100]  # Test different MA periods as filter
HORIZON = 10

DATA_PERIODS = {
    'in_sample': ('2020-01-01', '2023-12-31'),
    'out_of_sample': ('2024-01-01', '2025-12-30')
}

print("="*70)
print("RSI + MA TREND FILTER ANALYSIS")
print("="*70)
print(f"RSI Config: RSI({RSI_PERIOD}) {RSI_OVERSOLD}/{RSI_OVERBOUGHT}")
print(f"MA Periods to test: {MA_PERIODS}")
print(f"Horizon: {HORIZON} bars")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)
print(f"Total bars: {len(df):,}")

# =============================================================================
# CALCULATIONS
# =============================================================================

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sma(data, period):
    return data['close'].rolling(window=period).mean()

# Calculate indicators
print("\nCalculating indicators...")
df['rsi'] = calculate_rsi(df, RSI_PERIOD)
for period in MA_PERIODS:
    df[f'sma_{period}'] = calculate_sma(df, period)

# Detect RSI signals
df['oversold'] = df['rsi'] < RSI_OVERSOLD
df['overbought'] = df['rsi'] > RSI_OVERBOUGHT
df['rsi_oversold_cross'] = (df['oversold'] == True) & (df['oversold'].shift(1) == False)
df['rsi_overbought_cross'] = (df['overbought'] == True) & (df['overbought'].shift(1) == False)

# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================

def analyze_signals(data, signal_mask, signal_type='oversold', description=''):
    """
    Analyze performance of signals
    """
    signals = data[signal_mask].copy()

    results = {
        'description': description,
        'signal_type': signal_type,
        'total_signals': len(signals)
    }

    if len(signals) == 0:
        return results

    bounces = []
    mfes = []
    maes = []
    failures = 0

    for idx in signals.index:
        idx_pos = data.index.get_loc(idx)

        if idx_pos + HORIZON >= len(data):
            continue

        entry_price = data.iloc[idx_pos]['close']
        future_data = data.iloc[idx_pos + 1 : idx_pos + 1 + HORIZON]

        if len(future_data) == 0:
            continue

        if signal_type == 'oversold':
            # LONG: expect bounce UP
            max_price = future_data['high'].max()
            min_price = future_data['low'].min()
            mfe = (max_price - entry_price) / entry_price * 10000
            mae = (entry_price - min_price) / entry_price * 10000
            bounced = mfe > 0
        else:
            # SHORT: expect drop DOWN
            min_price = future_data['low'].min()
            max_price = future_data['high'].max()
            mfe = (entry_price - min_price) / entry_price * 10000
            mae = (max_price - entry_price) / entry_price * 10000
            bounced = mfe > 0

        bounces.append(bounced)
        mfes.append(mfe)
        maes.append(mae)

        if not bounced:
            failures += 1

    if bounces:
        results['signals_analyzed'] = len(bounces)
        results['bounce_rate'] = np.mean(bounces) * 100
        results['failures'] = failures
        results['failure_rate'] = (failures / len(bounces)) * 100
        results['avg_mfe'] = np.mean(mfes)
        results['avg_mae'] = np.mean(maes)
        results['median_mfe'] = np.median(mfes)
        results['median_mae'] = np.median(maes)
        results['mfe_mae_ratio'] = np.mean(mfes) / np.mean(maes) if np.mean(maes) > 0 else 0

        # Target achievement
        results['reached_12bp'] = np.mean([m >= 12 for m in mfes]) * 100
        results['reached_25bp'] = np.mean([m >= 25 for m in mfes]) * 100
        results['reached_50bp'] = np.mean([m >= 50 for m in mfes]) * 100

    return results

# =============================================================================
# RUN ANALYSIS
# =============================================================================

all_results = []

for period_name, (start_date, end_date) in DATA_PERIODS.items():
    print(f"\n{'='*70}")
    print(f"Analyzing {period_name}: {start_date} to {end_date}")
    print('='*70)

    period_data = df[start_date:end_date].copy()
    print(f"Bars: {len(period_data):,}")

    # =========================================================================
    # OVERSOLD SIGNALS (LONG)
    # =========================================================================
    print("\n--- OVERSOLD (LONG) SIGNALS ---")

    # 1. RSI Alone (baseline)
    rsi_alone_mask = period_data['rsi_oversold_cross'] == True
    result = analyze_signals(period_data, rsi_alone_mask, 'oversold', 'RSI Alone')
    result['period_name'] = period_name
    result['filter'] = 'none'
    result['ma_period'] = 0
    all_results.append(result)
    print(f"\nRSI Alone: {result['total_signals']} signals, "
          f"{result.get('bounce_rate', 0):.1f}% bounce, "
          f"{result.get('failures', 0)} failures")

    # 2. RSI + MA Filter (WITH trend only)
    for ma_period in MA_PERIODS:
        # Only take oversold signals when price > SMA (uptrend)
        with_trend_mask = (period_data['rsi_oversold_cross'] == True) & \
                         (period_data['close'] > period_data[f'sma_{ma_period}'])

        result = analyze_signals(period_data, with_trend_mask, 'oversold',
                                f'RSI + SMA({ma_period}) WITH trend')
        result['period_name'] = period_name
        result['filter'] = 'with_trend'
        result['ma_period'] = ma_period
        all_results.append(result)
        print(f"RSI + SMA({ma_period}) WITH trend: {result['total_signals']} signals, "
              f"{result.get('bounce_rate', 0):.1f}% bounce, "
              f"{result.get('failures', 0)} failures")

        # Also test AGAINST trend (to confirm it has more failures)
        against_trend_mask = (period_data['rsi_oversold_cross'] == True) & \
                            (period_data['close'] < period_data[f'sma_{ma_period}'])

        result = analyze_signals(period_data, against_trend_mask, 'oversold',
                                f'RSI + SMA({ma_period}) AGAINST trend')
        result['period_name'] = period_name
        result['filter'] = 'against_trend'
        result['ma_period'] = ma_period
        all_results.append(result)
        print(f"RSI + SMA({ma_period}) AGAINST trend: {result['total_signals']} signals, "
              f"{result.get('bounce_rate', 0):.1f}% bounce, "
              f"{result.get('failures', 0)} failures")

    # =========================================================================
    # OVERBOUGHT SIGNALS (SHORT)
    # =========================================================================
    print("\n--- OVERBOUGHT (SHORT) SIGNALS ---")

    # 1. RSI Alone (baseline)
    rsi_alone_mask = period_data['rsi_overbought_cross'] == True
    result = analyze_signals(period_data, rsi_alone_mask, 'overbought', 'RSI Alone')
    result['period_name'] = period_name
    result['filter'] = 'none'
    result['ma_period'] = 0
    result['signal_type'] = 'overbought'
    all_results.append(result)
    print(f"\nRSI Alone: {result['total_signals']} signals, "
          f"{result.get('bounce_rate', 0):.1f}% drop, "
          f"{result.get('failures', 0)} failures")

    # 2. RSI + MA Filter (WITH trend only - short in downtrend)
    for ma_period in MA_PERIODS:
        # Only take overbought signals when price < SMA (downtrend)
        with_trend_mask = (period_data['rsi_overbought_cross'] == True) & \
                         (period_data['close'] < period_data[f'sma_{ma_period}'])

        result = analyze_signals(period_data, with_trend_mask, 'overbought',
                                f'RSI + SMA({ma_period}) WITH trend')
        result['period_name'] = period_name
        result['filter'] = 'with_trend'
        result['ma_period'] = ma_period
        all_results.append(result)
        print(f"RSI + SMA({ma_period}) WITH trend: {result['total_signals']} signals, "
              f"{result.get('bounce_rate', 0):.1f}% drop, "
              f"{result.get('failures', 0)} failures")

# =============================================================================
# SAVE RESULTS
# =============================================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_PATH / 'results.csv', index=False)
print(f"\nSaved results to: {RESULTS_PATH / 'results.csv'}")

# =============================================================================
# SUMMARY TABLES
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: OVERSOLD (LONG) SIGNALS")
print("="*80)

for period_name in ['in_sample', 'out_of_sample']:
    print(f"\n{period_name.upper()}:")
    print("-"*75)
    print(f"{'Filter':<35} {'Signals':<10} {'Bounce%':<10} {'Failures':<10} {'Avg MFE':<10} {'Avg MAE':<10}")
    print("-"*75)

    period_results = results_df[(results_df['period_name'] == period_name) &
                                 (results_df['signal_type'] == 'oversold')]

    for _, row in period_results.iterrows():
        print(f"{row['description']:<35} {row.get('total_signals', 0):<10} "
              f"{row.get('bounce_rate', 0):.1f}%{'':<5} "
              f"{row.get('failures', 0):<10} "
              f"{row.get('avg_mfe', 0):.1f}{'':<5} "
              f"{row.get('avg_mae', 0):.1f}")

print("\n" + "="*80)
print("KEY COMPARISON: RSI Alone vs RSI + SMA(50) WITH Trend")
print("="*80)

for period_name in ['in_sample', 'out_of_sample']:
    print(f"\n{period_name.upper()} - OVERSOLD (LONG):")

    rsi_alone = results_df[(results_df['period_name'] == period_name) &
                           (results_df['signal_type'] == 'oversold') &
                           (results_df['filter'] == 'none')].iloc[0]

    rsi_with_ma = results_df[(results_df['period_name'] == period_name) &
                             (results_df['signal_type'] == 'oversold') &
                             (results_df['filter'] == 'with_trend') &
                             (results_df['ma_period'] == 50)]

    if len(rsi_with_ma) > 0:
        rsi_with_ma = rsi_with_ma.iloc[0]

        signals_filtered = rsi_alone['total_signals'] - rsi_with_ma['total_signals']
        failures_avoided = rsi_alone.get('failures', 0) - rsi_with_ma.get('failures', 0)

        print(f"  RSI Alone:        {rsi_alone['total_signals']} signals, "
              f"{rsi_alone.get('failures', 0)} failures, "
              f"MFE={rsi_alone.get('avg_mfe', 0):.1f}, MAE={rsi_alone.get('avg_mae', 0):.1f}")
        print(f"  RSI + SMA(50):    {rsi_with_ma['total_signals']} signals, "
              f"{rsi_with_ma.get('failures', 0)} failures, "
              f"MFE={rsi_with_ma.get('avg_mfe', 0):.1f}, MAE={rsi_with_ma.get('avg_mae', 0):.1f}")
        print(f"  --> Filtered out: {signals_filtered} signals")
        print(f"  --> Failures avoided: {failures_avoided}")

        if rsi_alone.get('avg_mae', 0) > 0 and rsi_with_ma.get('avg_mae', 0) > 0:
            mae_reduction = ((rsi_alone.get('avg_mae', 0) - rsi_with_ma.get('avg_mae', 0)) /
                            rsi_alone.get('avg_mae', 0)) * 100
            print(f"  --> MAE reduction: {mae_reduction:.1f}%")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
