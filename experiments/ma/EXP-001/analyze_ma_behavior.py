"""
EXP-001 (MA): Moving Average Behavior Analysis
Tests SMA, EMA, WMA standalone to see which best identifies trend

Question: Does price continue in the same direction after crossing a Moving Average?
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
RESULTS_PATH = Path("experiments/ma/EXP-001")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Configuration
MA_TYPES = ['SMA', 'EMA', 'WMA']
PERIODS = [7, 9, 20, 50, 100, 200]
HORIZONS = [3, 5, 10, 15, 20]  # bars to look forward

DATA_PERIODS = {
    'in_sample': ('2020-01-01', '2023-12-31'),
    'out_of_sample': ('2024-01-01', '2025-12-30')
}

print("="*60)
print("MOVING AVERAGE BEHAVIOR ANALYSIS")
print(f"MA Types: {MA_TYPES}")
print(f"Periods: {PERIODS}")
print(f"Horizons: {HORIZONS}")
print("="*60)

# Load data
print("\nLoading data...")
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)
print(f"Total bars: {len(df):,}")

# =============================================================================
# MOVING AVERAGE CALCULATIONS
# =============================================================================

def calculate_sma(data, period):
    """Simple Moving Average - equal weight"""
    return data['close'].rolling(window=period).mean()

def calculate_ema(data, period):
    """Exponential Moving Average - more weight to recent"""
    return data['close'].ewm(span=period, adjust=False).mean()

def calculate_wma(data, period):
    """Weighted Moving Average - linear weight"""
    weights = np.arange(1, period + 1)
    return data['close'].rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

def calculate_ma(data, ma_type, period):
    """Calculate any MA type"""
    if ma_type == 'SMA':
        return calculate_sma(data, period)
    elif ma_type == 'EMA':
        return calculate_ema(data, period)
    elif ma_type == 'WMA':
        return calculate_wma(data, period)
    else:
        raise ValueError(f"Unknown MA type: {ma_type}")

# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================

def analyze_ma(data, ma_type, period, horizons):
    """
    Analyze MA behavior: When price crosses MA, does trend continue?

    Signal:
    - Price crosses ABOVE MA → expect price to continue UP
    - Price crosses BELOW MA → expect price to continue DOWN
    """

    # Calculate MA
    col_name = f'{ma_type}_{period}'
    data = data.copy()
    data[col_name] = calculate_ma(data, ma_type, period)

    # Detect crosses
    data['above_ma'] = data['close'] > data[col_name]
    data['cross_above'] = (data['above_ma'] == True) & (data['above_ma'].shift(1) == False)
    data['cross_below'] = (data['above_ma'] == False) & (data['above_ma'].shift(1) == True)

    # Skip NaN rows
    data = data.dropna(subset=[col_name])

    results = {
        'ma_type': ma_type,
        'period': period,
        'total_bars': len(data),
        'cross_above_count': data['cross_above'].sum(),
        'cross_below_count': data['cross_below'].sum()
    }

    # Analyze cross above signals (expect continuation UP)
    cross_above_idx = data[data['cross_above'] == True].index
    for h in horizons:
        continuations = []
        moves = []

        for idx in cross_above_idx:
            idx_pos = data.index.get_loc(idx)
            if idx_pos + h >= len(data):
                continue

            entry_price = data.iloc[idx_pos]['close']
            future_data = data.iloc[idx_pos + 1 : idx_pos + 1 + h]

            if len(future_data) == 0:
                continue

            # MFE - max price reached (best case for LONG)
            max_price = future_data['high'].max()
            mfe_bps = (max_price - entry_price) / entry_price * 10000

            # Did trend continue? (price went higher)
            continued = mfe_bps > 0
            continuations.append(continued)
            moves.append(mfe_bps)

        if continuations:
            results[f'cross_above_continuation_rate_h{h}'] = np.mean(continuations) * 100
            results[f'cross_above_avg_move_h{h}'] = np.mean(moves)
            results[f'cross_above_median_move_h{h}'] = np.median(moves)
        else:
            results[f'cross_above_continuation_rate_h{h}'] = 0
            results[f'cross_above_avg_move_h{h}'] = 0
            results[f'cross_above_median_move_h{h}'] = 0

    # Analyze cross below signals (expect continuation DOWN)
    cross_below_idx = data[data['cross_below'] == True].index
    for h in horizons:
        continuations = []
        moves = []

        for idx in cross_below_idx:
            idx_pos = data.index.get_loc(idx)
            if idx_pos + h >= len(data):
                continue

            entry_price = data.iloc[idx_pos]['close']
            future_data = data.iloc[idx_pos + 1 : idx_pos + 1 + h]

            if len(future_data) == 0:
                continue

            # MFE for SHORT - min price reached (best case for SHORT)
            min_price = future_data['low'].min()
            mfe_bps = (entry_price - min_price) / entry_price * 10000

            # Did trend continue? (price went lower)
            continued = mfe_bps > 0
            continuations.append(continued)
            moves.append(mfe_bps)

        if continuations:
            results[f'cross_below_continuation_rate_h{h}'] = np.mean(continuations) * 100
            results[f'cross_below_avg_move_h{h}'] = np.mean(moves)
            results[f'cross_below_median_move_h{h}'] = np.median(moves)
        else:
            results[f'cross_below_continuation_rate_h{h}'] = 0
            results[f'cross_below_avg_move_h{h}'] = 0
            results[f'cross_below_median_move_h{h}'] = 0

    return results

# =============================================================================
# RUN ANALYSIS
# =============================================================================

all_results = []

for period_name, (start_date, end_date) in DATA_PERIODS.items():
    print(f"\n{'='*60}")
    print(f"Analyzing {period_name}: {start_date} to {end_date}")
    print('='*60)

    # Filter data
    period_data = df[start_date:end_date].copy()
    print(f"Bars in period: {len(period_data):,}")

    for ma_type in MA_TYPES:
        for period in PERIODS:
            print(f"  Analyzing {ma_type}({period})...", end=" ")

            result = analyze_ma(period_data, ma_type, period, HORIZONS)
            result['period_name'] = period_name
            result['period_start'] = start_date
            result['period_end'] = end_date

            all_results.append(result)

            # Quick summary
            cont_rate = result.get('cross_above_continuation_rate_h10', 0)
            avg_move = result.get('cross_above_avg_move_h10', 0)
            print(f"Cont. rate: {cont_rate:.1f}%, Avg move: {avg_move:.1f} bps")

# =============================================================================
# SAVE RESULTS
# =============================================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_PATH / 'results.csv', index=False)
print(f"\nSaved results to: {RESULTS_PATH / 'results.csv'}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: TREND CONTINUATION RATE AT H=10 (Cross Above = expect UP)")
print("="*80)

for period_name in ['in_sample', 'out_of_sample']:
    print(f"\n{period_name.upper()}:")
    print("-"*70)
    print(f"{'MA Type':<10} {'Period':<8} {'Signals':<10} {'Cont. Rate':<12} {'Avg Move':<12}")
    print("-"*70)

    period_results = results_df[results_df['period_name'] == period_name]

    for _, row in period_results.iterrows():
        print(f"{row['ma_type']:<10} {row['period']:<8} {int(row['cross_above_count']):<10} "
              f"{row['cross_above_continuation_rate_h10']:.1f}%{'':<6} "
              f"{row['cross_above_avg_move_h10']:.1f} bps")

print("\n" + "="*80)
print("SUMMARY: TREND CONTINUATION RATE AT H=10 (Cross Below = expect DOWN)")
print("="*80)

for period_name in ['in_sample', 'out_of_sample']:
    print(f"\n{period_name.upper()}:")
    print("-"*70)
    print(f"{'MA Type':<10} {'Period':<8} {'Signals':<10} {'Cont. Rate':<12} {'Avg Move':<12}")
    print("-"*70)

    period_results = results_df[results_df['period_name'] == period_name]

    for _, row in period_results.iterrows():
        print(f"{row['ma_type']:<10} {row['period']:<8} {int(row['cross_below_count']):<10} "
              f"{row['cross_below_continuation_rate_h10']:.1f}%{'':<6} "
              f"{row['cross_below_avg_move_h10']:.1f} bps")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
