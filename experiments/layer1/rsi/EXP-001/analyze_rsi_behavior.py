"""
EXP-001: RSI Mean Reversion Behavior Analysis

PURE RSI ANALYSIS - No other indicators or features

Test 7 RSI combinations:
- Periods: 7, 14, 21
- Thresholds: 20/80, 30/70, 40/60

Analyze:
1. How often RSI hits extremes
2. Does price reverse after extremes?
3. How much does price move after reversal?
4. Which combination works best?

Run on both:
- In-sample: 2020-2023
- Out-of-sample: 2024-2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EXP-001: RSI MEAN REVERSION BEHAVIOR ANALYSIS")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

# RSI combinations to test
RSI_COMBOS = [
    {'period': 7, 'oversold': 30, 'overbought': 70},
    {'period': 7, 'oversold': 20, 'overbought': 80},
    {'period': 14, 'oversold': 30, 'overbought': 70},
    {'period': 14, 'oversold': 20, 'overbought': 80},
    {'period': 14, 'oversold': 40, 'overbought': 60},
    {'period': 21, 'oversold': 30, 'overbought': 70},
    {'period': 21, 'oversold': 20, 'overbought': 80},
]

# Lookforward horizons to check reversals
HORIZONS = [3, 5, 10, 15, 20]  # bars to look ahead

# Data periods
PERIODS = {
    'in_sample': ('2020-01-01', '2023-12-31'),
    'out_of_sample': ('2024-01-01', '2025-12-30')
}

print(f"\nTesting {len(RSI_COMBOS)} RSI combinations")
print(f"Lookforward horizons: {HORIZONS} bars")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/3] Loading data...")
ohlcv_path = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"Loaded {len(ohlcv):,} candles")

# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================

def calculate_rsi(data, period):
    """Calculate RSI indicator"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analyze_rsi_combo(data, combo, horizons):
    """Analyze one RSI combination"""

    period = combo['period']
    oversold = combo['oversold']
    overbought = combo['overbought']

    # Calculate RSI
    data[f'rsi_{period}'] = calculate_rsi(data, period)

    # Detect extremes
    data['is_oversold'] = data[f'rsi_{period}'] < oversold
    data['is_overbought'] = data[f'rsi_{period}'] > overbought

    # Detect crosses (enter extreme zone)
    data['cross_oversold'] = (data['is_oversold'] == True) & (data['is_oversold'].shift(1) == False)
    data['cross_overbought'] = (data['is_overbought'] == True) & (data['is_overbought'].shift(1) == False)

    results = {
        'period': period,
        'oversold_threshold': oversold,
        'overbought_threshold': overbought,
        'total_bars': len(data),
        'oversold_crosses': data['cross_oversold'].sum(),
        'overbought_crosses': data['cross_overbought'].sum(),
    }

    # Analyze what happens after oversold (expecting bounce UP)
    oversold_indices = data[data['cross_oversold']].index

    for h in horizons:
        if len(oversold_indices) == 0:
            results[f'oversold_bounce_rate_h{h}'] = 0
            results[f'oversold_avg_move_h{h}'] = 0
            continue

        bounces = []
        moves = []

        for idx in oversold_indices:
            idx_pos = data.index.get_loc(idx)
            if idx_pos + h >= len(data):
                continue

            entry_price = data.iloc[idx_pos]['close']
            future_high = data.iloc[idx_pos + 1 : idx_pos + 1 + h]['high'].max()

            move_bps = (future_high - entry_price) / entry_price * 10000
            moves.append(move_bps)
            bounces.append(move_bps > 0)  # Did price go UP?

        if len(bounces) > 0:
            results[f'oversold_bounce_rate_h{h}'] = np.mean(bounces) * 100
            results[f'oversold_avg_move_h{h}'] = np.mean(moves)
        else:
            results[f'oversold_bounce_rate_h{h}'] = 0
            results[f'oversold_avg_move_h{h}'] = 0

    # Analyze what happens after overbought (expecting drop DOWN)
    overbought_indices = data[data['cross_overbought']].index

    for h in horizons:
        if len(overbought_indices) == 0:
            results[f'overbought_drop_rate_h{h}'] = 0
            results[f'overbought_avg_move_h{h}'] = 0
            continue

        drops = []
        moves = []

        for idx in overbought_indices:
            idx_pos = data.index.get_loc(idx)
            if idx_pos + h >= len(data):
                continue

            entry_price = data.iloc[idx_pos]['close']
            future_low = data.iloc[idx_pos + 1 : idx_pos + 1 + h]['low'].min()

            move_bps = (entry_price - future_low) / entry_price * 10000  # Positive if dropped
            moves.append(move_bps)
            drops.append(move_bps > 0)  # Did price go DOWN?

        if len(drops) > 0:
            results[f'overbought_drop_rate_h{h}'] = np.mean(drops) * 100
            results[f'overbought_avg_move_h{h}'] = np.mean(moves)
        else:
            results[f'overbought_drop_rate_h{h}'] = 0
            results[f'overbought_avg_move_h{h}'] = 0

    return results

# =============================================================================
# RUN ANALYSIS ON BOTH PERIODS
# =============================================================================

all_results = []

for period_name, (start, end) in PERIODS.items():
    print(f"\n[2/3] Analyzing {period_name}: {start} to {end}...")

    period_data = ohlcv[(ohlcv.index >= start) & (ohlcv.index <= end)].copy()
    print(f"  Data: {len(period_data):,} candles")

    for i, combo in enumerate(RSI_COMBOS, 1):
        print(f"  [{i}/{len(RSI_COMBOS)}] RSI({combo['period']}) < {combo['oversold']} / > {combo['overbought']}")

        results = analyze_rsi_combo(period_data, combo, HORIZONS)
        results['period_name'] = period_name
        results['period_start'] = start
        results['period_end'] = end

        all_results.append(results)

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n[3/3] Saving results...")

results_df = pd.DataFrame(all_results)

# Save to EXP-001 folder
output_path = Path("experiments/rsi/EXP-001/results.csv")
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# =============================================================================
# DISPLAY SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

for period_name in ['in_sample', 'out_of_sample']:
    print(f"\n{'='*80}")
    print(f"{period_name.upper().replace('_', ' ')}")
    print(f"{'='*80}\n")

    period_results = results_df[results_df['period_name'] == period_name]

    for _, row in period_results.iterrows():
        print(f"RSI({int(row['period'])}) < {int(row['oversold_threshold'])} / > {int(row['overbought_threshold'])}:")
        print(f"  Oversold crosses: {int(row['oversold_crosses'])}")
        print(f"  Overbought crosses: {int(row['overbought_crosses'])}")
        print(f"  Bounce rate (H=10): {row['oversold_bounce_rate_h10']:.1f}%")
        print(f"  Drop rate (H=10): {row['overbought_drop_rate_h10']:.1f}%")
        print(f"  Avg bounce (H=10): {row['oversold_avg_move_h10']:.1f} bps")
        print(f"  Avg drop (H=10): {row['overbought_avg_move_h10']:.1f} bps")
        print()

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("1. Review results.csv for detailed metrics")
print("2. Compare in_sample vs out_of_sample consistency")
print("3. Identify best RSI combination")
print("4. Update explanation.txt with findings")
print("=" * 80)
