"""
EXP-004 Part 2: Do Failures Eventually Recover?

For the 11 overbought failures:
- If we wait longer (50, 100, 200 bars), do they eventually recover?
- How long does recovery take?
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
RESULTS_PATH = Path("experiments/rsi/EXP-004")

RSI_PERIOD = 14
RSI_OVERBOUGHT = 80
EXTENDED_HORIZON = 200  # Check up to 200 bars (50 hours)

print("="*70)
print("FAILURE RECOVERY ANALYSIS")
print("="*70)

# Load data
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

# Calculate RSI
def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calculate_rsi(df, RSI_PERIOD)
df['overbought'] = df['rsi'] > RSI_OVERBOUGHT
df['rsi_overbought_cross'] = (df['overbought'] == True) & (df['overbought'].shift(1) == False)

# OOS data
oos_data = df['2024-01-01':'2025-12-30'].copy()

# Find overbought signals
signals = oos_data[oos_data['rsi_overbought_cross'] == True].copy()
print(f"\nTotal overbought signals: {len(signals)}")

# Analyze each signal for recovery
results = []

for idx in signals.index:
    idx_pos = oos_data.index.get_loc(idx)

    if idx_pos + EXTENDED_HORIZON >= len(oos_data):
        continue

    entry_price = oos_data.iloc[idx_pos]['close']

    # Track performance at various horizons
    recovered = False
    recovery_bar = None
    max_drawdown = 0

    for bar in range(1, EXTENDED_HORIZON + 1):
        bar_data = oos_data.iloc[idx_pos + bar]

        # For SHORT: profit if price drops, loss if price rises
        current_pnl = (entry_price - bar_data['close']) / entry_price * 10000
        low_pnl = (entry_price - bar_data['low']) / entry_price * 10000  # Best case in bar
        high_pnl = (entry_price - bar_data['high']) / entry_price * 10000  # Worst case in bar

        # Track max drawdown (worst loss)
        if high_pnl < max_drawdown:
            max_drawdown = high_pnl

        # Check if recovered (price dropped below entry = we're in profit)
        if low_pnl > 0 and not recovered:
            recovered = True
            recovery_bar = bar

    # Classification
    # Original H=10 failure = never went in profit within first 10 bars
    first_10_bars = oos_data.iloc[idx_pos + 1 : idx_pos + 11]
    h10_mfe = (entry_price - first_10_bars['low'].min()) / entry_price * 10000
    is_h10_failure = h10_mfe <= 0

    results.append({
        'timestamp': idx,
        'entry_price': entry_price,
        'is_h10_failure': is_h10_failure,
        'h10_mfe': h10_mfe,
        'max_drawdown': max_drawdown,
        'recovered': recovered,
        'recovery_bar': recovery_bar,
        'recovery_time_hours': recovery_bar * 0.25 if recovery_bar else None  # 15min bars
    })

results_df = pd.DataFrame(results)

# Filter to H=10 failures only
failures = results_df[results_df['is_h10_failure'] == True]
print(f"H=10 Failures: {len(failures)}")

# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("EACH FAILURE - RECOVERY ANALYSIS")
print("="*70)

for i, row in failures.iterrows():
    print(f"\nFailure: {row['timestamp']}")
    print(f"  Entry price: ${row['entry_price']:,.0f}")
    print(f"  Max drawdown: {row['max_drawdown']:.1f} bps")

    if row['recovered']:
        print(f"  RECOVERED: Yes!")
        print(f"  Recovery bar: {row['recovery_bar']} ({row['recovery_time_hours']:.1f} hours)")
    else:
        print(f"  RECOVERED: NO within {EXTENDED_HORIZON} bars ({EXTENDED_HORIZON * 0.25:.0f} hours)")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

recovered_count = failures['recovered'].sum()
total_failures = len(failures)

print(f"\nTotal H=10 failures: {total_failures}")
print(f"Eventually recovered: {recovered_count} ({recovered_count/total_failures*100:.0f}%)")
print(f"Never recovered: {total_failures - recovered_count}")

if recovered_count > 0:
    recovered_failures = failures[failures['recovered'] == True]

    print(f"\nAmong those that recovered:")
    print(f"  Avg recovery time: {recovered_failures['recovery_bar'].mean():.0f} bars ({recovered_failures['recovery_time_hours'].mean():.1f} hours)")
    print(f"  Min recovery time: {recovered_failures['recovery_bar'].min():.0f} bars ({recovered_failures['recovery_time_hours'].min():.1f} hours)")
    print(f"  Max recovery time: {recovered_failures['recovery_bar'].max():.0f} bars ({recovered_failures['recovery_time_hours'].max():.1f} hours)")
    print(f"  Avg max drawdown before recovery: {recovered_failures['max_drawdown'].mean():.1f} bps")

if total_failures - recovered_count > 0:
    not_recovered = failures[failures['recovered'] == False]
    print(f"\nAmong those that NEVER recovered:")
    print(f"  Avg max drawdown: {not_recovered['max_drawdown'].mean():.1f} bps")
    print(f"  Worst drawdown: {not_recovered['max_drawdown'].min():.1f} bps")

# =============================================================================
# THE KEY QUESTION: Is waiting worth it?
# =============================================================================

print("\n" + "="*70)
print("IS WAITING WORTH IT?")
print("="*70)

if recovered_count > 0:
    recovered_failures = failures[failures['recovered'] == True]
    avg_drawdown = abs(recovered_failures['max_drawdown'].mean())
    avg_recovery_hours = recovered_failures['recovery_time_hours'].mean()

    print(f"\nTo recover from a failure, you need to:")
    print(f"  1. Survive avg {avg_drawdown:.0f} bps drawdown")
    print(f"  2. Wait avg {avg_recovery_hours:.1f} hours ({avg_recovery_hours/24:.1f} days)")
    print(f"  3. Capital is locked during this time")

    print(f"\nOpportunity cost:")
    print(f"  - Could have done ~{avg_recovery_hours / 2.5:.0f} new trades in that time")
    print(f"  - At 50 bps avg profit = ~{avg_recovery_hours / 2.5 * 50:.0f} bps potential")

# Save results
failures.to_csv(RESULTS_PATH / 'failure_recovery_analysis.csv', index=False)
print(f"\nSaved to: {RESULTS_PATH / 'failure_recovery_analysis.csv'}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
