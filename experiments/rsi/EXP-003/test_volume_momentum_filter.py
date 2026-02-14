"""
EXP-003 Part 2: Test Volume + Momentum Filter

Hypothesis: Failures happen with WEAK signals (low volume, low momentum)
Test: Skip overbought signals if volume < threshold OR momentum < threshold
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
RESULTS_PATH = Path("experiments/rsi/EXP-003")

RSI_PERIOD = 14
RSI_OVERBOUGHT = 80
HORIZON = 10

print("="*70)
print("TESTING VOLUME + MOMENTUM FILTER")
print("="*70)

# Load and prepare data
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

# Calculate indicators
def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calculate_rsi(df, RSI_PERIOD)
df['volume_sma_20'] = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume'] / df['volume_sma_20']
df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 10000

df['overbought'] = df['rsi'] > RSI_OVERBOUGHT
df['rsi_overbought_cross'] = (df['overbought'] == True) & (df['overbought'].shift(1) == False)

# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_filter(data, vol_threshold=None, mom_threshold=None, description=""):
    """Test overbought signals with optional filters"""

    signals = data[data['rsi_overbought_cross'] == True].copy()

    # Apply filters
    if vol_threshold is not None:
        signals = signals[signals['volume_ratio'] >= vol_threshold]
    if mom_threshold is not None:
        signals = signals[signals['momentum_5'] >= mom_threshold]

    if len(signals) == 0:
        return {'description': description, 'signals': 0, 'failures': 0, 'success_rate': 0}

    successes = 0
    failures = 0

    for idx in signals.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + HORIZON >= len(data):
            continue

        entry_price = data.iloc[idx_pos]['close']
        future_data = data.iloc[idx_pos + 1 : idx_pos + 1 + HORIZON]

        if len(future_data) == 0:
            continue

        min_price = future_data['low'].min()
        dropped = min_price < entry_price

        if dropped:
            successes += 1
        else:
            failures += 1

    total = successes + failures
    success_rate = (successes / total * 100) if total > 0 else 0

    return {
        'description': description,
        'signals': total,
        'successes': successes,
        'failures': failures,
        'success_rate': success_rate
    }

# =============================================================================
# RUN TESTS
# =============================================================================

oos_data = df['2024-01-01':'2025-12-30'].copy()

print(f"\nOut-of-Sample Data: 2024-2025")
print(f"Total bars: {len(oos_data):,}")

results = []

# Baseline
results.append(test_filter(oos_data, description="No Filter (Baseline)"))

# Volume filters
for vol in [0.5, 0.75, 1.0, 1.25, 1.5]:
    results.append(test_filter(oos_data, vol_threshold=vol,
                               description=f"Volume >= {vol}x"))

# Momentum filters
for mom in [25, 50, 75, 100]:
    results.append(test_filter(oos_data, mom_threshold=mom,
                               description=f"Momentum >= {mom} bps"))

# Combined filters
results.append(test_filter(oos_data, vol_threshold=1.0, mom_threshold=50,
                           description="Vol >= 1.0x AND Mom >= 50 bps"))
results.append(test_filter(oos_data, vol_threshold=0.75, mom_threshold=25,
                           description="Vol >= 0.75x AND Mom >= 25 bps"))

# =============================================================================
# RESULTS TABLE
# =============================================================================

print("\n" + "="*70)
print("FILTER TEST RESULTS")
print("="*70)

print(f"\n{'Filter':<35} {'Signals':<10} {'Failures':<10} {'Success%':<12} {'Saved':<10}")
print("-"*77)

baseline_signals = results[0]['signals']
baseline_failures = results[0]['failures']

for r in results:
    signals_lost = baseline_signals - r['signals']
    failures_avoided = baseline_failures - r['failures']

    marker = ""
    if r['failures'] == 0 and r['signals'] > 0:
        marker = " <-- ZERO FAILURES!"

    print(f"{r['description']:<35} {r['signals']:<10} {r['failures']:<10} "
          f"{r['success_rate']:.1f}%{'':<6} {failures_avoided:+d}{marker}")

# =============================================================================
# TRADE-OFF ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("TRADE-OFF ANALYSIS")
print("="*70)

print("\nFor each filter, how many good signals do we lose per failure avoided?")
print(f"\n{'Filter':<35} {'Failures Avoided':<18} {'Good Signals Lost':<18} {'Ratio':<10}")
print("-"*80)

for r in results[1:]:  # Skip baseline
    failures_avoided = baseline_failures - r['failures']
    good_signals_lost = (baseline_signals - baseline_failures) - (r['signals'] - r['failures'])

    if failures_avoided > 0:
        ratio = good_signals_lost / failures_avoided
        print(f"{r['description']:<35} {failures_avoided:<18} {good_signals_lost:<18} {ratio:.1f}")
    else:
        print(f"{r['description']:<35} {failures_avoided:<18} {good_signals_lost:<18} N/A")

# =============================================================================
# BEST FILTER RECOMMENDATION
# =============================================================================

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

# Find filters with 0 failures
zero_failure_filters = [r for r in results if r['failures'] == 0 and r['signals'] > 0]

if zero_failure_filters:
    # Pick the one with most signals
    best = max(zero_failure_filters, key=lambda x: x['signals'])
    print(f"\nBest filter (zero failures, most signals): {best['description']}")
    print(f"  Signals: {best['signals']} (was {baseline_signals})")
    print(f"  Failures: {best['failures']} (was {baseline_failures})")
    print(f"  Success rate: {best['success_rate']:.1f}%")

    signals_kept_pct = best['signals'] / baseline_signals * 100
    print(f"  Signals retained: {signals_kept_pct:.1f}%")
else:
    # Find best balance
    best_ratio = None
    for r in results[1:]:
        if r['failures'] < baseline_failures and r['signals'] > 100:
            failures_avoided = baseline_failures - r['failures']
            good_lost = (baseline_signals - baseline_failures) - (r['signals'] - r['failures'])
            ratio = good_lost / failures_avoided if failures_avoided > 0 else float('inf')

            if best_ratio is None or ratio < best_ratio[1]:
                best_ratio = (r, ratio)

    if best_ratio:
        r, ratio = best_ratio
        print(f"\nBest balance: {r['description']}")
        print(f"  Lose {ratio:.1f} good signals per failure avoided")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_PATH / 'filter_test_results.csv', index=False)
print(f"\nResults saved to: {RESULTS_PATH / 'filter_test_results.csv'}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
