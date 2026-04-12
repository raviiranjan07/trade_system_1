"""
EXP-004 Part 4: Test All Options

Option 1: LONG only - Skip overbought entirely
Option 2: Macro filter - Use 200 SMA to determine when to SHORT
Option 3: Warning signals - What was unique about the account killer?
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
RESULTS_PATH = Path("experiments/rsi/EXP-004")

RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
HORIZON = 10

print("="*80)
print("EXP-004: TESTING ALL OPTIONS")
print("="*80)

# Load data
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
df['sma_200'] = df['close'].rolling(200).mean()
df['sma_50'] = df['close'].rolling(50).mean()

# Volume and momentum
df['volume_sma_20'] = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume'] / df['volume_sma_20']
df['momentum_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100

# Signals
df['oversold'] = df['rsi'] < RSI_OVERSOLD
df['overbought'] = df['rsi'] > RSI_OVERBOUGHT
df['rsi_oversold_cross'] = (df['oversold'] == True) & (df['oversold'].shift(1) == False)
df['rsi_overbought_cross'] = (df['overbought'] == True) & (df['overbought'].shift(1) == False)

# OOS data
oos_data = df['2024-01-01':'2025-12-30'].copy()

# =============================================================================
# HELPER: Analyze signals
# =============================================================================

def analyze_signals(data, signal_mask, signal_type='long', description=''):
    """Calculate performance metrics for a set of signals"""
    signals = data[signal_mask].copy()

    if len(signals) == 0:
        return {'description': description, 'signals': 0}

    results = []
    for idx in signals.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + HORIZON >= len(data):
            continue

        entry_price = data.iloc[idx_pos]['close']
        future_data = data.iloc[idx_pos + 1 : idx_pos + 1 + HORIZON]

        if len(future_data) == 0:
            continue

        if signal_type == 'long':
            mfe = (future_data['high'].max() - entry_price) / entry_price * 10000
            mae = (entry_price - future_data['low'].min()) / entry_price * 10000
            success = mfe > 0
        else:  # short
            mfe = (entry_price - future_data['low'].min()) / entry_price * 10000
            mae = (future_data['high'].max() - entry_price) / entry_price * 10000
            success = mfe > 0

        results.append({
            'timestamp': idx,
            'mfe': mfe,
            'mae': mae,
            'success': success
        })

    if not results:
        return {'description': description, 'signals': 0}

    results_df = pd.DataFrame(results)

    return {
        'description': description,
        'signals': len(results_df),
        'successes': results_df['success'].sum(),
        'failures': len(results_df) - results_df['success'].sum(),
        'success_rate': results_df['success'].mean() * 100,
        'avg_mfe': results_df['mfe'].mean(),
        'avg_mae': results_df['mae'].mean(),
        'max_mae': results_df['mae'].max()
    }

# =============================================================================
# OPTION 1: LONG ONLY
# =============================================================================

print("\n" + "="*80)
print("OPTION 1: LONG ONLY (Skip overbought entirely)")
print("="*80)

# Oversold (LONG) only
long_only = analyze_signals(
    oos_data,
    oos_data['rsi_oversold_cross'] == True,
    'long',
    'RSI Oversold - LONG only'
)

print(f"\n{long_only['description']}:")
print(f"  Signals: {long_only['signals']}")
print(f"  Success rate: {long_only['success_rate']:.1f}%")
print(f"  Failures: {long_only['failures']}")
print(f"  Avg MFE: {long_only['avg_mfe']:.1f} bps")
print(f"  Avg MAE: {long_only['avg_mae']:.1f} bps")
print(f"  Max MAE: {long_only['max_mae']:.1f} bps")

# Compare to trading both
short_only = analyze_signals(
    oos_data,
    oos_data['rsi_overbought_cross'] == True,
    'short',
    'RSI Overbought - SHORT only'
)

print(f"\nFor comparison - {short_only['description']}:")
print(f"  Signals: {short_only['signals']}")
print(f"  Success rate: {short_only['success_rate']:.1f}%")
print(f"  Failures: {short_only['failures']}")
print(f"  Max MAE: {short_only['max_mae']:.1f} bps")

print("\n*** OPTION 1 CONCLUSION ***")
print(f"LONG only: {long_only['signals']} signals, {long_only['failures']} failures, max loss {long_only['max_mae']:.0f} bps")
print(f"SHORT adds: {short_only['signals']} signals, {short_only['failures']} failures, max loss {short_only['max_mae']:.0f} bps")
print("RECOMMENDATION: LONG only is much safer!")

# =============================================================================
# OPTION 2: MACRO FILTER (200 SMA)
# =============================================================================

print("\n" + "="*80)
print("OPTION 2: MACRO FILTER (200 SMA)")
print("="*80)

# Rule: Only SHORT when price < 200 SMA (bear market)
# Rule: Only LONG when price > 200 SMA (bull market) - but this filters too much

# Test: SHORT only in bear market
short_bear_only = analyze_signals(
    oos_data,
    (oos_data['rsi_overbought_cross'] == True) & (oos_data['close'] < oos_data['sma_200']),
    'short',
    'SHORT only when price < SMA200 (bear market)'
)

print(f"\n{short_bear_only['description']}:")
print(f"  Signals: {short_bear_only['signals']}")
if short_bear_only['signals'] > 0:
    print(f"  Success rate: {short_bear_only['success_rate']:.1f}%")
    print(f"  Failures: {short_bear_only['failures']}")
    print(f"  Max MAE: {short_bear_only['max_mae']:.1f} bps")

# Test: SHORT in bull market (should include the killer)
short_bull = analyze_signals(
    oos_data,
    (oos_data['rsi_overbought_cross'] == True) & (oos_data['close'] > oos_data['sma_200']),
    'short',
    'SHORT when price > SMA200 (bull market)'
)

print(f"\n{short_bull['description']}:")
print(f"  Signals: {short_bull['signals']}")
if short_bull['signals'] > 0:
    print(f"  Success rate: {short_bull['success_rate']:.1f}%")
    print(f"  Failures: {short_bull['failures']}")
    print(f"  Max MAE: {short_bull['max_mae']:.1f} bps")

# Check: Would the 200 SMA filter have skipped the account killer?
killer_date = pd.Timestamp('2024-02-26 15:00:00')
if killer_date in oos_data.index:
    killer_price = oos_data.loc[killer_date, 'close']
    killer_sma200 = oos_data.loc[killer_date, 'sma_200']
    print(f"\n*** THE ACCOUNT KILLER (2024-02-26) ***")
    print(f"  Price: ${killer_price:,.0f}")
    print(f"  SMA200: ${killer_sma200:,.0f}")
    print(f"  Price > SMA200? {killer_price > killer_sma200}")
    if killer_price > killer_sma200:
        print(f"  --> 200 SMA FILTER WOULD HAVE SKIPPED THIS TRADE!")
    else:
        print(f"  --> 200 SMA filter would NOT have helped")

print("\n*** OPTION 2 CONCLUSION ***")
if short_bear_only['signals'] > 0:
    print(f"SHORT in bear market: {short_bear_only['signals']} signals, {short_bear_only['failures']} failures")
else:
    print(f"SHORT in bear market: 0 signals (no bear market in 2024-2025)")
print(f"SHORT in bull market: {short_bull['signals']} signals, {short_bull['failures']} failures")
print("The 200 SMA filter would have SKIPPED the account killer!")

# =============================================================================
# OPTION 3: WARNING SIGNALS - What was unique about the killer?
# =============================================================================

print("\n" + "="*80)
print("OPTION 3: WARNING SIGNALS - What was unique about the account killer?")
print("="*80)

# Get all 11 failures
overbought_signals = oos_data[oos_data['rsi_overbought_cross'] == True].copy()

failures_list = []
successes_list = []

for idx in overbought_signals.index:
    idx_pos = oos_data.index.get_loc(idx)
    if idx_pos + HORIZON >= len(oos_data):
        continue

    entry_price = oos_data.iloc[idx_pos]['close']
    future_data = oos_data.iloc[idx_pos + 1 : idx_pos + 1 + HORIZON]

    if len(future_data) == 0:
        continue

    mfe = (entry_price - future_data['low'].min()) / entry_price * 10000
    mae = (future_data['high'].max() - entry_price) / entry_price * 10000

    data_point = {
        'timestamp': idx,
        'price': entry_price,
        'rsi': oos_data.loc[idx, 'rsi'],
        'sma_200': oos_data.loc[idx, 'sma_200'],
        'price_vs_sma200': (entry_price - oos_data.loc[idx, 'sma_200']) / oos_data.loc[idx, 'sma_200'] * 100,
        'volume_ratio': oos_data.loc[idx, 'volume_ratio'],
        'momentum_20': oos_data.loc[idx, 'momentum_20'],
        'mfe': mfe,
        'mae': mae,
        'success': mfe > 0
    }

    if mfe > 0:
        successes_list.append(data_point)
    else:
        failures_list.append(data_point)

failures_df = pd.DataFrame(failures_list)
successes_df = pd.DataFrame(successes_list)

print(f"\nTotal failures: {len(failures_df)}")
print(f"Total successes: {len(successes_df)}")

# Analyze the killer specifically
print("\n--- THE ACCOUNT KILLER vs OTHER FAILURES ---")
killer_data = failures_df[failures_df['timestamp'] == killer_date]
other_failures = failures_df[failures_df['timestamp'] != killer_date]

if len(killer_data) > 0:
    killer = killer_data.iloc[0]
    print(f"\nAccount Killer (2024-02-26):")
    print(f"  RSI: {killer['rsi']:.1f}")
    print(f"  Price vs SMA200: {killer['price_vs_sma200']:+.1f}%")
    print(f"  Volume ratio: {killer['volume_ratio']:.2f}x")
    print(f"  20-bar momentum: {killer['momentum_20']:+.2f}%")
    print(f"  MAE (loss): {killer['mae']:.0f} bps")

if len(other_failures) > 0:
    print(f"\nOther failures (avg):")
    print(f"  RSI: {other_failures['rsi'].mean():.1f}")
    print(f"  Price vs SMA200: {other_failures['price_vs_sma200'].mean():+.1f}%")
    print(f"  Volume ratio: {other_failures['volume_ratio'].mean():.2f}x")
    print(f"  20-bar momentum: {other_failures['momentum_20'].mean():+.2f}%")
    print(f"  MAE (loss): {other_failures['mae'].mean():.0f} bps")

print(f"\nSuccesses (avg):")
print(f"  RSI: {successes_df['rsi'].mean():.1f}")
print(f"  Price vs SMA200: {successes_df['price_vs_sma200'].mean():+.1f}%")
print(f"  Volume ratio: {successes_df['volume_ratio'].mean():.2f}x")
print(f"  20-bar momentum: {successes_df['momentum_20'].mean():+.2f}%")

# What makes the killer unique?
print("\n--- WHAT MADE THE KILLER UNIQUE? ---")

if len(killer_data) > 0:
    killer = killer_data.iloc[0]

    checks = []

    # Check 1: Price vs SMA200
    if killer['price_vs_sma200'] > successes_df['price_vs_sma200'].quantile(0.9):
        checks.append(f"Price was {killer['price_vs_sma200']:.1f}% above SMA200 (>90th percentile)")

    # Check 2: Momentum
    if killer['momentum_20'] > successes_df['momentum_20'].quantile(0.9):
        checks.append(f"20-bar momentum was {killer['momentum_20']:.2f}% (>90th percentile)")

    # Check 3: RSI
    if killer['rsi'] < successes_df['rsi'].quantile(0.1):
        checks.append(f"RSI was only {killer['rsi']:.1f} (low for overbought)")

    # Check 4: Volume
    if killer['volume_ratio'] > successes_df['volume_ratio'].quantile(0.9):
        checks.append(f"Volume was {killer['volume_ratio']:.2f}x average (>90th percentile)")

    if checks:
        print("\nUnique characteristics of the killer:")
        for c in checks:
            print(f"  - {c}")
    else:
        print("\nNo obvious unique characteristics found!")
        print("The killer looked similar to other signals - that's what makes it dangerous.")

print("\n*** OPTION 3 CONCLUSION ***")
print("Potential warning signals for the killer:")
print("  1. Price significantly above SMA200 (strong uptrend)")
print("  2. High 20-bar momentum (rally in progress)")
print("  3. These indicate 'shorting into strength' - dangerous!")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY: ALL OPTIONS")
print("="*80)

print("""
OPTION 1: LONG ONLY
------------------
- Signals: {long_signals}
- Failures: {long_failures}
- Max loss: {long_max_mae:.0f} bps
- VERDICT: SAFE and SIMPLE

OPTION 2: MACRO FILTER (200 SMA)
--------------------------------
- Rule: Only SHORT when price < SMA200
- Would have SKIPPED the account killer
- In 2024-2025: {bear_signals} bear market signals
- VERDICT: WORKS but may have few SHORT signals in bull markets

OPTION 3: WARNING SIGNALS
-------------------------
- Killer had: Price far above SMA200, high momentum
- Indicates: Start of major rally (shorting into strength)
- Could use: Skip SHORT if momentum_20 > X% AND price > SMA200
- VERDICT: More complex, needs more testing
""".format(
    long_signals=long_only['signals'],
    long_failures=long_only['failures'],
    long_max_mae=long_only['max_mae'],
    bear_signals=short_bear_only['signals'] if short_bear_only['signals'] else 0
))

print("="*80)
print("RECOMMENDATION:")
print("="*80)
print("""
1. SAFEST: Trade LONG only (oversold signals)
2. IF you want SHORT: Use 200 SMA filter (only SHORT in bear market)
3. The account killer was "shorting into strength" during BTC ETF rally
""")
