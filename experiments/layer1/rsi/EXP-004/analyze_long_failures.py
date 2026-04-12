"""
EXP-004 Part 6: Analyze the 2 LONG (Oversold) Failures

Even "safest" option has 2 failures.
One had 1269 bps MAE - worse than SHORT failures!
What happened?
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")

RSI_PERIOD = 14
RSI_OVERSOLD = 20
HORIZON = 10
EXTENDED_HORIZON = 2000  # ~20 days

print("="*70)
print("ANALYZING THE 2 LONG (OVERSOLD) FAILURES")
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
df['sma_200'] = df['close'].rolling(200).mean()
df['oversold'] = df['rsi'] < RSI_OVERSOLD
df['rsi_oversold_cross'] = (df['oversold'] == True) & (df['oversold'].shift(1) == False)

# OOS data
oos_data = df['2024-01-01':'2025-12-30'].copy()

# Find oversold signals
signals = oos_data[oos_data['rsi_oversold_cross'] == True].copy()
print(f"\nTotal oversold signals: {len(signals)}")

# Find the 2 failures
failures = []

for idx in signals.index:
    idx_pos = oos_data.index.get_loc(idx)
    if idx_pos + HORIZON >= len(oos_data):
        continue

    entry_price = oos_data.iloc[idx_pos]['close']
    future_data = oos_data.iloc[idx_pos + 1 : idx_pos + 1 + HORIZON]

    if len(future_data) == 0:
        continue

    # For LONG: profit if price rises, loss if price drops
    mfe = (future_data['high'].max() - entry_price) / entry_price * 10000
    mae = (entry_price - future_data['low'].min()) / entry_price * 10000

    if mfe <= 0:  # Failure = never went in profit
        failures.append({
            'timestamp': idx,
            'entry_price': entry_price,
            'rsi': oos_data.loc[idx, 'rsi'],
            'sma_200': oos_data.loc[idx, 'sma_200'],
            'mfe': mfe,
            'mae': mae
        })

print(f"Failures found: {len(failures)}")

# Analyze each failure
for i, fail in enumerate(failures):
    print(f"\n{'='*70}")
    print(f"LONG FAILURE #{i+1}: {fail['timestamp']}")
    print("="*70)

    idx = fail['timestamp']
    idx_pos = oos_data.index.get_loc(idx)
    entry_price = fail['entry_price']

    print(f"Entry price: ${entry_price:,.0f}")
    print(f"RSI at entry: {fail['rsi']:.1f}")
    print(f"SMA200: ${fail['sma_200']:,.0f}")
    print(f"Price vs SMA200: {(entry_price - fail['sma_200']) / fail['sma_200'] * 100:+.1f}%")
    print(f"H=10 MFE: {fail['mfe']:.1f} bps")
    print(f"H=10 MAE: {fail['mae']:.1f} bps")

    # Extended analysis - does it recover?
    check_horizon = min(EXTENDED_HORIZON, len(oos_data) - idx_pos - 1)

    recovered = False
    recovery_bar = None
    max_drawdown = 0
    max_drawdown_bar = 0

    for bar in range(1, check_horizon + 1):
        bar_data = oos_data.iloc[idx_pos + bar]

        # For LONG: profit if price rises above entry
        high_pnl = (bar_data['high'] - entry_price) / entry_price * 10000
        low_pnl = (bar_data['low'] - entry_price) / entry_price * 10000

        # Track max drawdown
        if low_pnl < max_drawdown:
            max_drawdown = low_pnl
            max_drawdown_bar = bar

        # Check if recovered
        if high_pnl > 0 and not recovered:
            recovered = True
            recovery_bar = bar
            break

    print(f"\n--- EXTENDED ANALYSIS ---")
    print(f"Max drawdown: {max_drawdown:.1f} bps (at bar {max_drawdown_bar})")
    print(f"Max drawdown time: {max_drawdown_bar * 15 / 60:.1f} hours ({max_drawdown_bar * 15 / 60 / 24:.1f} days)")

    if recovered:
        recovery_hours = recovery_bar * 15 / 60
        print(f"\n*** RECOVERED! ***")
        print(f"Recovery bar: {recovery_bar}")
        print(f"Recovery time: {recovery_hours:.1f} hours ({recovery_hours/24:.1f} days)")
    else:
        print(f"\n*** NEVER RECOVERED within {check_horizon} bars ({check_horizon * 15 / 60 / 24:.1f} days) ***")
        # What's the status at end?
        last_bar = oos_data.iloc[idx_pos + check_horizon]
        current_pnl = (last_bar['close'] - entry_price) / entry_price * 10000
        print(f"PnL at end of check: {current_pnl:.1f} bps")

    # What was happening around this time?
    print(f"\n--- MARKET CONTEXT ---")
    # Look at 5 days before and after
    context_start = max(0, idx_pos - 480)  # 5 days before
    context_end = min(len(oos_data), idx_pos + 480)  # 5 days after

    price_before = oos_data.iloc[context_start]['close']
    price_after = oos_data.iloc[context_end - 1]['close']

    print(f"Price 5 days before: ${price_before:,.0f}")
    print(f"Entry price: ${entry_price:,.0f}")
    print(f"Price 5 days after: ${price_after:,.0f}")
    print(f"5-day move before: {(entry_price - price_before) / price_before * 100:+.1f}%")
    print(f"5-day move after: {(price_after - entry_price) / entry_price * 100:+.1f}%")

# Summary
print("\n" + "="*70)
print("SUMMARY: LONG FAILURES")
print("="*70)

if len(failures) > 0:
    print(f"\nTotal LONG failures: {len(failures)}")
    for i, fail in enumerate(failures):
        print(f"\n{i+1}. {fail['timestamp']}")
        print(f"   Entry: ${fail['entry_price']:,.0f}, MAE: {fail['mae']:.0f} bps")
