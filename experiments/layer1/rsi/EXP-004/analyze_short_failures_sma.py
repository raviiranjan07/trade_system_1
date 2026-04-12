"""
EXP-004: Check which market condition each SHORT failure occurred in
Rule: SHORT only when price < SMA200 (bear market)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")

RSI_PERIOD = 14
RSI_OVERBOUGHT = 80
HORIZON = 10

print("="*70)
print("SHORT FAILURES: BULL vs BEAR MARKET ANALYSIS")
print("="*70)

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
df['overbought'] = df['rsi'] > RSI_OVERBOUGHT
df['rsi_overbought_cross'] = (df['overbought'] == True) & (df['overbought'].shift(1) == False)

# OOS data
oos_data = df['2024-01-01':'2025-12-30'].copy()

# Find all SHORT signals and classify failures
signals = oos_data[oos_data['rsi_overbought_cross'] == True].copy()

failures = []
for idx in signals.index:
    idx_pos = oos_data.index.get_loc(idx)
    if idx_pos + HORIZON >= len(oos_data):
        continue

    entry_price = oos_data.iloc[idx_pos]['close']
    sma_200 = oos_data.iloc[idx_pos]['sma_200']
    future_data = oos_data.iloc[idx_pos + 1 : idx_pos + 1 + HORIZON]

    if len(future_data) == 0:
        continue

    # For SHORT: profit if price drops
    mfe = (entry_price - future_data['low'].min()) / entry_price * 10000
    mae = (future_data['high'].max() - entry_price) / entry_price * 10000

    if mfe <= 0:  # Failure
        market = "BULL" if entry_price > sma_200 else "BEAR"
        failures.append({
            'timestamp': idx,
            'entry_price': entry_price,
            'sma_200': sma_200,
            'price_vs_sma': (entry_price - sma_200) / sma_200 * 100,
            'market': market,
            'mae': mae
        })

print(f"\nTotal SHORT failures: {len(failures)}")

# Breakdown
bull_failures = [f for f in failures if f['market'] == 'BULL']
bear_failures = [f for f in failures if f['market'] == 'BEAR']

print(f"\nIn BULL market (price > SMA200): {len(bull_failures)} failures")
print(f"In BEAR market (price < SMA200): {len(bear_failures)} failures")

print("\n" + "="*70)
print("EACH SHORT FAILURE:")
print("="*70)

for i, f in enumerate(failures):
    is_killer = "*** ACCOUNT KILLER ***" if f['mae'] > 300 else ""
    print(f"\n{i+1}. {f['timestamp']}")
    print(f"   Price: ${f['entry_price']:,.0f}, SMA200: ${f['sma_200']:,.0f}")
    print(f"   Price vs SMA200: {f['price_vs_sma']:+.1f}%")
    print(f"   Market: {f['market']}")
    print(f"   MAE: {f['mae']:.0f} bps {is_killer}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"""
Rule: SHORT only when price < SMA200 (bear market)

BULL market failures (would be SKIPPED): {len(bull_failures)}
BEAR market failures (would still happen): {len(bear_failures)}

Failures avoided by filter: {len(bull_failures)}/{len(failures)} = {len(bull_failures)/len(failures)*100:.0f}%
""")

# Check if account killer would be avoided
killer = [f for f in failures if f['mae'] > 300]
if killer:
    print("ACCOUNT KILLER STATUS:")
    for k in killer:
        print(f"  {k['timestamp']} - Market: {k['market']}")
        if k['market'] == 'BULL':
            print("  --> WOULD BE SKIPPED BY 200 SMA FILTER!")
        else:
            print("  --> Would NOT be skipped")
