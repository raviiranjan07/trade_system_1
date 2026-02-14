"""
Profit/Loss Calculation for EMA + RSI Mean Reversion Strategy

Strategy:
- LONG when: Price below EMA9 + below EMA50 + RSI < 30 (oversold)
- SHORT when: Price above EMA9 + above EMA50 + RSI > 70 (overbought)

Exit: Next candle close (simple) OR fixed target/stop
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
print("=" * 80)
print("EMA + RSI MEAN REVERSION - PROFIT/LOSS CALCULATION")
print("=" * 80)

# Load 1-min data
data_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
df_1m = pd.read_parquet(data_path)
df_1m = df_1m.sort_index()

# Resample to 15-min
df = df_1m.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print(f"Data: {len(df):,} 15-minute candles")

# =============================================================================
# CALCULATE FEATURES
# =============================================================================
# EMAs
df['ema9'] = df['close'].ewm(span=9).mean()
df['ema50'] = df['close'].ewm(span=50).mean()

# RSI(17)
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(17).mean()
loss = (-delta.where(delta < 0, 0)).rolling(17).mean()
rs = gain / (loss + 1e-9)
df['rsi17'] = 100 - (100 / (1 + rs))

# Next candle data for exit
df['next_open'] = df['open'].shift(-1)
df['next_high'] = df['high'].shift(-1)
df['next_low'] = df['low'].shift(-1)
df['next_close'] = df['close'].shift(-1)

# =============================================================================
# DEFINE SIGNALS
# =============================================================================
# Bullish: below both EMAs + RSI oversold
df['bullish_signal'] = (
    (df['close'] < df['ema9']) &
    (df['close'] < df['ema50']) &
    (df['rsi17'] < 30)
)

# Bearish: above both EMAs + RSI overbought
df['bearish_signal'] = (
    (df['close'] > df['ema9']) &
    (df['close'] > df['ema50']) &
    (df['rsi17'] > 70)
)

df = df.dropna()

# =============================================================================
# SPLIT DATA
# =============================================================================
train_end = '2023-12-31'
df_train = df[df.index <= train_end].copy()
df_test = df[df.index > train_end].copy()

print(f"\nTrain (2020-2023): {len(df_train):,} candles")
print(f"Test (2024-2025): {len(df_test):,} candles")

# =============================================================================
# STRATEGY 1: Exit at Next Candle Close
# =============================================================================
print("\n" + "=" * 80)
print("STRATEGY 1: Exit at Next Candle Close")
print("=" * 80)

FEE_BPS = 8  # 8 bps round-trip

def calculate_pnl_next_close(data, signal_col, direction):
    """Calculate P&L for exiting at next candle close."""
    signals = data[data[signal_col]].copy()

    if len(signals) == 0:
        return None

    # Entry at next candle open, exit at next candle close
    signals['entry_price'] = signals['next_open']
    signals['exit_price'] = signals['next_close']

    if direction == 'LONG':
        signals['gross_pnl_bps'] = (signals['exit_price'] - signals['entry_price']) / signals['entry_price'] * 10000
    else:  # SHORT
        signals['gross_pnl_bps'] = (signals['entry_price'] - signals['exit_price']) / signals['entry_price'] * 10000

    signals['net_pnl_bps'] = signals['gross_pnl_bps'] - FEE_BPS

    return signals

# LONG trades (bullish signal)
long_trades_train = calculate_pnl_next_close(df_train, 'bullish_signal', 'LONG')
long_trades_test = calculate_pnl_next_close(df_test, 'bullish_signal', 'LONG')

# SHORT trades (bearish signal)
short_trades_train = calculate_pnl_next_close(df_train, 'bearish_signal', 'SHORT')
short_trades_test = calculate_pnl_next_close(df_test, 'bearish_signal', 'SHORT')

def print_results(trades, label):
    if trades is None or len(trades) == 0:
        print(f"\n{label}: No trades")
        return

    n_trades = len(trades)
    win_rate = (trades['gross_pnl_bps'] > 0).mean() * 100
    avg_gross = trades['gross_pnl_bps'].mean()
    avg_net = trades['net_pnl_bps'].mean()
    total_net = trades['net_pnl_bps'].sum()

    print(f"\n{label}:")
    print(f"  Trades: {n_trades:,}")
    print(f"  Win Rate (gross): {win_rate:.1f}%")
    print(f"  Avg Gross P&L: {avg_gross:.2f} bps")
    print(f"  Avg Net P&L (after {FEE_BPS}bp fees): {avg_net:.2f} bps")
    print(f"  Total Net P&L: {total_net:.1f} bps = {total_net/100:.2f}%")

print_results(long_trades_train, "LONG (Oversold) - Train")
print_results(long_trades_test, "LONG (Oversold) - Test (OOS)")
print_results(short_trades_train, "SHORT (Overbought) - Train")
print_results(short_trades_test, "SHORT (Overbought) - Test (OOS)")

# Combined
if long_trades_test is not None and short_trades_test is not None:
    all_test = pd.concat([long_trades_test, short_trades_test])
    print_results(all_test, "COMBINED (Long + Short) - Test (OOS)")

# =============================================================================
# STRATEGY 2: Fixed Target/Stop
# =============================================================================
print("\n" + "=" * 80)
print("STRATEGY 2: Fixed Target/Stop (within next candle)")
print("=" * 80)

def calculate_pnl_target_stop(data, signal_col, direction, target_bps, stop_bps):
    """Calculate P&L with fixed target and stop."""
    signals = data[data[signal_col]].copy()

    if len(signals) == 0:
        return None

    signals['entry_price'] = signals['next_open']

    results = []
    for idx, row in signals.iterrows():
        entry = row['entry_price']

        if direction == 'LONG':
            target_price = entry * (1 + target_bps / 10000)
            stop_price = entry * (1 - stop_bps / 10000)
            hit_target = row['next_high'] >= target_price
            hit_stop = row['next_low'] <= stop_price
        else:  # SHORT
            target_price = entry * (1 - target_bps / 10000)
            stop_price = entry * (1 + stop_bps / 10000)
            hit_target = row['next_low'] <= target_price
            hit_stop = row['next_high'] >= stop_price

        # Determine outcome
        if hit_target and not hit_stop:
            gross_pnl = target_bps
        elif hit_stop and not hit_target:
            gross_pnl = -stop_bps
        elif hit_target and hit_stop:
            # Both hit - use close to determine (simplified)
            if direction == 'LONG':
                gross_pnl = (row['next_close'] - entry) / entry * 10000
            else:
                gross_pnl = (entry - row['next_close']) / entry * 10000
        else:
            # Neither hit - exit at close
            if direction == 'LONG':
                gross_pnl = (row['next_close'] - entry) / entry * 10000
            else:
                gross_pnl = (entry - row['next_close']) / entry * 10000

        results.append({
            'gross_pnl_bps': gross_pnl,
            'net_pnl_bps': gross_pnl - FEE_BPS,
            'hit_target': hit_target and not hit_stop,
            'hit_stop': hit_stop and not hit_target
        })

    return pd.DataFrame(results)

# Test different target/stop combinations
print("\nLONG (Oversold) - Test Data - Different R:R Ratios:")
print("-" * 70)
print(f"{'Target':>8} {'Stop':>8} {'Trades':>8} {'Win%':>8} {'Avg Net':>10} {'Total':>12}")
print("-" * 70)

for target in [15, 20, 25, 30]:
    for stop in [15, 20, 25, 30]:
        result = calculate_pnl_target_stop(df_test, 'bullish_signal', 'LONG', target, stop)
        if result is not None and len(result) > 0:
            win_rate = (result['gross_pnl_bps'] > 0).mean() * 100
            avg_net = result['net_pnl_bps'].mean()
            total = result['net_pnl_bps'].sum()
            print(f"{target:>6}bp {stop:>6}bp {len(result):>8} {win_rate:>7.1f}% {avg_net:>9.2f}bp {total:>10.1f}bp")

print("\nSHORT (Overbought) - Test Data - Different R:R Ratios:")
print("-" * 70)
print(f"{'Target':>8} {'Stop':>8} {'Trades':>8} {'Win%':>8} {'Avg Net':>10} {'Total':>12}")
print("-" * 70)

for target in [15, 20, 25, 30]:
    for stop in [15, 20, 25, 30]:
        result = calculate_pnl_target_stop(df_test, 'bearish_signal', 'SHORT', target, stop)
        if result is not None and len(result) > 0:
            win_rate = (result['gross_pnl_bps'] > 0).mean() * 100
            avg_net = result['net_pnl_bps'].mean()
            total = result['net_pnl_bps'].sum()
            print(f"{target:>6}bp {stop:>6}bp {len(result):>8} {win_rate:>7.1f}% {avg_net:>9.2f}bp {total:>10.1f}bp")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if long_trades_test is not None:
    long_total = long_trades_test['net_pnl_bps'].sum()
    long_trades = len(long_trades_test)
    print(f"\nLONG (Oversold) over 2 years:")
    print(f"  {long_trades} trades, Total P&L: {long_total:.1f} bps = {long_total/100:.2f}%")

if short_trades_test is not None:
    short_total = short_trades_test['net_pnl_bps'].sum()
    short_trades = len(short_trades_test)
    print(f"\nSHORT (Overbought) over 2 years:")
    print(f"  {short_trades} trades, Total P&L: {short_total:.1f} bps = {short_total/100:.2f}%")

print("\nDONE!")
