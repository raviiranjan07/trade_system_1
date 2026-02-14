"""
TEST: Trailing Stop with Activation Threshold
Only START tracking trailing stop AFTER price reaches +X bps profit

User's idea:
- Entry at $70,000
- Trailing stop 15 bps
- Trailing stop INACTIVE until price reaches +15 bps first
- Once activated, track and exit when price drops 15 bps from peak
- Minimum exit = 0 bps (never loss from trailing stop trigger)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")

# Parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
SMA_PERIOD = 200
FEES_BPS = 8
MAX_BARS = 10

# Test different activation thresholds
ACTIVATION_THRESHOLDS = [0, 10, 15, 20, 25, 30]  # 0 = current V1 behavior

print("="*70)
print("TEST: Trailing Stop with Activation Threshold")
print("="*70)
print("""
Concept:
- Trailing stop is INACTIVE until price reaches +X bps
- Once activated, tracks peak and exits when price drops by trailing stop amount
- Guarantees minimum exit = activation threshold - trailing stop
""")

# Load data
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calculate_rsi(df, RSI_PERIOD)
df['sma_200'] = df['close'].rolling(SMA_PERIOD).mean()
df['rsi_oversold_cross'] = (df['rsi'] < RSI_OVERSOLD) & (df['rsi'].shift(1) >= RSI_OVERSOLD)
df['rsi_overbought_cross'] = (df['rsi'] > RSI_OVERBOUGHT) & (df['rsi'].shift(1) <= RSI_OVERBOUGHT)
df['bull_market'] = df['close'] > df['sma_200']
df['bear_market'] = df['close'] < df['sma_200']

oos_data = df['2024-01-01':'2025-12-31'].copy()

def execute_trade_with_activation(data, entry_idx, direction, trailing_stop_bps, activation_threshold):
    """
    Trailing stop with activation threshold.

    - Trailing stop is INACTIVE until price reaches +activation_threshold bps
    - Once activated, tracks peak and exits on trailing stop
    - If never activates, exit at Bar 10 (time exit)
    """

    entry_price = data.iloc[entry_idx + 1]['open']

    activated = False
    highest_profit_after_activation = 0
    exit_profit = None
    exit_bar = None
    exit_reason = None

    for bar in range(1, MAX_BARS + 1):
        future_idx = entry_idx + 1 + bar
        if future_idx >= len(data):
            break

        future_bar = data.iloc[future_idx]

        if direction == 'LONG':
            bar_mfe = (future_bar['high'] - entry_price) / entry_price * 10000
            bar_close_pnl = (future_bar['close'] - entry_price) / entry_price * 10000
        else:
            bar_mfe = (entry_price - future_bar['low']) / entry_price * 10000
            bar_close_pnl = (entry_price - future_bar['close']) / entry_price * 10000

        # Check activation
        if not activated:
            if bar_mfe >= activation_threshold:
                activated = True
                highest_profit_after_activation = bar_mfe

        # If activated, track trailing stop
        if activated:
            if bar_mfe > highest_profit_after_activation:
                highest_profit_after_activation = bar_mfe

            drawdown_from_peak = highest_profit_after_activation - bar_close_pnl

            if drawdown_from_peak >= trailing_stop_bps:
                exit_profit = highest_profit_after_activation - trailing_stop_bps
                exit_bar = bar
                exit_reason = 'TRAILING_STOP'
                break

    # Time exit if no trailing stop hit
    if exit_profit is None:
        final_bar = data.iloc[entry_idx + 1 + MAX_BARS]
        if direction == 'LONG':
            exit_profit = (final_bar['close'] - entry_price) / entry_price * 10000
        else:
            exit_profit = (entry_price - final_bar['close']) / entry_price * 10000
        exit_bar = MAX_BARS
        exit_reason = 'TIME_EXIT'

    return {
        'direction': direction,
        'gross_profit_bps': exit_profit,
        'net_profit_bps': exit_profit - FEES_BPS,
        'exit_bar': exit_bar,
        'exit_reason': exit_reason,
        'activated': activated
    }

def run_backtest(data, trailing_stop, activation_threshold):
    """Run backtest with specified parameters."""
    trades = []

    for idx in data.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + MAX_BARS + 1 >= len(data):
            continue

        current = data.iloc[idx_pos]

        if current['rsi_oversold_cross'] and current['bull_market']:
            trade = execute_trade_with_activation(data, idx_pos, 'LONG', trailing_stop, activation_threshold)
            if trade:
                trades.append(trade)
        elif current['rsi_overbought_cross'] and current['bear_market']:
            trade = execute_trade_with_activation(data, idx_pos, 'SHORT', trailing_stop, activation_threshold)
            if trade:
                trades.append(trade)

    return pd.DataFrame(trades)

# =============================================================================
# TEST ALL COMBINATIONS
# =============================================================================

results = []

# Test LONG with 20 bps trailing stop
print("\n" + "="*70)
print("LONG: 20 bps Trailing Stop + Various Activation Thresholds")
print("="*70)

print(f"\n{'Activation':>12} {'Trades':>7} {'Win%':>7} {'Winners':>8} {'Losers':>7} {'Total Net':>10} {'Avg Net':>9} {'Activated%':>10}")
print("-" * 85)

for activation in ACTIVATION_THRESHOLDS:
    trades = run_backtest(oos_data, 20, activation)
    long_trades = trades[trades['direction'] == 'LONG']

    if len(long_trades) == 0:
        continue

    winners = long_trades[long_trades['net_profit_bps'] > 0]
    activated_count = long_trades[long_trades['activated'] == True]

    result = {
        'type': 'LONG',
        'trailing_stop': 20,
        'activation': activation,
        'trades': len(long_trades),
        'winners': len(winners),
        'losers': len(long_trades) - len(winners),
        'win_rate': len(winners) / len(long_trades) * 100,
        'total_net': long_trades['net_profit_bps'].sum(),
        'avg_net': long_trades['net_profit_bps'].mean(),
        'activated_pct': len(activated_count) / len(long_trades) * 100
    }
    results.append(result)

    print(f"{activation:>12} {result['trades']:>7} {result['win_rate']:>6.1f}% {result['winners']:>8} {result['losers']:>7} {result['total_net']:>+10.0f} {result['avg_net']:>+9.1f} {result['activated_pct']:>9.1f}%")

# Test SHORT with 30 bps trailing stop
print("\n" + "="*70)
print("SHORT: 30 bps Trailing Stop + Various Activation Thresholds")
print("="*70)

print(f"\n{'Activation':>12} {'Trades':>7} {'Win%':>7} {'Winners':>8} {'Losers':>7} {'Total Net':>10} {'Avg Net':>9} {'Activated%':>10}")
print("-" * 85)

for activation in ACTIVATION_THRESHOLDS:
    trades = run_backtest(oos_data, 30, activation)
    short_trades = trades[trades['direction'] == 'SHORT']

    if len(short_trades) == 0:
        continue

    winners = short_trades[short_trades['net_profit_bps'] > 0]
    activated_count = short_trades[short_trades['activated'] == True]

    result = {
        'type': 'SHORT',
        'trailing_stop': 30,
        'activation': activation,
        'trades': len(short_trades),
        'winners': len(winners),
        'losers': len(short_trades) - len(winners),
        'win_rate': len(winners) / len(short_trades) * 100,
        'total_net': short_trades['net_profit_bps'].sum(),
        'avg_net': short_trades['net_profit_bps'].mean(),
        'activated_pct': len(activated_count) / len(short_trades) * 100
    }
    results.append(result)

    print(f"{activation:>12} {result['trades']:>7} {result['win_rate']:>6.1f}% {result['winners']:>8} {result['losers']:>7} {result['total_net']:>+10.0f} {result['avg_net']:>+9.1f} {result['activated_pct']:>9.1f}%")

# =============================================================================
# COMBINED RESULTS
# =============================================================================

print("\n" + "="*70)
print("COMBINED: Best Activation Threshold for Both")
print("="*70)

# Find best for LONG
long_results = [r for r in results if r['type'] == 'LONG']
best_long = max(long_results, key=lambda x: x['total_net'])
v1_long = [r for r in long_results if r['activation'] == 0][0]

print(f"\nLONG (20 bps trailing stop):")
print(f"  V1 (no activation):      {v1_long['total_net']:+.0f} bps, {v1_long['losers']} losers")
print(f"  Best ({best_long['activation']} bps activation): {best_long['total_net']:+.0f} bps, {best_long['losers']} losers")
print(f"  Change: {best_long['total_net'] - v1_long['total_net']:+.0f} bps")

# Find best for SHORT
short_results = [r for r in results if r['type'] == 'SHORT']
best_short = max(short_results, key=lambda x: x['total_net'])
v1_short = [r for r in short_results if r['activation'] == 0][0]

print(f"\nSHORT (30 bps trailing stop):")
print(f"  V1 (no activation):      {v1_short['total_net']:+.0f} bps, {v1_short['losers']} losers")
print(f"  Best ({best_short['activation']} bps activation): {best_short['total_net']:+.0f} bps, {best_short['losers']} losers")
print(f"  Change: {best_short['total_net'] - v1_short['total_net']:+.0f} bps")

# Total
v1_total = v1_long['total_net'] + v1_short['total_net']
best_total = best_long['total_net'] + best_short['total_net']

print(f"\nTOTAL:")
print(f"  V1: {v1_total:+.0f} bps")
print(f"  Best combination: {best_total:+.0f} bps")
print(f"  Change: {best_total - v1_total:+.0f} bps ({(best_total - v1_total)/v1_total*100:+.1f}%)")

# =============================================================================
# RECOMMENDATION
# =============================================================================

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

improvement = best_total - v1_total
if improvement > 100:
    print(f"""
IMPROVEMENT FOUND!

Best settings:
  LONG: {best_long['activation']} bps activation + 20 bps trailing stop
  SHORT: {best_short['activation']} bps activation + 30 bps trailing stop

Improvement: {improvement:+.0f} bps ({improvement/v1_total*100:+.1f}%)

Consider updating V1.
""")
elif improvement > 0:
    print(f"""
MINOR IMPROVEMENT

Improvement: {improvement:+.0f} bps ({improvement/v1_total*100:+.1f}%)

Not significant enough to change V1.
""")
else:
    print(f"""
NO IMPROVEMENT

Activation threshold doesn't help.
V1 (no activation) is already optimal.
""")
