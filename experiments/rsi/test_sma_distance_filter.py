"""
TEST: SMA200 Distance Filter
Skip trades when price is too close to SMA200
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
LONG_TRAILING_STOP = 20
SHORT_TRAILING_STOP = 30
FEES_BPS = 8
MAX_BARS = 10

# Test different distance thresholds
DISTANCE_THRESHOLDS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

print("="*70)
print("TEST: SMA200 Distance Filter")
print("="*70)
print("Skip trades when price is within X% of SMA200")

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

def execute_trade(data, entry_idx, direction, trailing_stop_bps):
    """Execute trade with trailing stop."""
    entry_price = data.iloc[entry_idx + 1]['open']
    highest_profit = 0
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

        if bar_mfe > highest_profit:
            highest_profit = bar_mfe

        drawdown_from_peak = highest_profit - bar_close_pnl

        if drawdown_from_peak >= trailing_stop_bps and highest_profit > 0:
            exit_profit = highest_profit - trailing_stop_bps
            exit_bar = bar
            exit_reason = 'TRAILING_STOP'
            break

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
    }

def run_backtest_with_distance_filter(data, min_distance_pct):
    """Run backtest, skipping trades within min_distance_pct of SMA200."""
    trades = []
    skipped = 0

    for idx in data.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + MAX_BARS + 1 >= len(data):
            continue

        current = data.iloc[idx_pos]
        direction = None

        if current['rsi_oversold_cross'] and current['bull_market']:
            direction = 'LONG'
            trailing_stop = LONG_TRAILING_STOP
        elif current['rsi_overbought_cross'] and current['bear_market']:
            direction = 'SHORT'
            trailing_stop = SHORT_TRAILING_STOP
        else:
            continue

        # Distance filter
        distance = abs((current['close'] - current['sma_200']) / current['sma_200'] * 100)
        if distance < min_distance_pct:
            skipped += 1
            continue

        trade = execute_trade(data, idx_pos, direction, trailing_stop)
        if trade:
            trades.append(trade)

    return pd.DataFrame(trades), skipped

# =============================================================================
# TEST ALL DISTANCE THRESHOLDS
# =============================================================================

print(f"\n{'Distance':>10} {'Trades':>7} {'Skipped':>8} {'Win%':>7} {'Winners':>8} {'Losers':>7} {'Total Net':>10} {'Avg Net':>9} {'Avg Win':>8} {'Avg Loss':>9} {'PF':>6}")
print("-" * 105)

results = []

for dist in DISTANCE_THRESHOLDS:
    trades, skipped = run_backtest_with_distance_filter(oos_data, dist)

    if len(trades) == 0:
        continue

    winners = trades[trades['net_profit_bps'] > 0]
    losers = trades[trades['net_profit_bps'] <= 0]

    pf = abs(winners['net_profit_bps'].sum() / losers['net_profit_bps'].sum()) if len(losers) > 0 and losers['net_profit_bps'].sum() != 0 else 999

    result = {
        'distance': dist,
        'trades': len(trades),
        'skipped': skipped,
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(trades) * 100,
        'total_net': trades['net_profit_bps'].sum(),
        'avg_net': trades['net_profit_bps'].mean(),
        'avg_winner': winners['net_profit_bps'].mean() if len(winners) > 0 else 0,
        'avg_loser': losers['net_profit_bps'].mean() if len(losers) > 0 else 0,
        'profit_factor': pf
    }
    results.append(result)

    print(f"{dist:>9.1f}% {result['trades']:>7} {result['skipped']:>8} {result['win_rate']:>6.1f}% {result['winners']:>8} {result['losers']:>7} {result['total_net']:>+10.0f} {result['avg_net']:>+9.1f} {result['avg_winner']:>+8.1f} {result['avg_loser']:>+9.1f} {result['profit_factor']:>6.2f}")

results_df = pd.DataFrame(results)

# =============================================================================
# COMPARISON TO V1
# =============================================================================

print("\n" + "="*70)
print("COMPARISON TO V1 (0% distance = no filter)")
print("="*70)

v1 = results_df[results_df['distance'] == 0].iloc[0]
best = results_df.loc[results_df['total_net'].idxmax()]

print(f"""
V1 Baseline (no distance filter):
  Trades:    {v1['trades']:.0f}
  Win Rate:  {v1['win_rate']:.1f}%
  Total Net: {v1['total_net']:+.0f} bps
  Avg Net:   {v1['avg_net']:+.1f} bps
  Losers:    {v1['losers']:.0f}
  PF:        {v1['profit_factor']:.2f}

Best Distance Filter ({best['distance']:.1f}%):
  Trades:    {best['trades']:.0f} ({v1['trades'] - best['trades']:.0f} skipped)
  Win Rate:  {best['win_rate']:.1f}%
  Total Net: {best['total_net']:+.0f} bps
  Avg Net:   {best['avg_net']:+.1f} bps
  Losers:    {best['losers']:.0f}
  PF:        {best['profit_factor']:.2f}

Changes:
  Total Net:  {best['total_net'] - v1['total_net']:+.0f} bps ({(best['total_net'] - v1['total_net'])/v1['total_net']*100:+.1f}%)
  Win Rate:   {best['win_rate'] - v1['win_rate']:+.1f}%
  Fewer Losers: {v1['losers'] - best['losers']:.0f}
  Trades Lost:  {v1['trades'] - best['trades']:.0f}
""")

# =============================================================================
# ALSO CHECK: By direction
# =============================================================================

print("="*70)
print("BREAKDOWN BY DIRECTION (at best distance)")
print("="*70)

best_dist = best['distance']
trades_at_best, _ = run_backtest_with_distance_filter(oos_data, best_dist)

for direction in ['LONG', 'SHORT']:
    dir_trades = trades_at_best[trades_at_best['direction'] == direction]
    if len(dir_trades) == 0:
        continue

    winners = dir_trades[dir_trades['net_profit_bps'] > 0]
    losers = dir_trades[dir_trades['net_profit_bps'] <= 0]

    print(f"\n{direction}:")
    print(f"  Trades:    {len(dir_trades)}")
    print(f"  Winners:   {len(winners)}")
    print(f"  Losers:    {len(losers)}")
    print(f"  Win Rate:  {len(winners)/len(dir_trades)*100:.1f}%")
    print(f"  Total Net: {dir_trades['net_profit_bps'].sum():+.0f} bps")
    print(f"  Avg Net:   {dir_trades['net_profit_bps'].mean():+.1f} bps")

# =============================================================================
# RECOMMENDATION
# =============================================================================

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

improvement = best['total_net'] - v1['total_net']
trades_lost = v1['trades'] - best['trades']

if improvement > 200 and trades_lost < 50:
    print(f"""
IMPROVEMENT FOUND!

Add distance filter: skip when price is within {best_dist:.1f}% of SMA200
  -> +{improvement:.0f} bps improvement ({improvement/v1['total_net']*100:.1f}%)
  -> Only lose {trades_lost:.0f} trades
  -> {v1['losers'] - best['losers']:.0f} fewer losers

RECOMMEND: Add this filter to V1.1
""")
elif improvement > 0:
    print(f"""
MINOR IMPROVEMENT ({improvement:+.0f} bps)

Filter at {best_dist:.1f}% removes {trades_lost:.0f} trades for {improvement:.0f} bps gain.
Not clearly worth the added complexity.
""")
else:
    print(f"""
NO IMPROVEMENT

Distance filter doesn't help overall.
KEEP V1 as is.
""")
