"""
TEST: Profit-Protected Trailing Stop
Only trigger trailing stop if exit would be PROFITABLE (after fees)

Current: Trailing stop triggers regardless of exit profit
New: Only trigger if (Peak - TS) > fees
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

print("="*70)
print("TEST: Profit-Protected Trailing Stop")
print("="*70)

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

def execute_trade_v1(data, entry_idx, direction, trailing_stop_bps):
    """V1: Original trailing stop (triggers even if exit is loss)"""

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

        # V1: Trigger if drawdown >= trailing stop (regardless of profit)
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
        'mfe': highest_profit
    }

def execute_trade_profit_protected(data, entry_idx, direction, trailing_stop_bps):
    """NEW: Profit-Protected Trailing Stop (only triggers if exit > fees)"""

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
        potential_exit = highest_profit - trailing_stop_bps

        # NEW: Only trigger if exit would be PROFITABLE (after fees)
        if drawdown_from_peak >= trailing_stop_bps and highest_profit > 0:
            if potential_exit > FEES_BPS:  # Only exit if net profit > 0
                exit_profit = potential_exit
                exit_bar = bar
                exit_reason = 'TRAILING_STOP'
                break
            # else: don't trigger, wait for better opportunity

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
        'mfe': highest_profit
    }

def run_backtest(data, execute_func):
    """Run backtest with specified execution function."""
    trades = []

    for idx in data.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + MAX_BARS + 1 >= len(data):
            continue

        current = data.iloc[idx_pos]

        if current['rsi_oversold_cross'] and current['bull_market']:
            trade = execute_func(data, idx_pos, 'LONG', LONG_TRAILING_STOP)
            if trade:
                trades.append(trade)
        elif current['rsi_overbought_cross'] and current['bear_market']:
            trade = execute_func(data, idx_pos, 'SHORT', SHORT_TRAILING_STOP)
            if trade:
                trades.append(trade)

    return pd.DataFrame(trades)

# Run both versions
print("\nRunning V1 (original)...")
v1_trades = run_backtest(oos_data, execute_trade_v1)

print("Running Profit-Protected version...")
pp_trades = run_backtest(oos_data, execute_trade_profit_protected)

# =============================================================================
# COMPARE RESULTS
# =============================================================================

def get_metrics(trades, name):
    winners = trades[trades['net_profit_bps'] > 0]
    losers = trades[trades['net_profit_bps'] <= 0]
    ts_exits = trades[trades['exit_reason'] == 'TRAILING_STOP']
    time_exits = trades[trades['exit_reason'] == 'TIME_EXIT']

    return {
        'name': name,
        'total': len(trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(trades) * 100,
        'total_net': trades['net_profit_bps'].sum(),
        'avg_net': trades['net_profit_bps'].mean(),
        'avg_winner': winners['net_profit_bps'].mean() if len(winners) > 0 else 0,
        'avg_loser': losers['net_profit_bps'].mean() if len(losers) > 0 else 0,
        'ts_exits': len(ts_exits),
        'time_exits': len(time_exits),
    }

v1_m = get_metrics(v1_trades, 'V1 (Original)')
pp_m = get_metrics(pp_trades, 'Profit-Protected')

print("\n" + "="*70)
print("COMPARISON: V1 vs Profit-Protected Trailing Stop")
print("="*70)

print(f"""
                           V1 (Original)    Profit-Protected
----------------------------------------------------------------
Total Trades:              {v1_m['total']:>12}    {pp_m['total']:>12}
Winners:                   {v1_m['winners']:>12}    {pp_m['winners']:>12}
Losers:                    {v1_m['losers']:>12}    {pp_m['losers']:>12}
Win Rate:                  {v1_m['win_rate']:>11.1f}%    {pp_m['win_rate']:>11.1f}%

Total Net Profit:          {v1_m['total_net']:>+12.0f}    {pp_m['total_net']:>+12.0f} bps
Avg Net/Trade:             {v1_m['avg_net']:>+12.1f}    {pp_m['avg_net']:>+12.1f} bps
Avg Winner:                {v1_m['avg_winner']:>+12.1f}    {pp_m['avg_winner']:>+12.1f} bps
Avg Loser:                 {v1_m['avg_loser']:>+12.1f}    {pp_m['avg_loser']:>+12.1f} bps

Trailing Stop Exits:       {v1_m['ts_exits']:>12}    {pp_m['ts_exits']:>12}
Time Exits:                {v1_m['time_exits']:>12}    {pp_m['time_exits']:>12}
""")

# =============================================================================
# ANALYSIS
# =============================================================================

print("="*70)
print("WHAT CHANGED?")
print("="*70)

# Find trades where behavior changed
changed = []
for i in range(len(v1_trades)):
    v1 = v1_trades.iloc[i]
    pp = pp_trades.iloc[i]
    if v1['exit_reason'] != pp['exit_reason'] or abs(v1['net_profit_bps'] - pp['net_profit_bps']) > 0.1:
        changed.append({
            'idx': i,
            'direction': v1['direction'],
            'v1_exit': v1['exit_reason'],
            'v1_bar': v1['exit_bar'],
            'v1_net': v1['net_profit_bps'],
            'pp_exit': pp['exit_reason'],
            'pp_bar': pp['exit_bar'],
            'pp_net': pp['net_profit_bps'],
            'diff': pp['net_profit_bps'] - v1['net_profit_bps']
        })

changed_df = pd.DataFrame(changed)

print(f"\nTrades with different behavior: {len(changed_df)}")

if len(changed_df) > 0:
    print("\nFirst 10 examples:")
    print(f"{'Dir':>6} {'V1 Exit':>12} {'V1 Bar':>7} {'V1 Net':>8} {'PP Exit':>12} {'PP Bar':>7} {'PP Net':>8} {'Diff':>8}")
    print("-" * 80)

    for _, row in changed_df.head(10).iterrows():
        print(f"{row['direction']:>6} {row['v1_exit']:>12} {row['v1_bar']:>7} {row['v1_net']:>+8.1f} {row['pp_exit']:>12} {row['pp_bar']:>7} {row['pp_net']:>+8.1f} {row['diff']:>+8.1f}")

    # Summary of changes
    improved = changed_df[changed_df['diff'] > 0]
    worsened = changed_df[changed_df['diff'] < 0]

    print(f"\n\nSummary of changed trades:")
    print(f"  Improved (PP better): {len(improved)} trades, total: {improved['diff'].sum():+.1f} bps")
    print(f"  Worsened (V1 better): {len(worsened)} trades, total: {worsened['diff'].sum():+.1f} bps")
    print(f"  Net change: {changed_df['diff'].sum():+.1f} bps")

# =============================================================================
# RECOMMENDATION
# =============================================================================

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

improvement = pp_m['total_net'] - v1_m['total_net']
improvement_pct = improvement / v1_m['total_net'] * 100

if improvement > 100:
    print(f"""
IMPROVEMENT FOUND!

Profit-Protected trailing stop is better:
  - Total: {pp_m['total_net']:+.0f} vs {v1_m['total_net']:+.0f} bps
  - Improvement: {improvement:+.0f} bps ({improvement_pct:+.1f}%)
  - Fewer losers: {v1_m['losers']} -> {pp_m['losers']}

RECOMMEND: Update to Profit-Protected trailing stop
""")
elif improvement > 0:
    print(f"""
MINOR IMPROVEMENT

Profit-Protected is slightly better:
  - Improvement: {improvement:+.0f} bps ({improvement_pct:+.1f}%)

Consider updating, but not significant.
""")
else:
    print(f"""
NO IMPROVEMENT / WORSE

Profit-Protected trailing stop: {pp_m['total_net']:+.0f} bps
V1 Original:                   {v1_m['total_net']:+.0f} bps
Change:                        {improvement:+.0f} bps

KEEP V1 as is.
""")
