"""
TEST: Patience Approach (potential V1.1)
Instead of exiting at Bar 10, WAIT for recovery

V1: Exit at Bar 10 if no trailing stop
V1.1: Wait until trailing stop hits (no time exit)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")

# Parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
SMA_PERIOD = 200
LONG_TRAILING_STOP = 20
SHORT_TRAILING_STOP = 30
FEES_BPS = 8

# Extended horizon for patience approach
MAX_BARS_V1 = 10       # V1: Exit at Bar 10
MAX_BARS_V11 = 200     # V1.1: Wait up to 50 hours for recovery

print("="*70)
print("TEST: PATIENCE APPROACH (V1.1)")
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

def run_backtest_patience(data, max_bars):
    """Run backtest with specified max bars."""
    trades = []

    for idx in data.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + max_bars >= len(data):
            continue

        current = data.iloc[idx_pos]

        # LONG signal
        if current['rsi_oversold_cross'] and current['bull_market']:
            trade = execute_trade_patience(data, idx_pos, 'LONG', LONG_TRAILING_STOP, max_bars)
            if trade:
                trades.append(trade)

        # SHORT signal
        elif current['rsi_overbought_cross'] and current['bear_market']:
            trade = execute_trade_patience(data, idx_pos, 'SHORT', SHORT_TRAILING_STOP, max_bars)
            if trade:
                trades.append(trade)

    return pd.DataFrame(trades)

def execute_trade_patience(data, entry_idx, direction, trailing_stop_bps, max_bars):
    """Execute trade with patience - wait for trailing stop or max bars."""

    entry_price = data.iloc[entry_idx + 1]['open']

    highest_profit = 0
    exit_profit = None
    exit_bar = None
    exit_reason = None

    for bar in range(1, max_bars + 1):
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

        # Check trailing stop
        drawdown_from_peak = highest_profit - bar_close_pnl

        if drawdown_from_peak >= trailing_stop_bps and highest_profit > 0:
            exit_profit = highest_profit - trailing_stop_bps
            exit_bar = bar
            exit_reason = 'TRAILING_STOP'
            break

    # If no trailing stop hit, exit at max bars
    if exit_profit is None:
        final_bar = data.iloc[entry_idx + 1 + max_bars]
        if direction == 'LONG':
            exit_profit = (final_bar['close'] - entry_price) / entry_price * 10000
        else:
            exit_profit = (entry_price - final_bar['close']) / entry_price * 10000
        exit_bar = max_bars
        exit_reason = 'TIME_EXIT'

    net_profit = exit_profit - FEES_BPS

    return {
        'direction': direction,
        'gross_profit_bps': exit_profit,
        'net_profit_bps': net_profit,
        'exit_bar': exit_bar,
        'exit_reason': exit_reason,
        'is_winner': net_profit > 0
    }

# =============================================================================
# RUN BOTH VERSIONS
# =============================================================================

print("\nRunning V1 (Bar 10 exit)...")
v1_trades = run_backtest_patience(oos_data, MAX_BARS_V1)

print("Running V1.1 (Bar 200 exit - patience)...")
v11_trades = run_backtest_patience(oos_data, MAX_BARS_V11)

# =============================================================================
# COMPARE RESULTS
# =============================================================================

def get_metrics(trades, version_name):
    winners = trades[trades['net_profit_bps'] > 0]
    losers = trades[trades['net_profit_bps'] <= 0]

    return {
        'version': version_name,
        'total_trades': len(trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(trades) * 100,
        'total_net': trades['net_profit_bps'].sum(),
        'avg_net': trades['net_profit_bps'].mean(),
        'avg_winner': winners['net_profit_bps'].mean() if len(winners) > 0 else 0,
        'avg_loser': losers['net_profit_bps'].mean() if len(losers) > 0 else 0,
        'avg_exit_bar': trades['exit_bar'].mean(),
        'trailing_stop_pct': len(trades[trades['exit_reason'] == 'TRAILING_STOP']) / len(trades) * 100,
        'time_exit_pct': len(trades[trades['exit_reason'] == 'TIME_EXIT']) / len(trades) * 100,
    }

v1_metrics = get_metrics(v1_trades, 'V1 (Bar 10)')
v11_metrics = get_metrics(v11_trades, 'V1.1 (Patience)')

print("\n" + "="*70)
print("COMPARISON: V1 vs V1.1 (Patience)")
print("="*70)

print(f"""
                            V1 (Bar 10)     V1.1 (Patience)
----------------------------------------------------------------
Total Trades:               {v1_metrics['total_trades']:>10}      {v11_metrics['total_trades']:>10}
Winners:                    {v1_metrics['winners']:>10}      {v11_metrics['winners']:>10}
Losers:                     {v1_metrics['losers']:>10}      {v11_metrics['losers']:>10}
Win Rate:                   {v1_metrics['win_rate']:>9.1f}%      {v11_metrics['win_rate']:>9.1f}%

Total Net Profit:           {v1_metrics['total_net']:>+10.1f}      {v11_metrics['total_net']:>+10.1f} bps
Avg Net/Trade:              {v1_metrics['avg_net']:>+10.1f}      {v11_metrics['avg_net']:>+10.1f} bps
Avg Winner:                 {v1_metrics['avg_winner']:>+10.1f}      {v11_metrics['avg_winner']:>+10.1f} bps
Avg Loser:                  {v1_metrics['avg_loser']:>+10.1f}      {v11_metrics['avg_loser']:>+10.1f} bps

Avg Exit Bar:               {v1_metrics['avg_exit_bar']:>10.1f}      {v11_metrics['avg_exit_bar']:>10.1f}
Trailing Stop Exits:        {v1_metrics['trailing_stop_pct']:>9.1f}%      {v11_metrics['trailing_stop_pct']:>9.1f}%
Time Exits:                 {v1_metrics['time_exit_pct']:>9.1f}%      {v11_metrics['time_exit_pct']:>9.1f}%
""")

# =============================================================================
# IMPROVEMENT ANALYSIS
# =============================================================================

print("="*70)
print("IMPROVEMENT ANALYSIS")
print("="*70)

win_rate_improvement = v11_metrics['win_rate'] - v1_metrics['win_rate']
total_profit_improvement = v11_metrics['total_net'] - v1_metrics['total_net']
avg_profit_improvement = v11_metrics['avg_net'] - v1_metrics['avg_net']
losers_reduced = v1_metrics['losers'] - v11_metrics['losers']

print(f"""
Win Rate:       {v1_metrics['win_rate']:.1f}% -> {v11_metrics['win_rate']:.1f}%  ({win_rate_improvement:+.1f}%)
Total Profit:   {v1_metrics['total_net']:.1f} -> {v11_metrics['total_net']:.1f}  ({total_profit_improvement:+.1f} bps)
Avg/Trade:      {v1_metrics['avg_net']:.1f} -> {v11_metrics['avg_net']:.1f}  ({avg_profit_improvement:+.1f} bps)
Losers:         {v1_metrics['losers']} -> {v11_metrics['losers']}  ({losers_reduced} fewer losers)
""")

# =============================================================================
# RECOMMENDATION
# =============================================================================

print("="*70)
print("RECOMMENDATION")
print("="*70)

if total_profit_improvement > 500 and win_rate_improvement > 5:
    print(f"""
SIGNIFICANT IMPROVEMENT FOUND!

V1.1 (Patience) is better:
- Win rate improved by {win_rate_improvement:.1f}%
- Total profit improved by {total_profit_improvement:.1f} bps
- {losers_reduced} fewer losing trades

RECOMMEND: Create V1.1
""")
elif total_profit_improvement > 0:
    print(f"""
MINOR IMPROVEMENT FOUND

V1.1 shows some improvement:
- Win rate: {win_rate_improvement:+.1f}%
- Total profit: {total_profit_improvement:+.1f} bps

But avg exit bar is {v11_metrics['avg_exit_bar']:.1f} (vs {v1_metrics['avg_exit_bar']:.1f})
Capital locked longer for marginal gain.

DECISION: Marginal - user choice whether to adopt V1.1
""")
else:
    print(f"""
NO IMPROVEMENT

V1.1 (Patience) is NOT better:
- Total profit change: {total_profit_improvement:+.1f} bps

KEEP V1 as is.
""")
