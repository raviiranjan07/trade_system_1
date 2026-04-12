"""
TEST: Different Trailing Stop Thresholds
Does wider trailing stop improve results?
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
FEES_BPS = 8
MAX_BARS = 10

# Test different trailing stops
TRAILING_STOPS_TO_TEST = [15, 20, 25, 30, 35, 40, 50]

print("="*70)
print("TEST: Trailing Stop Threshold Optimization")
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

def run_backtest(data, long_ts, short_ts):
    """Run backtest with specified trailing stops."""
    trades = []

    for idx in data.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + MAX_BARS + 1 >= len(data):
            continue

        current = data.iloc[idx_pos]

        # LONG signal
        if current['rsi_oversold_cross'] and current['bull_market']:
            trade = execute_trade(data, idx_pos, 'LONG', long_ts)
            if trade:
                trades.append(trade)

        # SHORT signal
        elif current['rsi_overbought_cross'] and current['bear_market']:
            trade = execute_trade(data, idx_pos, 'SHORT', short_ts)
            if trade:
                trades.append(trade)

    return pd.DataFrame(trades)

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

        # Check trailing stop
        drawdown_from_peak = highest_profit - bar_close_pnl

        if drawdown_from_peak >= trailing_stop_bps and highest_profit > 0:
            exit_profit = highest_profit - trailing_stop_bps
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

    net_profit = exit_profit - FEES_BPS

    return {
        'direction': direction,
        'gross_profit_bps': exit_profit,
        'net_profit_bps': net_profit,
        'exit_bar': exit_bar,
        'exit_reason': exit_reason,
        'mfe': highest_profit
    }

# =============================================================================
# TEST ALL COMBINATIONS
# =============================================================================

results = []

print("\nTesting trailing stop combinations...")
print("-" * 70)

for long_ts in TRAILING_STOPS_TO_TEST:
    for short_ts in TRAILING_STOPS_TO_TEST:
        trades = run_backtest(oos_data, long_ts, short_ts)

        if len(trades) == 0:
            continue

        long_trades = trades[trades['direction'] == 'LONG']
        short_trades = trades[trades['direction'] == 'SHORT']

        winners = trades[trades['net_profit_bps'] > 0]
        losers = trades[trades['net_profit_bps'] <= 0]

        results.append({
            'long_ts': long_ts,
            'short_ts': short_ts,
            'total_trades': len(trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(trades) * 100,
            'total_net': trades['net_profit_bps'].sum(),
            'avg_net': trades['net_profit_bps'].mean(),
            'avg_winner': winners['net_profit_bps'].mean() if len(winners) > 0 else 0,
            'avg_loser': losers['net_profit_bps'].mean() if len(losers) > 0 else 0,
            'long_net': long_trades['net_profit_bps'].sum() if len(long_trades) > 0 else 0,
            'short_net': short_trades['net_profit_bps'].sum() if len(short_trades) > 0 else 0,
        })

results_df = pd.DataFrame(results)

# Sort by total net profit
results_df = results_df.sort_values('total_net', ascending=False)

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

print("\n" + "="*70)
print("TOP 10 TRAILING STOP COMBINATIONS (by Total Net Profit)")
print("="*70)

print(f"\n{'LONG TS':>8} {'SHORT TS':>9} {'Trades':>7} {'Win%':>6} {'Winners':>8} {'Losers':>7} {'Total Net':>10} {'Avg/Trade':>10}")
print("-" * 75)

for _, row in results_df.head(10).iterrows():
    print(f"{row['long_ts']:>8} {row['short_ts']:>9} {row['total_trades']:>7} {row['win_rate']:>5.1f}% {row['winners']:>8} {row['losers']:>7} {row['total_net']:>+10.0f} {row['avg_net']:>+10.1f}")

# Compare to V1 baseline
v1_result = results_df[(results_df['long_ts'] == 20) & (results_df['short_ts'] == 30)].iloc[0]

print("\n" + "="*70)
print("COMPARISON TO V1 (LONG=20, SHORT=30)")
print("="*70)

best = results_df.iloc[0]

print(f"""
V1 Baseline (LONG=20, SHORT=30):
  Total Net: {v1_result['total_net']:+.0f} bps
  Win Rate:  {v1_result['win_rate']:.1f}%
  Winners:   {v1_result['winners']}
  Losers:    {v1_result['losers']}
  Avg/Trade: {v1_result['avg_net']:+.1f} bps

Best Found (LONG={best['long_ts']}, SHORT={best['short_ts']}):
  Total Net: {best['total_net']:+.0f} bps
  Win Rate:  {best['win_rate']:.1f}%
  Winners:   {best['winners']}
  Losers:    {best['losers']}
  Avg/Trade: {best['avg_net']:+.1f} bps

Improvement:
  Total Net: {best['total_net'] - v1_result['total_net']:+.0f} bps ({(best['total_net'] - v1_result['total_net'])/v1_result['total_net']*100:+.1f}%)
  Win Rate:  {best['win_rate'] - v1_result['win_rate']:+.1f}%
  Fewer Losers: {v1_result['losers'] - best['losers']}
""")

# =============================================================================
# ANALYSIS BY DIRECTION
# =============================================================================

print("="*70)
print("BEST TRAILING STOP BY DIRECTION")
print("="*70)

# Best for LONG (varying long_ts, fixed short_ts=30)
print("\nLONG Trailing Stop Analysis (SHORT fixed at 30):")
long_analysis = results_df[results_df['short_ts'] == 30].sort_values('long_net', ascending=False)
print(f"{'LONG TS':>8} {'LONG Net':>10}")
for _, row in long_analysis.head(5).iterrows():
    print(f"{row['long_ts']:>8} {row['long_net']:>+10.0f}")

# Best for SHORT (fixed long_ts=20, varying short_ts)
print("\nSHORT Trailing Stop Analysis (LONG fixed at 20):")
short_analysis = results_df[results_df['long_ts'] == 20].sort_values('short_net', ascending=False)
print(f"{'SHORT TS':>9} {'SHORT Net':>10}")
for _, row in short_analysis.head(5).iterrows():
    print(f"{row['short_ts']:>9} {row['short_net']:>+10.0f}")

# =============================================================================
# RECOMMENDATION
# =============================================================================

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

improvement_pct = (best['total_net'] - v1_result['total_net']) / v1_result['total_net'] * 100

if improvement_pct > 10:
    print(f"""
SIGNIFICANT IMPROVEMENT FOUND!

Best: LONG={best['long_ts']} bps, SHORT={best['short_ts']} bps
  -> +{improvement_pct:.1f}% better than V1
  -> {best['total_net']:+.0f} bps vs {v1_result['total_net']:+.0f} bps

RECOMMEND: Update to V1.1 with new trailing stops
""")
elif improvement_pct > 5:
    print(f"""
MINOR IMPROVEMENT FOUND

Best: LONG={best['long_ts']} bps, SHORT={best['short_ts']} bps
  -> +{improvement_pct:.1f}% better than V1

Consider updating, but not critical.
""")
else:
    print(f"""
NO SIGNIFICANT IMPROVEMENT

Best: LONG={best['long_ts']} bps, SHORT={best['short_ts']} bps
  -> Only +{improvement_pct:.1f}% better than V1

V1 trailing stops (20/30) are already near-optimal.
KEEP V1 as is.
""")
