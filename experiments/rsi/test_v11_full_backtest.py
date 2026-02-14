"""
FULL BACKTEST: V1 vs V1.1
V1.1 Change: SHORT trailing stop activates only after +20 bps profit

Validate on:
1. Full OOS period (2024-2025)
2. Year-by-year (2024 vs 2025) - check consistency
3. Direction breakdown
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

# V1.1 change
SHORT_ACTIVATION_THRESHOLD = 20  # Only activate trailing stop after +20 bps

print("="*70)
print("FULL BACKTEST: V1 vs V1.1")
print("="*70)
print("""
V1:   Trailing stop active from Bar 1 (both LONG and SHORT)
V1.1: LONG = same as V1
      SHORT = trailing stop activates only after +20 bps profit
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

def execute_trade_v1(data, entry_idx, direction, trailing_stop_bps):
    """V1: Trailing stop active from Bar 1."""
    entry_price = data.iloc[entry_idx + 1]['open']
    highest_profit = 0

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
            return {
                'direction': direction,
                'entry_date': data.index[entry_idx],
                'entry_price': entry_price,
                'gross_profit_bps': highest_profit - trailing_stop_bps,
                'net_profit_bps': highest_profit - trailing_stop_bps - FEES_BPS,
                'exit_bar': bar,
                'exit_reason': 'TRAILING_STOP',
                'mfe': highest_profit
            }

    # Time exit
    final_bar = data.iloc[entry_idx + 1 + MAX_BARS]
    if direction == 'LONG':
        exit_profit = (final_bar['close'] - entry_price) / entry_price * 10000
    else:
        exit_profit = (entry_price - final_bar['close']) / entry_price * 10000

    return {
        'direction': direction,
        'entry_date': data.index[entry_idx],
        'entry_price': entry_price,
        'gross_profit_bps': exit_profit,
        'net_profit_bps': exit_profit - FEES_BPS,
        'exit_bar': MAX_BARS,
        'exit_reason': 'TIME_EXIT',
        'mfe': highest_profit
    }

def execute_trade_v11(data, entry_idx, direction, trailing_stop_bps):
    """V1.1: SHORT trailing stop activates only after +20 bps."""
    entry_price = data.iloc[entry_idx + 1]['open']
    highest_profit = 0
    activated = (direction == 'LONG')  # LONG: always active. SHORT: needs activation

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

        # Activation check for SHORT
        if not activated and bar_mfe >= SHORT_ACTIVATION_THRESHOLD:
            activated = True

        if activated:
            if bar_mfe > highest_profit:
                highest_profit = bar_mfe

            drawdown_from_peak = highest_profit - bar_close_pnl
            if drawdown_from_peak >= trailing_stop_bps and highest_profit > 0:
                return {
                    'direction': direction,
                    'entry_date': data.index[entry_idx],
                    'entry_price': entry_price,
                    'gross_profit_bps': highest_profit - trailing_stop_bps,
                    'net_profit_bps': highest_profit - trailing_stop_bps - FEES_BPS,
                    'exit_bar': bar,
                    'exit_reason': 'TRAILING_STOP',
                    'mfe': highest_profit
                }
        else:
            # Track MFE even when not activated (for reference)
            if bar_mfe > highest_profit:
                highest_profit = bar_mfe

    # Time exit
    final_bar = data.iloc[entry_idx + 1 + MAX_BARS]
    if direction == 'LONG':
        exit_profit = (final_bar['close'] - entry_price) / entry_price * 10000
    else:
        exit_profit = (entry_price - final_bar['close']) / entry_price * 10000

    return {
        'direction': direction,
        'entry_date': data.index[entry_idx],
        'entry_price': entry_price,
        'gross_profit_bps': exit_profit,
        'net_profit_bps': exit_profit - FEES_BPS,
        'exit_bar': MAX_BARS,
        'exit_reason': 'TIME_EXIT',
        'mfe': highest_profit
    }

def run_backtest(data, version='v1'):
    """Run full backtest."""
    trades = []

    for idx in data.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + MAX_BARS + 1 >= len(data):
            continue

        current = data.iloc[idx_pos]

        if current['rsi_oversold_cross'] and current['bull_market']:
            if version == 'v1':
                trade = execute_trade_v1(data, idx_pos, 'LONG', LONG_TRAILING_STOP)
            else:
                trade = execute_trade_v11(data, idx_pos, 'LONG', LONG_TRAILING_STOP)
            trades.append(trade)

        elif current['rsi_overbought_cross'] and current['bear_market']:
            if version == 'v1':
                trade = execute_trade_v1(data, idx_pos, 'SHORT', SHORT_TRAILING_STOP)
            else:
                trade = execute_trade_v11(data, idx_pos, 'SHORT', SHORT_TRAILING_STOP)
            trades.append(trade)

    return pd.DataFrame(trades)

def print_metrics(trades, name):
    """Print comprehensive metrics."""
    winners = trades[trades['net_profit_bps'] > 0]
    losers = trades[trades['net_profit_bps'] <= 0]
    long_trades = trades[trades['direction'] == 'LONG']
    short_trades = trades[trades['direction'] == 'SHORT']

    pf = abs(winners['net_profit_bps'].sum() / losers['net_profit_bps'].sum()) if len(losers) > 0 and losers['net_profit_bps'].sum() != 0 else 999

    print(f"\n--- {name} ---")
    print(f"  Total Trades:    {len(trades)}")
    print(f"  Winners:         {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
    print(f"  Losers:          {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")
    print(f"  Total Net:       {trades['net_profit_bps'].sum():+.0f} bps")
    print(f"  Avg Net/Trade:   {trades['net_profit_bps'].mean():+.1f} bps")
    print(f"  Avg Winner:      {winners['net_profit_bps'].mean():+.1f} bps" if len(winners) > 0 else "  Avg Winner:      N/A")
    print(f"  Avg Loser:       {losers['net_profit_bps'].mean():+.1f} bps" if len(losers) > 0 else "  Avg Loser:       N/A")
    print(f"  Max Winner:      {trades['net_profit_bps'].max():+.1f} bps")
    print(f"  Max Loser:       {trades['net_profit_bps'].min():+.1f} bps")
    print(f"  Profit Factor:   {pf:.2f}")
    print(f"  LONG trades:     {len(long_trades)} (net: {long_trades['net_profit_bps'].sum():+.0f} bps)")
    print(f"  SHORT trades:    {len(short_trades)} (net: {short_trades['net_profit_bps'].sum():+.0f} bps)")

    return {
        'name': name,
        'trades': len(trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners)/len(trades)*100,
        'total_net': trades['net_profit_bps'].sum(),
        'avg_net': trades['net_profit_bps'].mean(),
        'pf': pf,
        'long_net': long_trades['net_profit_bps'].sum(),
        'short_net': short_trades['net_profit_bps'].sum(),
    }

# =============================================================================
# 1. FULL OOS PERIOD (2024-2025)
# =============================================================================

print("\n" + "="*70)
print("1. FULL OOS PERIOD (2024-2025)")
print("="*70)

v1_trades = run_backtest(oos_data, 'v1')
v11_trades = run_backtest(oos_data, 'v11')

v1_m = print_metrics(v1_trades, 'V1 (Original)')
v11_m = print_metrics(v11_trades, 'V1.1 (SHORT Activation)')

# =============================================================================
# 2. YEAR-BY-YEAR CONSISTENCY
# =============================================================================

print("\n" + "="*70)
print("2. YEAR-BY-YEAR CONSISTENCY CHECK")
print("="*70)

data_2024 = df['2024-01-01':'2024-12-31'].copy()
data_2025 = df['2025-01-01':'2025-12-31'].copy()

print("\n--- 2024 ---")
v1_2024 = run_backtest(data_2024, 'v1')
v11_2024 = run_backtest(data_2024, 'v11')

v1_m_2024 = print_metrics(v1_2024, 'V1 - 2024')
v11_m_2024 = print_metrics(v11_2024, 'V1.1 - 2024')

print(f"\n  2024 Change: {v11_m_2024['total_net'] - v1_m_2024['total_net']:+.0f} bps")

print("\n--- 2025 ---")
v1_2025 = run_backtest(data_2025, 'v1')
v11_2025 = run_backtest(data_2025, 'v11')

v1_m_2025 = print_metrics(v1_2025, 'V1 - 2025')
v11_m_2025 = print_metrics(v11_2025, 'V1.1 - 2025')

print(f"\n  2025 Change: {v11_m_2025['total_net'] - v1_m_2025['total_net']:+.0f} bps")

# =============================================================================
# 3. TRADE-BY-TRADE DIFFERENCES (SHORT only)
# =============================================================================

print("\n" + "="*70)
print("3. TRADE-BY-TRADE DIFFERENCES (SHORT only)")
print("="*70)

v1_short = v1_trades[v1_trades['direction'] == 'SHORT'].reset_index(drop=True)
v11_short = v11_trades[v11_trades['direction'] == 'SHORT'].reset_index(drop=True)

changed = 0
improved = 0
worsened = 0
improved_bps = 0
worsened_bps = 0

for i in range(len(v1_short)):
    v1_net = v1_short.iloc[i]['net_profit_bps']
    v11_net = v11_short.iloc[i]['net_profit_bps']
    diff = v11_net - v1_net

    if abs(diff) > 0.1:
        changed += 1
        if diff > 0:
            improved += 1
            improved_bps += diff
        else:
            worsened += 1
            worsened_bps += diff

print(f"""
SHORT Trades Changed: {changed}/{len(v1_short)}
  Improved (V1.1 better): {improved} trades ({improved_bps:+.0f} bps)
  Worsened (V1 better):   {worsened} trades ({worsened_bps:+.0f} bps)
  Net change:             {improved_bps + worsened_bps:+.0f} bps
""")

# =============================================================================
# 4. FINAL SUMMARY
# =============================================================================

print("="*70)
print("FINAL SUMMARY")
print("="*70)

total_improvement = v11_m['total_net'] - v1_m['total_net']
improvement_2024 = v11_m_2024['total_net'] - v1_m_2024['total_net']
improvement_2025 = v11_m_2025['total_net'] - v1_m_2025['total_net']

print(f"""
                              V1            V1.1          Change
─────────────────────────────────────────────────────────────────
Full Period (2024-2025):
  Total Net:              {v1_m['total_net']:>+8.0f} bps   {v11_m['total_net']:>+8.0f} bps   {total_improvement:>+8.0f} bps
  Win Rate:               {v1_m['win_rate']:>7.1f}%     {v11_m['win_rate']:>7.1f}%     {v11_m['win_rate']-v1_m['win_rate']:>+7.1f}%
  Profit Factor:          {v1_m['pf']:>8.2f}      {v11_m['pf']:>8.2f}      {v11_m['pf']-v1_m['pf']:>+8.2f}
  Losers:                 {v1_m['losers']:>8}      {v11_m['losers']:>8}      {v11_m['losers']-v1_m['losers']:>+8}

Year-by-Year:
  2024:                   {v1_m_2024['total_net']:>+8.0f} bps   {v11_m_2024['total_net']:>+8.0f} bps   {improvement_2024:>+8.0f} bps
  2025:                   {v1_m_2025['total_net']:>+8.0f} bps   {v11_m_2025['total_net']:>+8.0f} bps   {improvement_2025:>+8.0f} bps

Consistency: V1.1 better in {'BOTH' if improvement_2024 > 0 and improvement_2025 > 0 else 'ONE'} year(s)
""")

# =============================================================================
# RECOMMENDATION
# =============================================================================

print("="*70)
print("RECOMMENDATION")
print("="*70)

both_years_better = improvement_2024 > 0 and improvement_2025 > 0

if total_improvement > 100 and both_years_better:
    print(f"""
V1.1 IS VALIDATED!

Improvement consistent across both years:
  2024: {improvement_2024:+.0f} bps
  2025: {improvement_2025:+.0f} bps
  Total: {total_improvement:+.0f} bps ({total_improvement/v1_m['total_net']*100:.1f}%)

Change: SHORT trailing stop activates only after +20 bps profit
LONG: unchanged

RECOMMEND: Create V1.1
""")
elif total_improvement > 0 and both_years_better:
    print(f"""
V1.1 shows consistent but small improvement.

Both years positive:
  2024: {improvement_2024:+.0f} bps
  2025: {improvement_2025:+.0f} bps
  Total: {total_improvement:+.0f} bps

Consider adopting if improvement is meaningful enough.
""")
elif total_improvement > 0 and not both_years_better:
    print(f"""
CAUTION: V1.1 improvement is NOT consistent!

  2024: {improvement_2024:+.0f} bps
  2025: {improvement_2025:+.0f} bps

Improvement only in one year = possible OVERFITTING.
KEEP V1 as is.
""")
else:
    print(f"""
V1.1 is NOT better. KEEP V1.
""")
