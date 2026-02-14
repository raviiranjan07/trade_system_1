"""
Show details of the first losing trade on 2024-01-01
"""

import pandas as pd

# Load trades
trades = pd.read_csv('experiments/ema/ema_breakout_trades.csv')
trades['entry_time'] = pd.to_datetime(trades['entry_time'])

# Filter for 2024-01-01
day_trades = trades[
    (trades['entry_time'] >= '2024-01-01') &
    (trades['entry_time'] < '2024-01-02')
]

# Get losers only (TIME exits)
losers = day_trades[day_trades['exit_reason'] == 'TIME'].sort_values('entry_time')

print("=" * 80)
print("ALL LOSERS on 2024-01-01 (TIME exits):")
print("=" * 80)
print(losers[['entry_time', 'entry_price', 'exit_price', 'direction', 'pnl_bps', 'bars_held']].to_string(index=False))

print("\n" + "=" * 80)
print("FIRST LOSER - DETAILED INFO:")
print("=" * 80)

first = losers.iloc[0]
exit_time = first['entry_time'] + pd.Timedelta(minutes=int(first['bars_held']) * 15)

print(f"\nEntry Time:    {first['entry_time']}")
print(f"Entry Price:   ${first['entry_price']:.2f}")
print(f"Direction:     {first['direction']}")
print(f"\nBars Held:     {int(first['bars_held'])} bars = {int(first['bars_held']) * 15} minutes = {int(first['bars_held']) * 15 / 60:.1f} hours")
print(f"\nExit Time:     {exit_time}")
print(f"Exit Price:    ${first['exit_price']:.2f}")
print(f"\nP&L:           {first['pnl_bps']:.2f} bps")
print(f"Exit Reason:   {first['exit_reason']} (max time reached)")

# Calculate what would have been needed
if first['direction'] == 'LONG':
    tp_price = first['entry_price'] * 1.0012
    move_direction = "UP"
    actual_move_bps = (first['exit_price'] - first['entry_price']) / first['entry_price'] * 10000
    print(f"\nNeeded price:  ${tp_price:.2f} (move UP by 12 bps)")
    print(f"Actual move:   {actual_move_bps:.2f} bps (moved DOWN)")
else:
    tp_price = first['entry_price'] * 0.9988
    move_direction = "DOWN"
    actual_move_bps = (first['entry_price'] - first['exit_price']) / first['entry_price'] * 10000
    print(f"\nNeeded price:  ${tp_price:.2f} (move DOWN by 12 bps)")
    print(f"Actual move:   {actual_move_bps:.2f} bps (moved UP)")

print("\n" + "=" * 80)
