"""
Check ALL EMA7 crosses and trades for full day 2024-01-01
"""
import pandas as pd
from pathlib import Path

# Load full day data
data = pd.read_parquet(Path('data/ohlcv/BTCUSDT_15m_ohlcv.parquet'))
data = data[(data.index >= '2024-01-01') & (data.index < '2024-01-02')].copy()
data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()

# Detect ALL crosses
data['above_ema'] = data['close'] > data['ema7']
data['cross_above'] = (data['above_ema'] == True) & (data['above_ema'].shift(1) == False)
data['cross_below'] = (data['above_ema'] == False) & (data['above_ema'].shift(1) == True)

print('ALL EMA7 CROSSES ON 2024-01-01:')
print('=' * 80)

cross_above_times = []
cross_below_times = []

for idx, row in data.iterrows():
    time_str = idx.strftime('%H:%M')
    if row['cross_above']:
        cross_above_times.append(time_str)
        print(f'{time_str:<10} CROSS ABOVE  Price: ${row["close"]:>8.2f} EMA7: ${row["ema7"]:>8.2f}')
    elif row['cross_below']:
        cross_below_times.append(time_str)
        print(f'{time_str:<10} CROSS BELOW   Price: ${row["close"]:>8.2f} EMA7: ${row["ema7"]:>8.2f}')

print(f'\nTotal CROSS ABOVE: {len(cross_above_times)}')
print(f'Total CROSS BELOW: {len(cross_below_times)}')

print('\n' + '=' * 80)
print('TRADES THAT WERE ACTUALLY TAKEN:')
print('=' * 80)

# Load actual trades
trades = pd.read_csv('experiments/ema/two_stage_exit_trades.csv')
trades['entry_time'] = pd.to_datetime(trades['entry_time'])
day_trades = trades[(trades['entry_time'] >= '2024-01-01') & (trades['entry_time'] < '2024-01-02')]

for _, trade in day_trades.iterrows():
    time_str = trade['entry_time'].strftime('%H:%M')
    exit_type = trade['exit_reason']
    pnl = trade['pnl_bps']
    marker = '✓' if pnl > 0 else '✗'

    print(f"{time_str:<10} {trade['direction']:5s} Entry ${trade['entry_price']:>8.2f} -> {exit_type:4s} {marker} {pnl:>6.1f} bps")

print(f'\nTotal trades: {len(day_trades)}')
print(f'  TP exits: {(day_trades["exit_reason"] == "TP").sum()}')
print(f'  MAE exits: {(day_trades["exit_reason"] == "MAE").sum()}')
print(f'  TIME exits: {(day_trades["exit_reason"] == "TIME").sum()}')
