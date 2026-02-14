"""
Check what happened between 00:00 to 04:00
"""
import pandas as pd
from pathlib import Path

# Load data
data = pd.read_parquet(Path('data/ohlcv/BTCUSDT_15m_ohlcv.parquet'))
data = data[(data.index >= '2024-01-01 00:00') & (data.index <= '2024-01-01 04:00')].copy()
data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()

data['above_ema'] = data['close'] > data['ema7']
data['cross_above'] = (data['above_ema'] == True) & (data['above_ema'].shift(1) == False)
data['cross_below'] = (data['above_ema'] == False) & (data['above_ema'].shift(1) == True)

print('WHAT HAPPENED 00:00 to 04:00:')
print('=' * 80)

for idx, row in data.iterrows():
    time_str = idx.strftime('%H:%M')
    relation = 'ABOVE' if row['close'] > row['ema7'] else 'BELOW'

    marker = ''
    if row['cross_above']:
        marker = '<<< CROSS ABOVE (LONG entry)'
    elif row['cross_below']:
        marker = '<<< CROSS BELOW (SHORT entry)'

    print(f'{time_str}  Price: ${row["close"]:>8.2f}  EMA7: ${row["ema7"]:>8.2f}  {relation:5s}  {marker}')

print('\n' + '=' * 80)
print('SUMMARY:')
print(f'Start (00:00): Price ${data.iloc[0]["close"]:.2f}')
print(f'End (04:00):   Price ${data.iloc[-1]["close"]:.2f}')
print(f'Change: ${data.iloc[-1]["close"] - data.iloc[0]["close"]:.2f} ({(data.iloc[-1]["close"] - data.iloc[0]["close"]) / data.iloc[0]["close"] * 10000:.1f} bps)')
print(f'Low: ${data["close"].min():.2f} at {data["close"].idxmin().strftime("%H:%M")}')
print(f'High: ${data["close"].max():.2f} at {data["close"].idxmax().strftime("%H:%M")}')

# Count crosses
crosses_above = data['cross_above'].sum()
crosses_below = data['cross_below'].sum()
print(f'\nCrosses: {crosses_above} ABOVE, {crosses_below} BELOW')
