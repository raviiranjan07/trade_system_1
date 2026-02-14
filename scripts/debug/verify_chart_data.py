"""
Verify exact data plotted on the middle chart
"""
import pandas as pd
from pathlib import Path

# Load the same data used in visualization
ohlcv_path = Path('data/ohlcv/BTCUSDT_15m_ohlcv.parquet')
data = pd.read_parquet(ohlcv_path)

# Filter to Jan 1, 2024 - same as visualization
data = data[(data.index >= '2024-01-01') & (data.index < '2024-01-02')].copy()
data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()

# Extract 12:00-16:00 period
period = data[(data.index >= '2024-01-01 12:00') & (data.index <= '2024-01-01 16:00')]

print('EXACT DATA PLOTTED ON MIDDLE CHART (12:00-16:00):')
print('=' * 70)
print(f"{'Time':<20} {'Close':>10} {'EMA7':>10} {'Relation':<15}")
print('=' * 70)

for idx, row in period.iterrows():
    time_str = idx.strftime('%H:%M')
    relation = 'ABOVE EMA7' if row['close'] > row['ema7'] else 'BELOW EMA7'
    print(f"{time_str:<20} {row['close']:>10.2f} {row['ema7']:>10.2f} {relation:<15}")

print('\n' + '=' * 70)
print('PRICE MOVEMENT:')
print(f"12:00 price: ${period.iloc[0]['close']:.2f}")
print(f"16:00 price: ${period.iloc[-1]['close']:.2f}")
print(f"Net change: ${period.iloc[-1]['close'] - period.iloc[0]['close']:.2f}")
print(f"Low in period: ${period['close'].min():.2f}")
print(f"High in period: ${period['close'].max():.2f}")
