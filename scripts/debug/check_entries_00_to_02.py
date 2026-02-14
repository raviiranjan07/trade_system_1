"""
Check entries between 00:00 to 02:00 on 2024-01-01
Show what the filtered strategy would do
"""
import pandas as pd
from pathlib import Path

# Load data
ohlcv_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
data = pd.read_parquet(ohlcv_path)

# Filter to 00:00-02:00 on 2024-01-01
data = data[(data.index >= "2024-01-01 00:00") & (data.index <= "2024-01-01 02:00")].copy()
data['ema7'] = data['close'].ewm(span=7, adjust=False).mean()

# Calculate crosses
data['above_ema'] = data['close'] > data['ema7']
data['below_ema'] = data['close'] < data['ema7']

data['cross_above'] = (data['above_ema'] == True) & (data['above_ema'].shift(1) == False)
data['cross_below'] = (data['below_ema'] == True) & (data['below_ema'].shift(1) == False)

# Calculate distance from EMA (for filter)
MIN_DISTANCE_BPS = 5
data['distance_from_ema_bps'] = abs(data['close'] - data['ema7']) / data['close'] * 10000

# Apply filter
data['valid_cross_above'] = data['cross_above'] & (data['distance_from_ema_bps'] >= MIN_DISTANCE_BPS)
data['valid_cross_below'] = data['cross_below'] & (data['distance_from_ema_bps'] >= MIN_DISTANCE_BPS)

print("=" * 80)
print("ENTRIES BETWEEN 00:00 to 02:00 on 2024-01-01")
print("=" * 80)
print("\nSTRATEGY: Cross EMA7 with >= 5 bps distance\n")

# Show all bars
for idx, row in data.iterrows():
    time_str = idx.strftime('%H:%M')
    price = row['close']
    ema = row['ema7']
    distance = row['distance_from_ema_bps']
    relation = 'ABOVE' if row['close'] > row['ema7'] else 'BELOW'

    marker = ''
    if row['valid_cross_above']:
        marker = '>>> LONG ENTRY (cross above, distance={:.1f} bps)'.format(distance)
    elif row['valid_cross_below']:
        marker = '>>> SHORT ENTRY (cross below, distance={:.1f} bps)'.format(distance)
    elif row['cross_above']:
        marker = 'XXX SKIPPED (cross above but distance={:.1f} bps < 5)'.format(distance)
    elif row['cross_below']:
        marker = 'XXX SKIPPED (cross below but distance={:.1f} bps < 5)'.format(distance)

    print(f'{time_str}  Price: ${price:>8.2f}  EMA7: ${ema:>8.2f}  {relation:5s}  Dist: {distance:>5.1f} bps  {marker}')

# Summary
long_entries = data['valid_cross_above'].sum()
short_entries = data['valid_cross_below'].sum()
skipped_above = (data['cross_above'] & ~data['valid_cross_above']).sum()
skipped_below = (data['cross_below'] & ~data['valid_cross_below']).sum()

print("\n" + "=" * 80)
print("SUMMARY:")
print(f"LONG entries: {long_entries}")
print(f"SHORT entries: {short_entries}")
print(f"Total entries: {long_entries + short_entries}")
print(f"Skipped (too close to EMA): {skipped_above + skipped_below}")
print("=" * 80)
