"""
CHECK: The 14 trades that NEVER reach profit
- Are they above/below SMA200?
- What happened to them?
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")

RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
SMA_PERIOD = 200
FEES_BPS = 8

print("="*70)
print("CHECK: 14 Trades That NEVER Reach Profit")
print("="*70)

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

# Find the 14 never-recover trades
never_recover_trades = []

for idx in oos_data.index:
    idx_pos = oos_data.index.get_loc(idx)
    if idx_pos + 210 >= len(oos_data):
        continue

    current = oos_data.iloc[idx_pos]
    direction = None

    if current['rsi_oversold_cross'] and current['bull_market']:
        direction = 'LONG'
        trailing_stop = 20
    elif current['rsi_overbought_cross'] and current['bear_market']:
        direction = 'SHORT'
        trailing_stop = 30
    else:
        continue

    entry_price = oos_data.iloc[idx_pos + 1]['open']
    entry_sma = current['sma_200']
    entry_distance_from_sma = (entry_price - entry_sma) / entry_sma * 100  # percentage

    # Check if it EVER reaches net profit within 200 bars
    ever_profit = False
    max_mfe = 0
    max_mae = 0
    bar_200_profit = None

    # Also check: does SMA200 relationship change during trade?
    sma_flipped = False
    sma_flip_bar = None

    for bar in range(1, 201):
        future_idx = idx_pos + 1 + bar
        if future_idx >= len(oos_data):
            break

        future_bar = oos_data.iloc[future_idx]

        if direction == 'LONG':
            bar_mfe = (future_bar['high'] - entry_price) / entry_price * 10000
            bar_close_pnl = (future_bar['close'] - entry_price) / entry_price * 10000
            # Check SMA flip
            if not sma_flipped and future_bar['close'] < future_bar['sma_200'] if 'sma_200' in oos_data.columns else False:
                sma_flipped = True
                sma_flip_bar = bar
        else:
            bar_mfe = (entry_price - future_bar['low']) / entry_price * 10000
            bar_close_pnl = (entry_price - future_bar['close']) / entry_price * 10000
            # Check SMA flip
            if not sma_flipped and future_bar['close'] > future_bar['sma_200']:
                sma_flipped = True
                sma_flip_bar = bar

        if bar_mfe > max_mfe:
            max_mfe = bar_mfe
        if bar_close_pnl < max_mae:
            max_mae = bar_close_pnl

        if bar_close_pnl - FEES_BPS > 0:
            ever_profit = True

        if bar == 200:
            bar_200_profit = bar_close_pnl

    if not ever_profit:
        never_recover_trades.append({
            'date': idx.strftime('%Y-%m-%d %H:%M'),
            'direction': direction,
            'entry_price': entry_price,
            'entry_sma200': entry_sma,
            'above_sma200': 'YES' if entry_price > entry_sma else 'NO',
            'distance_from_sma200_pct': entry_distance_from_sma,
            'max_mfe_bps': max_mfe,
            'max_mae_bps': max_mae,
            'bar_200_profit': bar_200_profit,
            'sma_flipped': sma_flipped,
            'sma_flip_bar': sma_flip_bar,
            'rsi_value': current['rsi']
        })

nr_df = pd.DataFrame(never_recover_trades)

print(f"\nTotal trades that NEVER reach net profit: {len(nr_df)}")

print(f"\nBy direction:")
print(f"  LONG:  {len(nr_df[nr_df['direction'] == 'LONG'])}")
print(f"  SHORT: {len(nr_df[nr_df['direction'] == 'SHORT'])}")

print(f"\nAbove SMA200 at entry?")
print(f"  YES: {len(nr_df[nr_df['above_sma200'] == 'YES'])}")
print(f"  NO:  {len(nr_df[nr_df['above_sma200'] == 'NO'])}")

print(f"\nSMA200 relationship flipped DURING trade?")
print(f"  Flipped: {len(nr_df[nr_df['sma_flipped'] == True])}")
print(f"  Stayed:  {len(nr_df[nr_df['sma_flipped'] == False])}")

print("\n" + "="*70)
print("DETAILED VIEW OF ALL 14 TRADES")
print("="*70)

for i, (_, trade) in enumerate(nr_df.iterrows()):
    print(f"\n--- Trade #{i+1} ---")
    print(f"  Date:      {trade['date']}")
    print(f"  Direction: {trade['direction']}")
    print(f"  Entry:     ${trade['entry_price']:.0f}")
    print(f"  SMA200:    ${trade['entry_sma200']:.0f}")
    print(f"  Above SMA: {trade['above_sma200']}")
    print(f"  Distance from SMA200: {trade['distance_from_sma200_pct']:+.2f}%")
    print(f"  RSI at entry: {trade['rsi_value']:.1f}")
    print(f"  Max MFE:   {trade['max_mfe_bps']:+.1f} bps (best it ever got)")
    print(f"  Max MAE:   {trade['max_mae_bps']:+.1f} bps (worst drawdown)")
    print(f"  Bar 200:   {trade['bar_200_profit']:+.1f} bps" if trade['bar_200_profit'] is not None else "  Bar 200: N/A")
    print(f"  SMA flipped? {trade['sma_flipped']}", end="")
    if trade['sma_flipped']:
        print(f" at Bar {trade['sma_flip_bar']}")
    else:
        print()

# Summary stats
print("\n" + "="*70)
print("SUMMARY STATS FOR NEVER-RECOVER TRADES")
print("="*70)

print(f"""
Avg distance from SMA200: {nr_df['distance_from_sma200_pct'].mean():.2f}%
Avg Max MFE (best ever):  {nr_df['max_mfe_bps'].mean():.1f} bps
Avg Max MAE (worst ever):  {nr_df['max_mae_bps'].mean():.1f} bps
Avg Bar 200 profit:        {nr_df['bar_200_profit'].mean():.1f} bps

SMA200 flipped during trade: {len(nr_df[nr_df['sma_flipped']==True])}/{len(nr_df)} ({len(nr_df[nr_df['sma_flipped']==True])/len(nr_df)*100:.0f}%)
""")

# Check if any pattern emerges
print("="*70)
print("PATTERN CHECK: Can we filter these out?")
print("="*70)

print(f"\nDistance from SMA200 at entry:")
for _, trade in nr_df.iterrows():
    print(f"  {trade['date']} {trade['direction']:>5} distance: {trade['distance_from_sma200_pct']:+.2f}%  MFE: {trade['max_mfe_bps']:+.1f}")
