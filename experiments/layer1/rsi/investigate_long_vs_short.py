"""
INVESTIGATION: V1 LONG vs SHORT Deep Analysis
==============================================
Option B: Why is LONG weak? Can WHEN filters help?
Option D: Why does SHORT work when WHAT says "overbought doesn't work"?

Run: python experiments/rsi/investigate_long_vs_short.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
TRADES_PATH = Path("experiments/rsi/backtest_v1_trades.csv")

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("INVESTIGATION: V1 LONG vs SHORT")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

trades = pd.read_csv(TRADES_PATH)
trades['entry_time'] = pd.to_datetime(trades['entry_time'])
trades['exit_time'] = pd.to_datetime(trades['exit_time'])

# =============================================================================
# Add WHEN features to each trade
# =============================================================================
print("\nCalculating WHEN features for each trade...")

# Calculate indicators on 15min data
df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
df['sma200'] = df['close'].rolling(200).mean()

# ATR
df['tr'] = np.maximum(
    df['high'] - df['low'],
    np.maximum(
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    )
)
df['atr14'] = df['tr'].rolling(14).mean()
df['atr_bps'] = df['atr14'] / df['close'] * 10000

# ATR percentile (rolling 200-bar)
df['atr_percentile'] = df['atr_bps'].rolling(200).rank(pct=True) * 100

# EMA separation (trend strength)
df['ema_separation'] = abs(df['ema50'] - df['ema200']) / df['close'] * 100

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# Time features
df['hour_utc'] = df.index.hour
df['day_of_week'] = df.index.dayofweek  # 0=Mon, 6=Sun
df['is_weekend'] = df['day_of_week'] >= 5
df['is_asia_night'] = (df['hour_utc'] >= 0) & (df['hour_utc'] < 4)
df['is_europe'] = (df['hour_utc'] >= 8) & (df['hour_utc'] < 12)
df['is_saturday'] = df['day_of_week'] == 5

# Entry bar range
df['bar_range_bps'] = (df['high'] - df['low']) / df['close'] * 10000

# Distance from SMA200
df['sma200_dist_pct'] = (df['close'] - df['sma200']) / df['sma200'] * 100

# Map features to trades
for col in ['atr_bps', 'atr_percentile', 'ema_separation', 'hour_utc',
            'day_of_week', 'is_weekend', 'is_asia_night', 'is_europe',
            'is_saturday', 'bar_range_bps', 'sma200_dist_pct', 'rsi']:
    trade_values = []
    for _, trade in trades.iterrows():
        entry_time = trade['entry_time']
        # Find the signal bar (one bar before entry)
        signal_idx = df.index.get_indexer([entry_time], method='ffill')[0] - 1
        if signal_idx >= 0 and signal_idx < len(df):
            trade_values.append(df.iloc[signal_idx][col])
        else:
            trade_values.append(np.nan)
    trades[col] = trade_values

print(f"Features mapped to {len(trades)} trades")

# =============================================================================
# OPTION D: Why does SHORT work when WHAT says overbought doesn't work?
# =============================================================================
print("\n" + "=" * 70)
print("OPTION D: WHY DOES V1 SHORT WORK?")
print("=" * 70)

long_trades = trades[trades['direction'] == 'LONG']
short_trades = trades[trades['direction'] == 'SHORT']

print("\n--- KEY DIFFERENCE: V1 SHORT != just RSI>80 ---")
print("WHAT analysis tested: RSI>80 on 1-min bars, ANY market condition")
print("V1 SHORT uses:        RSI>80 + Price < SMA200 (bear market only)")

print(f"\n--- SMA200 Distance at Entry ---")
print(f"LONG  (price>SMA200): avg distance = {long_trades['sma200_dist_pct'].mean():+.2f}%")
print(f"SHORT (price<SMA200): avg distance = {short_trades['sma200_dist_pct'].mean():+.2f}%")

# SHORT: How far below SMA200?
print(f"\nSHORT distance from SMA200 distribution:")
print(f"  25th pctl: {short_trades['sma200_dist_pct'].quantile(0.25):.2f}%")
print(f"  50th pctl: {short_trades['sma200_dist_pct'].quantile(0.50):.2f}%")
print(f"  75th pctl: {short_trades['sma200_dist_pct'].quantile(0.75):.2f}%")

# SHORT: ATR at entry
print(f"\nSHORT ATR percentile at entry:")
print(f"  Mean: {short_trades['atr_percentile'].mean():.1f}%")
print(f"  < 25th: {(short_trades['atr_percentile'] < 25).sum()} trades ({(short_trades['atr_percentile'] < 25).mean()*100:.1f}%)")
print(f"  > 75th: {(short_trades['atr_percentile'] > 75).sum()} trades ({(short_trades['atr_percentile'] > 75).mean()*100:.1f}%)")

# SHORT: EMA separation at entry
print(f"\nSHORT EMA separation (trend strength) at entry:")
print(f"  Mean: {short_trades['ema_separation'].mean():.2f}%")
print(f"  > 0.5%: {(short_trades['ema_separation'] > 0.5).sum()} trades ({(short_trades['ema_separation'] > 0.5).mean()*100:.1f}%)")
print(f"  > 1.0%: {(short_trades['ema_separation'] > 1.0).sum()} trades ({(short_trades['ema_separation'] > 1.0).mean()*100:.1f}%)")
print(f"  > 2.0%: {(short_trades['ema_separation'] > 2.0).sum()} trades ({(short_trades['ema_separation'] > 2.0).mean()*100:.1f}%)")

# SHORT winners vs losers - what's different?
short_winners = short_trades[short_trades['net_profit_bps'] > 0]
short_losers = short_trades[short_trades['net_profit_bps'] <= 0]

print(f"\n--- SHORT Winners vs Losers ---")
print(f"{'Metric':<25} {'Winners':>12} {'Losers':>12}")
print(f"{'-'*25} {'-'*12} {'-'*12}")
print(f"{'Count':<25} {len(short_winners):>12} {len(short_losers):>12}")
print(f"{'Avg SMA200 dist %':<25} {short_winners['sma200_dist_pct'].mean():>12.2f} {short_losers['sma200_dist_pct'].mean():>12.2f}")
print(f"{'Avg ATR percentile':<25} {short_winners['atr_percentile'].mean():>12.1f} {short_losers['atr_percentile'].mean():>12.1f}")
print(f"{'Avg EMA separation':<25} {short_winners['ema_separation'].mean():>12.2f} {short_losers['ema_separation'].mean():>12.2f}")
print(f"{'Avg bar range (bps)':<25} {short_winners['bar_range_bps'].mean():>12.1f} {short_losers['bar_range_bps'].mean():>12.1f}")
print(f"{'Avg RSI at signal':<25} {short_winners['rsi'].mean():>12.1f} {short_losers['rsi'].mean():>12.1f}")

# WHAT said: "overbought + momentum continues UP"
# V1 says:  "overbought + BEAR market = price goes DOWN"
# The SMA200 filter is the key differentiator
print(f"\n--- THE EXPLANATION ---")
print(f"WHAT analysis: RSI>80 alone = momentum continues UP (no reversal)")
print(f"V1 SHORT:      RSI>80 + price<SMA200 = overbought WITHIN a downtrend")
print(f"")
print(f"V1 is NOT betting on reversal from overbought.")
print(f"V1 is betting on CONTINUATION of downtrend after a relief rally.")
print(f"RSI>80 below SMA200 = 'relief rally exhaustion in bear market'")

# =============================================================================
# OPTION B: Why is LONG weak? What conditions make it worse?
# =============================================================================
print("\n" + "=" * 70)
print("OPTION B: WHY IS LONG WEAK?")
print("=" * 70)

long_winners = long_trades[long_trades['net_profit_bps'] > 0]
long_losers = long_trades[long_trades['net_profit_bps'] <= 0]

print(f"\n--- LONG Winners vs Losers ---")
print(f"{'Metric':<25} {'Winners':>12} {'Losers':>12}")
print(f"{'-'*25} {'-'*12} {'-'*12}")
print(f"{'Count':<25} {len(long_winners):>12} {len(long_losers):>12}")
print(f"{'Avg SMA200 dist %':<25} {long_winners['sma200_dist_pct'].mean():>12.2f} {long_losers['sma200_dist_pct'].mean():>12.2f}")
print(f"{'Avg ATR percentile':<25} {long_winners['atr_percentile'].mean():>12.1f} {long_losers['atr_percentile'].mean():>12.1f}")
print(f"{'Avg EMA separation':<25} {long_winners['ema_separation'].mean():>12.2f} {long_losers['ema_separation'].mean():>12.2f}")
print(f"{'Avg bar range (bps)':<25} {long_winners['bar_range_bps'].mean():>12.1f} {long_losers['bar_range_bps'].mean():>12.1f}")
print(f"{'Avg RSI at signal':<25} {long_winners['rsi'].mean():>12.1f} {long_losers['rsi'].mean():>12.1f}")
print(f"{'Weekend %':<25} {long_winners['is_weekend'].mean()*100:>11.1f}% {long_losers['is_weekend'].mean()*100:>11.1f}%")
print(f"{'Asia Night %':<25} {long_winners['is_asia_night'].mean()*100:>11.1f}% {long_losers['is_asia_night'].mean()*100:>11.1f}%")

# WHEN filter analysis for LONG
print(f"\n--- LONG: WHEN Filter Impact ---")

# ATR filter
print(f"\n1. ATR FILTER:")
for threshold_name, condition in [
    ("ATR < 25th pctl", long_trades['atr_percentile'] < 25),
    ("ATR 25-75th pctl", (long_trades['atr_percentile'] >= 25) & (long_trades['atr_percentile'] <= 75)),
    ("ATR > 75th pctl", long_trades['atr_percentile'] > 75)
]:
    subset = long_trades[condition]
    if len(subset) > 0:
        winners = subset[subset['net_profit_bps'] > 0]
        print(f"  {threshold_name}: {len(subset)} trades | Win: {len(winners)/len(subset)*100:.1f}% | "
              f"Avg: {subset['net_profit_bps'].mean():+.1f} bps | Total: {subset['net_profit_bps'].sum():+.1f} bps")

# EMA separation filter
print(f"\n2. TREND STRENGTH (EMA separation):")
for threshold_name, condition in [
    ("EMA sep < 0.5%", long_trades['ema_separation'] < 0.5),
    ("EMA sep 0.5-1%", (long_trades['ema_separation'] >= 0.5) & (long_trades['ema_separation'] < 1.0)),
    ("EMA sep > 1%", long_trades['ema_separation'] >= 1.0)
]:
    subset = long_trades[condition]
    if len(subset) > 0:
        winners = subset[subset['net_profit_bps'] > 0]
        print(f"  {threshold_name}: {len(subset)} trades | Win: {len(winners)/len(subset)*100:.1f}% | "
              f"Avg: {subset['net_profit_bps'].mean():+.1f} bps | Total: {subset['net_profit_bps'].sum():+.1f} bps")

# Time filter
print(f"\n3. TIME FILTER:")
for threshold_name, condition in [
    ("Asia Night (00-04)", long_trades['is_asia_night'] == True),
    ("Europe (08-12)", long_trades['is_europe'] == True),
    ("Other hours", (long_trades['is_asia_night'] == False) & (long_trades['is_europe'] == False)),
    ("Weekend", long_trades['is_weekend'] == True),
    ("Weekday", long_trades['is_weekend'] == False)
]:
    subset = long_trades[condition]
    if len(subset) > 0:
        winners = subset[subset['net_profit_bps'] > 0]
        print(f"  {threshold_name}: {len(subset)} trades | Win: {len(winners)/len(subset)*100:.1f}% | "
              f"Avg: {subset['net_profit_bps'].mean():+.1f} bps | Total: {subset['net_profit_bps'].sum():+.1f} bps")

# SMA200 distance filter
print(f"\n4. SMA200 DISTANCE:")
for threshold_name, condition in [
    ("< 1% above SMA200", long_trades['sma200_dist_pct'] < 1.0),
    ("1-3% above SMA200", (long_trades['sma200_dist_pct'] >= 1.0) & (long_trades['sma200_dist_pct'] < 3.0)),
    ("3-5% above SMA200", (long_trades['sma200_dist_pct'] >= 3.0) & (long_trades['sma200_dist_pct'] < 5.0)),
    ("> 5% above SMA200", long_trades['sma200_dist_pct'] >= 5.0)
]:
    subset = long_trades[condition]
    if len(subset) > 0:
        winners = subset[subset['net_profit_bps'] > 0]
        print(f"  {threshold_name}: {len(subset)} trades | Win: {len(winners)/len(subset)*100:.1f}% | "
              f"Avg: {subset['net_profit_bps'].mean():+.1f} bps | Total: {subset['net_profit_bps'].sum():+.1f} bps")

# Bar range filter
print(f"\n5. ENTRY BAR RANGE:")
for threshold_name, condition in [
    ("Bar range < 15bps", long_trades['bar_range_bps'] < 15),
    ("Bar range 15-30bps", (long_trades['bar_range_bps'] >= 15) & (long_trades['bar_range_bps'] < 30)),
    ("Bar range > 30bps", long_trades['bar_range_bps'] >= 30)
]:
    subset = long_trades[condition]
    if len(subset) > 0:
        winners = subset[subset['net_profit_bps'] > 0]
        print(f"  {threshold_name}: {len(subset)} trades | Win: {len(winners)/len(subset)*100:.1f}% | "
              f"Avg: {subset['net_profit_bps'].mean():+.1f} bps | Total: {subset['net_profit_bps'].sum():+.1f} bps")

# =============================================================================
# SAME WHEN FILTER ANALYSIS FOR SHORT (for comparison)
# =============================================================================
print("\n" + "=" * 70)
print("SHORT: WHEN Filter Impact (for comparison)")
print("=" * 70)

print(f"\n1. ATR FILTER:")
for threshold_name, condition in [
    ("ATR < 25th pctl", short_trades['atr_percentile'] < 25),
    ("ATR 25-75th pctl", (short_trades['atr_percentile'] >= 25) & (short_trades['atr_percentile'] <= 75)),
    ("ATR > 75th pctl", short_trades['atr_percentile'] > 75)
]:
    subset = short_trades[condition]
    if len(subset) > 0:
        winners = subset[subset['net_profit_bps'] > 0]
        print(f"  {threshold_name}: {len(subset)} trades | Win: {len(winners)/len(subset)*100:.1f}% | "
              f"Avg: {subset['net_profit_bps'].mean():+.1f} bps | Total: {subset['net_profit_bps'].sum():+.1f} bps")

print(f"\n2. TREND STRENGTH (EMA separation):")
for threshold_name, condition in [
    ("EMA sep < 0.5%", short_trades['ema_separation'] < 0.5),
    ("EMA sep 0.5-1%", (short_trades['ema_separation'] >= 0.5) & (short_trades['ema_separation'] < 1.0)),
    ("EMA sep > 1%", short_trades['ema_separation'] >= 1.0)
]:
    subset = short_trades[condition]
    if len(subset) > 0:
        winners = subset[subset['net_profit_bps'] > 0]
        print(f"  {threshold_name}: {len(subset)} trades | Win: {len(winners)/len(subset)*100:.1f}% | "
              f"Avg: {subset['net_profit_bps'].mean():+.1f} bps | Total: {subset['net_profit_bps'].sum():+.1f} bps")

print(f"\n3. SMA200 DISTANCE:")
for threshold_name, condition in [
    ("< 1% below SMA200", short_trades['sma200_dist_pct'] > -1.0),
    ("1-3% below SMA200", (short_trades['sma200_dist_pct'] <= -1.0) & (short_trades['sma200_dist_pct'] > -3.0)),
    ("3-5% below SMA200", (short_trades['sma200_dist_pct'] <= -3.0) & (short_trades['sma200_dist_pct'] > -5.0)),
    ("> 5% below SMA200", short_trades['sma200_dist_pct'] <= -5.0)
]:
    subset = short_trades[condition]
    if len(subset) > 0:
        winners = subset[subset['net_profit_bps'] > 0]
        print(f"  {threshold_name}: {len(subset)} trades | Win: {len(winners)/len(subset)*100:.1f}% | "
              f"Avg: {subset['net_profit_bps'].mean():+.1f} bps | Total: {subset['net_profit_bps'].sum():+.1f} bps")

# =============================================================================
# COMBINED: What if we apply WHEN filters to LONG only?
# =============================================================================
print("\n" + "=" * 70)
print("SIMULATION: LONG with WHEN filters vs V1 original")
print("=" * 70)

# Test various filter combinations on LONG
filters = {
    "V1 Original (no filter)": long_trades,
    "Skip ATR < 25th": long_trades[long_trades['atr_percentile'] >= 25],
    "Skip EMA sep < 0.5%": long_trades[long_trades['ema_separation'] >= 0.5],
    "Skip Weekend": long_trades[long_trades['is_weekend'] == False],
    "Skip Asia Night": long_trades[long_trades['is_asia_night'] == False],
    "Skip close to SMA200 (<1%)": long_trades[long_trades['sma200_dist_pct'] >= 1.0],
    "ATR>=25 + EMA>=0.5%": long_trades[(long_trades['atr_percentile'] >= 25) & (long_trades['ema_separation'] >= 0.5)],
    "ATR>=25 + No Weekend": long_trades[(long_trades['atr_percentile'] >= 25) & (long_trades['is_weekend'] == False)],
    "ATR>=25 + EMA>=0.5% + No Wknd": long_trades[(long_trades['atr_percentile'] >= 25) & (long_trades['ema_separation'] >= 0.5) & (long_trades['is_weekend'] == False)],
    "ALL filters combined": long_trades[
        (long_trades['atr_percentile'] >= 25) &
        (long_trades['ema_separation'] >= 0.5) &
        (long_trades['is_weekend'] == False) &
        (long_trades['is_asia_night'] == False)
    ],
}

print(f"\n{'Filter':<35} {'Trades':>7} {'Win%':>7} {'Avg':>8} {'Total':>10} {'PF':>6}")
print(f"{'-'*35} {'-'*7} {'-'*7} {'-'*8} {'-'*10} {'-'*6}")

for name, subset in filters.items():
    if len(subset) == 0:
        print(f"{name:<35} {'0':>7} {'N/A':>7} {'N/A':>8} {'N/A':>10} {'N/A':>6}")
        continue

    winners = subset[subset['net_profit_bps'] > 0]
    losers = subset[subset['net_profit_bps'] <= 0]
    win_pct = len(winners) / len(subset) * 100
    avg = subset['net_profit_bps'].mean()
    total = subset['net_profit_bps'].sum()
    gross_win = winners['net_profit_bps'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['net_profit_bps'].sum()) if len(losers) > 0 else 1
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    print(f"{name:<35} {len(subset):>7} {win_pct:>6.1f}% {avg:>+7.1f} {total:>+9.1f} {pf:>6.2f}")

# What's the combined system performance?
print("\n" + "=" * 70)
print("PROJECTED V1.2: LONG (filtered) + SHORT (unchanged)")
print("=" * 70)

best_filter = long_trades[
    (long_trades['atr_percentile'] >= 25) &
    (long_trades['ema_separation'] >= 0.5) &
    (long_trades['is_weekend'] == False) &
    (long_trades['is_asia_night'] == False)
]

# Year-by-year for filtered LONG
print(f"\n--- Filtered LONG year-by-year ---")
for year in [2024, 2025]:
    subset = best_filter[best_filter['entry_time'].dt.year == year]
    if len(subset) > 0:
        winners = subset[subset['net_profit_bps'] > 0]
        print(f"  {year}: {len(subset)} trades | Win: {len(winners)/len(subset)*100:.1f}% | "
              f"Total: {subset['net_profit_bps'].sum():+.1f} bps | Avg: {subset['net_profit_bps'].mean():+.1f} bps")

# Combined projection
filtered_long_total = best_filter['net_profit_bps'].sum()
short_total = short_trades['net_profit_bps'].sum()

print(f"\n--- V1.2 Projected Performance ---")
print(f"  Filtered LONG:  {len(best_filter)} trades, {filtered_long_total:+.1f} bps")
print(f"  SHORT (no change): {len(short_trades)} trades, {short_total:+.1f} bps")
print(f"  COMBINED:       {len(best_filter) + len(short_trades)} trades, {filtered_long_total + short_total:+.1f} bps")
print(f"\n  vs V1 original: {len(trades)} trades, {trades['net_profit_bps'].sum():+.1f} bps")
print(f"  Difference:     {(filtered_long_total + short_total) - trades['net_profit_bps'].sum():+.1f} bps")
