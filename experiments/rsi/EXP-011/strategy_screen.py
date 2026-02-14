"""
EXP-011: New Strategy Screen — 4 Alternative Strategies
=========================================================
Test 4 completely different strategy ideas on 15min BTCUSDT.
Same exit mechanics as V1.2 (trailing stop + time exit + 8bps fees).

Strategies:
  1. Volume Squeeze Breakout — compressed volume then expansion
  2. Volume Spike + Direction — big volume bar, follow its direction
  3. Candlestick Patterns — engulfing, hammer, shooting star
  4. MACD Divergence — price/MACD divergence signals

All signals pre-computed vectorized for speed.

Run: python experiments/rsi/EXP-011/strategy_screen.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
OUTPUT_PATH = Path("experiments/rsi/EXP-011")
FEES_BPS = 8
MAX_BARS = 10
LONG_TS = 20
SHORT_TS = 30

TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

print("=" * 80)
print("EXP-011: NEW STRATEGY SCREEN — 4 ALTERNATIVE STRATEGIES")
print("=" * 80)

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)
print(f"Full data: {df.index[0].date()} to {df.index[-1].date()} | {len(df)} bars")

# Common indicators (vectorized)
df['bar_body_bps'] = (df['close'] - df['open']) / df['close'] * 10000
df['bar_range_bps'] = (df['high'] - df['low']) / df['close'] * 10000
df['is_bullish'] = df['close'] > df['open']
df['is_bearish'] = df['close'] < df['open']
df['body'] = abs(df['close'] - df['open'])
df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
df['bar_range'] = df['high'] - df['low']

# Volume
df['vol_sma20'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_sma20']

# MACD
df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = df['ema12'] - df['ema26']
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']

# Filter to OOS
test = df[TEST_START:TEST_END].copy()
print(f"Test data: {test.index[0].date()} to {test.index[-1].date()} | {len(test)} bars")

# Convert to numpy arrays for fast access in trade execution
close_arr = test['close'].values
high_arr = test['high'].values
low_arr = test['low'].values
open_arr = test['open'].values
times = test.index.values


# =============================================================================
# FAST TRADE EXECUTION (numpy, no pandas)
# =============================================================================
def execute_trade_fast(idx, direction, trailing_stop_bps):
    """Execute trade using numpy arrays. Returns dict or None."""
    if idx + 1 >= len(close_arr) or idx + 1 + MAX_BARS >= len(close_arr):
        return None

    entry_price = open_arr[idx + 1]
    entry_time = times[idx + 1]
    signal_time = times[idx]
    highest_profit = 0.0
    exit_profit = None
    exit_bar = None
    exit_reason = None
    mae = 0.0
    mfe = 0.0

    for bar in range(1, MAX_BARS + 1):
        fi = idx + 1 + bar
        if fi >= len(close_arr):
            return None

        if direction == 'LONG':
            bar_mfe = (high_arr[fi] - entry_price) / entry_price * 10000
            bar_mae = (low_arr[fi] - entry_price) / entry_price * 10000
            bar_close = (close_arr[fi] - entry_price) / entry_price * 10000
        else:
            bar_mfe = (entry_price - low_arr[fi]) / entry_price * 10000
            bar_mae = (entry_price - high_arr[fi]) / entry_price * 10000
            bar_close = (entry_price - close_arr[fi]) / entry_price * 10000

        if bar_mfe > mfe: mfe = bar_mfe
        if bar_mae < mae: mae = bar_mae
        if bar_mfe > highest_profit: highest_profit = bar_mfe

        dd = highest_profit - bar_close
        if dd >= trailing_stop_bps and highest_profit > 0:
            exit_profit = highest_profit - trailing_stop_bps
            exit_bar = bar
            exit_reason = 'TRAILING_STOP'
            break

    if exit_profit is None:
        li = min(idx + 1 + MAX_BARS, len(close_arr) - 1)
        if direction == 'LONG':
            exit_profit = (close_arr[li] - entry_price) / entry_price * 10000
        else:
            exit_profit = (entry_price - close_arr[li]) / entry_price * 10000
        exit_bar = li - idx - 1
        exit_reason = 'TIME_EXIT'

    exit_time = times[min(idx + 1 + exit_bar, len(times) - 1)]

    return {
        'signal_time': signal_time,
        'entry_time': entry_time,
        'exit_time': exit_time,
        'direction': direction,
        'entry_price': entry_price,
        'gross_profit_bps': exit_profit,
        'net_profit_bps': exit_profit - FEES_BPS,
        'mfe_bps': mfe,
        'mae_bps': mae,
        'exit_bar': exit_bar,
        'exit_reason': exit_reason,
    }


def run_signals(signal_indices, directions):
    """Given signal indices and directions, execute trades skipping overlaps."""
    trades = []
    next_allowed = 0

    for i, d in zip(signal_indices, directions):
        if i < next_allowed:
            continue
        ts = LONG_TS if d == 'LONG' else SHORT_TS
        trade = execute_trade_fast(i, d, ts)
        if trade:
            trade['strategy'] = ''
            trades.append(trade)
            next_allowed = i + MAX_BARS + 1

    return trades


# =============================================================================
# PRE-COMPUTE ALL SIGNALS (vectorized)
# =============================================================================
print("\nPre-computing signals...")
t0 = time.time()

# --- Strategy 1: Volume Squeeze Breakout ---
# Squeeze = vol_ratio < 0.5 for 5+ consecutive bars, then spike >= 2.0
vol_low = (test['vol_ratio'] < 0.5).astype(int)
vol_low_streak = vol_low.rolling(5).sum()  # 5 = all 5 bars were low
squeeze_active = vol_low_streak.shift(1) >= 5  # squeeze was active on previous bar cluster

# Breakout = spike after squeeze
vol_spike_mask = test['vol_ratio'] >= 2.0
squeeze_breakout = squeeze_active & vol_spike_mask
sq_directions = np.where(test['is_bullish'], 'LONG', 'SHORT')

sq_idx = np.where(squeeze_breakout.values)[0]
sq_dirs = sq_directions[sq_idx]

# --- Strategy 2: Volume Spike + Direction ---
vol_spike_sig = (test['vol_ratio'] >= 2.0) & (abs(test['bar_body_bps']) >= 5)
vs_directions = np.where(test['is_bullish'], 'LONG', 'SHORT')

vs_idx = np.where(vol_spike_sig.values)[0]
vs_dirs = vs_directions[vs_idx]

# --- Strategy 3: Candlestick Patterns ---
prev_bearish = test['is_bearish'].shift(1).fillna(False)
prev_bullish = test['is_bullish'].shift(1).fillna(False)
prev_open = test['open'].shift(1)
prev_close = test['close'].shift(1)
prev_body = test['body'].shift(1)

# Bullish engulfing
bull_engulf = (
    prev_bearish &
    test['is_bullish'] &
    (test['close'] > prev_open) &
    (test['open'] < prev_close) &
    (test['body'] > prev_body)
)

# Bearish engulfing
bear_engulf = (
    prev_bullish &
    test['is_bearish'] &
    (test['close'] < prev_open) &
    (test['open'] > prev_close) &
    (test['body'] > prev_body)
)

# Hammer: long lower wick, small upper wick, body > 0, range > 5bps
hammer = (
    (test['lower_wick'] > 2 * test['body']) &
    (test['upper_wick'] < test['body'] * 0.5) &
    (test['body'] > 0) &
    (test['bar_range_bps'] > 5)
)

# Shooting star: long upper wick, small lower wick
shooting_star = (
    (test['upper_wick'] > 2 * test['body']) &
    (test['lower_wick'] < test['body'] * 0.5) &
    (test['body'] > 0) &
    (test['bar_range_bps'] > 5)
)

# Combine: LONG signals (bull engulf or hammer), SHORT signals (bear engulf or shooting star)
candle_long = bull_engulf | hammer
candle_short = bear_engulf | shooting_star
# Priority: if both, pick the engulfing (stronger pattern)
candle_signal = candle_long | candle_short
candle_dir = np.where(candle_long, 'LONG', np.where(candle_short, 'SHORT', 'NONE'))

cd_idx = np.where(candle_signal.values)[0]
cd_dirs = candle_dir[cd_idx]

# --- Strategy 4: MACD Divergence ---
lookback = 20
price_low_roll = test['low'].rolling(lookback).min()
price_high_roll = test['high'].rolling(lookback).max()
macd_low_roll = test['macd_hist'].rolling(lookback).min()
macd_high_roll = test['macd_hist'].rolling(lookback).max()

macd_bull = (
    (test['low'] <= price_low_roll * 1.001) &
    (test['macd_hist'] > macd_low_roll) &
    (macd_low_roll < 0)
)
macd_bear = (
    (test['high'] >= price_high_roll * 0.999) &
    (test['macd_hist'] < macd_high_roll) &
    (macd_high_roll > 0)
)

macd_signal = macd_bull | macd_bear
macd_dir = np.where(macd_bull, 'LONG', np.where(macd_bear, 'SHORT', 'NONE'))

md_idx = np.where(macd_signal.values)[0]
md_dirs = macd_dir[md_idx]

print(f"Signals computed in {time.time() - t0:.1f}s")
print(f"  Vol Squeeze: {len(sq_idx)} signals")
print(f"  Vol Spike:   {len(vs_idx)} signals")
print(f"  Candlestick: {len(cd_idx)} signals")
print(f"  MACD Div:    {len(md_idx)} signals")


# =============================================================================
# RUN ALL STRATEGIES
# =============================================================================
strategy_configs = [
    ('Vol Squeeze Breakout', sq_idx, sq_dirs),
    ('Volume Spike + Direction', vs_idx, vs_dirs),
    ('Candlestick Patterns', cd_idx, cd_dirs),
    ('MACD Divergence', md_idx, md_dirs),
]

all_trades = {}

for name, sig_idx, sig_dirs in strategy_configs:
    t0 = time.time()
    print(f"\nRunning {name}...")
    trades = run_signals(sig_idx, sig_dirs)

    for t in trades:
        t['strategy'] = name

    tdf = pd.DataFrame(trades)
    all_trades[name] = tdf

    if len(tdf) == 0:
        print(f"  NO TRADES")
        continue

    # Save trades
    safe_name = name.lower().replace(' ', '_').replace('+', '_')
    tdf.to_csv(OUTPUT_PATH / f"{safe_name}_trades.csv", index=False)

    w = tdf[tdf['net_profit_bps'] > 0]
    l = tdf[tdf['net_profit_bps'] <= 0]
    gw = w['net_profit_bps'].sum() if len(w) > 0 else 0
    gl = abs(l['net_profit_bps'].sum()) if len(l) > 0 else 1
    pf = gw / gl

    lt = tdf[tdf['direction'] == 'LONG']
    st = tdf[tdf['direction'] == 'SHORT']

    print(f"  Trades: {len(tdf)} | Win: {len(w)/len(tdf)*100:.1f}% | Net: {tdf['net_profit_bps'].sum():+.0f} bps | PF: {pf:.2f}  ({time.time()-t0:.1f}s)")
    print(f"  LONG:  {len(lt)} trades, {lt['net_profit_bps'].sum():+.0f} bps")
    print(f"  SHORT: {len(st)} trades, {st['net_profit_bps'].sum():+.0f} bps")

    # Year split
    tdf['year'] = pd.to_datetime(tdf['entry_time']).dt.year
    for y in sorted(tdf['year'].unique()):
        yt = tdf[tdf['year'] == y]
        yw = yt[yt['net_profit_bps'] > 0]
        yl = yt[yt['net_profit_bps'] <= 0]
        ygw = yw['net_profit_bps'].sum() if len(yw) > 0 else 0
        ygl = abs(yl['net_profit_bps'].sum()) if len(yl) > 0 else 1
        print(f"    {y}: {len(yt)}t | Win: {len(yw)/len(yt)*100:.1f}% | Net: {yt['net_profit_bps'].sum():+.0f} bps | PF: {ygw/ygl:.2f}")

    # Exit reasons
    for reason in tdf['exit_reason'].unique():
        rt = tdf[tdf['exit_reason'] == reason]
        print(f"    {reason}: {len(rt)} ({len(rt)/len(tdf)*100:.0f}%) avg {rt['net_profit_bps'].mean():+.1f} bps")


# =============================================================================
# COMPARISON TABLE
# =============================================================================
print("\n" + "=" * 100)
print("COMPARISON: ALL STRATEGIES vs V1.2")
print("=" * 100)

print(f"\n{'Strategy':<30} {'Trades':>7} {'Win%':>7} {'Net bps':>10} {'PF':>7} {'Avg/Tr':>8} {'L_Net':>8} {'S_Net':>8}")
print("-" * 90)

# V1.2 reference
print(f"{'V1.2 (RSI + filters)':<30} {'202':>7} {'56.9%':>7} {'+3,250':>10} {'2.47':>7} {'+16.1':>8} {'+1,047':>8} {'+2,203':>8}")

for name, tdf in all_trades.items():
    if len(tdf) == 0:
        print(f"{name:<30} {'NO TRADES':>7}")
        continue
    w = tdf[tdf['net_profit_bps'] > 0]
    l = tdf[tdf['net_profit_bps'] <= 0]
    gw = w['net_profit_bps'].sum() if len(w) > 0 else 0
    gl = abs(l['net_profit_bps'].sum()) if len(l) > 0 else 1
    lt = tdf[tdf['direction'] == 'LONG']
    st = tdf[tdf['direction'] == 'SHORT']
    print(f"{name:<30} {len(tdf):>7} {len(w)/len(tdf)*100:>6.1f}% {tdf['net_profit_bps'].sum():>+9.0f} "
          f"{gw/gl:>7.2f} {tdf['net_profit_bps'].mean():>+7.1f} "
          f"{lt['net_profit_bps'].sum():>+7.0f} {st['net_profit_bps'].sum():>+7.0f}")


print("\n" + "=" * 70)
print("DONE — Run visualize_strategies.py for charts")
print("=" * 70)
