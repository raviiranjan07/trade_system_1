"""
EXP-012: V1.2 + Volume Spike Combination + Re-entry
=====================================================
Test combining V1.2 (RSI + filters) with Volume Spike strategy.
Also test re-entry RE(1,2) on each combination.

Combinations:
  A: V1.2 only (baseline)
  B: Parallel — take signals from EITHER V1.2 OR Vol Spike
  C: Confluence — only enter when BOTH agree
  D: V1.2 + Vol Spike filter (V1.2 signal + require vol > 2x avg)

Each tested with and without RE(1,2): 1 max re-entry, 2 bar cooldown.

Run: python experiments/rsi/EXP-012/combined_strategy.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
OUTPUT_PATH = Path("experiments/rsi/EXP-012")
OUTPUT_PATH.mkdir(exist_ok=True)

FEES_BPS = 8
MAX_BARS = 10
LONG_TS = 20
SHORT_TS = 30

# Re-entry params
RE_MAX = 1       # max 1 re-entry per original signal
RE_COOLDOWN = 2  # wait 2 bars after exit before re-entering

TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

print("=" * 80)
print("EXP-012: V1.2 + VOLUME SPIKE + RE-ENTRY")
print("=" * 80)

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# SMA200
df['sma_200'] = df['close'].rolling(200).mean()

# EMA50/200 for separation filter
df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
df['ema_separation'] = abs(df['ema50'] - df['ema200']) / df['close'] * 100

# ATR
df['tr'] = np.maximum(df['high'] - df['low'],
                      np.maximum(abs(df['high'] - df['close'].shift(1)),
                                 abs(df['low'] - df['close'].shift(1))))
df['atr14'] = df['tr'].rolling(14).mean()
df['atr_bps'] = df['atr14'] / df['close'] * 10000
df['atr_percentile'] = df['atr_bps'].rolling(200).rank(pct=True) * 100

# Volume
df['vol_sma20'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_sma20']

# Bar properties
df['bar_body_bps'] = (df['close'] - df['open']) / df['close'] * 10000
df['is_bullish'] = df['close'] > df['open']

# V1.2 signals (vectorized)
df['rsi_oversold'] = df['rsi'] < 20
df['rsi_overbought'] = df['rsi'] > 80
df['rsi_oversold_cross'] = df['rsi_oversold'] & ~df['rsi_oversold'].shift(1).fillna(False)
df['rsi_overbought_cross'] = df['rsi_overbought'] & ~df['rsi_overbought'].shift(1).fillna(False)
df['bull_market'] = df['close'] > df['sma_200']
df['bear_market'] = df['close'] < df['sma_200']

# V1.2 LONG signal (with filters)
df['v12_long'] = (
    df['rsi_oversold_cross'] &
    df['bull_market'] &
    (df['atr_percentile'] >= 25) &
    (df['ema_separation'] >= 0.5)
)

# V1.2 SHORT signal (no filters)
df['v12_short'] = df['rsi_overbought_cross'] & df['bear_market']

# Volume Spike signals
df['vol_spike_long'] = (df['vol_ratio'] >= 2.0) & (abs(df['bar_body_bps']) >= 5) & df['is_bullish']
df['vol_spike_short'] = (df['vol_ratio'] >= 2.0) & (abs(df['bar_body_bps']) >= 5) & ~df['is_bullish']

# Confluence: V1.2 signal + volume spike within ±2 bars
# (volume spike on same bar OR 1-2 bars before the RSI signal)
vol_spike_any = (df['vol_ratio'] >= 2.0) & (abs(df['bar_body_bps']) >= 5)
vol_spike_nearby = vol_spike_any | vol_spike_any.shift(1).fillna(False) | vol_spike_any.shift(2).fillna(False)

df['confluence_long'] = df['v12_long'] & vol_spike_nearby
df['confluence_short'] = df['v12_short'] & vol_spike_nearby

# V1.2 + vol filter: V1.2 signal + volume > 2x (not necessarily a spike bar, just high vol environment)
df['v12_volfilt_long'] = df['v12_long'] & (df['vol_ratio'] >= 1.5)
df['v12_volfilt_short'] = df['v12_short'] & (df['vol_ratio'] >= 1.5)

# Filter to OOS
test = df[TEST_START:TEST_END].copy()
print(f"Test data: {test.index[0].date()} to {test.index[-1].date()} | {len(test)} bars")

# Numpy arrays for fast execution
close_arr = test['close'].values
high_arr = test['high'].values
low_arr = test['low'].values
open_arr = test['open'].values
times = test.index.values


# =============================================================================
# TRADE EXECUTION
# =============================================================================
def execute_trade_fast(idx, direction, trailing_stop_bps):
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

    exit_idx = min(idx + 1 + exit_bar, len(times) - 1)

    return {
        'signal_time': signal_time,
        'entry_time': entry_time,
        'exit_time': times[exit_idx],
        'exit_idx': exit_idx,
        'direction': direction,
        'entry_price': entry_price,
        'gross_profit_bps': exit_profit,
        'net_profit_bps': exit_profit - FEES_BPS,
        'mfe_bps': mfe,
        'mae_bps': mae,
        'exit_bar': exit_bar,
        'exit_reason': exit_reason,
        'is_reentry': False,
    }


def run_strategy(long_mask, short_mask, with_reentry=False):
    """Run backtest given boolean signal masks. Optionally add re-entry."""
    long_idx = np.where(long_mask.values)[0]
    short_idx = np.where(short_mask.values)[0]

    # Merge and sort all signals
    all_signals = []
    for i in long_idx:
        all_signals.append((i, 'LONG'))
    for i in short_idx:
        all_signals.append((i, 'SHORT'))
    all_signals.sort(key=lambda x: x[0])

    trades = []
    next_allowed = 0

    for sig_idx, direction in all_signals:
        if sig_idx < next_allowed:
            continue

        ts = LONG_TS if direction == 'LONG' else SHORT_TS
        trade = execute_trade_fast(sig_idx, direction, ts)
        if trade is None:
            continue

        trades.append(trade)
        exit_idx = trade['exit_idx']
        next_allowed = exit_idx + 1

        # Re-entry logic
        if with_reentry and trade['exit_reason'] == 'TRAILING_STOP':
            re_count = 0
            re_exit_idx = exit_idx

            while re_count < RE_MAX:
                # Wait cooldown bars
                re_entry_idx = re_exit_idx + RE_COOLDOWN

                if re_entry_idx >= len(close_arr) - MAX_BARS - 2:
                    break

                # Check regime still valid
                if direction == 'LONG':
                    # Still in bull market?
                    if re_entry_idx < len(test) and not test.iloc[re_entry_idx]['bull_market']:
                        break
                else:
                    # Still in bear market?
                    if re_entry_idx < len(test) and not test.iloc[re_entry_idx]['bear_market']:
                        break

                re_trade = execute_trade_fast(re_entry_idx, direction, ts)
                if re_trade is None:
                    break

                re_trade['is_reentry'] = True
                trades.append(re_trade)
                re_exit_idx = re_trade['exit_idx']
                re_count += 1

                # Only re-enter again if this trade also hit trailing stop
                if re_trade['exit_reason'] != 'TRAILING_STOP':
                    break

            next_allowed = re_exit_idx + 1

    return pd.DataFrame(trades)


# =============================================================================
# RUN ALL CONFIGURATIONS
# =============================================================================
configs = [
    ("A: V1.2 only",            'v12_long', 'v12_short', False),
    ("A+RE: V1.2 + RE(1,2)",    'v12_long', 'v12_short', True),

    ("B: Parallel (V1.2 OR Vol)", None, None, False),  # special handling
    ("B+RE: Parallel + RE(1,2)",  None, None, True),

    ("C: Confluence (V1.2 AND Vol)", 'confluence_long', 'confluence_short', False),
    ("C+RE: Confluence + RE(1,2)",   'confluence_long', 'confluence_short', True),

    ("D: V1.2 + Vol Filter",       'v12_volfilt_long', 'v12_volfilt_short', False),
    ("D+RE: V1.2 + VolFilt + RE",  'v12_volfilt_long', 'v12_volfilt_short', True),
]

all_results = []

for name, long_col, short_col, with_re in configs:
    t0 = time.time()

    if 'Parallel' in name:
        # Parallel: combine V1.2 and Vol Spike signals (OR logic)
        long_mask = test['v12_long'] | test['vol_spike_long']
        short_mask = test['v12_short'] | test['vol_spike_short']
        tdf = run_strategy(long_mask, short_mask, with_re)
    else:
        long_mask = test[long_col]
        short_mask = test[short_col]
        tdf = run_strategy(long_mask, short_mask, with_re)

    elapsed = time.time() - t0

    if len(tdf) == 0:
        print(f"\n{name}: NO TRADES")
        continue

    # Save trades
    safe = name.split(':')[0].strip().replace('+', '_').replace(' ', '_')
    tdf.to_csv(OUTPUT_PATH / f"{safe}_trades.csv", index=False)

    # Metrics
    w = tdf[tdf['net_profit_bps'] > 0]
    l = tdf[tdf['net_profit_bps'] <= 0]
    gw = w['net_profit_bps'].sum() if len(w) > 0 else 0
    gl = abs(l['net_profit_bps'].sum()) if len(l) > 0 else 1
    pf = gw / gl

    lt = tdf[tdf['direction'] == 'LONG']
    st = tdf[tdf['direction'] == 'SHORT']
    orig = tdf[~tdf['is_reentry']]
    re_trades = tdf[tdf['is_reentry']]

    # Max drawdown
    equity = tdf['net_profit_bps'].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = drawdown.min()

    result = {
        'name': name,
        'trades': len(tdf),
        'orig_trades': len(orig),
        're_trades': len(re_trades),
        'win_pct': len(w) / len(tdf) * 100,
        'net_bps': tdf['net_profit_bps'].sum(),
        'pf': pf,
        'avg_bps': tdf['net_profit_bps'].mean(),
        'long_n': len(lt),
        'long_net': lt['net_profit_bps'].sum() if len(lt) > 0 else 0,
        'short_n': len(st),
        'short_net': st['net_profit_bps'].sum() if len(st) > 0 else 0,
        'max_dd': max_dd,
    }
    all_results.append(result)

    print(f"\n{name}  ({elapsed:.1f}s)")
    print(f"  Total: {len(tdf)}t ({len(orig)} orig + {len(re_trades)} RE) | "
          f"Win: {len(w)/len(tdf)*100:.1f}% | Net: {tdf['net_profit_bps'].sum():+.0f} bps | "
          f"PF: {pf:.2f} | Avg: {tdf['net_profit_bps'].mean():+.1f} | DD: {max_dd:+.0f}")
    print(f"  LONG:  {len(lt)}t, {lt['net_profit_bps'].sum():+.0f} bps")
    print(f"  SHORT: {len(st)}t, {st['net_profit_bps'].sum():+.0f} bps")

    if len(re_trades) > 0:
        re_w = re_trades[re_trades['net_profit_bps'] > 0]
        print(f"  RE trades: {len(re_trades)}t | Win: {len(re_w)/len(re_trades)*100:.1f}% | "
              f"Net: {re_trades['net_profit_bps'].sum():+.0f} bps | "
              f"Avg: {re_trades['net_profit_bps'].mean():+.1f}")

    # Year split
    tdf['year'] = pd.to_datetime(tdf['entry_time']).dt.year
    for y in sorted(tdf['year'].unique()):
        yt = tdf[tdf['year'] == y]
        yw = yt[yt['net_profit_bps'] > 0]
        yl = yt[yt['net_profit_bps'] <= 0]
        ygw = yw['net_profit_bps'].sum() if len(yw) > 0 else 0
        ygl = abs(yl['net_profit_bps'].sum()) if len(yl) > 0 else 1
        print(f"    {y}: {len(yt)}t | Win: {len(yw)/len(yt)*100:.1f}% | "
              f"Net: {yt['net_profit_bps'].sum():+.0f} bps | PF: {ygw/ygl:.2f}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 120)
print("SUMMARY: ALL CONFIGURATIONS")
print("=" * 120)

print(f"\n{'Config':<32} {'Total':>6} {'Orig':>5} {'RE':>4} {'Win%':>6} {'Net bps':>10} {'PF':>6} {'Avg':>7} {'L_Net':>8} {'S_Net':>8} {'MaxDD':>8}")
print("-" * 115)

for r in all_results:
    print(f"{r['name']:<32} {r['trades']:>6} {r['orig_trades']:>5} {r['re_trades']:>4} "
          f"{r['win_pct']:>5.1f}% {r['net_bps']:>+9.0f} {r['pf']:>6.2f} {r['avg_bps']:>+6.1f} "
          f"{r['long_net']:>+7.0f} {r['short_net']:>+7.0f} {r['max_dd']:>+7.0f}")


# =============================================================================
# RE-ENTRY VALUE-ADD
# =============================================================================
print("\n" + "=" * 80)
print("RE-ENTRY VALUE-ADD")
print("=" * 80)

pairs = [('A', 'A+RE'), ('B', 'B+RE'), ('C', 'C+RE'), ('D', 'D+RE')]
results_dict = {r['name'].split(':')[0].strip(): r for r in all_results}

for base_key, re_key in pairs:
    base = results_dict.get(base_key)
    re = results_dict.get(re_key)
    if base and re:
        delta_bps = re['net_bps'] - base['net_bps']
        delta_pf = re['pf'] - base['pf']
        print(f"  {base['name']:<30} -> {re['name']:<30}: "
              f"{delta_bps:+.0f} bps | PF {base['pf']:.2f}→{re['pf']:.2f}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
