"""
EXP-007: V1.3 - SHORT Side WHEN Filters Test
==============================================
Building on V1.2 (LONG already filtered).
Test: Does adding WHEN filters to SHORT improve the already strong side?

Versions tested:
  V1:   Original (no filters)
  V1.2: LONG filtered (ATR>=25pctl + EMA sep>=0.5%)
  V1.3: LONG filtered + SHORT filtered (various combos)

Run: python experiments/rsi/EXP-007/backtest_v13.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
OUTPUT_PATH = Path("experiments/rsi/EXP-007")

# =============================================================================
# PARAMETERS
# =============================================================================
RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
SMA_PERIOD = 200
LONG_TRAILING_STOP = 20
SHORT_TRAILING_STOP = 30
MAX_BARS = 10
FEES_BPS = 8

# V1.2 LONG filters (already proven)
LONG_ATR_MIN = 25
LONG_EMA_SEP_MIN = 0.5

print("=" * 70)
print("EXP-007: V1.3 - SHORT SIDE WHEN FILTERS TEST")
print("=" * 70)

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# SMA200
df['sma_200'] = df['close'].rolling(SMA_PERIOD).mean()

# EMA50/200 for separation
df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
df['ema_separation'] = abs(df['ema50'] - df['ema200']) / df['close'] * 100

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
df['atr_percentile'] = df['atr_bps'].rolling(200).rank(pct=True) * 100

# Time features
df['hour_utc'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = df['day_of_week'] >= 5
df['is_asia_night'] = (df['hour_utc'] >= 0) & (df['hour_utc'] < 4)

# Signals
df['rsi_oversold'] = df['rsi'] < RSI_OVERSOLD
df['rsi_overbought'] = df['rsi'] > RSI_OVERBOUGHT
df['rsi_oversold_cross'] = (df['rsi_oversold'] == True) & (df['rsi_oversold'].shift(1) == False)
df['rsi_overbought_cross'] = (df['rsi_overbought'] == True) & (df['rsi_overbought'].shift(1) == False)
df['bull_market'] = df['close'] > df['sma_200']
df['bear_market'] = df['close'] < df['sma_200']

# OOS data
oos_data = df['2024-01-01':'2025-12-31'].copy()
print(f"Data: 2024-01-01 to 2025-12-31 | {len(oos_data)} bars")


# =============================================================================
# TRADE EXECUTION (same as V1)
# =============================================================================
def execute_trade(data, entry_idx, direction, trailing_stop_bps):
    entry_bar = data.iloc[entry_idx]
    entry_price = data.iloc[entry_idx + 1]['open']
    entry_time = data.index[entry_idx + 1]

    highest_profit = 0
    exit_profit = None
    exit_bar = None
    exit_reason = None
    mae = 0
    mfe = 0

    for bar in range(1, MAX_BARS + 1):
        future_idx = entry_idx + 1 + bar
        if future_idx >= len(data):
            break

        future_bar = data.iloc[future_idx]

        if direction == 'LONG':
            bar_mfe = (future_bar['high'] - entry_price) / entry_price * 10000
            bar_mae = (future_bar['low'] - entry_price) / entry_price * 10000
            bar_close_pnl = (future_bar['close'] - entry_price) / entry_price * 10000
        else:
            bar_mfe = (entry_price - future_bar['low']) / entry_price * 10000
            bar_mae = (entry_price - future_bar['high']) / entry_price * 10000
            bar_close_pnl = (entry_price - future_bar['close']) / entry_price * 10000

        if bar_mfe > mfe:
            mfe = bar_mfe
        if bar_mae < mae:
            mae = bar_mae
        if bar_mfe > highest_profit:
            highest_profit = bar_mfe

        drawdown_from_peak = highest_profit - bar_close_pnl

        if drawdown_from_peak >= trailing_stop_bps and highest_profit > 0:
            exit_profit = highest_profit - trailing_stop_bps
            exit_bar = bar
            exit_reason = 'TRAILING_STOP'
            break

    if exit_profit is None:
        final_bar = data.iloc[entry_idx + 1 + MAX_BARS]
        if direction == 'LONG':
            exit_profit = (final_bar['close'] - entry_price) / entry_price * 10000
        else:
            exit_profit = (entry_price - final_bar['close']) / entry_price * 10000
        exit_bar = MAX_BARS
        exit_reason = 'TIME_EXIT'

    net_profit = exit_profit - FEES_BPS

    return {
        'entry_time': entry_time,
        'exit_time': data.index[entry_idx + 1 + exit_bar],
        'direction': direction,
        'entry_price': entry_price,
        'gross_profit_bps': exit_profit,
        'net_profit_bps': net_profit,
        'mfe_bps': mfe,
        'mae_bps': mae,
        'exit_bar': exit_bar,
        'exit_reason': exit_reason,
        'is_winner': net_profit > 0
    }


# =============================================================================
# BACKTEST WITH CONFIGURABLE FILTERS
# =============================================================================
def run_backtest(data, long_filters=None, short_filters=None, label=""):
    """
    long_filters/short_filters: dict with keys like 'atr_min', 'ema_sep_min',
    'skip_weekend', 'skip_asia_night'
    """
    trades = []
    skipped_long = 0
    skipped_short = 0

    if long_filters is None:
        long_filters = {}
    if short_filters is None:
        short_filters = {}

    for idx in data.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + MAX_BARS + 1 >= len(data):
            continue

        current = data.iloc[idx_pos]

        # LONG signal
        if current['rsi_oversold_cross'] and current['bull_market']:
            skip = False

            if 'atr_min' in long_filters:
                if pd.isna(current['atr_percentile']) or current['atr_percentile'] < long_filters['atr_min']:
                    skip = True
            if 'ema_sep_min' in long_filters:
                if pd.isna(current['ema_separation']) or current['ema_separation'] < long_filters['ema_sep_min']:
                    skip = True
            if long_filters.get('skip_weekend') and current['is_weekend']:
                skip = True
            if long_filters.get('skip_asia_night') and current['is_asia_night']:
                skip = True

            if skip:
                skipped_long += 1
            else:
                trade = execute_trade(data, idx_pos, 'LONG', LONG_TRAILING_STOP)
                if trade:
                    trades.append(trade)

        # SHORT signal
        elif current['rsi_overbought_cross'] and current['bear_market']:
            skip = False

            if 'atr_min' in short_filters:
                if pd.isna(current['atr_percentile']) or current['atr_percentile'] < short_filters['atr_min']:
                    skip = True
            if 'ema_sep_min' in short_filters:
                if pd.isna(current['ema_separation']) or current['ema_separation'] < short_filters['ema_sep_min']:
                    skip = True
            if short_filters.get('skip_weekend') and current['is_weekend']:
                skip = True
            if short_filters.get('skip_asia_night') and current['is_asia_night']:
                skip = True

            if skip:
                skipped_short += 1
            else:
                trade = execute_trade(data, idx_pos, 'SHORT', SHORT_TRAILING_STOP)
                if trade:
                    trades.append(trade)

    return pd.DataFrame(trades), skipped_long, skipped_short


# =============================================================================
# METRICS
# =============================================================================
def calc_metrics(trades):
    if len(trades) == 0:
        return None

    long_t = trades[trades['direction'] == 'LONG']
    short_t = trades[trades['direction'] == 'SHORT']
    winners = trades[trades['net_profit_bps'] > 0]
    losers = trades[trades['net_profit_bps'] <= 0]
    gw = winners['net_profit_bps'].sum() if len(winners) > 0 else 0
    gl = abs(losers['net_profit_bps'].sum()) if len(losers) > 0 else 1

    m = {
        'trades': len(trades),
        'long': len(long_t),
        'short': len(short_t),
        'win_pct': len(winners) / len(trades) * 100,
        'total_bps': trades['net_profit_bps'].sum(),
        'avg_bps': trades['net_profit_bps'].mean(),
        'pf': gw / gl if gl > 0 else float('inf'),
    }

    # LONG
    if len(long_t) > 0:
        lw = long_t[long_t['net_profit_bps'] > 0]
        ll = long_t[long_t['net_profit_bps'] <= 0]
        lgw = lw['net_profit_bps'].sum() if len(lw) > 0 else 0
        lgl = abs(ll['net_profit_bps'].sum()) if len(ll) > 0 else 1
        m['long_win'] = len(lw) / len(long_t) * 100
        m['long_total'] = long_t['net_profit_bps'].sum()
        m['long_pf'] = lgw / lgl if lgl > 0 else float('inf')
    else:
        m['long_win'] = 0
        m['long_total'] = 0
        m['long_pf'] = 0

    # SHORT
    if len(short_t) > 0:
        sw = short_t[short_t['net_profit_bps'] > 0]
        sl = short_t[short_t['net_profit_bps'] <= 0]
        sgw = sw['net_profit_bps'].sum() if len(sw) > 0 else 0
        sgl = abs(sl['net_profit_bps'].sum()) if len(sl) > 0 else 1
        m['short_win'] = len(sw) / len(short_t) * 100
        m['short_total'] = short_t['net_profit_bps'].sum()
        m['short_pf'] = sgw / sgl if sgl > 0 else float('inf')
    else:
        m['short_win'] = 0
        m['short_total'] = 0
        m['short_pf'] = 0

    # Year split
    trades_copy = trades.copy()
    trades_copy['year'] = pd.to_datetime(trades_copy['entry_time']).dt.year
    for year in [2024, 2025]:
        yt = trades_copy[trades_copy['year'] == year]
        if len(yt) > 0:
            m[f'y{year}_trades'] = len(yt)
            m[f'y{year}_total'] = yt['net_profit_bps'].sum()
        else:
            m[f'y{year}_trades'] = 0
            m[f'y{year}_total'] = 0

    return m


# =============================================================================
# TEST ALL CONFIGURATIONS
# =============================================================================
print("\nRunning all configurations...")

# V1.2 LONG filters (always applied)
long_f = {'atr_min': LONG_ATR_MIN, 'ema_sep_min': LONG_EMA_SEP_MIN}

configs = [
    ("V1 (original)", {}, {}),
    ("V1.2 (LONG filtered)", long_f, {}),
    ("V1.3a: SHORT ATR>=25", long_f, {'atr_min': 25}),
    ("V1.3b: SHORT EMA>=0.5%", long_f, {'ema_sep_min': 0.5}),
    ("V1.3c: SHORT no weekend", long_f, {'skip_weekend': True}),
    ("V1.3d: SHORT no Asia night", long_f, {'skip_asia_night': True}),
    ("V1.3e: SHORT ATR+EMA", long_f, {'atr_min': 25, 'ema_sep_min': 0.5}),
    ("V1.3f: SHORT ATR+no wknd", long_f, {'atr_min': 25, 'skip_weekend': True}),
    ("V1.3g: SHORT all filters", long_f, {'atr_min': 25, 'ema_sep_min': 0.5, 'skip_weekend': True, 'skip_asia_night': True}),
]

results = []

for label, lf, sf in configs:
    trades_df, skip_l, skip_s = run_backtest(oos_data, lf, sf, label)
    m = calc_metrics(trades_df)
    if m:
        m['label'] = label
        m['skip_long'] = skip_l
        m['skip_short'] = skip_s
        results.append(m)

# =============================================================================
# PRINT RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS: ALL CONFIGURATIONS")
print("=" * 70)

print(f"\n{'Config':<32} {'Trades':>7} {'Win%':>6} {'Total':>9} {'PF':>6} {'S.Trades':>9} {'S.Win%':>7} {'S.Total':>9} {'S.PF':>6}")
print(f"{'-'*32} {'-'*7} {'-'*6} {'-'*9} {'-'*6} {'-'*9} {'-'*7} {'-'*9} {'-'*6}")

for m in results:
    print(f"{m['label']:<32} {m['trades']:>7} {m['win_pct']:>5.1f}% {m['total_bps']:>+8.0f} {m['pf']:>6.2f} "
          f"{m['short']:>9} {m['short_win']:>6.1f}% {m['short_total']:>+8.0f} {m['short_pf']:>6.2f}")

# =============================================================================
# YEAR-BY-YEAR
# =============================================================================
print("\n" + "=" * 70)
print("YEAR-BY-YEAR COMPARISON")
print("=" * 70)

print(f"\n{'Config':<32} {'2024 T':>7} {'2024 Net':>10} {'2025 T':>7} {'2025 Net':>10}")
print(f"{'-'*32} {'-'*7} {'-'*10} {'-'*7} {'-'*10}")

for m in results:
    print(f"{m['label']:<32} {m['y2024_trades']:>7} {m['y2024_total']:>+9.0f} {m['y2025_trades']:>7} {m['y2025_total']:>+9.0f}")

# =============================================================================
# FILTERED OUT SHORT TRADES ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("WHAT GETS FILTERED OUT FROM SHORT?")
print("=" * 70)

# Run V1.2 to get all SHORT trades
v12_trades, _, _ = run_backtest(oos_data, long_f, {})
v12_short = v12_trades[v12_trades['direction'] == 'SHORT']

# Run V1.3e (ATR+EMA on SHORT) to see what's removed
v13e_trades, _, _ = run_backtest(oos_data, long_f, {'atr_min': 25, 'ema_sep_min': 0.5})
v13e_short = v13e_trades[v13e_trades['direction'] == 'SHORT']

v12_short_times = set(v12_short['entry_time'].values)
v13e_short_times = set(v13e_short['entry_time'].values)

filtered_out_times = v12_short_times - v13e_short_times
filtered_out = v12_short[v12_short['entry_time'].isin(filtered_out_times)]
kept = v12_short[v12_short['entry_time'].isin(v13e_short_times)]

print(f"\nSHORT trades in V1.2:     {len(v12_short)}")
print(f"SHORT trades in V1.3e:    {len(v13e_short)}")
print(f"Filtered out:             {len(filtered_out)}")

if len(filtered_out) > 0:
    fo_w = filtered_out[filtered_out['net_profit_bps'] > 0]
    print(f"\nFiltered out SHORT trades:")
    print(f"  Winners: {len(fo_w)} | Losers: {len(filtered_out) - len(fo_w)}")
    print(f"  Win Rate: {len(fo_w)/len(filtered_out)*100:.1f}%")
    print(f"  Total Net: {filtered_out['net_profit_bps'].sum():+.1f} bps")
    print(f"  Avg Net:   {filtered_out['net_profit_bps'].mean():+.1f} bps")

if len(kept) > 0:
    k_w = kept[kept['net_profit_bps'] > 0]
    print(f"\nKept SHORT trades:")
    print(f"  Winners: {len(k_w)} | Losers: {len(kept) - len(k_w)}")
    print(f"  Win Rate: {len(k_w)/len(kept)*100:.1f}%")
    print(f"  Total Net: {kept['net_profit_bps'].sum():+.1f} bps")
    print(f"  Avg Net:   {kept['net_profit_bps'].mean():+.1f} bps")

# =============================================================================
# CONCLUSION
# =============================================================================
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

v12 = results[1]  # V1.2
best_v13 = max(results[2:], key=lambda x: x['total_bps'])

print(f"\nV1.2:       {v12['trades']} trades | {v12['win_pct']:.1f}% win | {v12['total_bps']:+.0f} bps | PF {v12['pf']:.2f}")
print(f"Best V1.3:  {best_v13['trades']} trades | {best_v13['win_pct']:.1f}% win | {best_v13['total_bps']:+.0f} bps | PF {best_v13['pf']:.2f}  ({best_v13['label']})")
print(f"Difference: {best_v13['total_bps'] - v12['total_bps']:+.0f} bps")

if best_v13['total_bps'] > v12['total_bps']:
    print("\nSHORT filtering HELPS - consider V1.3")
else:
    print("\nSHORT filtering does NOT help - keep V1.2 as is")
