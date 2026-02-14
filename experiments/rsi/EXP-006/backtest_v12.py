"""
EXP-006: V1.2 Backtest - LONG Side WHEN Filters
=================================================
V1.2 = V1 + two filters on LONG side only:
  1. ATR percentile >= 25th (skip quiet markets)
  2. EMA separation >= 0.5% (skip choppy markets)

SHORT side: NO CHANGES

Run: python experiments/rsi/EXP-006/backtest_v12.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
OUTPUT_PATH = Path("experiments/rsi/EXP-006")

# =============================================================================
# V1 PARAMETERS (unchanged)
# =============================================================================
RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
SMA_PERIOD = 200
LONG_TRAILING_STOP = 20  # bps
SHORT_TRAILING_STOP = 30  # bps
MAX_BARS = 10
FEES_BPS = 8

# V1.2 NEW FILTERS (LONG only)
ATR_MIN_PERCENTILE = 25  # skip LONG when ATR below 25th percentile
EMA_SEP_MIN = 0.5  # skip LONG when EMA separation below 0.5%

print("=" * 70)
print("EXP-006: V1.2 BACKTEST")
print("V1 + LONG filters (ATR >= 25th pctl, EMA sep >= 0.5%)")
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

# EMA50 and EMA200 for separation
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

# Signal detection
df['rsi_oversold'] = df['rsi'] < RSI_OVERSOLD
df['rsi_overbought'] = df['rsi'] > RSI_OVERBOUGHT
df['rsi_oversold_cross'] = (df['rsi_oversold'] == True) & (df['rsi_oversold'].shift(1) == False)
df['rsi_overbought_cross'] = (df['rsi_overbought'] == True) & (df['rsi_overbought'].shift(1) == False)

# Market condition
df['bull_market'] = df['close'] > df['sma_200']
df['bear_market'] = df['close'] < df['sma_200']

# OOS data
oos_start = '2024-01-01'
oos_end = '2025-12-31'
oos_data = df[oos_start:oos_end].copy()

print(f"\nData: {oos_start} to {oos_end}")
print(f"Total bars: {len(oos_data)}")


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
    exit_price = None
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
            exit_price = entry_price * (1 + exit_profit / 10000) if direction == 'LONG' else entry_price * (1 - exit_profit / 10000)
            exit_reason = 'TRAILING_STOP'
            break

    if exit_profit is None:
        final_bar = data.iloc[entry_idx + 1 + MAX_BARS]
        if direction == 'LONG':
            exit_profit = (final_bar['close'] - entry_price) / entry_price * 10000
        else:
            exit_profit = (entry_price - final_bar['close']) / entry_price * 10000
        exit_bar = MAX_BARS
        exit_price = final_bar['close']
        exit_reason = 'TIME_EXIT'

    exit_time = data.index[entry_idx + 1 + exit_bar]
    net_profit = exit_profit - FEES_BPS

    return {
        'entry_time': entry_time,
        'exit_time': exit_time,
        'direction': direction,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'rsi_at_entry': entry_bar['rsi'],
        'gross_profit_bps': exit_profit,
        'net_profit_bps': net_profit,
        'mfe_bps': mfe,
        'mae_bps': mae,
        'exit_bar': exit_bar,
        'exit_reason': exit_reason,
        'is_winner': net_profit > 0
    }


# =============================================================================
# RUN BOTH V1 AND V1.2
# =============================================================================
def run_backtest(data, version="V1"):
    trades = []
    skipped_long = 0

    for idx in data.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + MAX_BARS + 1 >= len(data):
            continue

        current = data.iloc[idx_pos]

        # LONG signal
        if current['rsi_oversold_cross'] and current['bull_market']:

            # V1.2: Check WHEN filters for LONG
            if version == "V1.2":
                atr_pctl = current['atr_percentile']
                ema_sep = current['ema_separation']

                if pd.isna(atr_pctl) or pd.isna(ema_sep):
                    skipped_long += 1
                    continue

                if atr_pctl < ATR_MIN_PERCENTILE:
                    skipped_long += 1
                    continue

                if ema_sep < EMA_SEP_MIN:
                    skipped_long += 1
                    continue

            trade = execute_trade(data, idx_pos, 'LONG', LONG_TRAILING_STOP)
            if trade:
                trades.append(trade)

        # SHORT signal (SAME for V1 and V1.2 - no changes)
        elif current['rsi_overbought_cross'] and current['bear_market']:
            trade = execute_trade(data, idx_pos, 'SHORT', SHORT_TRAILING_STOP)
            if trade:
                trades.append(trade)

    if skipped_long > 0:
        print(f"  LONG trades skipped by filter: {skipped_long}")

    return pd.DataFrame(trades)


print("\n--- Running V1 (original) ---")
v1_trades = run_backtest(oos_data, "V1")

print("\n--- Running V1.2 (LONG filtered) ---")
v12_trades = run_backtest(oos_data, "V1.2")


# =============================================================================
# METRICS FUNCTION
# =============================================================================
def calc_metrics(trades, label=""):
    if len(trades) == 0:
        return {}

    long_t = trades[trades['direction'] == 'LONG']
    short_t = trades[trades['direction'] == 'SHORT']
    winners = trades[trades['net_profit_bps'] > 0]
    losers = trades[trades['net_profit_bps'] <= 0]

    gross_win = winners['net_profit_bps'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['net_profit_bps'].sum()) if len(losers) > 0 else 1
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    m = {
        'label': label,
        'total_trades': len(trades),
        'long_trades': len(long_t),
        'short_trades': len(short_t),
        'win_rate': len(winners) / len(trades) * 100,
        'total_net_bps': trades['net_profit_bps'].sum(),
        'avg_net_bps': trades['net_profit_bps'].mean(),
        'avg_winner': winners['net_profit_bps'].mean() if len(winners) > 0 else 0,
        'avg_loser': losers['net_profit_bps'].mean() if len(losers) > 0 else 0,
        'profit_factor': pf,
    }

    # LONG metrics
    if len(long_t) > 0:
        long_w = long_t[long_t['net_profit_bps'] > 0]
        long_l = long_t[long_t['net_profit_bps'] <= 0]
        long_gw = long_w['net_profit_bps'].sum() if len(long_w) > 0 else 0
        long_gl = abs(long_l['net_profit_bps'].sum()) if len(long_l) > 0 else 1
        m['long_win_rate'] = len(long_w) / len(long_t) * 100
        m['long_total'] = long_t['net_profit_bps'].sum()
        m['long_avg'] = long_t['net_profit_bps'].mean()
        m['long_pf'] = long_gw / long_gl if long_gl > 0 else float('inf')
    else:
        m['long_win_rate'] = 0
        m['long_total'] = 0
        m['long_avg'] = 0
        m['long_pf'] = 0

    # SHORT metrics
    if len(short_t) > 0:
        short_w = short_t[short_t['net_profit_bps'] > 0]
        short_l = short_t[short_t['net_profit_bps'] <= 0]
        short_gw = short_w['net_profit_bps'].sum() if len(short_w) > 0 else 0
        short_gl = abs(short_l['net_profit_bps'].sum()) if len(short_l) > 0 else 1
        m['short_win_rate'] = len(short_w) / len(short_t) * 100
        m['short_total'] = short_t['net_profit_bps'].sum()
        m['short_avg'] = short_t['net_profit_bps'].mean()
        m['short_pf'] = short_gw / short_gl if short_gl > 0 else float('inf')
    else:
        m['short_win_rate'] = 0
        m['short_total'] = 0
        m['short_avg'] = 0
        m['short_pf'] = 0

    return m


# =============================================================================
# COMPARE V1 vs V1.2
# =============================================================================
m1 = calc_metrics(v1_trades, "V1")
m12 = calc_metrics(v12_trades, "V1.2")

print("\n" + "=" * 70)
print("COMPARISON: V1 vs V1.2")
print("=" * 70)

print(f"\n{'Metric':<25} {'V1':>12} {'V1.2':>12} {'Diff':>12}")
print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}")

comparisons = [
    ('Total Trades', 'total_trades', 'd'),
    ('  LONG trades', 'long_trades', 'd'),
    ('  SHORT trades', 'short_trades', 'd'),
    ('Win Rate %', 'win_rate', 'f'),
    ('Total Net (bps)', 'total_net_bps', 'f'),
    ('Avg Net/Trade (bps)', 'avg_net_bps', 'f'),
    ('Profit Factor', 'profit_factor', 'f2'),
    ('Avg Winner (bps)', 'avg_winner', 'f'),
    ('Avg Loser (bps)', 'avg_loser', 'f'),
    ('LONG Win Rate %', 'long_win_rate', 'f'),
    ('LONG Total (bps)', 'long_total', 'f'),
    ('LONG Avg/Trade', 'long_avg', 'f'),
    ('LONG PF', 'long_pf', 'f2'),
    ('SHORT Win Rate %', 'short_win_rate', 'f'),
    ('SHORT Total (bps)', 'short_total', 'f'),
    ('SHORT Avg/Trade', 'short_avg', 'f'),
    ('SHORT PF', 'short_pf', 'f2'),
]

for label, key, fmt in comparisons:
    v1_val = m1[key]
    v12_val = m12[key]
    diff = v12_val - v1_val

    if fmt == 'd':
        print(f"{label:<25} {v1_val:>12.0f} {v12_val:>12.0f} {diff:>+12.0f}")
    elif fmt == 'f':
        print(f"{label:<25} {v1_val:>12.1f} {v12_val:>12.1f} {diff:>+12.1f}")
    elif fmt == 'f2':
        print(f"{label:<25} {v1_val:>12.2f} {v12_val:>12.2f} {diff:>+12.2f}")


# =============================================================================
# YEAR-BY-YEAR COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("YEAR-BY-YEAR COMPARISON")
print("=" * 70)

for year in [2024, 2025]:
    print(f"\n--- {year} ---")
    for version_name, version_trades in [("V1", v1_trades), ("V1.2", v12_trades)]:
        yt = version_trades[pd.to_datetime(version_trades['entry_time']).dt.year == year]
        if len(yt) == 0:
            continue

        long_t = yt[yt['direction'] == 'LONG']
        short_t = yt[yt['direction'] == 'SHORT']

        long_str = f"LONG: {len(long_t)}t, {long_t['net_profit_bps'].sum():+.0f}bps" if len(long_t) > 0 else "LONG: 0t"
        short_str = f"SHORT: {len(short_t)}t, {short_t['net_profit_bps'].sum():+.0f}bps" if len(short_t) > 0 else "SHORT: 0t"

        winners = yt[yt['net_profit_bps'] > 0]
        losers = yt[yt['net_profit_bps'] <= 0]
        gw = winners['net_profit_bps'].sum() if len(winners) > 0 else 0
        gl = abs(losers['net_profit_bps'].sum()) if len(losers) > 0 else 1

        print(f"  {version_name}: {len(yt)} trades | Win: {len(winners)/len(yt)*100:.1f}% | "
              f"Total: {yt['net_profit_bps'].sum():+.0f}bps | PF: {gw/gl:.2f} | {long_str} | {short_str}")


# =============================================================================
# QUARTERLY COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("QUARTERLY COMPARISON")
print("=" * 70)

v1_trades['quarter'] = pd.to_datetime(v1_trades['entry_time']).dt.to_period('Q')
v12_trades['quarter'] = pd.to_datetime(v12_trades['entry_time']).dt.to_period('Q')

all_quarters = sorted(set(v1_trades['quarter'].unique()) | set(v12_trades['quarter'].unique()))

print(f"\n{'Quarter':<10} {'V1 Trades':>10} {'V1 Net':>10} {'V1.2 Trades':>12} {'V1.2 Net':>10} {'Diff':>10}")
print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")

for q in all_quarters:
    v1_q = v1_trades[v1_trades['quarter'] == q]
    v12_q = v12_trades[v12_trades['quarter'] == q]

    v1_net = v1_q['net_profit_bps'].sum() if len(v1_q) > 0 else 0
    v12_net = v12_q['net_profit_bps'].sum() if len(v12_q) > 0 else 0

    print(f"{str(q):<10} {len(v1_q):>10} {v1_net:>+10.0f} {len(v12_q):>12} {v12_net:>+10.0f} {v12_net - v1_net:>+10.0f}")

total_v1 = v1_trades['net_profit_bps'].sum()
total_v12 = v12_trades['net_profit_bps'].sum()
print(f"{'TOTAL':<10} {len(v1_trades):>10} {total_v1:>+10.0f} {len(v12_trades):>12} {total_v12:>+10.0f} {total_v12 - total_v1:>+10.0f}")


# =============================================================================
# WHAT TRADES WERE FILTERED OUT?
# =============================================================================
print("\n" + "=" * 70)
print("FILTERED OUT LONG TRADES ANALYSIS")
print("=" * 70)

# Find LONG trades in V1 that are NOT in V1.2
v1_long = v1_trades[v1_trades['direction'] == 'LONG']
v12_long = v12_trades[v12_trades['direction'] == 'LONG']

v1_long_times = set(v1_long['entry_time'].values)
v12_long_times = set(v12_long['entry_time'].values)

filtered_out_times = v1_long_times - v12_long_times
kept_times = v1_long_times & v12_long_times

filtered_out = v1_long[v1_long['entry_time'].isin(filtered_out_times)]
kept = v1_long[v1_long['entry_time'].isin(kept_times)]

print(f"\nLONG trades in V1:        {len(v1_long)}")
print(f"LONG trades in V1.2:      {len(v12_long)}")
print(f"Filtered out:             {len(filtered_out)}")

if len(filtered_out) > 0:
    fo_winners = filtered_out[filtered_out['net_profit_bps'] > 0]
    fo_losers = filtered_out[filtered_out['net_profit_bps'] <= 0]
    print(f"\nFiltered out trades:")
    print(f"  Winners: {len(fo_winners)} | Losers: {len(fo_losers)}")
    print(f"  Win Rate: {len(fo_winners)/len(filtered_out)*100:.1f}%")
    print(f"  Total Net: {filtered_out['net_profit_bps'].sum():+.1f} bps")
    print(f"  Avg Net:   {filtered_out['net_profit_bps'].mean():+.1f} bps")

if len(kept) > 0:
    k_winners = kept[kept['net_profit_bps'] > 0]
    k_losers = kept[kept['net_profit_bps'] <= 0]
    print(f"\nKept trades:")
    print(f"  Winners: {len(k_winners)} | Losers: {len(k_losers)}")
    print(f"  Win Rate: {len(k_winners)/len(kept)*100:.1f}%")
    print(f"  Total Net: {kept['net_profit_bps'].sum():+.1f} bps")
    print(f"  Avg Net:   {kept['net_profit_bps'].mean():+.1f} bps")


# =============================================================================
# SAVE RESULTS
# =============================================================================
v12_trades.to_csv(OUTPUT_PATH / 'v12_trades.csv', index=False)
print(f"\nSaved: {OUTPUT_PATH / 'v12_trades.csv'}")

# Save metrics
with open(OUTPUT_PATH / 'results.txt', 'w') as f:
    f.write("EXP-006: V1.2 Results\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"V1:  {m1['total_trades']} trades | Win: {m1['win_rate']:.1f}% | Net: {m1['total_net_bps']:+.1f} bps | PF: {m1['profit_factor']:.2f}\n")
    f.write(f"V1.2: {m12['total_trades']} trades | Win: {m12['win_rate']:.1f}% | Net: {m12['total_net_bps']:+.1f} bps | PF: {m12['profit_factor']:.2f}\n\n")
    f.write(f"LONG improvement:\n")
    f.write(f"  V1 LONG:  {m1['long_trades']} trades | PF: {m1['long_pf']:.2f} | Total: {m1['long_total']:+.1f}\n")
    f.write(f"  V1.2 LONG: {m12['long_trades']} trades | PF: {m12['long_pf']:.2f} | Total: {m12['long_total']:+.1f}\n")

print(f"Saved: {OUTPUT_PATH / 'results.txt'}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"\nV1:   {m1['total_trades']} trades | {m1['win_rate']:.1f}% win | {m1['total_net_bps']:+.0f} bps | PF {m1['profit_factor']:.2f}")
print(f"V1.2: {m12['total_trades']} trades | {m12['win_rate']:.1f}% win | {m12['total_net_bps']:+.0f} bps | PF {m12['profit_factor']:.2f}")
print(f"\nLONG:  PF {m1['long_pf']:.2f} -> {m12['long_pf']:.2f}")
print(f"SHORT: PF {m1['short_pf']:.2f} -> {m12['short_pf']:.2f} (unchanged)")
