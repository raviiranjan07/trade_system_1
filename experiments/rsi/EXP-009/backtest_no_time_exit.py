"""
EXP-009: Remove TIME_EXIT from V1.2
=====================================
V1.2 baseline: trailing stop + bar 10 time exit
EXP-009: trailing stop ONLY, no time limit

Hold until trailing stop triggers. No forced exit.

Run: python experiments/rsi/EXP-009/backtest_no_time_exit.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
OUTPUT_PATH = Path("experiments/rsi/EXP-009")

# =============================================================================
# PARAMETERS
# =============================================================================
RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
SMA_PERIOD = 200
LONG_TRAILING_STOP = 20  # bps
SHORT_TRAILING_STOP = 30  # bps
FEES_BPS = 8

# V1.2 LONG filters
ATR_MIN_PERCENTILE = 25
EMA_SEP_MIN = 0.5

# Test configs: different max bar limits
# 500 bars = ~5 days of 15min candles (effectively "no limit")
CONFIGS = [
    ("V1.2 (bar 10)", 10),
    ("Bar 20", 20),
    ("Bar 40", 40),
    ("Bar 100", 100),
    ("No Limit (500)", 500),
]

print("=" * 70)
print("EXP-009: REMOVE TIME_EXIT FROM V1.2")
print("Hold until trailing stop triggers")
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
# TRADE EXECUTION - configurable max bars
# =============================================================================
def execute_trade(data, entry_idx, direction, trailing_stop_bps, max_bars):
    entry_price = data.iloc[entry_idx + 1]['open']
    entry_time = data.index[entry_idx + 1]
    entry_bar = data.iloc[entry_idx]

    highest_profit = 0
    exit_profit = None
    exit_bar = None
    exit_price = None
    exit_reason = None
    mae = 0
    mfe = 0

    for bar in range(1, max_bars + 1):
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

    # If no trailing stop triggered, force exit at last available bar
    if exit_profit is None:
        last_bar_idx = min(entry_idx + 1 + max_bars, len(data) - 1)
        actual_bars = last_bar_idx - entry_idx - 1
        final_bar = data.iloc[last_bar_idx]
        if direction == 'LONG':
            exit_profit = (final_bar['close'] - entry_price) / entry_price * 10000
        else:
            exit_profit = (entry_price - final_bar['close']) / entry_price * 10000
        exit_bar = actual_bars
        exit_price = final_bar['close']
        exit_reason = 'TIME_EXIT'

    exit_time = data.index[min(entry_idx + 1 + exit_bar, len(data) - 1)]
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
# RUN BACKTEST FOR EACH CONFIG
# =============================================================================
def run_backtest(data, max_bars):
    trades = []

    for idx in data.index:
        idx_pos = data.index.get_loc(idx)
        if idx_pos + max_bars + 1 >= len(data):
            continue

        current = data.iloc[idx_pos]

        # LONG signal (V1.2 filters)
        if current['rsi_oversold_cross'] and current['bull_market']:
            atr_pctl = current['atr_percentile']
            ema_sep = current['ema_separation']

            if pd.isna(atr_pctl) or pd.isna(ema_sep):
                continue
            if atr_pctl < ATR_MIN_PERCENTILE:
                continue
            if ema_sep < EMA_SEP_MIN:
                continue

            trade = execute_trade(data, idx_pos, 'LONG', LONG_TRAILING_STOP, max_bars)
            if trade:
                trades.append(trade)

        # SHORT signal (no filter)
        elif current['rsi_overbought_cross'] and current['bear_market']:
            trade = execute_trade(data, idx_pos, 'SHORT', SHORT_TRAILING_STOP, max_bars)
            if trade:
                trades.append(trade)

    return pd.DataFrame(trades)


all_results = []

for config_name, max_bars in CONFIGS:
    print(f"\n--- Running {config_name} (max_bars={max_bars}) ---")
    trades = run_backtest(oos_data, max_bars)

    if len(trades) == 0:
        print("  No trades!")
        continue

    winners = trades[trades['net_profit_bps'] > 0]
    losers = trades[trades['net_profit_bps'] <= 0]
    long_t = trades[trades['direction'] == 'LONG']
    short_t = trades[trades['direction'] == 'SHORT']

    gross_win = winners['net_profit_bps'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['net_profit_bps'].sum()) if len(losers) > 0 else 1
    pf = gross_win / gross_loss

    ts_trades = trades[trades['exit_reason'] == 'TRAILING_STOP']
    te_trades = trades[trades['exit_reason'] == 'TIME_EXIT']

    avg_exit_bar = trades['exit_bar'].mean()
    max_exit_bar = trades['exit_bar'].max()

    # Holding time in hours
    trades['hold_hours'] = (pd.to_datetime(trades['exit_time']) - pd.to_datetime(trades['entry_time'])).dt.total_seconds() / 3600

    result = {
        'config': config_name,
        'max_bars': max_bars,
        'total_trades': len(trades),
        'win_rate': len(winners) / len(trades) * 100,
        'total_net_bps': trades['net_profit_bps'].sum(),
        'profit_factor': pf,
        'avg_winner': winners['net_profit_bps'].mean() if len(winners) > 0 else 0,
        'avg_loser': losers['net_profit_bps'].mean() if len(losers) > 0 else 0,
        'ts_count': len(ts_trades),
        'te_count': len(te_trades),
        'avg_exit_bar': avg_exit_bar,
        'max_exit_bar': max_exit_bar,
        'avg_hold_hours': trades['hold_hours'].mean(),
        'max_hold_hours': trades['hold_hours'].max(),
        'long_trades': len(long_t),
        'long_net': long_t['net_profit_bps'].sum() if len(long_t) > 0 else 0,
        'short_trades': len(short_t),
        'short_net': short_t['net_profit_bps'].sum() if len(short_t) > 0 else 0,
    }
    all_results.append(result)

    # Save trades for the "no limit" config
    if max_bars == 500:
        trades.to_csv(OUTPUT_PATH / 'v12_no_time_exit_trades.csv', index=False)
        print(f"  Saved trades to {OUTPUT_PATH / 'v12_no_time_exit_trades.csv'}")


# =============================================================================
# COMPARISON TABLE
# =============================================================================
print("\n" + "=" * 110)
print("COMPARISON: V1.2 WITH DIFFERENT TIME LIMITS")
print("=" * 110)

print(f"{'Config':<22} {'Trades':>7} {'Win%':>7} {'Net bps':>10} {'PF':>7} {'TS':>5} {'TE':>5} "
      f"{'AvgBar':>7} {'MaxBar':>7} {'AvgHrs':>7} {'MaxHrs':>7}")
print("-" * 110)

for r in all_results:
    print(f"{r['config']:<22} {r['total_trades']:>7} {r['win_rate']:>6.1f}% {r['total_net_bps']:>+9.0f} "
          f"{r['profit_factor']:>7.2f} {r['ts_count']:>5} {r['te_count']:>5} "
          f"{r['avg_exit_bar']:>7.1f} {r['max_exit_bar']:>7} {r['avg_hold_hours']:>7.1f} {r['max_hold_hours']:>7.1f}")


# =============================================================================
# LONG vs SHORT BREAKDOWN
# =============================================================================
print("\n" + "=" * 80)
print("LONG vs SHORT BREAKDOWN")
print("=" * 80)

print(f"{'Config':<22} {'LONG N':>7} {'LONG Net':>10} {'SHORT N':>8} {'SHORT Net':>10}")
print("-" * 60)
for r in all_results:
    print(f"{r['config']:<22} {r['long_trades']:>7} {r['long_net']:>+9.0f} {r['short_trades']:>8} {r['short_net']:>+9.0f}")


# =============================================================================
# WHAT HAPPENED TO THE TIME_EXIT LOSERS?
# =============================================================================
print("\n" + "=" * 80)
print("WHAT HAPPENED TO FORMER TIME_EXIT TRADES?")
print("=" * 80)

# Compare bar 10 trades vs no-limit
if len(all_results) >= 2:
    baseline_trades = run_backtest(oos_data, 10)
    nolimit_trades = run_backtest(oos_data, 500)

    # Find trades that were TIME_EXIT losers in V1.2
    te_losers_v12 = baseline_trades[
        (baseline_trades['exit_reason'] == 'TIME_EXIT') &
        (~baseline_trades['is_winner'])
    ]

    print(f"\nTIME_EXIT losers in V1.2: {len(te_losers_v12)} trades, total {te_losers_v12['net_profit_bps'].sum():+.0f} bps")

    # Find what happened to these same entries in the no-limit version
    matched = []
    for _, te_trade in te_losers_v12.iterrows():
        match = nolimit_trades[nolimit_trades['entry_time'] == te_trade['entry_time']]
        if len(match) > 0:
            m = match.iloc[0]
            matched.append({
                'entry_time': te_trade['entry_time'],
                'direction': te_trade['direction'],
                'v12_net': te_trade['net_profit_bps'],
                'v12_bar': te_trade['exit_bar'],
                'nolimit_net': m['net_profit_bps'],
                'nolimit_bar': m['exit_bar'],
                'nolimit_reason': m['exit_reason'],
                'improved': m['net_profit_bps'] > te_trade['net_profit_bps']
            })

    matched_df = pd.DataFrame(matched)

    if len(matched_df) > 0:
        print(f"Matched in no-limit version: {len(matched_df)}")

        improved = matched_df[matched_df['improved']]
        not_improved = matched_df[~matched_df['improved']]

        print(f"\n  Improved (no-limit better than V1.2): {len(improved)} trades")
        if len(improved) > 0:
            print(f"    V1.2 total:     {improved['v12_net'].sum():+.0f} bps")
            print(f"    No-limit total: {improved['nolimit_net'].sum():+.0f} bps")
            print(f"    Gain:           {improved['nolimit_net'].sum() - improved['v12_net'].sum():+.0f} bps")

        print(f"\n  NOT improved (no-limit same or worse): {len(not_improved)} trades")
        if len(not_improved) > 0:
            print(f"    V1.2 total:     {not_improved['v12_net'].sum():+.0f} bps")
            print(f"    No-limit total: {not_improved['nolimit_net'].sum():+.0f} bps")

        # Show each trade
        print(f"\n  Detail - each former TIME_EXIT loser:")
        print(f"  {'Entry':>20} {'Dir':>6} {'V1.2 bps':>10} {'V1.2 Bar':>9} {'NoLim bps':>10} {'NoLim Bar':>10} {'NoLim Exit':>12} {'Better?':>8}")
        print("  " + "-" * 95)
        for _, t in matched_df.sort_values('v12_net').iterrows():
            better = "YES" if t['improved'] else "NO"
            print(f"  {str(t['entry_time'])[:16]:>20} {t['direction']:>6} {t['v12_net']:>+9.1f} {int(t['v12_bar']):>9} "
                  f"{t['nolimit_net']:>+9.1f} {int(t['nolimit_bar']):>10} {t['nolimit_reason']:>12} {better:>8}")

        # How many became winners?
        became_winners = matched_df[matched_df['nolimit_net'] > 0]
        still_losers = matched_df[matched_df['nolimit_net'] <= 0]
        print(f"\n  Became WINNERS: {len(became_winners)}/{len(matched_df)} ({len(became_winners)/len(matched_df)*100:.0f}%)")
        print(f"  Still LOSERS:   {len(still_losers)}/{len(matched_df)} ({len(still_losers)/len(matched_df)*100:.0f}%)")


# =============================================================================
# YEARLY BREAKDOWN FOR NO-LIMIT
# =============================================================================
print("\n" + "=" * 80)
print("YEARLY BREAKDOWN: V1.2 vs NO LIMIT")
print("=" * 80)

for max_bars_test in [10, 500]:
    trades = run_backtest(oos_data, max_bars_test)
    label = "V1.2 (bar 10)" if max_bars_test == 10 else "No Limit (500)"
    print(f"\n{label}:")
    for year in [2024, 2025]:
        yt = trades[pd.to_datetime(trades['entry_time']).dt.year == year]
        if len(yt) == 0:
            continue
        w = yt[yt['net_profit_bps'] > 0]
        l = yt[yt['net_profit_bps'] <= 0]
        gw = w['net_profit_bps'].sum() if len(w) > 0 else 0
        gl = abs(l['net_profit_bps'].sum()) if len(l) > 0 else 1
        print(f"  {year}: {len(yt)} trades | Win: {len(w)/len(yt)*100:.1f}% | Net: {yt['net_profit_bps'].sum():+.0f} bps | PF: {gw/gl:.2f}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
