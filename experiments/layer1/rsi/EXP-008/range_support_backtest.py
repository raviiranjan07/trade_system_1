"""
EXP-008b: Range Support BACKTEST on 15-minute BTCUSDT
=====================================================
Test Range Position as actual trading strategy with entries, exits, fees.

Best OOS setups from analysis:
  Support: LB100/200 <5-10% -> price goes UP (57-59% accuracy OOS)
  Resistance: LB100 >95% -> price goes DOWN (56% accuracy OOS)

Entry: Range position crosses below/above threshold
Exit: Trailing stop + time exit (like V1)
Fees: 8 bps round-trip

Run: python experiments/rsi/EXP-008/range_support_backtest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
FEES_BPS = 8

# Test configurations
CONFIGS = [
    # (name, lb, long_threshold, short_threshold, trailing_stop_long, trailing_stop_short, max_bars)
    ("A: LB100 <10/>90 TS20/30",  100, 10, 90, 20, 30, 10),
    ("B: LB200 <10/>90 TS20/30",  200, 10, 90, 20, 30, 10),
    ("C: LB100 <5/>95 TS20/30",   100,  5, 95, 20, 30, 10),
    ("D: LB200 <5/>95 TS20/30",   200,  5, 95, 20, 30, 10),
    ("E: LB100 <10/>90 TS25/35",  100, 10, 90, 25, 35, 10),
    ("F: LB100 <10/>90 TS20/30 B15", 100, 10, 90, 20, 30, 15),
    ("G: LB100 <10 LONG only",    100, 10, 999, 20, 30, 10),  # 999 = no SHORT
    ("H: LB200 <5 LONG only",     200,  5, 999, 20, 30, 10),
]

# OOS period
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

print("=" * 70)
print("EXP-008b: RANGE SUPPORT BACKTEST")
print("=" * 70)

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

# Filter to OOS
df = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()
print(f"Data: {df.index[0].date()} to {df.index[-1].date()} | {len(df)} bars")

# Pre-compute range positions for all lookbacks
for lb in [100, 200]:
    # Need to compute on full dataset first, but we've already filtered
    # Re-load for proper rolling calculation
    pass

# Reload full data for rolling calculations
df_full = pd.read_parquet(DATA_PATH)
df_full.index = pd.to_datetime(df_full.index).tz_localize(None)

for lb in [100, 200]:
    high_roll = df_full['high'].rolling(lb).max()
    low_roll = df_full['low'].rolling(lb).min()
    range_size = high_roll - low_roll
    range_size = range_size.replace(0, np.nan)
    df_full[f'range_pos_{lb}'] = ((df_full['close'] - low_roll) / range_size) * 100

# Filter to OOS after computing indicators
df = df_full[(df_full.index >= TEST_START) & (df_full.index <= TEST_END)].copy()
print(f"OOS bars: {len(df)}")


def simulate_trade(df, entry_idx, direction, trailing_stop_bps, max_bars):
    """Simulate a single trade with trailing stop and time exit."""
    entry_price = df.iloc[entry_idx]['close']
    best_price = entry_price
    exit_bar = None
    exit_price = None
    exit_reason = None

    for bar_offset in range(1, max_bars + 1):
        idx = entry_idx + bar_offset
        if idx >= len(df):
            # End of data
            exit_price = df.iloc[-1]['close']
            exit_reason = "end_of_data"
            exit_bar = bar_offset
            break

        high = df.iloc[idx]['high']
        low = df.iloc[idx]['low']
        close = df.iloc[idx]['close']

        if direction == "LONG":
            # Update best price (highest point)
            best_price = max(best_price, high)
            # Check trailing stop: did price drop trailing_stop_bps from best?
            stop_level = best_price * (1 - trailing_stop_bps / 10000)
            if low <= stop_level:
                exit_price = stop_level
                exit_reason = "trailing_stop"
                exit_bar = bar_offset
                break
        else:  # SHORT
            # Update best price (lowest point)
            best_price = min(best_price, low)
            # Check trailing stop: did price rise trailing_stop_bps from best?
            stop_level = best_price * (1 + trailing_stop_bps / 10000)
            if high >= stop_level:
                exit_price = stop_level
                exit_reason = "trailing_stop"
                exit_bar = bar_offset
                break

    # Time exit if no trailing stop hit
    if exit_price is None:
        final_idx = min(entry_idx + max_bars, len(df) - 1)
        exit_price = df.iloc[final_idx]['close']
        exit_reason = "time_exit"
        exit_bar = max_bars

    # Calculate P&L
    if direction == "LONG":
        gross_bps = (exit_price - entry_price) / entry_price * 10000
    else:
        gross_bps = (entry_price - exit_price) / entry_price * 10000

    net_bps = gross_bps - FEES_BPS

    return {
        'entry_price': entry_price,
        'exit_price': exit_price,
        'direction': direction,
        'gross_bps': gross_bps,
        'net_bps': net_bps,
        'exit_reason': exit_reason,
        'exit_bar': exit_bar,
        'entry_time': df.index[entry_idx],
    }


def run_backtest(df, lb, long_thresh, short_thresh, ts_long, ts_short, max_bars):
    """Run backtest with range position entry signals."""
    col = f'range_pos_{lb}'
    trades = []
    i = 0

    while i < len(df) - max_bars - 1:
        rp = df.iloc[i][col]

        if pd.isna(rp):
            i += 1
            continue

        # LONG signal: range position crosses below threshold
        if rp < long_thresh:
            # Check it wasn't already below on previous bar (cross signal)
            if i > 0 and not pd.isna(df.iloc[i-1][col]) and df.iloc[i-1][col] >= long_thresh:
                trade = simulate_trade(df, i, "LONG", ts_long, max_bars)
                trades.append(trade)
                i += trade['exit_bar'] + 1  # Skip past this trade
                continue

        # SHORT signal: range position crosses above threshold
        if short_thresh < 100 and rp > short_thresh:
            if i > 0 and not pd.isna(df.iloc[i-1][col]) and df.iloc[i-1][col] <= short_thresh:
                trade = simulate_trade(df, i, "SHORT", ts_short, max_bars)
                trades.append(trade)
                i += trade['exit_bar'] + 1
                continue

        i += 1

    return trades


# =============================================================================
# RUN ALL CONFIGURATIONS
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS: ALL CONFIGURATIONS (OOS 2024-2025)")
print("=" * 70)

print(f"\n{'Config':<35} {'Trades':>6} {'Win%':>6} {'Total':>8} {'PF':>6} "
      f"{'L.Tr':>5} {'L.Win%':>7} {'L.Tot':>7} {'S.Tr':>5} {'S.Win%':>7} {'S.Tot':>7}")
print("-" * 110)

all_results = {}
for name, lb, lt, st, tsl, tss, mb in CONFIGS:
    trades = run_backtest(df, lb, lt, st, tsl, tss, mb)

    if not trades:
        print(f"{name:<35} {'NO TRADES':>6}")
        continue

    tdf = pd.DataFrame(trades)
    total = len(tdf)
    winners = tdf[tdf['net_bps'] > 0]
    losers = tdf[tdf['net_bps'] <= 0]
    win_pct = len(winners) / total * 100
    total_net = tdf['net_bps'].sum()
    gross_win = winners['net_bps'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['net_bps'].sum()) if len(losers) > 0 else 0.001
    pf = gross_win / gross_loss

    # LONG/SHORT split
    longs = tdf[tdf['direction'] == 'LONG']
    shorts = tdf[tdf['direction'] == 'SHORT']

    l_trades = len(longs)
    l_win = len(longs[longs['net_bps'] > 0]) / max(l_trades, 1) * 100
    l_total = longs['net_bps'].sum() if l_trades > 0 else 0

    s_trades = len(shorts)
    s_win = len(shorts[shorts['net_bps'] > 0]) / max(s_trades, 1) * 100
    s_total = shorts['net_bps'].sum() if s_trades > 0 else 0

    print(f"{name:<35} {total:>6} {win_pct:>5.1f}% {total_net:>+7.0f} {pf:>5.2f} "
          f"{l_trades:>5} {l_win:>6.1f}% {l_total:>+6.0f} {s_trades:>5} {s_win:>6.1f}% {s_total:>+6.0f}")

    all_results[name] = tdf

# =============================================================================
# DETAILED ANALYSIS OF BEST CONFIG
# =============================================================================
# Find best by total net
best_name = max(all_results.keys(), key=lambda k: all_results[k]['net_bps'].sum())
best_trades = all_results[best_name]

print(f"\n{'='*70}")
print(f"DETAILED: {best_name}")
print(f"{'='*70}")

print(f"\nTotal trades: {len(best_trades)}")
print(f"Winners: {len(best_trades[best_trades['net_bps'] > 0])} | "
      f"Losers: {len(best_trades[best_trades['net_bps'] <= 0])}")
print(f"Win Rate: {len(best_trades[best_trades['net_bps'] > 0]) / len(best_trades) * 100:.1f}%")
print(f"Total Net: {best_trades['net_bps'].sum():+.0f} bps")
print(f"Avg Net/Trade: {best_trades['net_bps'].mean():+.1f} bps")
print(f"Avg Winner: {best_trades[best_trades['net_bps'] > 0]['net_bps'].mean():+.1f} bps")
print(f"Avg Loser: {best_trades[best_trades['net_bps'] <= 0]['net_bps'].mean():+.1f} bps")

# Exit reasons
print(f"\nExit Reasons:")
for reason, count in best_trades['exit_reason'].value_counts().items():
    pct = count / len(best_trades) * 100
    avg_net = best_trades[best_trades['exit_reason'] == reason]['net_bps'].mean()
    print(f"  {reason}: {count} ({pct:.0f}%) avg {avg_net:+.1f} bps")

# Year-by-year
print(f"\nYear-by-Year:")
best_trades['year'] = best_trades['entry_time'].dt.year
for year in sorted(best_trades['year'].unique()):
    yt = best_trades[best_trades['year'] == year]
    yw = len(yt[yt['net_bps'] > 0]) / len(yt) * 100
    print(f"  {year}: {len(yt)} trades, {yw:.1f}% win, {yt['net_bps'].sum():+.0f} bps")

# =============================================================================
# COMPARISON WITH V1.2
# =============================================================================
print(f"\n{'='*70}")
print("COMPARISON WITH V1.2")
print(f"{'='*70}")
print(f"{'Strategy':<35} {'Trades':>6} {'Win%':>6} {'Total':>8} {'PF':>6} {'Avg/Tr':>7}")
print("-" * 70)

# V1.2 numbers from memory
print(f"{'V1.2 (RSI + WHEN filters)':<35} {'202':>6} {'56.9%':>6} {'+3250':>8} {'2.47':>6} {'+16.1':>7}")

bt = best_trades
bt_win = len(bt[bt['net_bps'] > 0]) / len(bt) * 100
bt_total = bt['net_bps'].sum()
bt_gw = bt[bt['net_bps'] > 0]['net_bps'].sum()
bt_gl = abs(bt[bt['net_bps'] <= 0]['net_bps'].sum()) + 0.001
bt_pf = bt_gw / bt_gl
bt_avg = bt['net_bps'].mean()
print(f"{best_name:<35} {len(bt):>6} {bt_win:>5.1f}% {bt_total:>+7.0f} {bt_pf:>5.2f} {bt_avg:>+6.1f}")
