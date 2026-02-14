"""
EXP-010: EMA Crossover as Trend Reversal Strategy
===================================================
Test if EMA crossovers can signal trend reversals on 15min BTCUSDT.

Entry: Fast EMA crosses Slow EMA
  - Fast crosses ABOVE slow = trend turning UP → LONG
  - Fast crosses BELOW slow = trend turning DOWN → SHORT

Test multiple EMA pairs and full backtest with trailing stops + fees.

Run: python experiments/rsi/EXP-010/ema_crossover_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
OUTPUT_PATH = Path("experiments/rsi/EXP-010")

# =============================================================================
# PARAMETERS
# =============================================================================
FEES_BPS = 8
MAX_BARS = 10
LONG_TRAILING_STOP = 20   # bps
SHORT_TRAILING_STOP = 30  # bps

# EMA pairs to test (fast, slow)
EMA_PAIRS = [
    (5, 13),
    (8, 21),
    (9, 21),
    (12, 26),
    (20, 50),
    (50, 200),
]

TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

print("=" * 80)
print("EXP-010: EMA CROSSOVER AS TREND REVERSAL STRATEGY")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)
print(f"Data: {df.index[0].date()} to {df.index[-1].date()} | {len(df)} bars")

# =============================================================================
# PHASE 1: SIGNAL COUNT & DIRECTIONAL ACCURACY
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 1: SIGNAL COUNT & DIRECTIONAL ACCURACY")
print("=" * 80)

# Calculate all EMAs
ema_periods = sorted(set([p for pair in EMA_PAIRS for p in pair]))
for p in ema_periods:
    df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()

# Future returns for measuring direction
for h in [5, 10, 20]:
    df[f'future_return_{h}'] = (df['close'].shift(-h) - df['close']) / df['close'] * 10000
    df[f'future_max_{h}'] = df['high'].shift(-1).rolling(h).max().shift(-h+1)
    df[f'future_max_bps_{h}'] = (df[f'future_max_{h}'] - df['close']) / df['close'] * 10000

# Compute ALL crossover columns BEFORE splitting
for fast, slow in EMA_PAIRS:
    fast_col = f'ema_{fast}'
    slow_col = f'ema_{slow}'
    df[f'cross_{fast}_{slow}_bull'] = (df[fast_col] > df[slow_col]) & (df[fast_col].shift(1) <= df[slow_col].shift(1))
    df[f'cross_{fast}_{slow}_bear'] = (df[fast_col] < df[slow_col]) & (df[fast_col].shift(1) >= df[slow_col].shift(1))

# Split
train = df[df.index <= TRAIN_END].copy()
test = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()

print(f"Train: {len(train)} bars | Test: {len(test)} bars")

print(f"\n{'Pair':>10} {'Period':>8} {'Signals':>8} {'LONG':>6} {'SHORT':>7} "
      f"{'L UP%':>7} {'S DN%':>7} {'L Avg':>8} {'S Avg':>8}")
print("-" * 85)

phase1_results = []

for fast, slow in EMA_PAIRS:
    for period_name, data in [("TRAIN", train), ("TEST", test)]:
        bull_signals = data[data[f'cross_{fast}_{slow}_bull'] == True]
        bear_signals = data[data[f'cross_{fast}_{slow}_bear'] == True]

        # Directional accuracy at horizon 10 (same as V1.2's max bars)
        h = 10
        bull_up = bull_signals[f'future_return_{h}'].dropna()
        bear_dn = bear_signals[f'future_return_{h}'].dropna()

        bull_up_pct = (bull_up > 0).mean() * 100 if len(bull_up) > 0 else 0
        bear_dn_pct = (bear_dn < 0).mean() * 100 if len(bear_dn) > 0 else 0

        bull_avg = bull_up.mean() if len(bull_up) > 0 else 0
        bear_avg = bear_dn.mean() if len(bear_dn) > 0 else 0

        phase1_results.append({
            'pair': f'{fast}/{slow}',
            'period': period_name,
            'total_signals': len(bull_signals) + len(bear_signals),
            'long_signals': len(bull_signals),
            'short_signals': len(bear_signals),
            'long_up_pct': bull_up_pct,
            'short_dn_pct': bear_dn_pct,
            'long_avg_bps': bull_avg,
            'short_avg_bps': bear_avg,
        })

        print(f"{fast}/{slow:>3} {period_name:>8} {len(bull_signals)+len(bear_signals):>8} "
              f"{len(bull_signals):>6} {len(bear_signals):>7} "
              f"{bull_up_pct:>6.1f}% {bear_dn_pct:>6.1f}% "
              f"{bull_avg:>+7.1f} {bear_avg:>+7.1f}")


# =============================================================================
# PHASE 2: FULL BACKTEST WITH TRAILING STOPS
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 2: FULL BACKTEST (Trailing Stop + TIME_EXIT + 8bps fees)")
print("=" * 80)

def execute_trade(data, entry_idx, direction, trailing_stop_bps, max_bars):
    entry_price = data.iloc[entry_idx + 1]['open']
    entry_time = data.index[entry_idx + 1]

    highest_profit = 0
    exit_profit = None
    exit_bar = None
    exit_reason = None
    mae = 0
    mfe = 0

    for bar in range(1, max_bars + 1):
        future_idx = entry_idx + 1 + bar
        if future_idx >= len(data):
            return None

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
        last_idx = min(entry_idx + 1 + max_bars, len(data) - 1)
        actual_bars = last_idx - entry_idx - 1
        final_bar = data.iloc[last_idx]
        if direction == 'LONG':
            exit_profit = (final_bar['close'] - entry_price) / entry_price * 10000
        else:
            exit_profit = (entry_price - final_bar['close']) / entry_price * 10000
        exit_bar = actual_bars
        exit_reason = 'TIME_EXIT'

    net_profit = exit_profit - FEES_BPS
    exit_time = data.index[min(entry_idx + 1 + exit_bar, len(data) - 1)]

    return {
        'entry_time': entry_time,
        'exit_time': exit_time,
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


# Test each EMA pair
backtest_results = []

for fast, slow in EMA_PAIRS:
    bull_col = f'cross_{fast}_{slow}_bull'
    bear_col = f'cross_{fast}_{slow}_bear'

    trades = []
    for idx_pos in range(len(test)):
        if idx_pos + MAX_BARS + 2 >= len(test):
            break

        current = test.iloc[idx_pos]

        if current[bull_col]:
            trade = execute_trade(test, idx_pos, 'LONG', LONG_TRAILING_STOP, MAX_BARS)
            if trade:
                trades.append(trade)

        elif current[bear_col]:
            trade = execute_trade(test, idx_pos, 'SHORT', SHORT_TRAILING_STOP, MAX_BARS)
            if trade:
                trades.append(trade)

    if len(trades) == 0:
        continue

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df['is_winner']]
    losers = trades_df[~trades_df['is_winner']]
    long_t = trades_df[trades_df['direction'] == 'LONG']
    short_t = trades_df[trades_df['direction'] == 'SHORT']

    gw = winners['net_profit_bps'].sum() if len(winners) > 0 else 0
    gl = abs(losers['net_profit_bps'].sum()) if len(losers) > 0 else 1
    pf = gw / gl

    result = {
        'pair': f'{fast}/{slow}',
        'trades': len(trades_df),
        'win_rate': len(winners) / len(trades_df) * 100,
        'total_net': trades_df['net_profit_bps'].sum(),
        'pf': pf,
        'avg_winner': winners['net_profit_bps'].mean() if len(winners) > 0 else 0,
        'avg_loser': losers['net_profit_bps'].mean() if len(losers) > 0 else 0,
        'long_n': len(long_t),
        'long_net': long_t['net_profit_bps'].sum() if len(long_t) > 0 else 0,
        'short_n': len(short_t),
        'short_net': short_t['net_profit_bps'].sum() if len(short_t) > 0 else 0,
    }
    backtest_results.append(result)

# Print results
print(f"\n{'Pair':>8} {'Trades':>7} {'Win%':>7} {'Net bps':>10} {'PF':>7} "
      f"{'AvgW':>7} {'AvgL':>7} {'L_N':>5} {'L_Net':>8} {'S_N':>5} {'S_Net':>8}")
print("-" * 95)

for r in backtest_results:
    print(f"{r['pair']:>8} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['total_net']:>+9.0f} "
          f"{r['pf']:>7.2f} {r['avg_winner']:>+6.1f} {r['avg_loser']:>+6.1f} "
          f"{r['long_n']:>5} {r['long_net']:>+7.0f} {r['short_n']:>5} {r['short_net']:>+7.0f}")


# =============================================================================
# PHASE 3: BEST EMA + SMA200 FILTER (like V1.2 has)
# =============================================================================
print("\n" + "=" * 80)
print("PHASE 3: EMA CROSSOVER + SMA200 TREND FILTER")
print("(LONG only in bull, SHORT only in bear - same as V1.2)")
print("=" * 80)

# SMA200
df['sma_200'] = df['close'].rolling(200).mean()
df['bull_market'] = df['close'] > df['sma_200']
df['bear_market'] = df['close'] < df['sma_200']

test_filtered = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()

filtered_results = []

for fast, slow in EMA_PAIRS:
    bull_col = f'cross_{fast}_{slow}_bull'
    bear_col = f'cross_{fast}_{slow}_bear'

    trades = []
    for idx_pos in range(len(test_filtered)):
        if idx_pos + MAX_BARS + 2 >= len(test_filtered):
            break

        current = test_filtered.iloc[idx_pos]

        # LONG only in bull market
        if current[bull_col] and current['bull_market']:
            trade = execute_trade(test_filtered, idx_pos, 'LONG', LONG_TRAILING_STOP, MAX_BARS)
            if trade:
                trades.append(trade)

        # SHORT only in bear market
        elif current[bear_col] and current['bear_market']:
            trade = execute_trade(test_filtered, idx_pos, 'SHORT', SHORT_TRAILING_STOP, MAX_BARS)
            if trade:
                trades.append(trade)

    if len(trades) == 0:
        filtered_results.append({
            'pair': f'{fast}/{slow}', 'trades': 0, 'win_rate': 0,
            'total_net': 0, 'pf': 0, 'long_n': 0, 'long_net': 0,
            'short_n': 0, 'short_net': 0,
        })
        continue

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df['is_winner']]
    losers = trades_df[~trades_df['is_winner']]
    long_t = trades_df[trades_df['direction'] == 'LONG']
    short_t = trades_df[trades_df['direction'] == 'SHORT']

    gw = winners['net_profit_bps'].sum() if len(winners) > 0 else 0
    gl = abs(losers['net_profit_bps'].sum()) if len(losers) > 0 else 1
    pf = gw / gl

    filtered_results.append({
        'pair': f'{fast}/{slow}',
        'trades': len(trades_df),
        'win_rate': len(winners) / len(trades_df) * 100,
        'total_net': trades_df['net_profit_bps'].sum(),
        'pf': pf,
        'long_n': len(long_t),
        'long_net': long_t['net_profit_bps'].sum() if len(long_t) > 0 else 0,
        'short_n': len(short_t),
        'short_net': short_t['net_profit_bps'].sum() if len(short_t) > 0 else 0,
    })

print(f"\n{'Pair':>8} {'Trades':>7} {'Win%':>7} {'Net bps':>10} {'PF':>7} "
      f"{'L_N':>5} {'L_Net':>8} {'S_N':>5} {'S_Net':>8}")
print("-" * 75)

for r in filtered_results:
    if r['trades'] == 0:
        print(f"{r['pair']:>8} {'NO TRADES':>7}")
        continue
    print(f"{r['pair']:>8} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['total_net']:>+9.0f} "
          f"{r['pf']:>7.2f} {r['long_n']:>5} {r['long_net']:>+7.0f} "
          f"{r['short_n']:>5} {r['short_net']:>+7.0f}")


# =============================================================================
# PHASE 4: COMPARE BEST EMA CROSSOVER vs V1.2
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON vs V1.2")
print("=" * 80)
print(f"V1.2 (RSI): 202 trades | 56.9% win | +3,250 bps | PF 2.47")
print()

# Best from unfiltered
if backtest_results:
    best_unf = max(backtest_results, key=lambda x: x['total_net'])
    print(f"Best EMA (unfiltered):  {best_unf['pair']} | {best_unf['trades']} trades | "
          f"{best_unf['win_rate']:.1f}% win | {best_unf['total_net']:+.0f} bps | PF {best_unf['pf']:.2f}")

# Best from filtered
if filtered_results:
    valid_filtered = [r for r in filtered_results if r['trades'] > 0]
    if valid_filtered:
        best_flt = max(valid_filtered, key=lambda x: x['total_net'])
        print(f"Best EMA (+SMA200):     {best_flt['pair']} | {best_flt['trades']} trades | "
              f"{best_flt['win_rate']:.1f}% win | {best_flt['total_net']:+.0f} bps | PF {best_flt['pf']:.2f}")


# =============================================================================
# PHASE 5: YEARLY BREAKDOWN FOR BEST CONFIG
# =============================================================================
print("\n" + "=" * 80)
print("YEARLY BREAKDOWN - BEST CONFIGS")
print("=" * 80)

# Run yearly for best unfiltered pair
if backtest_results:
    best_pair = best_unf['pair']
    fast, slow = [int(x) for x in best_pair.split('/')]
    bull_col = f'cross_{fast}_{slow}_bull'
    bear_col = f'cross_{fast}_{slow}_bear'

    print(f"\nBest Unfiltered: {best_pair}")
    for year in [2024, 2025]:
        year_data = test_filtered[test_filtered.index.year == year]
        trades = []
        for idx_pos in range(len(year_data)):
            if idx_pos + MAX_BARS + 2 >= len(year_data):
                break
            current = year_data.iloc[idx_pos]
            if current[bull_col]:
                trade = execute_trade(year_data, idx_pos, 'LONG', LONG_TRAILING_STOP, MAX_BARS)
                if trade:
                    trades.append(trade)
            elif current[bear_col]:
                trade = execute_trade(year_data, idx_pos, 'SHORT', SHORT_TRAILING_STOP, MAX_BARS)
                if trade:
                    trades.append(trade)

        if trades:
            tdf = pd.DataFrame(trades)
            w = tdf[tdf['is_winner']]
            l = tdf[~tdf['is_winner']]
            gw = w['net_profit_bps'].sum() if len(w) > 0 else 0
            gl = abs(l['net_profit_bps'].sum()) if len(l) > 0 else 1
            print(f"  {year}: {len(tdf)} trades | Win: {len(w)/len(tdf)*100:.1f}% | "
                  f"Net: {tdf['net_profit_bps'].sum():+.0f} bps | PF: {gw/gl:.2f}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
