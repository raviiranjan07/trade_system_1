"""
EMA7 Bounce Strategy Backtest - Theoretical Performance

Strategy:
- Entry: Price touches EMA7 (within 0.2%)
- Direction: EMA50 > EMA200 = LONG, else SHORT
- Take profit: 30 bps
- Stop loss: None
- Max holding: H=10 bars (2.5 hours)
- Fees: 8 bps per trade

Tracks theoretical performance in bps, then applies to $100.

Run: python scripts/debug/ema_bounce_backtest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EMA7 BOUNCE STRATEGY BACKTEST - 2024-2025")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_PERIOD = 7
TOUCH_THRESHOLD_MAX = 0.2  # 0.2% from EMA
TAKE_PROFIT_BPS = 12  # bps
STOP_LOSS_BPS = 999999  # NO STOP LOSS (very high value)
MAX_HOLDING_BARS = 10  # H=10 = 2.5 hours
FEES_BPS = 8  # Round-trip fees
TEST_START = "2024-01-01"

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/5] Loading data...")
ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)

# Resample to 15-minute
ohlcv = ohlcv_1m.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Filter test period only
test = ohlcv[ohlcv.index >= TEST_START].copy()
print(f"Test data: {len(test):,} candles ({test.index[0]} to {test.index[-1]})")

# =============================================================================
# CALCULATE INDICATORS
# =============================================================================
print("\n[2/5] Calculating indicators...")
test['ema7'] = test['close'].ewm(span=7, adjust=False).mean()

# Distance from EMA7
test['dist_from_ema7'] = abs(test['close'] - test['ema7']) / test['ema7'] * 100

# Touch event
test['touch'] = test['dist_from_ema7'] <= TOUCH_THRESHOLD_MAX

# CORRECT LOGIC: Check approach direction (where price came from)
LOOKBACK = 1
test['was_above_ema'] = test['close'].shift(LOOKBACK) > test['ema7'].shift(LOOKBACK)
test['was_below_ema'] = test['close'].shift(LOOKBACK) < test['ema7'].shift(LOOKBACK)

# Determine entry direction based on approach
test['entry_direction'] = 0  # 0 = no entry
test.loc[test['touch'] & test['was_above_ema'], 'entry_direction'] = 1   # LONG
test.loc[test['touch'] & test['was_below_ema'], 'entry_direction'] = -1  # SHORT

touch_count = (test['entry_direction'] != 0).sum()
long_count = (test['entry_direction'] == 1).sum()
short_count = (test['entry_direction'] == -1).sum()
print(f"EMA7 entries detected: {touch_count:,} ({long_count:,} LONG, {short_count:,} SHORT)")

# =============================================================================
# SIMULATE TRADES
# =============================================================================
print("\n[3/5] Simulating trades...")

@njit
def simulate_trade(entry_price, direction, future_highs, future_lows, future_opens, tp_bps, sl_bps, max_bars):
    """
    Simulate a single trade with REALISTIC within-bar order.

    When both TP and SL are hit in the same bar, determines which was hit first
    based on distance from open price (more realistic than always checking TP first).

    Returns: (exit_price, bars_held, exit_reason)
    exit_reason: 'TP' (take profit), 'SL' (stop loss), 'TIME' (max holding time)
    """
    for i in range(min(len(future_highs), max_bars)):
        high = future_highs[i]
        low = future_lows[i]
        open_price = future_opens[i] if i < len(future_opens) else entry_price

        if direction == 1:  # LONG
            tp_price = entry_price * (1 + tp_bps / 10000)
            sl_price = entry_price * (1 - sl_bps / 10000)

            # Check if TP and/or SL were hit
            tp_hit = high >= tp_price
            sl_hit = low <= sl_price

            if tp_hit and sl_hit:
                # Both hit - determine which was closer to open (hit first)
                dist_to_tp = abs(tp_price - open_price)
                dist_to_sl = abs(sl_price - open_price)

                if dist_to_tp < dist_to_sl:
                    exit_price = tp_price
                    return exit_price, i + 1, 1  # TP
                else:
                    exit_price = sl_price
                    return exit_price, i + 1, 3  # SL
            elif tp_hit:
                exit_price = tp_price
                return exit_price, i + 1, 1  # TP
            elif sl_hit:
                exit_price = sl_price
                return exit_price, i + 1, 3  # SL

        else:  # SHORT
            tp_price = entry_price * (1 - tp_bps / 10000)
            sl_price = entry_price * (1 + sl_bps / 10000)

            # Check if TP and/or SL were hit
            tp_hit = low <= tp_price
            sl_hit = high >= sl_price

            if tp_hit and sl_hit:
                # Both hit - determine which was closer to open (hit first)
                dist_to_tp = abs(tp_price - open_price)
                dist_to_sl = abs(sl_price - open_price)

                if dist_to_tp < dist_to_sl:
                    exit_price = tp_price
                    return exit_price, i + 1, 1  # TP
                else:
                    exit_price = sl_price
                    return exit_price, i + 1, 3  # SL
            elif tp_hit:
                exit_price = tp_price
                return exit_price, i + 1, 1  # TP
            elif sl_hit:
                exit_price = sl_price
                return exit_price, i + 1, 3  # SL

    # Max holding time reached - exit at last close
    return 0.0, max_bars, 2  # 2 = TIME, 0.0 = placeholder (will use actual close)

# Find all entry events (touch + valid direction)
entry_indices = test[test['entry_direction'] != 0].index.tolist()
print(f"Processing {len(entry_indices):,} entry events...")

trades = []
test_array = test.reset_index()

for idx, entry_time in enumerate(entry_indices):
    if idx % 5000 == 0 and idx > 0:
        print(f"  Processed {idx:,} / {len(entry_indices):,} trades...")

    # Get entry bar
    entry_idx = test.index.get_loc(entry_time)

    # Check if we have enough future bars
    if entry_idx + MAX_HOLDING_BARS >= len(test):
        continue

    entry_row = test.iloc[entry_idx]
    entry_price = entry_row['close']
    direction = entry_row['entry_direction']  # CORRECTED: Use approach direction

    # Get future data
    future_data = test.iloc[entry_idx + 1 : entry_idx + 1 + MAX_HOLDING_BARS]
    future_highs = future_data['high'].values
    future_lows = future_data['low'].values
    future_opens = future_data['open'].values
    future_closes = future_data['close'].values

    # Simulate trade
    exit_price, bars_held, exit_reason = simulate_trade(
        entry_price,
        direction,
        future_highs,
        future_lows,
        future_opens,
        TAKE_PROFIT_BPS,
        STOP_LOSS_BPS,
        MAX_HOLDING_BARS
    )

    # If TIME exit, use actual close price
    if exit_reason == 2:
        exit_price = future_closes[-1] if len(future_closes) > 0 else entry_price

    # Calculate P&L in bps
    if direction == 1:  # LONG
        pnl_bps = (exit_price - entry_price) / entry_price * 10000
    else:  # SHORT
        pnl_bps = (entry_price - exit_price) / entry_price * 10000

    # Subtract fees
    pnl_bps -= FEES_BPS

    # Record trade
    exit_reason_str = 'TP' if exit_reason == 1 else ('TIME' if exit_reason == 2 else 'SL')
    trades.append({
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'direction': 'LONG' if direction == 1 else 'SHORT',
        'bars_held': bars_held,
        'exit_reason': exit_reason_str,
        'pnl_bps': pnl_bps,
        'winner': pnl_bps > 0
    })

trades_df = pd.DataFrame(trades)
print(f"\nTotal trades executed: {len(trades_df):,}")

# =============================================================================
# CALCULATE STATISTICS
# =============================================================================
print("\n[4/5] Calculating statistics...")

total_trades = len(trades_df)
winners = trades_df[trades_df['winner']]
losers = trades_df[~trades_df['winner']]

win_rate = len(winners) / total_trades * 100
total_pnl_bps = trades_df['pnl_bps'].sum()
avg_pnl_bps = trades_df['pnl_bps'].mean()

avg_win_bps = winners['pnl_bps'].mean() if len(winners) > 0 else 0
avg_loss_bps = losers['pnl_bps'].mean() if len(losers) > 0 else 0

max_win_bps = trades_df['pnl_bps'].max()
max_loss_bps = trades_df['pnl_bps'].min()

# Cumulative P&L
trades_df['cumulative_pnl_bps'] = trades_df['pnl_bps'].cumsum()
max_cumulative = trades_df['cumulative_pnl_bps'].cummax()
drawdown_bps = (trades_df['cumulative_pnl_bps'] - max_cumulative)
max_drawdown_bps = drawdown_bps.min()

# Exit reason breakdown
tp_exits = (trades_df['exit_reason'] == 'TP').sum()
sl_exits = (trades_df['exit_reason'] == 'SL').sum()
time_exits = (trades_df['exit_reason'] == 'TIME').sum()

# Direction breakdown
long_trades = trades_df[trades_df['direction'] == 'LONG']
short_trades = trades_df[trades_df['direction'] == 'SHORT']

long_win_rate = (long_trades['winner'].sum() / len(long_trades) * 100) if len(long_trades) > 0 else 0
short_win_rate = (short_trades['winner'].sum() / len(short_trades) * 100) if len(short_trades) > 0 else 0

# =============================================================================
# APPLY TO $100 CAPITAL
# =============================================================================
print("\n[5/5] Applying to $100 capital...")

# Convert total bps to percentage
total_return_pct = total_pnl_bps / 100  # 100 bps = 1%

# Calculate final balance
starting_capital = 100
final_balance = starting_capital * (1 + total_return_pct / 100)
profit_loss = final_balance - starting_capital

# =============================================================================
# SAVE RESULTS
# =============================================================================
output_dir = Path("experiments/ema")
output_dir.mkdir(parents=True, exist_ok=True)

trades_path = output_dir / "ema7_backtest_trades.csv"
trades_df.to_csv(trades_path, index=False)

# =============================================================================
# DISPLAY RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("BACKTEST RESULTS - EMA7 BOUNCE STRATEGY")
print("=" * 80)

print("\n--- STRATEGY PARAMETERS ---")
print(f"Entry: EMA7 touch (within {TOUCH_THRESHOLD_MAX}%)")
print(f"Direction: APPROACH DIRECTION (from above = LONG, from below = SHORT)")
print(f"Take profit: {TAKE_PROFIT_BPS} bps")
print(f"Stop loss: {STOP_LOSS_BPS} bps")
print(f"Max holding: {MAX_HOLDING_BARS} bars (2.5 hours)")
print(f"Fees: {FEES_BPS} bps per trade")
print(f"Within-bar order: REALISTIC (uses open price)")

print("\n--- TRADE STATISTICS ---")
print(f"Total trades: {total_trades:,}")
print(f"Winners: {len(winners):,} ({win_rate:.2f}%)")
print(f"Losers: {len(losers):,} ({100-win_rate:.2f}%)")
print(f"Long trades: {len(long_trades):,} ({long_win_rate:.2f}% win rate)")
print(f"Short trades: {len(short_trades):,} ({short_win_rate:.2f}% win rate)")

print("\n--- EXIT REASONS ---")
print(f"Take profit exits: {tp_exits:,} ({tp_exits/total_trades*100:.1f}%)")
print(f"Stop loss exits: {sl_exits:,} ({sl_exits/total_trades*100:.1f}%)")
print(f"Time exits: {time_exits:,} ({time_exits/total_trades*100:.1f}%)")

print("\n--- PERFORMANCE (BPS) ---")
print(f"Total P&L: {total_pnl_bps:,.0f} bps")
print(f"Average trade: {avg_pnl_bps:.2f} bps")
print(f"Average winner: {avg_win_bps:.2f} bps")
print(f"Average loser: {avg_loss_bps:.2f} bps")
print(f"Best trade: {max_win_bps:.2f} bps")
print(f"Worst trade: {max_loss_bps:.2f} bps")
print(f"Max drawdown: {max_drawdown_bps:.0f} bps")

print("\n--- $100 CAPITAL PERFORMANCE ---")
print(f"Starting capital: ${starting_capital:.2f}")
print(f"Final balance: ${final_balance:.2f}")
print(f"Profit/Loss: ${profit_loss:.2f} ({total_return_pct:+.2f}%)")

print("\n--- METRICS ---")
if avg_loss_bps != 0:
    profit_factor = abs(winners['pnl_bps'].sum() / losers['pnl_bps'].sum()) if len(losers) > 0 else float('inf')
    print(f"Profit factor: {profit_factor:.2f}")

if avg_loss_bps != 0:
    risk_reward = abs(avg_win_bps / avg_loss_bps)
    print(f"Risk/Reward ratio: {risk_reward:.2f}")

print(f"\nTrades saved to: {trades_path}")

print("\n" + "=" * 80)
print("BACKTEST COMPLETE!")
print("=" * 80)
