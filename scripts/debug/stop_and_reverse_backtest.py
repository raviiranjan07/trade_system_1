"""
STOP AND REVERSE Strategy

Entry: Cross above/below EMA7
TP: 12 bps
SL: 12 bps
If SL hit → Reverse position and continue

Example:
- Enter LONG at $100
- SL hit at $98.80 (-12 bps) → EXIT LONG, ENTER SHORT at $98.80
- SHORT TP at $97.62 (+12 bps from $98.80)
- Or SHORT SL at $99.98 → Reverse back to LONG
- Continue until TP or max time

Run: python scripts/debug/stop_and_reverse_backtest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STOP AND REVERSE STRATEGY BACKTEST")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
TP_BPS = 12
SL_BPS = 12
MAX_HOLDING_BARS = 10
FEES_BPS = 8
TEST_START = "2024-01-01"

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/5] Loading data...")
ohlcv_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)

test = ohlcv[ohlcv.index >= TEST_START].copy()
print(f"Test data: {len(test):,} candles ({test.index[0]} to {test.index[-1]})")

# =============================================================================
# CALCULATE INDICATORS
# =============================================================================
print("\n[2/5] Calculating indicators...")
test['ema7'] = test['close'].ewm(span=7, adjust=False).mean()

test['above_ema'] = test['close'] > test['ema7']
test['below_ema'] = test['close'] < test['ema7']

test['cross_above'] = (test['above_ema'] == True) & (test['above_ema'].shift(1) == False)
test['cross_below'] = (test['below_ema'] == True) & (test['below_ema'].shift(1) == False)

cross_above_count = test['cross_above'].sum()
cross_below_count = test['cross_below'].sum()
print(f"Cross above: {cross_above_count:,}")
print(f"Cross below: {cross_below_count:,}")
print(f"Total signals: {cross_above_count + cross_below_count:,}")

# =============================================================================
# SIMULATE TRADES WITH REVERSALS
# =============================================================================
print("\n[3/5] Simulating trades with stop-and-reverse...")

@njit
def simulate_stop_and_reverse(entry_price, initial_direction, future_highs, future_lows,
                               tp_bps, sl_bps, max_bars):
    """
    Simulate trade with stop and reverse logic

    Returns:
        exit_price: final exit price
        bars_held: total bars held
        exit_reason: 1=TP, 2=TIME
        num_reversals: number of times position reversed
        total_pnl_bps: cumulative P&L from all positions
    """
    current_price = entry_price
    direction = initial_direction
    bars_held = 0
    num_reversals = 0
    total_pnl_bps = 0.0

    for i in range(min(len(future_highs), max_bars)):
        high = future_highs[i]
        low = future_lows[i]
        bars_held += 1

        if direction == 1:  # LONG
            tp_price = current_price * (1 + tp_bps / 10000)
            sl_price = current_price * (1 - sl_bps / 10000)

            # Check TP first
            if high >= tp_price:
                # TP HIT - EXIT with profit
                pnl = tp_bps
                total_pnl_bps += pnl
                return tp_price, bars_held, 1, num_reversals, total_pnl_bps

            # Check SL
            if low <= sl_price:
                # SL HIT - Record loss and REVERSE to SHORT
                pnl = -sl_bps
                total_pnl_bps += pnl
                num_reversals += 1
                current_price = sl_price
                direction = -1  # Flip to SHORT
                continue

        else:  # SHORT
            tp_price = current_price * (1 - tp_bps / 10000)
            sl_price = current_price * (1 + sl_bps / 10000)

            # Check TP first
            if low <= tp_price:
                # TP HIT - EXIT with profit
                pnl = tp_bps
                total_pnl_bps += pnl
                return tp_price, bars_held, 1, num_reversals, total_pnl_bps

            # Check SL
            if high >= sl_price:
                # SL HIT - Record loss and REVERSE to LONG
                pnl = -sl_bps
                total_pnl_bps += pnl
                num_reversals += 1
                current_price = sl_price
                direction = 1  # Flip to LONG
                continue

    # TIME EXIT - close at last close price
    # Calculate final position P&L
    if i < len(future_lows):
        final_close = (future_highs[i] + future_lows[i]) / 2  # Approximation
    else:
        final_close = current_price

    if direction == 1:  # LONG at time exit
        final_pnl = (final_close - current_price) / current_price * 10000
    else:  # SHORT at time exit
        final_pnl = (current_price - final_close) / current_price * 10000

    total_pnl_bps += final_pnl

    return final_close, max_bars, 2, num_reversals, total_pnl_bps

trades = []

for idx in range(len(test)):
    if idx + MAX_HOLDING_BARS >= len(test):
        continue

    row = test.iloc[idx]
    entry_time = row.name
    entry_price = row['close']

    # Determine initial direction
    if row['cross_above']:
        direction = 1  # LONG
    elif row['cross_below']:
        direction = -1  # SHORT
    else:
        continue

    # Get future data
    future_data = test.iloc[idx + 1 : idx + 1 + MAX_HOLDING_BARS]
    future_highs = future_data['high'].values
    future_lows = future_data['low'].values

    # Simulate trade with reversals
    exit_price, bars_held, exit_reason, num_reversals, gross_pnl_bps = simulate_stop_and_reverse(
        entry_price,
        direction,
        future_highs,
        future_lows,
        TP_BPS,
        SL_BPS,
        MAX_HOLDING_BARS
    )

    # Calculate net P&L (subtract fees for EACH position opened)
    # FEES_BPS = 8 is ROUND-TRIP cost (entry + exit)
    # Each reversal adds one more round-trip
    num_positions = 1 + num_reversals  # Initial + reversals
    total_fees = FEES_BPS * num_positions  # Each position costs 8 bps round-trip
    net_pnl_bps = gross_pnl_bps - total_fees

    # Record trade
    exit_reason_str = 'TP' if exit_reason == 1 else 'TIME'
    trades.append({
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'initial_direction': 'LONG' if direction == 1 else 'SHORT',
        'bars_held': bars_held,
        'num_reversals': num_reversals,
        'exit_reason': exit_reason_str,
        'gross_pnl_bps': gross_pnl_bps,
        'total_fees_bps': total_fees,
        'net_pnl_bps': net_pnl_bps,
        'winner': net_pnl_bps > 0
    })

trades_df = pd.DataFrame(trades)
print(f"Total trades executed: {len(trades_df):,}")

# =============================================================================
# CALCULATE STATISTICS
# =============================================================================
print("\n[4/5] Calculating statistics...")

total_trades = len(trades_df)
winners = trades_df[trades_df['winner']]
losers = trades_df[~trades_df['winner']]

win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
total_pnl_bps = trades_df['net_pnl_bps'].sum()
avg_pnl_bps = trades_df['net_pnl_bps'].mean()

avg_win_bps = winners['net_pnl_bps'].mean() if len(winners) > 0 else 0
avg_loss_bps = losers['net_pnl_bps'].mean() if len(losers) > 0 else 0

max_win_bps = trades_df['net_pnl_bps'].max()
max_loss_bps = trades_df['net_pnl_bps'].min()

# Reversal stats
avg_reversals = trades_df['num_reversals'].mean()
max_reversals = trades_df['num_reversals'].max()
no_reversals = (trades_df['num_reversals'] == 0).sum()
with_reversals = (trades_df['num_reversals'] > 0).sum()

# Exit reason breakdown
tp_exits = (trades_df['exit_reason'] == 'TP').sum()
time_exits = (trades_df['exit_reason'] == 'TIME').sum()

# =============================================================================
# APPLY TO $100 CAPITAL
# =============================================================================
print("\n[5/5] Applying to $100 capital...")

total_return_pct = total_pnl_bps / 100
starting_capital = 100
final_balance = starting_capital * (1 + total_return_pct / 100)
profit_loss = final_balance - starting_capital

# =============================================================================
# SAVE RESULTS
# =============================================================================
output_dir = Path("experiments/ema")
output_dir.mkdir(parents=True, exist_ok=True)

trades_path = output_dir / "stop_and_reverse_trades.csv"
trades_df.to_csv(trades_path, index=False)

# =============================================================================
# DISPLAY RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("BACKTEST RESULTS - STOP AND REVERSE STRATEGY")
print("=" * 80)

print("\n--- STRATEGY PARAMETERS ---")
print(f"Entry: Cross above/below EMA7")
print(f"Take profit: {TP_BPS} bps")
print(f"Stop loss: {SL_BPS} bps (REVERSE position)")
print(f"Max holding: {MAX_HOLDING_BARS} bars (2.5 hours)")
print(f"Fees: {FEES_BPS} bps per position entry/exit")

print("\n--- TRADE STATISTICS ---")
print(f"Total trades: {total_trades:,}")
print(f"Winners: {len(winners):,} ({win_rate:.2f}%)")
print(f"Losers: {len(losers):,} ({100-win_rate:.2f}%)")

print("\n--- REVERSAL STATISTICS ---")
print(f"Trades with NO reversals: {no_reversals:,} ({no_reversals/total_trades*100:.1f}%)")
print(f"Trades WITH reversals: {with_reversals:,} ({with_reversals/total_trades*100:.1f}%)")
print(f"Average reversals per trade: {avg_reversals:.2f}")
print(f"Max reversals in single trade: {int(max_reversals)}")

print("\n--- EXIT REASONS ---")
print(f"Take profit exits: {tp_exits:,} ({tp_exits/total_trades*100:.1f}%)")
print(f"Time exits: {time_exits:,} ({time_exits/total_trades*100:.1f}%)")

print("\n--- PERFORMANCE (BPS) ---")
print(f"Total P&L: {total_pnl_bps:,.0f} bps")
print(f"Average trade: {avg_pnl_bps:.2f} bps")
print(f"Average winner: {avg_win_bps:.2f} bps")
print(f"Average loser: {avg_loss_bps:.2f} bps")
print(f"Best trade: {max_win_bps:.2f} bps")
print(f"Worst trade: {max_loss_bps:.2f} bps")

print("\n--- $100 CAPITAL PERFORMANCE ---")
print(f"Starting capital: ${starting_capital:.2f}")
print(f"Final balance: ${final_balance:.2f}")
print(f"Profit/Loss: ${profit_loss:.2f} ({total_return_pct:+.2f}%)")

print("\n--- METRICS ---")
if len(losers) > 0 and losers['net_pnl_bps'].sum() != 0:
    profit_factor = abs(winners['net_pnl_bps'].sum() / losers['net_pnl_bps'].sum())
    print(f"Profit factor: {profit_factor:.2f}")

if avg_loss_bps != 0:
    risk_reward = abs(avg_win_bps / avg_loss_bps)
    print(f"Risk/Reward ratio: {risk_reward:.2f}")

print(f"\nTrades saved to: {trades_path}")

print("\n" + "=" * 80)
print("BACKTEST COMPLETE!")
print("=" * 80)
