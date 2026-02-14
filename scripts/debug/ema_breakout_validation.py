"""
EMA7 BREAKOUT Strategy VALIDATION (Train Data)

Validate results on 2020-2023 train data
Compare with test results from 2024-2025

Entry: Price crosses above/below EMA7
Exit: 12 bps profit target
Fees: 8 bps

Run: python scripts/debug/ema_breakout_validation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EMA7 BREAKOUT STRATEGY VALIDATION (TRAIN DATA)")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
TAKE_PROFIT_BPS = 12
MAX_HOLDING_BARS = 10  # Max 2.5 hours
FEES_BPS = 8
TRAIN_START = "2020-01-01"
TRAIN_END = "2024-01-01"

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/5] Loading data...")
ohlcv_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)

# Filter TRAIN period
train = ohlcv[(ohlcv.index >= TRAIN_START) & (ohlcv.index < TRAIN_END)].copy()
print(f"Train data: {len(train):,} candles ({train.index[0]} to {train.index[-1]})")

# =============================================================================
# CALCULATE INDICATORS
# =============================================================================
print("\n[2/5] Calculating indicators...")
train['ema7'] = train['close'].ewm(span=7, adjust=False).mean()

# Detect crosses
train['above_ema'] = train['close'] > train['ema7']
train['below_ema'] = train['close'] < train['ema7']

# Cross detection
train['cross_above'] = (train['above_ema'] == True) & (train['above_ema'].shift(1) == False)
train['cross_below'] = (train['below_ema'] == True) & (train['below_ema'].shift(1) == False)

cross_above_count = train['cross_above'].sum()
cross_below_count = train['cross_below'].sum()
print(f"Cross above: {cross_above_count:,}")
print(f"Cross below: {cross_below_count:,}")
print(f"Total signals: {cross_above_count + cross_below_count:,}")

# =============================================================================
# SIMULATE TRADES
# =============================================================================
print("\n[3/5] Simulating trades...")

@njit
def simulate_breakout_trade(entry_price, direction, future_highs, future_lows, tp_bps, max_bars):
    """
    Simulate breakout trade
    Returns: (exit_price, bars_held, exit_reason)
    exit_reason: 1=TP, 2=TIME
    """
    for i in range(min(len(future_highs), max_bars)):
        high = future_highs[i]
        low = future_lows[i]

        if direction == 1:  # LONG
            tp_price = entry_price * (1 + tp_bps / 10000)
            if high >= tp_price:
                return tp_price, i + 1, 1  # TP
        else:  # SHORT
            tp_price = entry_price * (1 - tp_bps / 10000)
            if low <= tp_price:
                return tp_price, i + 1, 1  # TP

    # Max holding time - exit at close
    return 0.0, max_bars, 2  # TIME

trades = []

for idx in range(len(train)):
    if idx + MAX_HOLDING_BARS >= len(train):
        continue

    row = train.iloc[idx]
    entry_time = row.name
    entry_price = row['close']

    # Determine direction
    if row['cross_above']:
        direction = 1  # LONG
    elif row['cross_below']:
        direction = -1  # SHORT
    else:
        continue

    # Get future data
    future_data = train.iloc[idx + 1 : idx + 1 + MAX_HOLDING_BARS]
    future_highs = future_data['high'].values
    future_lows = future_data['low'].values
    future_closes = future_data['close'].values

    # Simulate trade
    exit_price, bars_held, exit_reason = simulate_breakout_trade(
        entry_price,
        direction,
        future_highs,
        future_lows,
        TAKE_PROFIT_BPS,
        MAX_HOLDING_BARS
    )

    # If TIME exit, use actual close
    if exit_reason == 2:
        exit_price = future_closes[-1] if len(future_closes) > 0 else entry_price

    # Calculate P&L
    if direction == 1:  # LONG
        pnl_bps = (exit_price - entry_price) / entry_price * 10000
    else:  # SHORT
        pnl_bps = (entry_price - exit_price) / entry_price * 10000

    # Subtract fees
    pnl_bps -= FEES_BPS

    # Record trade
    exit_reason_str = 'TP' if exit_reason == 1 else 'TIME'
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
print(f"Total trades executed: {len(trades_df):,}")

# =============================================================================
# CALCULATE STATISTICS
# =============================================================================
print("\n[4/5] Calculating statistics...")

total_trades = len(trades_df)
winners = trades_df[trades_df['winner']]
losers = trades_df[~trades_df['winner']]

win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
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

total_return_pct = total_pnl_bps / 100
starting_capital = 100
final_balance = starting_capital * (1 + total_return_pct / 100)
profit_loss = final_balance - starting_capital

# =============================================================================
# SAVE RESULTS
# =============================================================================
output_dir = Path("experiments/ema")
output_dir.mkdir(parents=True, exist_ok=True)

trades_path = output_dir / "ema_breakout_trades_TRAIN.csv"
trades_df.to_csv(trades_path, index=False)

# =============================================================================
# DISPLAY RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("VALIDATION RESULTS - EMA7 BREAKOUT STRATEGY (TRAIN DATA)")
print("=" * 80)

print("\n--- STRATEGY PARAMETERS ---")
print(f"Entry: Cross above/below EMA7")
print(f"Take profit: {TAKE_PROFIT_BPS} bps")
print(f"Max holding: {MAX_HOLDING_BARS} bars (2.5 hours)")
print(f"Fees: {FEES_BPS} bps per trade")

print("\n--- TRADE STATISTICS ---")
print(f"Total trades: {total_trades:,}")
print(f"Winners: {len(winners):,} ({win_rate:.2f}%)")
print(f"Losers: {len(losers):,} ({100-win_rate:.2f}%)")
print(f"Long trades: {len(long_trades):,} ({long_win_rate:.2f}% win rate)")
print(f"Short trades: {len(short_trades):,} ({short_win_rate:.2f}% win rate)")

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
print(f"Max drawdown: {max_drawdown_bps:.0f} bps")

print("\n--- $100 CAPITAL PERFORMANCE ---")
print(f"Starting capital: ${starting_capital:.2f}")
print(f"Final balance: ${final_balance:.2f}")
print(f"Profit/Loss: ${profit_loss:.2f} ({total_return_pct:+.2f}%)")

print("\n--- METRICS ---")
if len(losers) > 0:
    profit_factor = abs(winners['pnl_bps'].sum() / losers['pnl_bps'].sum()) if losers['pnl_bps'].sum() != 0 else float('inf')
    print(f"Profit factor: {profit_factor:.2f}")

if avg_loss_bps != 0:
    risk_reward = abs(avg_win_bps / avg_loss_bps)
    print(f"Risk/Reward ratio: {risk_reward:.2f}")

print(f"\nTrades saved to: {trades_path}")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE!")
print("=" * 80)
