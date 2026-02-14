"""
TWO-STAGE EXIT Strategy (Option 3)

Stage 1: Wait through drawdown for WAIT_BARS
Stage 2: After WAIT_BARS, check MAE:
  - If MAE < MAE_THRESHOLD: Keep waiting (likely recovers)
  - If MAE > MAE_THRESHOLD: Exit (wrong direction)

This aligns with WHAT phase findings:
- 84-90% positions eventually recover
- Need to wait through "dirty wins"
- Only exit if clearly wrong direction (high MAE)

Entry: Cross above/below EMA7
TP: 12 bps
Exit: Two-stage logic

Run: python scripts/debug/two_stage_exit_backtest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TWO-STAGE EXIT STRATEGY BACKTEST")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
TP_BPS = 12
WAIT_BARS = 20  # Stage 1: Wait this many bars before checking MAE
MAE_THRESHOLD_BPS = 50  # Stage 2: Exit if MAE > this after WAIT_BARS
MAX_HOLDING_BARS = 40  # Maximum time (give more room for recovery)
FEES_BPS = 8
TEST_START = "2024-01-01"

print(f"\nParameters:")
print(f"  TP Target: {TP_BPS} bps")
print(f"  Stage 1 - Wait: {WAIT_BARS} bars ({WAIT_BARS * 15} min = {WAIT_BARS * 15 / 60:.1f} hours)")
print(f"  Stage 2 - MAE Threshold: {MAE_THRESHOLD_BPS} bps")
print(f"  Max holding: {MAX_HOLDING_BARS} bars ({MAX_HOLDING_BARS * 15} min = {MAX_HOLDING_BARS * 15 / 60:.1f} hours)")

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
# SIMULATE TRADES WITH TWO-STAGE EXIT
# =============================================================================
print("\n[3/5] Simulating trades with two-stage exit...")

@njit
def simulate_two_stage_exit(entry_price, direction, future_highs, future_lows,
                             tp_bps, wait_bars, mae_threshold_bps, max_bars):
    """
    Two-stage exit logic:

    Stage 1 (bars 0 to wait_bars):
      - Check for TP only
      - Track MAE but don't exit

    Stage 2 (bars wait_bars+1 to max_bars):
      - Check for TP
      - If MAE > mae_threshold_bps: Exit

    Returns:
        exit_price: final exit price
        bars_held: bars held
        exit_reason: 1=TP, 2=MAE, 3=TIME
        mae_bps: maximum adverse excursion
    """
    mae_bps = 0.0

    for i in range(min(len(future_highs), max_bars)):
        high = future_highs[i]
        low = future_lows[i]

        # Update MAE
        if direction == 1:  # LONG
            current_mae = (entry_price - low) / entry_price * 10000
        else:  # SHORT
            current_mae = (high - entry_price) / entry_price * 10000

        if current_mae > mae_bps:
            mae_bps = current_mae

        # Check TP (always active)
        if direction == 1:  # LONG
            tp_price = entry_price * (1 + tp_bps / 10000)
            if high >= tp_price:
                return tp_price, i + 1, 1, mae_bps  # TP
        else:  # SHORT
            tp_price = entry_price * (1 - tp_bps / 10000)
            if low <= tp_price:
                return tp_price, i + 1, 1, mae_bps  # TP

        # Stage 2: After WAIT_BARS, check MAE threshold
        if i >= wait_bars:
            if mae_bps > mae_threshold_bps:
                # MAE too high - exit at current close
                exit_price = (high + low) / 2
                return exit_price, i + 1, 2, mae_bps  # MAE exit

    # TIME EXIT - max bars reached
    final_close = (future_highs[-1] + future_lows[-1]) / 2 if len(future_highs) > 0 else entry_price
    return final_close, max_bars, 3, mae_bps  # TIME

trades = []

for idx in range(len(test)):
    if idx + MAX_HOLDING_BARS >= len(test):
        continue

    row = test.iloc[idx]
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
    future_data = test.iloc[idx + 1 : idx + 1 + MAX_HOLDING_BARS]
    future_highs = future_data['high'].values
    future_lows = future_data['low'].values

    # Simulate trade
    exit_price, bars_held, exit_reason, mae_bps = simulate_two_stage_exit(
        entry_price,
        direction,
        future_highs,
        future_lows,
        TP_BPS,
        WAIT_BARS,
        MAE_THRESHOLD_BPS,
        MAX_HOLDING_BARS
    )

    # Calculate P&L
    if direction == 1:  # LONG
        pnl_bps = (exit_price - entry_price) / entry_price * 10000
    else:  # SHORT
        pnl_bps = (entry_price - exit_price) / entry_price * 10000

    # Subtract fees
    pnl_bps -= FEES_BPS

    # Record trade
    exit_reason_map = {1: 'TP', 2: 'MAE', 3: 'TIME'}
    exit_reason_str = exit_reason_map.get(exit_reason, 'UNKNOWN')

    trades.append({
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'direction': 'LONG' if direction == 1 else 'SHORT',
        'bars_held': bars_held,
        'exit_reason': exit_reason_str,
        'mae_bps': mae_bps,
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

# MAE stats
avg_mae = trades_df['mae_bps'].mean()
median_mae = trades_df['mae_bps'].median()

# Exit reason breakdown
tp_exits = (trades_df['exit_reason'] == 'TP').sum()
mae_exits = (trades_df['exit_reason'] == 'MAE').sum()
time_exits = (trades_df['exit_reason'] == 'TIME').sum()

# Average holding time by exit reason
avg_bars_tp = trades_df[trades_df['exit_reason'] == 'TP']['bars_held'].mean()
avg_bars_mae = trades_df[trades_df['exit_reason'] == 'MAE']['bars_held'].mean() if mae_exits > 0 else 0
avg_bars_time = trades_df[trades_df['exit_reason'] == 'TIME']['bars_held'].mean() if time_exits > 0 else 0

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

trades_path = output_dir / "two_stage_exit_trades.csv"
trades_df.to_csv(trades_path, index=False)

# =============================================================================
# DISPLAY RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("BACKTEST RESULTS - TWO-STAGE EXIT STRATEGY")
print("=" * 80)

print("\n--- STRATEGY PARAMETERS ---")
print(f"Entry: Cross above/below EMA7")
print(f"Take profit: {TP_BPS} bps")
print(f"Stage 1 (Wait): {WAIT_BARS} bars ({WAIT_BARS * 15 / 60:.1f} hours)")
print(f"Stage 2 (MAE threshold): {MAE_THRESHOLD_BPS} bps")
print(f"Max holding: {MAX_HOLDING_BARS} bars ({MAX_HOLDING_BARS * 15 / 60:.1f} hours)")
print(f"Fees: {FEES_BPS} bps per trade")

print("\n--- TRADE STATISTICS ---")
print(f"Total trades: {total_trades:,}")
print(f"Winners: {len(winners):,} ({win_rate:.2f}%)")
print(f"Losers: {len(losers):,} ({100-win_rate:.2f}%)")

print("\n--- EXIT REASONS ---")
print(f"TP exits: {tp_exits:,} ({tp_exits/total_trades*100:.1f}%) - avg {avg_bars_tp:.1f} bars")
print(f"MAE exits: {mae_exits:,} ({mae_exits/total_trades*100:.1f}%) - avg {avg_bars_mae:.1f} bars")
print(f"TIME exits: {time_exits:,} ({time_exits/total_trades*100:.1f}%) - avg {avg_bars_time:.1f} bars")

print("\n--- MAE STATISTICS ---")
print(f"Average MAE: {avg_mae:.2f} bps")
print(f"Median MAE: {median_mae:.2f} bps")

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
if len(losers) > 0 and losers['pnl_bps'].sum() != 0:
    profit_factor = abs(winners['pnl_bps'].sum() / losers['pnl_bps'].sum())
    print(f"Profit factor: {profit_factor:.2f}")

if avg_loss_bps != 0:
    risk_reward = abs(avg_win_bps / avg_loss_bps)
    print(f"Risk/Reward ratio: {risk_reward:.2f}")

print(f"\nTrades saved to: {trades_path}")

print("\n" + "=" * 80)
print("BACKTEST COMPLETE!")
print("=" * 80)
