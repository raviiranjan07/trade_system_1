"""
FILTERED ENTRY + TRAILING STOP (WITH ACTIVATION THRESHOLD)

Entry Filter:
- Only enter if cross is >= 5 bps away from EMA (avoid choppy signals)

Exit Logic:
- Wait until gross profit >= 12 bps (= 4 bps net after 8 bps fees)
- THEN activate trailing stop: Exit if price drops 10 bps from peak
- This avoids exiting tiny profits that would be losers after fees

Run: python scripts/debug/filtered_trailing_with_activation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FILTERED ENTRY + TRAILING STOP WITH ACTIVATION THRESHOLD")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
MIN_DISTANCE_BPS = 5  # Skip entry if within 5 bps of EMA
ACTIVATION_BPS = 12  # Activate trailing stop after 12 bps gross profit (4 bps net)
TRAILING_STOP_BPS = 10  # Trail by 10 bps from peak
MAX_HOLDING_BARS = 60  # Safety limit
FEES_BPS = 8
TEST_START = "2024-01-01"

print(f"\nParameters:")
print(f"  Entry filter: >= {MIN_DISTANCE_BPS} bps from EMA")
print(f"  Activation threshold: {ACTIVATION_BPS} bps gross profit ({ACTIVATION_BPS - FEES_BPS} bps net)")
print(f"  Trailing stop: {TRAILING_STOP_BPS} bps from peak (after activation)")
print(f"  Max holding: {MAX_HOLDING_BARS} bars")
print(f"  Fees: {FEES_BPS} bps")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/4] Loading data...")
ohlcv_path = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)

test = ohlcv[ohlcv.index >= TEST_START].copy()
print(f"Test data: {len(test):,} candles ({test.index[0]} to {test.index[-1]})")

# =============================================================================
# CALCULATE INDICATORS
# =============================================================================
print("\n[2/4] Calculating indicators...")
test['ema7'] = test['close'].ewm(span=7, adjust=False).mean()

test['above_ema'] = test['close'] > test['ema7']
test['below_ema'] = test['close'] < test['ema7']

test['cross_above'] = (test['above_ema'] == True) & (test['above_ema'].shift(1) == False)
test['cross_below'] = (test['below_ema'] == True) & (test['below_ema'].shift(1) == False)

# Calculate distance from EMA
test['distance_from_ema_bps'] = abs(test['close'] - test['ema7']) / test['close'] * 10000

# Filter crosses by distance
test['valid_cross_above'] = test['cross_above'] & (test['distance_from_ema_bps'] >= MIN_DISTANCE_BPS)
test['valid_cross_below'] = test['cross_below'] & (test['distance_from_ema_bps'] >= MIN_DISTANCE_BPS)

valid_above = test['valid_cross_above'].sum()
valid_below = test['valid_cross_below'].sum()
skipped_above = test['cross_above'].sum() - valid_above
skipped_below = test['cross_below'].sum() - valid_below

print(f"Cross above: {test['cross_above'].sum():,} (valid: {valid_above:,}, skipped: {skipped_above:,})")
print(f"Cross below: {test['cross_below'].sum():,} (valid: {valid_below:,}, skipped: {skipped_below:,})")
print(f"Total valid signals: {valid_above + valid_below:,}")

# =============================================================================
# SIMULATE TRADES WITH ACTIVATION THRESHOLD
# =============================================================================
print("\n[3/4] Simulating trades with activation threshold...")

@njit
def simulate_trailing_stop_with_activation(entry_price, direction, future_highs, future_lows,
                                           activation_bps, trailing_bps, max_bars):
    """
    Trailing stop logic with activation threshold:
    - First, wait until gross profit >= activation_bps (12 bps = 4 bps net after fees)
    - Then activate trailing stop of trailing_bps from peak
    - Before activation, just track peak but don't exit

    Returns:
        exit_price, bars_held, exit_reason, peak_pnl_bps
        exit_reason: 1=TRAILING_STOP, 2=TIME
    """
    peak_price = entry_price
    peak_pnl_bps = 0.0
    trailing_active = False

    for i in range(min(len(future_highs), max_bars)):
        high = future_highs[i]
        low = future_lows[i]

        if direction == 1:  # LONG
            # Update peak
            if high > peak_price:
                peak_price = high
                peak_pnl_bps = (peak_price - entry_price) / entry_price * 10000

            # Check if we should activate trailing stop
            if not trailing_active and peak_pnl_bps >= activation_bps:
                trailing_active = True

            # Only check trailing stop if activated
            if trailing_active:
                trailing_price = peak_price * (1 - trailing_bps / 10000)
                if low <= trailing_price:
                    return trailing_price, i + 1, 1, peak_pnl_bps

        else:  # SHORT
            # Update best SHORT price (lowest is best for SHORT)
            if low < peak_price:
                peak_price = low
                peak_pnl_bps = (entry_price - peak_price) / entry_price * 10000

            # Check if we should activate trailing stop
            if not trailing_active and peak_pnl_bps >= activation_bps:
                trailing_active = True

            # Only check trailing stop if activated
            if trailing_active:
                trailing_price = peak_price * (1 + trailing_bps / 10000)
                if high >= trailing_price:
                    return trailing_price, i + 1, 1, peak_pnl_bps

    # TIME EXIT
    final_close = (future_highs[-1] + future_lows[-1]) / 2 if len(future_highs) > 0 else entry_price
    return final_close, max_bars, 2, peak_pnl_bps

trades = []
skipped_entries = 0

for idx in range(len(test)):
    if idx + MAX_HOLDING_BARS >= len(test):
        continue

    row = test.iloc[idx]
    entry_time = row.name
    entry_price = row['close']

    # Check for VALID crosses only
    if row['valid_cross_above']:
        direction = 1  # LONG
    elif row['valid_cross_below']:
        direction = -1  # SHORT
    else:
        if row['cross_above'] or row['cross_below']:
            skipped_entries += 1
        continue

    # Get future data
    future_data = test.iloc[idx + 1 : idx + 1 + MAX_HOLDING_BARS]
    future_highs = future_data['high'].values
    future_lows = future_data['low'].values

    # Simulate trade
    exit_price, bars_held, exit_reason, peak_pnl_bps = simulate_trailing_stop_with_activation(
        entry_price,
        direction,
        future_highs,
        future_lows,
        ACTIVATION_BPS,
        TRAILING_STOP_BPS,
        MAX_HOLDING_BARS
    )

    # Calculate final P&L
    if direction == 1:  # LONG
        pnl_bps = (exit_price - entry_price) / entry_price * 10000
    else:  # SHORT
        pnl_bps = (entry_price - exit_price) / entry_price * 10000

    # Subtract fees
    pnl_bps -= FEES_BPS

    # Record trade
    exit_reason_str = 'TRAILING' if exit_reason == 1 else 'TIME'
    trades.append({
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'direction': 'LONG' if direction == 1 else 'SHORT',
        'bars_held': bars_held,
        'exit_reason': exit_reason_str,
        'peak_pnl_bps': peak_pnl_bps,
        'final_pnl_bps': pnl_bps,
        'winner': pnl_bps > 0
    })

trades_df = pd.DataFrame(trades)
print(f"Total trades executed: {len(trades_df):,}")
print(f"Entries skipped (< 5 bps from EMA): {skipped_entries:,}")

# =============================================================================
# CALCULATE STATISTICS
# =============================================================================
print("\n[4/4] Calculating statistics...")

total_trades = len(trades_df)
winners = trades_df[trades_df['winner']]
losers = trades_df[~trades_df['winner']]

win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
total_pnl_bps = trades_df['final_pnl_bps'].sum()
avg_pnl_bps = trades_df['final_pnl_bps'].mean()

avg_win_bps = winners['final_pnl_bps'].mean() if len(winners) > 0 else 0
avg_loss_bps = losers['final_pnl_bps'].mean() if len(losers) > 0 else 0

max_win_bps = trades_df['final_pnl_bps'].max()
max_loss_bps = trades_df['final_pnl_bps'].min()

# Exit reason breakdown
trailing_exits = (trades_df['exit_reason'] == 'TRAILING').sum()
time_exits = (trades_df['exit_reason'] == 'TIME').sum()

# Peak analysis
avg_peak_pnl = trades_df['peak_pnl_bps'].mean()
avg_peak_winners = winners['peak_pnl_bps'].mean() if len(winners) > 0 else 0

# Activation analysis
activated_trades = trades_df[trades_df['peak_pnl_bps'] >= ACTIVATION_BPS]
activation_rate = len(activated_trades) / total_trades * 100 if total_trades > 0 else 0

# =============================================================================
# APPLY TO $100 CAPITAL
# =============================================================================
total_return_pct = total_pnl_bps / 100
starting_capital = 100
final_balance = starting_capital * (1 + total_return_pct / 100)
profit_loss = final_balance - starting_capital

# =============================================================================
# SAVE RESULTS
# =============================================================================
output_dir = Path("experiments/ema")
trades_path = output_dir / "filtered_trailing_activation_trades.csv"
trades_df.to_csv(trades_path, index=False)

# =============================================================================
# DISPLAY RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("BACKTEST RESULTS - FILTERED ENTRY + TRAILING STOP WITH ACTIVATION")
print("=" * 80)

print("\n--- STRATEGY PARAMETERS ---")
print(f"Entry: Cross EMA7 with >= {MIN_DISTANCE_BPS} bps distance")
print(f"Activation: Trailing stop activates at {ACTIVATION_BPS} bps gross profit")
print(f"Exit: Trailing stop {TRAILING_STOP_BPS} bps from peak (after activation)")
print(f"Max holding: {MAX_HOLDING_BARS} bars")
print(f"Fees: {FEES_BPS} bps per trade")

print("\n--- ENTRY FILTER RESULTS ---")
print(f"Entries skipped: {skipped_entries:,} (too close to EMA)")
print(f"Entries taken: {total_trades:,}")
print(f"Filter ratio: {skipped_entries/(skipped_entries+total_trades)*100:.1f}% filtered out")

print("\n--- TRADE STATISTICS ---")
print(f"Total trades: {total_trades:,}")
print(f"Winners: {len(winners):,} ({win_rate:.2f}%)")
print(f"Losers: {len(losers):,} ({100-win_rate:.2f}%)")

print("\n--- EXIT REASONS ---")
print(f"Trailing stop: {trailing_exits:,} ({trailing_exits/total_trades*100:.1f}%)")
print(f"Time exits: {time_exits:,} ({time_exits/total_trades*100:.1f}%)")

print("\n--- ACTIVATION ANALYSIS ---")
print(f"Trades that reached activation ({ACTIVATION_BPS} bps): {len(activated_trades):,} ({activation_rate:.1f}%)")
print(f"Trades that never activated: {total_trades - len(activated_trades):,} ({100-activation_rate:.1f}%)")

print("\n--- PERFORMANCE (BPS) ---")
print(f"Total P&L: {total_pnl_bps:,.0f} bps")
print(f"Average trade: {avg_pnl_bps:.2f} bps")
print(f"Average winner: {avg_win_bps:.2f} bps")
print(f"Average loser: {avg_loss_bps:.2f} bps")
print(f"Best trade: {max_win_bps:.2f} bps")
print(f"Worst trade: {max_loss_bps:.2f} bps")

print("\n--- PEAK ANALYSIS ---")
print(f"Avg peak P&L (all): {avg_peak_pnl:.2f} bps")
print(f"Avg peak P&L (winners): {avg_peak_winners:.2f} bps")

print("\n--- $100 CAPITAL PERFORMANCE ---")
print(f"Starting capital: ${starting_capital:.2f}")
print(f"Final balance: ${final_balance:.2f}")
print(f"Profit/Loss: ${profit_loss:.2f} ({total_return_pct:+.2f}%)")

print("\n--- METRICS ---")
if len(losers) > 0 and losers['final_pnl_bps'].sum() != 0:
    profit_factor = abs(winners['final_pnl_bps'].sum() / losers['final_pnl_bps'].sum())
    print(f"Profit factor: {profit_factor:.2f}")

if avg_loss_bps != 0:
    risk_reward = abs(avg_win_bps / avg_loss_bps)
    print(f"Risk/Reward ratio: {risk_reward:.2f}")

print(f"\nTrades saved to: {trades_path}")

print("\n" + "=" * 80)
print("STRATEGY IMPROVEMENTS:")
print("[OK] Entry filter reduces choppy signals")
print("[OK] Activation threshold prevents exiting tiny profits")
print("[OK] Trailing stop lets winners run while protecting profits")
print("=" * 80)
