"""
EXP-004 Part 3: Do the 2 Catastrophic Failures EVER Recover?

The 2 failures that didn't recover within 50 hours:
1. 2024-02-26 15:00:00 - Max drawdown -2301 bps
2. 2024-07-25 23:30:00 - Max drawdown -565 bps

Let's check if they EVER recovered, even if it took days/weeks.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
RESULTS_PATH = Path("experiments/rsi/EXP-004")

# The 2 catastrophic failures
CATASTROPHIC_FAILURES = [
    '2024-02-26 15:00:00',
    '2024-07-25 23:30:00'
]

MAX_HORIZON = 2000  # ~500 hours = ~20 days

print("="*70)
print("CATASTROPHIC FAILURE ANALYSIS")
print("Do the 2 worst failures EVER recover?")
print("="*70)

# Load data
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

for failure_time in CATASTROPHIC_FAILURES:
    print(f"\n{'='*70}")
    print(f"FAILURE: {failure_time}")
    print("="*70)

    idx = pd.Timestamp(failure_time)

    if idx not in df.index:
        print(f"Timestamp not found!")
        continue

    idx_pos = df.index.get_loc(idx)
    entry_price = df.iloc[idx_pos]['close']

    print(f"Entry price: ${entry_price:,.0f}")

    # Track recovery
    recovered = False
    recovery_bar = None
    max_drawdown = 0
    max_drawdown_bar = 0

    # Check up to MAX_HORIZON bars
    check_horizon = min(MAX_HORIZON, len(df) - idx_pos - 1)

    for bar in range(1, check_horizon + 1):
        bar_data = df.iloc[idx_pos + bar]

        # For SHORT: profit if price drops below entry
        low_pnl = (entry_price - bar_data['low']) / entry_price * 10000
        high_pnl = (entry_price - bar_data['high']) / entry_price * 10000

        # Track max drawdown
        if high_pnl < max_drawdown:
            max_drawdown = high_pnl
            max_drawdown_bar = bar

        # Check if recovered
        if low_pnl > 0 and not recovered:
            recovered = True
            recovery_bar = bar
            break

    print(f"\nMax drawdown: {max_drawdown:.1f} bps (at bar {max_drawdown_bar})")
    print(f"Max drawdown in $: ${abs(max_drawdown / 10000 * entry_price):,.0f}")
    print(f"Max drawdown time: {max_drawdown_bar * 15 / 60:.1f} hours ({max_drawdown_bar * 15 / 60 / 24:.1f} days)")

    if recovered:
        recovery_hours = recovery_bar * 15 / 60
        recovery_days = recovery_hours / 24

        print(f"\n*** RECOVERED! ***")
        print(f"Recovery bar: {recovery_bar}")
        print(f"Recovery time: {recovery_hours:.1f} hours ({recovery_days:.1f} days)")

        # What was the price at recovery?
        recovery_price = df.iloc[idx_pos + recovery_bar]['low']
        print(f"Recovery price: ${recovery_price:,.0f}")
    else:
        print(f"\n*** NEVER RECOVERED within {check_horizon} bars ({check_horizon * 15 / 60:.0f} hours / {check_horizon * 15 / 60 / 24:.1f} days) ***")

        # What's the current price relative to entry?
        last_bar = df.iloc[idx_pos + check_horizon]
        current_pnl = (entry_price - last_bar['close']) / entry_price * 10000
        print(f"PnL at end of data: {current_pnl:.1f} bps")

    # Show price path
    print(f"\nPrice journey:")
    milestones = [1, 10, 50, 100, 200, 500, 1000]
    for m in milestones:
        if m < check_horizon:
            bar_data = df.iloc[idx_pos + m]
            pnl = (entry_price - bar_data['close']) / entry_price * 10000
            hours = m * 15 / 60
            print(f"  Bar {m:4d} ({hours:6.1f}h): ${bar_data['close']:,.0f} = {pnl:+.1f} bps")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
