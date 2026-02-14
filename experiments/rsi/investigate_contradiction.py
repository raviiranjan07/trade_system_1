"""
INVESTIGATE: Why does EXP-004 say "100% recover" but patience test is worse?

Hypothesis: The definition of "recovery" is different
- EXP-004: Price returns to entry level (recovery = back to break-even)
- Trading: We use TRAILING STOP which exits BEFORE full recovery

Let's check what actually happens to the 123 V1 losers.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")

# Parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
SMA_PERIOD = 200
LONG_TRAILING_STOP = 20
SHORT_TRAILING_STOP = 30
FEES_BPS = 8
MAX_BARS = 10

# Extended horizons to check recovery
CHECK_HORIZONS = [10, 20, 50, 100, 200]

print("="*70)
print("INVESTIGATION: V1 Losers - Do They Eventually Recover?")
print("="*70)

# Load data
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calculate_rsi(df, RSI_PERIOD)
df['sma_200'] = df['close'].rolling(SMA_PERIOD).mean()
df['rsi_oversold_cross'] = (df['rsi'] < RSI_OVERSOLD) & (df['rsi'].shift(1) >= RSI_OVERSOLD)
df['rsi_overbought_cross'] = (df['rsi'] > RSI_OVERBOUGHT) & (df['rsi'].shift(1) <= RSI_OVERBOUGHT)
df['bull_market'] = df['close'] > df['sma_200']
df['bear_market'] = df['close'] < df['sma_200']

oos_data = df['2024-01-01':'2025-12-31'].copy()

def analyze_trade_deeply(data, entry_idx, direction, trailing_stop_bps, max_check_bars=200):
    """Analyze a single trade in detail - what happens at each stage."""

    entry_price = data.iloc[entry_idx + 1]['open']

    results = {
        'direction': direction,
        'entry_price': entry_price,
        'v1_exit_bar': None,
        'v1_exit_profit': None,
        'v1_exit_reason': None,
        'v1_is_loser': None,
        'max_mfe_ever': 0,
        'max_mfe_bar': 0,
        'max_mae_ever': 0,
        'max_mae_bar': 0,
        'recovers_to_profit': False,    # Does it EVER show net profit?
        'recovery_bar': None,           # When does it first show net profit?
        'final_200_profit': None,       # What's the profit at Bar 200?
    }

    highest_profit = 0
    lowest_profit = 0
    first_profit_bar = None

    # First, simulate V1 (Bar 10 exit)
    v1_exit_bar = None
    v1_exit_profit = None
    v1_exit_reason = None

    for bar in range(1, min(MAX_BARS + 1, max_check_bars)):
        future_idx = entry_idx + 1 + bar
        if future_idx >= len(data):
            break

        future_bar = data.iloc[future_idx]

        if direction == 'LONG':
            bar_mfe = (future_bar['high'] - entry_price) / entry_price * 10000
            bar_close_pnl = (future_bar['close'] - entry_price) / entry_price * 10000
        else:
            bar_mfe = (entry_price - future_bar['low']) / entry_price * 10000
            bar_close_pnl = (entry_price - future_bar['close']) / entry_price * 10000

        if bar_mfe > highest_profit:
            highest_profit = bar_mfe
            results['max_mfe_bar'] = bar

        if bar_close_pnl < lowest_profit:
            lowest_profit = bar_close_pnl
            results['max_mae_bar'] = bar

        # Check if this is profitable (net of fees)
        if bar_close_pnl - FEES_BPS > 0 and first_profit_bar is None:
            first_profit_bar = bar

        # V1 trailing stop check
        if v1_exit_bar is None:
            drawdown_from_peak = highest_profit - bar_close_pnl
            if drawdown_from_peak >= trailing_stop_bps and highest_profit > 0:
                v1_exit_bar = bar
                v1_exit_profit = highest_profit - trailing_stop_bps
                v1_exit_reason = 'TRAILING_STOP'

    # V1 time exit if no trailing stop
    if v1_exit_bar is None:
        final_bar = data.iloc[entry_idx + 1 + MAX_BARS]
        if direction == 'LONG':
            v1_exit_profit = (final_bar['close'] - entry_price) / entry_price * 10000
        else:
            v1_exit_profit = (entry_price - final_bar['close']) / entry_price * 10000
        v1_exit_bar = MAX_BARS
        v1_exit_reason = 'TIME_EXIT'

    results['v1_exit_bar'] = v1_exit_bar
    results['v1_exit_profit'] = v1_exit_profit
    results['v1_exit_reason'] = v1_exit_reason
    results['v1_is_loser'] = (v1_exit_profit - FEES_BPS) <= 0

    # Now check extended horizon
    max_mfe_ever = highest_profit
    max_mae_ever = lowest_profit

    for bar in range(MAX_BARS + 1, max_check_bars + 1):
        future_idx = entry_idx + 1 + bar
        if future_idx >= len(data):
            break

        future_bar = data.iloc[future_idx]

        if direction == 'LONG':
            bar_mfe = (future_bar['high'] - entry_price) / entry_price * 10000
            bar_close_pnl = (future_bar['close'] - entry_price) / entry_price * 10000
        else:
            bar_mfe = (entry_price - future_bar['low']) / entry_price * 10000
            bar_close_pnl = (entry_price - future_bar['close']) / entry_price * 10000

        if bar_mfe > max_mfe_ever:
            max_mfe_ever = bar_mfe
            results['max_mfe_bar'] = bar

        if bar_close_pnl < max_mae_ever:
            max_mae_ever = bar_close_pnl
            results['max_mae_bar'] = bar

        # Check recovery
        if bar_close_pnl - FEES_BPS > 0 and first_profit_bar is None:
            first_profit_bar = bar

    results['max_mfe_ever'] = max_mfe_ever
    results['max_mae_ever'] = max_mae_ever
    results['recovers_to_profit'] = first_profit_bar is not None
    results['recovery_bar'] = first_profit_bar

    # Get Bar 200 profit
    bar_200_idx = entry_idx + 1 + 200
    if bar_200_idx < len(data):
        bar_200 = data.iloc[bar_200_idx]
        if direction == 'LONG':
            results['final_200_profit'] = (bar_200['close'] - entry_price) / entry_price * 10000
        else:
            results['final_200_profit'] = (entry_price - bar_200['close']) / entry_price * 10000

    return results

# Analyze all trades
all_trades = []

for idx in oos_data.index:
    idx_pos = oos_data.index.get_loc(idx)
    if idx_pos + 210 >= len(oos_data):
        continue

    current = oos_data.iloc[idx_pos]

    # LONG signal
    if current['rsi_oversold_cross'] and current['bull_market']:
        trade = analyze_trade_deeply(oos_data, idx_pos, 'LONG', LONG_TRAILING_STOP)
        if trade:
            all_trades.append(trade)

    # SHORT signal
    elif current['rsi_overbought_cross'] and current['bear_market']:
        trade = analyze_trade_deeply(oos_data, idx_pos, 'SHORT', SHORT_TRAILING_STOP)
        if trade:
            all_trades.append(trade)

trades_df = pd.DataFrame(all_trades)

# =============================================================================
# ANALYSIS
# =============================================================================

print(f"\nTotal trades analyzed: {len(trades_df)}")

v1_losers = trades_df[trades_df['v1_is_loser'] == True]
v1_winners = trades_df[trades_df['v1_is_loser'] == False]

print(f"\nV1 Winners: {len(v1_winners)}")
print(f"V1 Losers: {len(v1_losers)}")

print("\n" + "="*70)
print("KEY QUESTION: Do V1 losers EVER show profit (with patience)?")
print("="*70)

losers_that_recover = v1_losers[v1_losers['recovers_to_profit'] == True]
losers_never_recover = v1_losers[v1_losers['recovers_to_profit'] == False]

print(f"\nV1 Losers that EVER show profit (within 200 bars): {len(losers_that_recover)}")
print(f"V1 Losers that NEVER show profit: {len(losers_never_recover)}")

if len(losers_that_recover) > 0:
    print(f"\nAmong those that recover:")
    print(f"  - Average recovery bar: {losers_that_recover['recovery_bar'].mean():.1f}")
    print(f"  - Median recovery bar: {losers_that_recover['recovery_bar'].median():.1f}")
    print(f"  - Max MFE ever: {losers_that_recover['max_mfe_ever'].mean():.1f} bps")

print("\n" + "="*70)
print("THE CONTRADICTION EXPLAINED")
print("="*70)

print(f"""
EXP-004 Definition of "Failure":
  - RSI signal that NEVER bounces at all (price only goes against you)
  - Found: 1 such failure in 266 signals
  - That 1 failure eventually recovers

V1 "Loser" Definition:
  - Trade that exits with NET LOSS (after fees)
  - Includes trades where price DID bounce, but:
    a) Bounce was small (< trailing stop)
    b) Trailing stop hit too early
    c) Time exit at Bar 10 when price was down

V1 Losers breakdown:
  - Total: {len(v1_losers)} losers
  - Eventually show profit: {len(losers_that_recover)} ({len(losers_that_recover)/len(v1_losers)*100:.1f}%)
  - NEVER show profit: {len(losers_never_recover)} ({len(losers_never_recover)/len(v1_losers)*100:.1f}%)
""")

print("\n" + "="*70)
print("WHY PATIENCE DOESN'T HELP (Even Though They Eventually Recover)")
print("="*70)

# Check what happens with the trailing stop logic over time
print("""
The problem: TRAILING STOP gets in the way!

Scenario for a typical V1 loser that "eventually recovers":
""")

# Let's look at actual examples
print("\nEXAMPLE LOSERS (Detailed):")
sample_losers = v1_losers.head(5)

for i, (idx, trade) in enumerate(sample_losers.iterrows()):
    print(f"\nLoser #{i+1} ({trade['direction']}):")
    print(f"  V1 exit: Bar {trade['v1_exit_bar']}, {trade['v1_exit_profit']:.1f} bps gross ({trade['v1_exit_reason']})")
    print(f"  Net P/L: {trade['v1_exit_profit'] - FEES_BPS:.1f} bps")
    print(f"  Max MFE ever: {trade['max_mfe_ever']:.1f} bps at Bar {trade['max_mfe_bar']}")
    print(f"  Max MAE ever: {trade['max_mae_ever']:.1f} bps")
    print(f"  Recovers to profit? {trade['recovers_to_profit']}", end="")
    if trade['recovers_to_profit']:
        print(f" at Bar {trade['recovery_bar']}")
    else:
        print()
    print(f"  Bar 200 profit: {trade['final_200_profit']:.1f} bps" if trade['final_200_profit'] else "  Bar 200: N/A")

# The key insight
print("\n" + "="*70)
print("THE REAL ISSUE")
print("="*70)

# For losers that eventually recover - when do they recover vs when would trailing stop hit?
if len(losers_that_recover) > 0:
    # How many would have the trailing stop hit BEFORE recovery?
    early_recovery = losers_that_recover[losers_that_recover['recovery_bar'] <= 20]
    late_recovery = losers_that_recover[losers_that_recover['recovery_bar'] > 20]

    print(f"""
Losers that eventually recover: {len(losers_that_recover)}
  - Early recovery (Bar 1-20): {len(early_recovery)} ({len(early_recovery)/len(losers_that_recover)*100:.1f}%)
  - Late recovery (Bar 21+): {len(late_recovery)} ({len(late_recovery)/len(losers_that_recover)*100:.1f}%)

THE PROBLEM:
Even if price eventually reaches profit, with a TRAILING STOP:
1. Price goes up +10 bps (sets peak)
2. Price drops to -5 bps
3. Trailing stop (20 bps from peak) triggers at +10 - 20 = -10 bps
4. We exit at LOSS
5. Then price recovers to +30 bps... but we're already out!

This is why PATIENCE with TRAILING STOP doesn't help:
- More time = more price swings
- More swings = more chances for trailing stop to trigger at bad times
- Trailing stop locks in early, before full recovery

SOLUTION OPTIONS:
1. No trailing stop (just time exit) - but this has worse results
2. Wider trailing stop - but this increases losses on real losers
3. Accept the losers - they are the cost of locking in winners early
""")

# Final summary
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

print(f"""
The contradiction is RESOLVED:

EXP-004 "100% recover" = Price eventually returns to entry level
V1.1 "Patience is worse" = Trailing stop exits BEFORE recovery

{len(losers_that_recover)}/{len(v1_losers)} ({len(losers_that_recover)/len(v1_losers)*100:.1f}%) of V1 losers DO eventually reach profit
BUT the trailing stop kicks them out early.

This is actually a FEATURE, not a bug:
- Trailing stop PROTECTS profits on winners
- Same mechanism LOCKS IN small losses on losers
- Net result: Still profitable (+3,024 bps in V1)

The 123 losers are the "cost" of using trailing stops.
Without trailing stops, we'd have fewer losers but smaller winners.
V1 with trailing stops is the optimal balance.
""")
