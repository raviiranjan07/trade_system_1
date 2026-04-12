"""
TRADING SETUP V1 - Full Backtest
================================
Entry:
  LONG:  RSI < 20 AND Price > SMA200
  SHORT: RSI > 80 AND Price < SMA200

Exit:
  LONG:  20 bps trailing stop OR Bar 10
  SHORT: 30 bps trailing stop OR Bar 10

Fees: 8 bps round-trip
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
OUTPUT_PATH = Path("experiments/rsi")

# =============================================================================
# PARAMETERS
# =============================================================================
RSI_PERIOD = 14
RSI_OVERSOLD = 20
RSI_OVERBOUGHT = 80
SMA_PERIOD = 200
LONG_TRAILING_STOP = 20  # bps
SHORT_TRAILING_STOP = 30  # bps
MAX_BARS = 10
FEES_BPS = 8  # round-trip fees

print("="*70)
print("TRADING SETUP V1 - FULL BACKTEST")
print("="*70)
print(f"\nParameters:")
print(f"  RSI Period: {RSI_PERIOD}")
print(f"  RSI Oversold: {RSI_OVERSOLD}")
print(f"  RSI Overbought: {RSI_OVERBOUGHT}")
print(f"  SMA Period: {SMA_PERIOD}")
print(f"  LONG Trailing Stop: {LONG_TRAILING_STOP} bps")
print(f"  SHORT Trailing Stop: {SHORT_TRAILING_STOP} bps")
print(f"  Max Holding: {MAX_BARS} bars")
print(f"  Fees: {FEES_BPS} bps")

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
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

print(f"\nData Period: {oos_start} to {oos_end}")
print(f"Total bars: {len(oos_data)}")

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(data):
    """Run full backtest with all rules."""

    trades = []

    for idx in data.index:
        idx_pos = data.index.get_loc(idx)

        # Skip if not enough future data
        if idx_pos + MAX_BARS >= len(data):
            continue

        current = data.iloc[idx_pos]

        # Check for LONG signal
        if current['rsi_oversold_cross'] and current['bull_market']:
            trade = execute_trade(data, idx_pos, 'LONG', LONG_TRAILING_STOP)
            if trade:
                trades.append(trade)

        # Check for SHORT signal
        elif current['rsi_overbought_cross'] and current['bear_market']:
            trade = execute_trade(data, idx_pos, 'SHORT', SHORT_TRAILING_STOP)
            if trade:
                trades.append(trade)

    return pd.DataFrame(trades)

def execute_trade(data, entry_idx, direction, trailing_stop_bps):
    """Execute a single trade with trailing stop."""

    entry_bar = data.iloc[entry_idx]
    entry_price = data.iloc[entry_idx + 1]['open']  # Enter at next bar open
    entry_time = data.index[entry_idx + 1]

    highest_profit = 0
    exit_profit = None
    exit_bar = None
    exit_price = None
    exit_reason = None
    mae = 0  # Max Adverse Excursion
    mfe = 0  # Max Favorable Excursion

    for bar in range(1, MAX_BARS + 1):
        future_idx = entry_idx + 1 + bar
        if future_idx >= len(data):
            break

        future_bar = data.iloc[future_idx]

        if direction == 'LONG':
            # Calculate current profit using high (for MFE) and close (for exit check)
            bar_mfe = (future_bar['high'] - entry_price) / entry_price * 10000
            bar_mae = (future_bar['low'] - entry_price) / entry_price * 10000
            bar_close_pnl = (future_bar['close'] - entry_price) / entry_price * 10000
        else:  # SHORT
            bar_mfe = (entry_price - future_bar['low']) / entry_price * 10000
            bar_mae = (entry_price - future_bar['high']) / entry_price * 10000
            bar_close_pnl = (entry_price - future_bar['close']) / entry_price * 10000

        # Track MFE and MAE
        if bar_mfe > mfe:
            mfe = bar_mfe
        if bar_mae < mae:
            mae = bar_mae

        # Update highest profit
        if bar_mfe > highest_profit:
            highest_profit = bar_mfe

        # Check trailing stop
        drawdown_from_peak = highest_profit - bar_close_pnl

        if drawdown_from_peak >= trailing_stop_bps and highest_profit > 0:
            # Trailing stop hit
            exit_profit = highest_profit - trailing_stop_bps
            exit_bar = bar
            exit_price = entry_price * (1 + exit_profit / 10000) if direction == 'LONG' else entry_price * (1 - exit_profit / 10000)
            exit_reason = 'TRAILING_STOP'
            break

    # If no trailing stop hit, exit at Bar 10 close
    if exit_profit is None:
        final_bar = data.iloc[entry_idx + 1 + MAX_BARS]
        if direction == 'LONG':
            exit_profit = (final_bar['close'] - entry_price) / entry_price * 10000
        else:
            exit_profit = (entry_price - final_bar['close']) / entry_price * 10000
        exit_bar = MAX_BARS
        exit_price = final_bar['close']
        exit_reason = 'TIME_EXIT'

    exit_time = data.index[entry_idx + 1 + exit_bar]

    # Calculate net profit after fees
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
# RUN BACKTEST
# =============================================================================
print("\nRunning backtest...")
trades_df = run_backtest(oos_data)

print(f"\nTotal trades: {len(trades_df)}")

# =============================================================================
# CALCULATE METRICS
# =============================================================================

def calculate_metrics(trades):
    """Calculate all performance metrics."""

    if len(trades) == 0:
        return {}

    metrics = {}

    # Basic counts
    metrics['total_trades'] = len(trades)
    metrics['long_trades'] = len(trades[trades['direction'] == 'LONG'])
    metrics['short_trades'] = len(trades[trades['direction'] == 'SHORT'])

    # Win/Loss
    winners = trades[trades['net_profit_bps'] > 0]
    losers = trades[trades['net_profit_bps'] <= 0]

    metrics['winners'] = len(winners)
    metrics['losers'] = len(losers)
    metrics['win_rate'] = len(winners) / len(trades) * 100 if len(trades) > 0 else 0

    # Profit metrics
    metrics['total_gross_profit_bps'] = trades['gross_profit_bps'].sum()
    metrics['total_net_profit_bps'] = trades['net_profit_bps'].sum()
    metrics['avg_gross_profit_bps'] = trades['gross_profit_bps'].mean()
    metrics['avg_net_profit_bps'] = trades['net_profit_bps'].mean()

    # Winner/Loser stats
    if len(winners) > 0:
        metrics['avg_winner_bps'] = winners['net_profit_bps'].mean()
        metrics['max_winner_bps'] = winners['net_profit_bps'].max()
    else:
        metrics['avg_winner_bps'] = 0
        metrics['max_winner_bps'] = 0

    if len(losers) > 0:
        metrics['avg_loser_bps'] = losers['net_profit_bps'].mean()
        metrics['max_loser_bps'] = losers['net_profit_bps'].min()
    else:
        metrics['avg_loser_bps'] = 0
        metrics['max_loser_bps'] = 0

    # Profit Factor
    gross_profits = winners['net_profit_bps'].sum() if len(winners) > 0 else 0
    gross_losses = abs(losers['net_profit_bps'].sum()) if len(losers) > 0 else 1
    metrics['profit_factor'] = gross_profits / gross_losses if gross_losses > 0 else float('inf')

    # MFE/MAE
    metrics['avg_mfe_bps'] = trades['mfe_bps'].mean()
    metrics['avg_mae_bps'] = trades['mae_bps'].mean()

    # Exit reasons
    metrics['trailing_stop_exits'] = len(trades[trades['exit_reason'] == 'TRAILING_STOP'])
    metrics['time_exits'] = len(trades[trades['exit_reason'] == 'TIME_EXIT'])

    # By direction
    long_trades = trades[trades['direction'] == 'LONG']
    short_trades = trades[trades['direction'] == 'SHORT']

    if len(long_trades) > 0:
        metrics['long_win_rate'] = len(long_trades[long_trades['net_profit_bps'] > 0]) / len(long_trades) * 100
        metrics['long_avg_net'] = long_trades['net_profit_bps'].mean()
    else:
        metrics['long_win_rate'] = 0
        metrics['long_avg_net'] = 0

    if len(short_trades) > 0:
        metrics['short_win_rate'] = len(short_trades[short_trades['net_profit_bps'] > 0]) / len(short_trades) * 100
        metrics['short_avg_net'] = short_trades['net_profit_bps'].mean()
    else:
        metrics['short_win_rate'] = 0
        metrics['short_avg_net'] = 0

    return metrics

metrics = calculate_metrics(trades_df)

# =============================================================================
# PRINT RESULTS
# =============================================================================

print("\n" + "="*70)
print("BACKTEST RESULTS")
print("="*70)

print(f"""
TRADE SUMMARY:
--------------
Total Trades:     {metrics['total_trades']}
  LONG:           {metrics['long_trades']}
  SHORT:          {metrics['short_trades']}

Winners:          {metrics['winners']}
Losers:           {metrics['losers']}
Win Rate:         {metrics['win_rate']:.1f}%

PROFIT METRICS:
---------------
Total Gross:      {metrics['total_gross_profit_bps']:+.1f} bps
Total Net:        {metrics['total_net_profit_bps']:+.1f} bps
Avg Gross/Trade:  {metrics['avg_gross_profit_bps']:+.1f} bps
Avg Net/Trade:    {metrics['avg_net_profit_bps']:+.1f} bps

Avg Winner:       {metrics['avg_winner_bps']:+.1f} bps
Avg Loser:        {metrics['avg_loser_bps']:+.1f} bps
Max Winner:       {metrics['max_winner_bps']:+.1f} bps
Max Loser:        {metrics['max_loser_bps']:+.1f} bps

Profit Factor:    {metrics['profit_factor']:.2f}

BY DIRECTION:
-------------
LONG:
  Win Rate:       {metrics['long_win_rate']:.1f}%
  Avg Net:        {metrics['long_avg_net']:+.1f} bps

SHORT:
  Win Rate:       {metrics['short_win_rate']:.1f}%
  Avg Net:        {metrics['short_avg_net']:+.1f} bps

EXIT ANALYSIS:
--------------
Trailing Stop:    {metrics['trailing_stop_exits']} ({metrics['trailing_stop_exits']/metrics['total_trades']*100:.1f}%)
Time Exit:        {metrics['time_exits']} ({metrics['time_exits']/metrics['total_trades']*100:.1f}%)

MFE/MAE:
--------
Avg MFE:          {metrics['avg_mfe_bps']:+.1f} bps
Avg MAE:          {metrics['avg_mae_bps']:+.1f} bps
""")

# =============================================================================
# EQUITY CURVE
# =============================================================================

trades_df['cumulative_net_bps'] = trades_df['net_profit_bps'].cumsum()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Equity Curve
ax1 = axes[0, 0]
ax1.plot(trades_df['exit_time'], trades_df['cumulative_net_bps'], 'b-', linewidth=2)
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.fill_between(trades_df['exit_time'], 0, trades_df['cumulative_net_bps'],
                  where=trades_df['cumulative_net_bps'] >= 0, color='green', alpha=0.3)
ax1.fill_between(trades_df['exit_time'], 0, trades_df['cumulative_net_bps'],
                  where=trades_df['cumulative_net_bps'] < 0, color='red', alpha=0.3)
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative Net Profit (bps)')
ax1.set_title(f'Equity Curve - Total: {metrics["total_net_profit_bps"]:+.1f} bps')
ax1.grid(True, alpha=0.3)

# Plot 2: Trade Distribution
ax2 = axes[0, 1]
ax2.hist(trades_df['net_profit_bps'], bins=30, edgecolor='black', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', linewidth=2)
ax2.axvline(trades_df['net_profit_bps'].mean(), color='green', linestyle='--', linewidth=2,
            label=f'Mean: {trades_df["net_profit_bps"].mean():.1f} bps')
ax2.set_xlabel('Net Profit (bps)')
ax2.set_ylabel('Count')
ax2.set_title('Trade Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: By Direction
ax3 = axes[1, 0]
long_trades = trades_df[trades_df['direction'] == 'LONG']
short_trades = trades_df[trades_df['direction'] == 'SHORT']

categories = ['LONG', 'SHORT']
win_rates = [metrics['long_win_rate'], metrics['short_win_rate']]
avg_profits = [metrics['long_avg_net'], metrics['short_avg_net']]

x = np.arange(len(categories))
width = 0.35

bars1 = ax3.bar(x - width/2, win_rates, width, label='Win Rate %', color='blue', alpha=0.7)
ax3_twin = ax3.twinx()
bars2 = ax3_twin.bar(x + width/2, avg_profits, width, label='Avg Net (bps)', color='green', alpha=0.7)

ax3.set_ylabel('Win Rate %', color='blue')
ax3_twin.set_ylabel('Avg Net Profit (bps)', color='green')
ax3.set_title('Performance by Direction')
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')

# Plot 4: Monthly Performance
ax4 = axes[1, 1]
trades_df['month'] = trades_df['exit_time'].dt.to_period('M')
monthly = trades_df.groupby('month')['net_profit_bps'].sum()

colors = ['green' if v > 0 else 'red' for v in monthly.values]
ax4.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7)
ax4.set_xticks(range(len(monthly)))
ax4.set_xticklabels([str(m) for m in monthly.index], rotation=45, ha='right')
ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel('Month')
ax4.set_ylabel('Net Profit (bps)')
ax4.set_title('Monthly Performance')
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle('TRADING SETUP V1 - BACKTEST RESULTS', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'backtest_v1_results.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {OUTPUT_PATH / 'backtest_v1_results.png'}")

# =============================================================================
# SAVE TRADES TO CSV
# =============================================================================
trades_df.to_csv(OUTPUT_PATH / 'backtest_v1_trades.csv', index=False)
print(f"Saved: {OUTPUT_PATH / 'backtest_v1_trades.csv'}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"""
Trading Setup V1 Performance (2024-2025 OOS):

  Total Trades:     {metrics['total_trades']}
  Win Rate:         {metrics['win_rate']:.1f}%
  Profit Factor:    {metrics['profit_factor']:.2f}

  Total Net Profit: {metrics['total_net_profit_bps']:+.1f} bps
  Avg Net/Trade:    {metrics['avg_net_profit_bps']:+.1f} bps

  LONG Avg:         {metrics['long_avg_net']:+.1f} bps
  SHORT Avg:        {metrics['short_avg_net']:+.1f} bps
""")
