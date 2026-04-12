"""
EXP-011: Strategy Visualization
=================================
Price charts with entry/exit markers for each strategy.
Shows: sample trades, equity curves, trade distribution.

Run: python experiments/rsi/EXP-011/visualize_strategies.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_PATH = Path("data/raw/BTCUSDT_15m_ohlcv.parquet")
OUTPUT_PATH = Path("experiments/rsi/EXP-011/plots")
OUTPUT_PATH.mkdir(exist_ok=True)

# Load price data
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

# Load trade CSVs
trade_files = {
    'Vol Squeeze': 'vol_squeeze_breakout_trades.csv',
    'Vol Spike': 'volume_spike___direction_trades.csv',
    'Candlestick': 'candlestick_patterns_trades.csv',
    'MACD Div': 'macd_divergence_trades.csv',
}

all_trades = {}
for name, fname in trade_files.items():
    fpath = Path("experiments/rsi/EXP-011") / fname
    if fpath.exists():
        tdf = pd.read_csv(fpath)
        tdf['entry_time'] = pd.to_datetime(tdf['entry_time'])
        tdf['exit_time'] = pd.to_datetime(tdf['exit_time'])
        tdf['signal_time'] = pd.to_datetime(tdf['signal_time'])
        all_trades[name] = tdf
        print(f"Loaded {name}: {len(tdf)} trades")
    else:
        print(f"Missing: {fpath}")


# =============================================================================
# PLOT 1: EQUITY CURVES — ALL STRATEGIES
# =============================================================================
fig, ax = plt.subplots(figsize=(16, 8))

colors = {'Vol Squeeze': '#e74c3c', 'Vol Spike': '#2ecc71', 'Candlestick': '#3498db', 'MACD Div': '#9b59b6'}

for name, tdf in all_trades.items():
    tdf_sorted = tdf.sort_values('entry_time')
    equity = tdf_sorted['net_profit_bps'].cumsum()
    ax.plot(tdf_sorted['entry_time'].values, equity.values, label=f"{name} ({len(tdf)}t, {equity.iloc[-1]:+.0f} bps)",
            color=colors.get(name, 'gray'), linewidth=1.5)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_title('EXP-011: Equity Curves — All 4 Strategies (OOS 2024-2025)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Net Profit (bps)')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_PATH / '01_equity_curves_all.png', dpi=150)
plt.close()
print("Saved: 01_equity_curves_all.png")


# =============================================================================
# PLOT 2-5: PRICE CHARTS WITH ENTRIES — SAMPLE TRADES FOR EACH STRATEGY
# =============================================================================
def plot_sample_trades(strategy_name, trades_df, df_price, plot_num, n_samples=6):
    """Plot sample trades on 15min price chart with candlesticks."""
    if len(trades_df) == 0:
        return

    # Pick trades: 3 best winners, 3 worst losers
    sorted_t = trades_df.sort_values('net_profit_bps')
    worst = sorted_t.head(min(3, len(sorted_t)))
    best = sorted_t.tail(min(3, len(sorted_t)))
    samples = pd.concat([worst, best]).sort_values('entry_time')

    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(18, 5 * n))
    if n == 1:
        axes = [axes]

    for idx, (_, trade) in enumerate(samples.iterrows()):
        ax = axes[idx]

        # Get price data around the trade: 10 bars before signal, to 5 bars after exit
        sig_time = trade['signal_time']
        exit_time = trade['exit_time']

        # Find indices
        try:
            sig_idx = df_price.index.get_indexer([sig_time], method='ffill')[0]
        except:
            continue

        start_idx = max(0, sig_idx - 10)
        end_idx = min(len(df_price) - 1, sig_idx + 25)

        window = df_price.iloc[start_idx:end_idx + 1]

        # Draw candlesticks
        for j in range(len(window)):
            bar = window.iloc[j]
            t = mdates.date2num(window.index[j])
            color = '#2ecc71' if bar['close'] >= bar['open'] else '#e74c3c'
            edge_color = '#1a9641' if bar['close'] >= bar['open'] else '#d73027'

            # Body
            body_bottom = min(bar['open'], bar['close'])
            body_height = abs(bar['close'] - bar['open'])
            ax.bar(t, body_height, bottom=body_bottom, width=0.006,
                   color=color, edgecolor=edge_color, linewidth=0.5)
            # Wicks
            ax.plot([t, t], [bar['low'], body_bottom], color=edge_color, linewidth=0.7)
            ax.plot([t, t], [body_bottom + body_height, bar['high']], color=edge_color, linewidth=0.7)

        # Mark entry
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        entry_t = mdates.date2num(entry_time)

        marker_color = '#00aa00' if trade['direction'] == 'LONG' else '#cc0000'
        marker = '^' if trade['direction'] == 'LONG' else 'v'
        ax.plot(entry_t, entry_price, marker=marker, color=marker_color,
                markersize=14, markeredgecolor='black', markeredgewidth=1, zorder=5)

        # Mark exit (approximate from exit_time)
        exit_t = mdates.date2num(exit_time)
        # Calculate exit price from entry + profit
        if trade['direction'] == 'LONG':
            exit_price = entry_price * (1 + trade['gross_profit_bps'] / 10000)
        else:
            exit_price = entry_price * (1 - trade['gross_profit_bps'] / 10000)

        ax.plot(exit_t, exit_price, marker='x', color=marker_color,
                markersize=12, markeredgewidth=2.5, zorder=5)

        # Draw line from entry to exit
        ax.plot([entry_t, exit_t], [entry_price, exit_price],
                color=marker_color, linewidth=1.5, linestyle='--', alpha=0.6)

        # Add volume bars at bottom
        ax2 = ax.twinx()
        vol_colors = ['#2ecc71' if window.iloc[j]['close'] >= window.iloc[j]['open'] else '#e74c3c'
                       for j in range(len(window))]
        vol_times = [mdates.date2num(window.index[j]) for j in range(len(window))]
        ax2.bar(vol_times, window['volume'].values, width=0.006, color=vol_colors, alpha=0.2)
        ax2.set_ylim(0, window['volume'].max() * 4)
        ax2.set_yticks([])

        # Labels
        profit_str = f"{trade['net_profit_bps']:+.1f} bps"
        win_str = "WIN" if trade['net_profit_bps'] > 0 else "LOSS"
        ax.set_title(f"{trade['direction']} | {win_str} {profit_str} | "
                     f"Exit: {trade['exit_reason']} bar {int(trade['exit_bar'])} | "
                     f"{str(entry_time)[:16]}",
                     fontsize=11, color='green' if trade['net_profit_bps'] > 0 else 'red')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'EXP-011: {strategy_name} — Sample Trades (3 Best, 3 Worst)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH / f'{plot_num:02d}_{strategy_name.lower().replace(" ", "_")}_samples.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_num:02d}_{strategy_name.lower().replace(' ', '_')}_samples.png")


for i, (name, tdf) in enumerate(all_trades.items()):
    plot_sample_trades(name, tdf, df, plot_num=i + 2)


# =============================================================================
# PLOT 6: TRADE DISTRIBUTION — ALL STRATEGIES
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (name, tdf) in enumerate(all_trades.items()):
    ax = axes[idx]

    winners = tdf[tdf['net_profit_bps'] > 0]['net_profit_bps']
    losers = tdf[tdf['net_profit_bps'] <= 0]['net_profit_bps']

    bins = np.arange(-100, 150, 5)
    ax.hist(winners, bins=bins, color='#2ecc71', alpha=0.7, label=f'Winners ({len(winners)})')
    ax.hist(losers, bins=bins, color='#e74c3c', alpha=0.7, label=f'Losers ({len(losers)})')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=tdf['net_profit_bps'].mean(), color='blue', linewidth=1.5, linestyle='--',
               label=f'Avg: {tdf["net_profit_bps"].mean():+.1f} bps')

    gw = winners.sum() if len(winners) > 0 else 0
    gl = abs(losers.sum()) if len(losers) > 0 else 1
    pf = gw / gl

    ax.set_title(f'{name} | {len(tdf)}t | PF {pf:.2f} | {tdf["net_profit_bps"].sum():+.0f} bps',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Net Profit (bps)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

fig.suptitle('EXP-011: Trade Distribution — All 4 Strategies', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(OUTPUT_PATH / '06_trade_distributions.png', dpi=150)
plt.close()
print("Saved: 06_trade_distributions.png")


# =============================================================================
# PLOT 7: MONTHLY RETURNS HEATMAP — VOLUME SPIKE (best strategy)
# =============================================================================
if 'Vol Spike' in all_trades:
    tdf = all_trades['Vol Spike'].copy()
    tdf['month'] = pd.to_datetime(tdf['entry_time']).dt.to_period('M')
    monthly = tdf.groupby('month').agg(
        trades=('net_profit_bps', 'count'),
        net_bps=('net_profit_bps', 'sum'),
        win_rate=('net_profit_bps', lambda x: (x > 0).mean() * 100)
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    months = [str(m) for m in monthly.index]
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in monthly['net_bps']]

    ax1.bar(months, monthly['net_bps'], color=colors, edgecolor='white', linewidth=0.5)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_title('Volume Spike + Direction — Monthly Net Profit', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Net bps')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.2, axis='y')

    # Add trade count labels
    for i, (m, row) in enumerate(monthly.iterrows()):
        ax1.text(i, row['net_bps'] + 50 * np.sign(row['net_bps']),
                f"{int(row['trades'])}t", ha='center', fontsize=8)

    ax2.bar(months, monthly['win_rate'], color='#3498db', edgecolor='white', linewidth=0.5)
    ax2.axhline(y=50, color='red', linewidth=1, linestyle='--', label='50%')
    ax2.set_title('Volume Spike + Direction — Monthly Win Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Win Rate %')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH / '07_vol_spike_monthly.png', dpi=150)
    plt.close()
    print("Saved: 07_vol_spike_monthly.png")


# =============================================================================
# PLOT 8: LONG vs SHORT COMPARISON — ALL STRATEGIES
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

strat_names = []
long_nets = []
short_nets = []

for name, tdf in all_trades.items():
    lt = tdf[tdf['direction'] == 'LONG']
    st = tdf[tdf['direction'] == 'SHORT']
    strat_names.append(name)
    long_nets.append(lt['net_profit_bps'].sum() if len(lt) > 0 else 0)
    short_nets.append(st['net_profit_bps'].sum() if len(st) > 0 else 0)

x = np.arange(len(strat_names))
width = 0.35

ax1.bar(x - width/2, long_nets, width, label='LONG', color='#2ecc71', edgecolor='white')
ax1.bar(x + width/2, short_nets, width, label='SHORT', color='#e74c3c', edgecolor='white')
ax1.set_xticks(x)
ax1.set_xticklabels(strat_names, rotation=15)
ax1.set_ylabel('Net Profit (bps)')
ax1.set_title('LONG vs SHORT Net Profit', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.2, axis='y')
ax1.axhline(y=0, color='black', linewidth=0.5)

# Trade counts
long_counts = [len(tdf[tdf['direction'] == 'LONG']) for tdf in all_trades.values()]
short_counts = [len(tdf[tdf['direction'] == 'SHORT']) for tdf in all_trades.values()]

ax2.bar(x - width/2, long_counts, width, label='LONG', color='#2ecc71', edgecolor='white')
ax2.bar(x + width/2, short_counts, width, label='SHORT', color='#e74c3c', edgecolor='white')
ax2.set_xticks(x)
ax2.set_xticklabels(strat_names, rotation=15)
ax2.set_ylabel('Trade Count')
ax2.set_title('LONG vs SHORT Trade Counts', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.2, axis='y')

fig.suptitle('EXP-011: LONG vs SHORT by Strategy', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(OUTPUT_PATH / '08_long_vs_short.png', dpi=150)
plt.close()
print("Saved: 08_long_vs_short.png")


print("\n" + "=" * 50)
print("ALL VISUALIZATIONS COMPLETE")
print("=" * 50)
