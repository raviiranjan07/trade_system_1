"""
EXP-001: RSI Signals on Price Action
Shows RSI oversold/overbought signals with actual price movement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path

# Setup paths
DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
PLOTS_PATH = Path("experiments/rsi/EXP-001/plots")
PLOTS_PATH.mkdir(exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
# Index is already 'time' as datetime
df.index = pd.to_datetime(df.index).tz_localize(None)  # Remove timezone for plotting

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate RSI(14)
print("Calculating RSI...")
df['rsi'] = calculate_rsi(df, period=14)

# Detect signals
df['oversold'] = df['rsi'] < 20
df['overbought'] = df['rsi'] > 80
df['oversold_cross'] = (df['oversold'] == True) & (df['oversold'].shift(1) == False)
df['overbought_cross'] = (df['overbought'] == True) & (df['overbought'].shift(1) == False)

def plot_price_with_rsi(data, start_date, end_date, title, filename, horizon=10):
    """
    Plot price action with RSI signals
    """
    # Filter data
    mask = (data.index >= start_date) & (data.index <= end_date)
    plot_data = data[mask].copy()

    if len(plot_data) == 0:
        print(f"No data for {start_date} to {end_date}")
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1], sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # =========================================================================
    # Top: Price Chart
    # =========================================================================
    ax1.plot(plot_data.index, plot_data['close'], color='#2c3e50', linewidth=1.5, label='BTC Price')
    ax1.fill_between(plot_data.index, plot_data['low'], plot_data['high'], alpha=0.3, color='#3498db')

    # Mark oversold signals (expect bounce UP)
    oversold_signals = plot_data[plot_data['oversold_cross'] == True]
    for idx in oversold_signals.index:
        # Entry price
        entry_price = plot_data.loc[idx, 'close']

        # Get next H bars
        idx_pos = plot_data.index.get_loc(idx)
        if idx_pos + horizon < len(plot_data):
            future_data = plot_data.iloc[idx_pos:idx_pos + horizon + 1]
            max_price = future_data['high'].max()
            move_bps = (max_price - entry_price) / entry_price * 10000

            # Plot entry point
            ax1.scatter(idx, entry_price, color='green', s=150, zorder=5, marker='^',
                       edgecolors='white', linewidth=2)

            # Plot horizontal line to show potential move
            ax1.hlines(y=entry_price, xmin=idx, xmax=future_data.index[-1],
                      colors='green', linestyles='--', alpha=0.5)

            # Annotate with move size
            ax1.annotate(f'+{move_bps:.0f}bp', xy=(idx, entry_price),
                        xytext=(10, 20), textcoords='offset points',
                        fontsize=9, color='green', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Mark overbought signals (expect drop DOWN)
    overbought_signals = plot_data[plot_data['overbought_cross'] == True]
    for idx in overbought_signals.index:
        entry_price = plot_data.loc[idx, 'close']

        idx_pos = plot_data.index.get_loc(idx)
        if idx_pos + horizon < len(plot_data):
            future_data = plot_data.iloc[idx_pos:idx_pos + horizon + 1]
            min_price = future_data['low'].min()
            move_bps = (entry_price - min_price) / entry_price * 10000

            # Plot entry point
            ax1.scatter(idx, entry_price, color='red', s=150, zorder=5, marker='v',
                       edgecolors='white', linewidth=2)

            # Plot horizontal line
            ax1.hlines(y=entry_price, xmin=idx, xmax=future_data.index[-1],
                      colors='red', linestyles='--', alpha=0.5)

            # Annotate
            ax1.annotate(f'-{move_bps:.0f}bp', xy=(idx, entry_price),
                        xytext=(10, -25), textcoords='offset points',
                        fontsize=9, color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax1.set_ylabel('BTC Price (USDT)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Format y-axis with comma separator
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # =========================================================================
    # Bottom: RSI Chart
    # =========================================================================
    ax2.plot(plot_data.index, plot_data['rsi'], color='#8e44ad', linewidth=1.5, label='RSI(14)')

    # Oversold/Overbought zones
    ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Overbought (80)')
    ax2.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Oversold (20)')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)

    # Fill zones
    ax2.fill_between(plot_data.index, 80, 100, alpha=0.2, color='red')
    ax2.fill_between(plot_data.index, 0, 20, alpha=0.2, color='green')

    # Mark signal points on RSI
    ax2.scatter(oversold_signals.index, oversold_signals['rsi'],
               color='green', s=100, zorder=5, marker='^', edgecolors='white', linewidth=2)
    ax2.scatter(overbought_signals.index, overbought_signals['rsi'],
               color='red', s=100, zorder=5, marker='v', edgecolors='white', linewidth=2)

    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {filename}")

# =============================================================================
# Create Multiple Sample Charts
# =============================================================================

print("\nGenerating price action charts with RSI signals...")

# Chart 1: In-sample period example (2023)
plot_price_with_rsi(
    df,
    start_date='2023-03-10',
    end_date='2023-03-20',
    title='RSI(14) 20/80 Signals on BTC 15min - March 2023 (In-Sample)\n▲ Oversold Entry (expect bounce) | ▼ Overbought Entry (expect drop)',
    filename='7_price_action_2023_march.png',
    horizon=10
)

# Chart 2: Another in-sample period (2022 - volatile)
plot_price_with_rsi(
    df,
    start_date='2022-06-10',
    end_date='2022-06-20',
    title='RSI(14) 20/80 Signals on BTC 15min - June 2022 (In-Sample, High Volatility)\n▲ Oversold Entry (expect bounce) | ▼ Overbought Entry (expect drop)',
    filename='8_price_action_2022_june.png',
    horizon=10
)

# Chart 3: Out-of-sample period (2024)
plot_price_with_rsi(
    df,
    start_date='2024-03-01',
    end_date='2024-03-10',
    title='RSI(14) 20/80 Signals on BTC 15min - March 2024 (Out-of-Sample)\n▲ Oversold Entry (expect bounce) | ▼ Overbought Entry (expect drop)',
    filename='9_price_action_2024_march.png',
    horizon=10
)

# Chart 4: Recent period (2024 late)
plot_price_with_rsi(
    df,
    start_date='2024-11-01',
    end_date='2024-11-15',
    title='RSI(14) 20/80 Signals on BTC 15min - November 2024 (Out-of-Sample)\n▲ Oversold Entry (expect bounce) | ▼ Overbought Entry (expect drop)',
    filename='10_price_action_2024_november.png',
    horizon=10
)

# Chart 5: Zoomed in view - single day with multiple signals
plot_price_with_rsi(
    df,
    start_date='2023-08-17',
    end_date='2023-08-19',
    title='RSI(14) 20/80 Signals - Zoomed View (2 Days)\n▲ Oversold Entry (expect bounce) | ▼ Overbought Entry (expect drop)',
    filename='11_price_action_zoomed.png',
    horizon=10
)

print("\n" + "="*50)
print("Price action visualizations complete!")
print("="*50)
