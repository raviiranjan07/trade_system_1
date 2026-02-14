"""
EXP-001 (MA): Price Action Visualizations with Moving Averages
Shows MA crossover signals on actual price charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Setup paths
DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")
PLOTS_PATH = Path("experiments/ma/EXP-001/plots")
PLOTS_PATH.mkdir(exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)

def calculate_sma(data, period):
    return data['close'].rolling(window=period).mean()

def calculate_ema(data, period):
    return data['close'].ewm(span=period, adjust=False).mean()

def calculate_wma(data, period):
    weights = np.arange(1, period + 1)
    return data['close'].rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

# Calculate MAs
print("Calculating Moving Averages...")
df['SMA_20'] = calculate_sma(df, 20)
df['SMA_50'] = calculate_sma(df, 50)
df['EMA_20'] = calculate_ema(df, 20)
df['EMA_50'] = calculate_ema(df, 50)

# Detect crossovers for SMA(50)
df['above_sma50'] = df['close'] > df['SMA_50']
df['cross_above_sma50'] = (df['above_sma50'] == True) & (df['above_sma50'].shift(1) == False)
df['cross_below_sma50'] = (df['above_sma50'] == False) & (df['above_sma50'].shift(1) == True)

plt.style.use('seaborn-v0_8-whitegrid')

def plot_price_with_ma(data, start_date, end_date, title, filename, horizon=10):
    """
    Plot price action with Moving Averages and crossover signals
    """
    # Filter data
    mask = (data.index >= start_date) & (data.index <= end_date)
    plot_data = data[mask].copy()

    if len(plot_data) == 0:
        print(f"No data for {start_date} to {end_date}")
        return

    # Create figure - LARGER size for better visibility
    fig, ax = plt.subplots(figsize=(20, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Plot price
    ax.plot(plot_data.index, plot_data['close'], color='#2c3e50', linewidth=1.5, label='BTC Price', zorder=2)
    ax.fill_between(plot_data.index, plot_data['low'], plot_data['high'], alpha=0.2, color='#3498db')

    # Plot MAs
    ax.plot(plot_data.index, plot_data['SMA_20'], color='#e74c3c', linewidth=1.5, label='SMA(20)', alpha=0.8)
    ax.plot(plot_data.index, plot_data['SMA_50'], color='#2ecc71', linewidth=2, label='SMA(50)', alpha=0.9)
    ax.plot(plot_data.index, plot_data['EMA_50'], color='#9b59b6', linewidth=1.5, label='EMA(50)', linestyle='--', alpha=0.8)

    # Mark cross above signals (bullish)
    cross_above = plot_data[plot_data['cross_above_sma50'] == True]
    for idx in cross_above.index:
        entry_price = plot_data.loc[idx, 'close']
        idx_pos = plot_data.index.get_loc(idx)

        if idx_pos + horizon < len(plot_data):
            future_data = plot_data.iloc[idx_pos + 1 : idx_pos + 1 + horizon]
            max_price = future_data['high'].max()
            move_bps = (max_price - entry_price) / entry_price * 10000

            # Plot signal
            ax.scatter(idx, entry_price, color='green', s=150, zorder=5, marker='^',
                      edgecolors='white', linewidth=2)
            ax.annotate(f'+{move_bps:.0f}bp', xy=(idx, entry_price),
                       xytext=(10, 15), textcoords='offset points',
                       fontsize=9, color='green', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Mark cross below signals (bearish)
    cross_below = plot_data[plot_data['cross_below_sma50'] == True]
    for idx in cross_below.index:
        entry_price = plot_data.loc[idx, 'close']
        idx_pos = plot_data.index.get_loc(idx)

        if idx_pos + horizon < len(plot_data):
            future_data = plot_data.iloc[idx_pos + 1 : idx_pos + 1 + horizon]
            min_price = future_data['low'].min()
            move_bps = (entry_price - min_price) / entry_price * 10000

            # Plot signal
            ax.scatter(idx, entry_price, color='red', s=150, zorder=5, marker='v',
                      edgecolors='white', linewidth=2)
            ax.annotate(f'-{move_bps:.0f}bp', xy=(idx, entry_price),
                       xytext=(10, -20), textcoords='offset points',
                       fontsize=9, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_ylabel('BTC Price (USDT)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {filename}")

def plot_ma_comparison(data, start_date, end_date, title, filename):
    """
    Plot comparing SMA vs EMA on same chart
    """
    mask = (data.index >= start_date) & (data.index <= end_date)
    plot_data = data[mask].copy()

    if len(plot_data) == 0:
        print(f"No data for {start_date} to {end_date}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Top: Price with SMA
    ax = axes[0]
    ax.plot(plot_data.index, plot_data['close'], color='#2c3e50', linewidth=1.5, label='BTC Price')
    ax.plot(plot_data.index, plot_data['SMA_20'], color='#e74c3c', linewidth=1.5, label='SMA(20)')
    ax.plot(plot_data.index, plot_data['SMA_50'], color='#2ecc71', linewidth=2, label='SMA(50)')
    ax.fill_between(plot_data.index, plot_data['low'], plot_data['high'], alpha=0.2, color='#3498db')
    ax.set_ylabel('Price (USDT)', fontsize=11)
    ax.set_title('Simple Moving Average (SMA)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Bottom: Price with EMA
    ax = axes[1]
    ax.plot(plot_data.index, plot_data['close'], color='#2c3e50', linewidth=1.5, label='BTC Price')
    ax.plot(plot_data.index, plot_data['EMA_20'], color='#e74c3c', linewidth=1.5, label='EMA(20)', linestyle='--')
    ax.plot(plot_data.index, plot_data['EMA_50'], color='#9b59b6', linewidth=2, label='EMA(50)', linestyle='--')
    ax.fill_between(plot_data.index, plot_data['low'], plot_data['high'], alpha=0.2, color='#3498db')
    ax.set_ylabel('Price (USDT)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_title('Exponential Moving Average (EMA)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Created: {filename}")

# =============================================================================
# Create Charts
# =============================================================================

print("\nGenerating price action charts with MA signals...")

# Chart 1: In-sample with MA signals (5 days for clarity)
plot_price_with_ma(
    df,
    start_date='2023-03-05',
    end_date='2023-03-10',
    title='SMA(50) Crossover Signals on BTC 15min - March 2023 (In-Sample)\n▲ Cross Above (bullish) | ▼ Cross Below (bearish)',
    filename='6_price_action_2023_march.png',
    horizon=10
)

# Chart 2: Volatile period (2022 crash) - 5 days
plot_price_with_ma(
    df,
    start_date='2022-06-12',
    end_date='2022-06-17',
    title='SMA(50) Crossover Signals on BTC 15min - June 2022 (In-Sample, Crash)\n▲ Cross Above (bullish) | ▼ Cross Below (bearish)',
    filename='7_price_action_2022_june.png',
    horizon=10
)

# Chart 3: Out-of-sample (2024) - 5 days
plot_price_with_ma(
    df,
    start_date='2024-03-05',
    end_date='2024-03-10',
    title='SMA(50) Crossover Signals on BTC 15min - March 2024 (Out-of-Sample)\n▲ Cross Above (bullish) | ▼ Cross Below (bearish)',
    filename='8_price_action_2024_march.png',
    horizon=10
)

# Chart 4: Recent (November 2024) - 5 days
plot_price_with_ma(
    df,
    start_date='2024-11-05',
    end_date='2024-11-10',
    title='SMA(50) Crossover Signals on BTC 15min - November 2024 (Out-of-Sample)\n▲ Cross Above (bullish) | ▼ Cross Below (bearish)',
    filename='9_price_action_2024_november.png',
    horizon=10
)

# Chart 5: SMA vs EMA comparison - 7 days
plot_ma_comparison(
    df,
    start_date='2023-06-05',
    end_date='2023-06-12',
    title='SMA vs EMA Comparison - June 2023\nNotice how EMA reacts faster to price changes',
    filename='10_sma_vs_ema_comparison.png'
)

# Chart 6: Another SMA vs EMA comparison (trending market) - 7 days
plot_ma_comparison(
    df,
    start_date='2024-02-25',
    end_date='2024-03-03',
    title='SMA vs EMA Comparison - Feb-Mar 2024 (Strong Uptrend)\nBoth MAs follow price during strong trends',
    filename='11_sma_vs_ema_trending.png'
)

print("\n" + "="*50)
print("Price action visualizations complete!")
print("="*50)
