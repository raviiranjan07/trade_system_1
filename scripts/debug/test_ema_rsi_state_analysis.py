"""
Historical State Analysis: EMA9, EMA50, RSI(17)

For each state combination, count what happened next (UP or DOWN).
This is descriptive statistics, not prediction.

States:
- Price vs EMA9: above/below
- Price vs EMA50: above/below
- EMA9 vs EMA50: uptrend/downtrend
- Trend magnitude: strong/weak (EMA separation)
- RSI(17): oversold/neutral/overbought

Direction:
- UP = next candle close > current close
- DOWN = next candle close < current close
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
print("=" * 80)
print("EMA + RSI STATE HISTORICAL ANALYSIS")
print("=" * 80)

# Load 1-min data
data_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
df_1m = pd.read_parquet(data_path)
df_1m = df_1m.sort_index()
print(f"\nLoaded {len(df_1m):,} 1-minute candles")

# Resample to 15-min
df = df_1m.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()
print(f"Resampled to {len(df):,} 15-minute candles")

# =============================================================================
# CALCULATE FEATURES
# =============================================================================
print("\nCalculating features...")

# EMAs
df['ema9'] = df['close'].ewm(span=9).mean()
df['ema50'] = df['close'].ewm(span=50).mean()

# RSI(17)
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(17).mean()
loss = (-delta.where(delta < 0, 0)).rolling(17).mean()
rs = gain / (loss + 1e-9)
df['rsi17'] = 100 - (100 / (1 + rs))

# EMA separation (magnitude) in bps
df['ema_separation_bps'] = ((df['ema9'] - df['ema50']) / df['close']) * 10000

# Price distance from EMAs in bps
df['price_to_ema9_bps'] = ((df['close'] - df['ema9']) / df['close']) * 10000
df['price_to_ema50_bps'] = ((df['close'] - df['ema50']) / df['close']) * 10000

# =============================================================================
# DEFINE STATES
# =============================================================================
print("Defining states...")

# Price vs EMA9
df['price_vs_ema9'] = np.where(df['close'] > df['ema9'], 'above', 'below')

# Price vs EMA50
df['price_vs_ema50'] = np.where(df['close'] > df['ema50'], 'above', 'below')

# EMA trend (EMA9 vs EMA50)
df['ema_trend'] = np.where(df['ema9'] > df['ema50'], 'uptrend', 'downtrend')

# Trend magnitude (based on EMA separation)
# Strong = |separation| > 50bps, Weak = |separation| <= 50bps
df['trend_magnitude'] = np.where(abs(df['ema_separation_bps']) > 50, 'strong', 'weak')

# RSI state
df['rsi_state'] = pd.cut(df['rsi17'],
                          bins=[0, 30, 70, 100],
                          labels=['oversold', 'neutral', 'overbought'])

# =============================================================================
# CALCULATE DIRECTION (what happened next)
# =============================================================================
print("Calculating direction (next candle)...")

# Direction = next close vs current close
df['next_close'] = df['close'].shift(-1)
df['direction'] = np.where(df['next_close'] > df['close'], 'UP', 'DOWN')

# Drop NaN
df = df.dropna()

# =============================================================================
# SPLIT DATA
# =============================================================================
train_end = '2023-12-31'
df_train = df[df.index <= train_end].copy()
df_test = df[df.index > train_end].copy()

print(f"\nTrain data (2020-2023): {len(df_train):,} candles")
print(f"Test data (2024-2025): {len(df_test):,} candles")

# =============================================================================
# ANALYSIS 1: Basic EMA Position
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 1: Price Position vs EMAs")
print("=" * 80)

def analyze_state(data, group_cols, title):
    """Analyze UP/DOWN % for each state group."""
    grouped = data.groupby(group_cols)['direction'].value_counts().unstack(fill_value=0)
    grouped['total'] = grouped.sum(axis=1)
    grouped['UP%'] = (grouped.get('UP', 0) / grouped['total'] * 100).round(1)
    grouped['DOWN%'] = (grouped.get('DOWN', 0) / grouped['total'] * 100).round(1)
    grouped['edge'] = (grouped['UP%'] - 50).round(1)

    print(f"\n{title}")
    print("-" * 70)
    print(grouped[['total', 'UP%', 'DOWN%', 'edge']].to_string())
    return grouped

# Price vs EMA9 only
analyze_state(df_train, ['price_vs_ema9'], "Price vs EMA9 (Train)")
analyze_state(df_test, ['price_vs_ema9'], "Price vs EMA9 (Test/OOS)")

# Price vs EMA50 only
analyze_state(df_train, ['price_vs_ema50'], "Price vs EMA50 (Train)")
analyze_state(df_test, ['price_vs_ema50'], "Price vs EMA50 (Test/OOS)")

# =============================================================================
# ANALYSIS 2: Combined EMA Position
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 2: Combined EMA Position")
print("=" * 80)

analyze_state(df_train, ['price_vs_ema9', 'price_vs_ema50'], "Price vs Both EMAs (Train)")
analyze_state(df_test, ['price_vs_ema9', 'price_vs_ema50'], "Price vs Both EMAs (Test/OOS)")

# =============================================================================
# ANALYSIS 3: EMA Trend + Magnitude
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 3: EMA Trend + Magnitude")
print("=" * 80)

analyze_state(df_train, ['ema_trend', 'trend_magnitude'], "EMA Trend + Magnitude (Train)")
analyze_state(df_test, ['ema_trend', 'trend_magnitude'], "EMA Trend + Magnitude (Test/OOS)")

# =============================================================================
# ANALYSIS 4: RSI State
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 4: RSI(17) State")
print("=" * 80)

analyze_state(df_train, ['rsi_state'], "RSI(17) State (Train)")
analyze_state(df_test, ['rsi_state'], "RSI(17) State (Test/OOS)")

# =============================================================================
# ANALYSIS 5: Full State (EMA + RSI)
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 5: Full State (EMA Position + Trend + RSI)")
print("=" * 80)

analyze_state(df_train, ['price_vs_ema9', 'price_vs_ema50', 'ema_trend', 'rsi_state'],
              "Full State (Train)")
analyze_state(df_test, ['price_vs_ema9', 'price_vs_ema50', 'ema_trend', 'rsi_state'],
              "Full State (Test/OOS)")

# =============================================================================
# ANALYSIS 6: Best/Worst States
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 6: States with Strongest Edge (Train)")
print("=" * 80)

full_train = df_train.groupby(['price_vs_ema9', 'price_vs_ema50', 'ema_trend', 'trend_magnitude', 'rsi_state'])['direction'].value_counts().unstack(fill_value=0)
full_train['total'] = full_train.sum(axis=1)
full_train['UP%'] = (full_train.get('UP', 0) / full_train['total'] * 100).round(1)
full_train['edge'] = (full_train['UP%'] - 50).round(1)

# Filter for meaningful sample size
full_train_filtered = full_train[full_train['total'] >= 1000].copy()

print("\nTop 10 States with HIGHEST UP% (bullish states):")
print("-" * 80)
top_bullish = full_train_filtered.nlargest(10, 'UP%')[['total', 'UP%', 'edge']]
print(top_bullish.to_string())

print("\nTop 10 States with LOWEST UP% (bearish states):")
print("-" * 80)
top_bearish = full_train_filtered.nsmallest(10, 'UP%')[['total', 'UP%', 'edge']]
print(top_bearish.to_string())

# =============================================================================
# ANALYSIS 7: Validate Best States on Test Data
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 7: Validate Best States on Test Data (OOS)")
print("=" * 80)

full_test = df_test.groupby(['price_vs_ema9', 'price_vs_ema50', 'ema_trend', 'trend_magnitude', 'rsi_state'])['direction'].value_counts().unstack(fill_value=0)
full_test['total'] = full_test.sum(axis=1)
full_test['UP%'] = (full_test.get('UP', 0) / full_test['total'] * 100).round(1)
full_test['edge'] = (full_test['UP%'] - 50).round(1)

# Get top bullish states from train
top_bullish_states = top_bullish.index.tolist()

print("\nTop Bullish States from Train - How they performed in Test:")
print("-" * 80)
for state in top_bullish_states:
    if state in full_test.index:
        train_up = top_bullish.loc[state, 'UP%']
        test_up = full_test.loc[state, 'UP%']
        test_count = full_test.loc[state, 'total']
        print(f"{state}")
        print(f"  Train UP%: {train_up}%  |  Test UP%: {test_up}%  |  Test samples: {test_count}")
    else:
        print(f"{state} - NOT FOUND in test data")

# Get top bearish states from train
top_bearish_states = top_bearish.index.tolist()

print("\nTop Bearish States from Train - How they performed in Test:")
print("-" * 80)
for state in top_bearish_states:
    if state in full_test.index:
        train_up = top_bearish.loc[state, 'UP%']
        test_up = full_test.loc[state, 'UP%']
        test_count = full_test.loc[state, 'total']
        print(f"{state}")
        print(f"  Train UP%: {train_up}%  |  Test UP%: {test_up}%  |  Test samples: {test_count}")
    else:
        print(f"{state} - NOT FOUND in test data")

print("\n" + "=" * 80)
print("DONE!")
print("=" * 80)
