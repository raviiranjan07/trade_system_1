"""
Investigation: Why Analysis (77% success) vs Backtest (49% win rate)?

This script runs both methods on the same data to find discrepancies.

Run: python scripts/debug/investigate_ema_discrepancy.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("INVESTIGATION: ANALYSIS vs BACKTEST DISCREPANCY")
print("=" * 80)

# =============================================================================
# LOAD DATA (TEST SET ONLY - 2024-2025)
# =============================================================================
print("\n[1/5] Loading test data (2024-2025)...")
ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)

# Resample to 15-minute
ohlcv = ohlcv_1m.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Filter to test period
test = ohlcv[ohlcv.index >= "2024-01-01"].copy()
print(f"Test data: {len(test):,} candles")

# =============================================================================
# CALCULATE INDICATORS
# =============================================================================
print("\n[2/5] Calculating indicators...")
test['ema7'] = test['close'].ewm(span=7, adjust=False).mean()
test['ema50'] = test['close'].ewm(span=50, adjust=False).mean()
test['ema200'] = test['close'].ewm(span=200, adjust=False).mean()
test['trend'] = np.where(test['ema50'] > test['ema200'], 1, -1)
test['dist_from_ema7'] = abs(test['close'] - test['ema7']) / test['ema7'] * 100
test['touch'] = test['dist_from_ema7'] <= 0.2

touch_count = test['touch'].sum()
print(f"EMA7 touches: {touch_count:,}")

# =============================================================================
# METHOD 1: ANALYSIS METHOD (Like ema_bounce_validation.py)
# =============================================================================
print("\n[3/5] Running ANALYSIS method (MFE >= threshold)...")

@njit
def calculate_forward_mfe_mae(highs, lows, close_prices, horizon):
    n = len(close_prices)
    mfe_long = np.full(n, np.nan)
    mae_long = np.full(n, np.nan)
    mfe_short = np.full(n, np.nan)
    mae_short = np.full(n, np.nan)

    for i in range(n - horizon):
        entry = close_prices[i]
        future_highs = highs[i+1:i+1+horizon]
        future_lows = lows[i+1:i+1+horizon]

        # LONG
        max_high = np.max(future_highs)
        min_low = np.min(future_lows)
        mfe_long[i] = (max_high - entry) / entry * 10000
        mae_long[i] = (min_low - entry) / entry * 10000

        # SHORT
        mfe_short[i] = (entry - min_low) / entry * 10000
        mae_short[i] = (max_high - entry) / entry * 10000

    return mfe_long, mae_long, mfe_short, mae_short

# Calculate MFE/MAE
mfe_long, mae_long, mfe_short, mae_short = calculate_forward_mfe_mae(
    test['high'].values,
    test['low'].values,
    test['close'].values,
    10  # H=10
)

test['mfe_long'] = mfe_long
test['mae_long'] = mae_long
test['mfe_short'] = mfe_short
test['mae_short'] = mae_short

# Filter touches
touches_analysis = test[test['touch']].copy()
touches_analysis = touches_analysis.dropna(subset=['mfe_long', 'mfe_short'])

# Test different thresholds
thresholds = [12, 20]
analysis_results = {}

for threshold in thresholds:
    # UPTREND
    uptrend = touches_analysis[touches_analysis['trend'] == 1]
    uptrend_success = (uptrend['mfe_long'] >= threshold).mean() * 100

    # DOWNTREND
    downtrend = touches_analysis[touches_analysis['trend'] == -1]
    downtrend_success = (downtrend['mfe_short'] >= threshold).mean() * 100

    # OVERALL
    overall_success = (
        ((uptrend['mfe_long'] >= threshold).sum() +
         (downtrend['mfe_short'] >= threshold).sum()) /
        len(touches_analysis) * 100
    )

    analysis_results[threshold] = {
        'method': 'ANALYSIS',
        'threshold': threshold,
        'total_touches': len(touches_analysis),
        'overall_success': overall_success,
        'uptrend_success': uptrend_success,
        'downtrend_success': downtrend_success
    }

    print(f"\nAnalysis with {threshold} bps threshold:")
    print(f"  Total touches: {len(touches_analysis):,}")
    print(f"  Overall success: {overall_success:.1f}%")
    print(f"  Uptrend: {uptrend_success:.1f}%")
    print(f"  Downtrend: {downtrend_success:.1f}%")

# =============================================================================
# METHOD 2: BACKTEST METHOD (TP before SL)
# =============================================================================
print("\n[4/5] Running BACKTEST method (TP before SL or TIME)...")

@njit
def simulate_trade_v1(entry_price, direction, future_highs, future_lows, tp_bps, sl_bps, max_bars):
    """Version 1: Check TP first, then SL (current backtest logic)"""
    for i in range(min(len(future_highs), max_bars)):
        high = future_highs[i]
        low = future_lows[i]

        if direction == 1:  # LONG
            profit_bps = (high - entry_price) / entry_price * 10000
            if profit_bps >= tp_bps:
                return 1  # TP

            loss_bps = (entry_price - low) / entry_price * 10000
            if loss_bps >= sl_bps:
                return 3  # SL
        else:  # SHORT
            profit_bps = (entry_price - low) / entry_price * 10000
            if profit_bps >= tp_bps:
                return 1  # TP

            loss_bps = (high - entry_price) / entry_price * 10000
            if loss_bps >= sl_bps:
                return 3  # SL

    return 2  # TIME

@njit
def simulate_trade_v2(entry_price, direction, future_highs, future_lows, future_opens, tp_bps, sl_bps, max_bars):
    """Version 2: Check which is hit first based on open (more realistic)"""
    for i in range(min(len(future_highs), max_bars)):
        high = future_highs[i]
        low = future_lows[i]
        open_price = future_opens[i] if i < len(future_opens) else entry_price

        if direction == 1:  # LONG
            tp_price = entry_price * (1 + tp_bps / 10000)
            sl_price = entry_price * (1 - sl_bps / 10000)

            # Check if both hit
            tp_hit = high >= tp_price
            sl_hit = low <= sl_price

            if tp_hit and sl_hit:
                # Both hit - check which is closer to open
                dist_to_tp = abs(tp_price - open_price)
                dist_to_sl = abs(sl_price - open_price)
                return 1 if dist_to_tp < dist_to_sl else 3
            elif tp_hit:
                return 1
            elif sl_hit:
                return 3
        else:  # SHORT
            tp_price = entry_price * (1 - tp_bps / 10000)
            sl_price = entry_price * (1 + sl_bps / 10000)

            tp_hit = low <= tp_price
            sl_hit = high >= sl_price

            if tp_hit and sl_hit:
                dist_to_tp = abs(tp_price - open_price)
                dist_to_sl = abs(sl_price - open_price)
                return 1 if dist_to_tp < dist_to_sl else 3
            elif tp_hit:
                return 1
            elif sl_hit:
                return 3

    return 2  # TIME

# Run backtests with different TP/SL levels
touch_indices = test[test['touch']].index.tolist()
backtest_configs = [
    {'tp': 12, 'sl': 12},
    {'tp': 20, 'sl': 20}
]

backtest_results = {}

for config in backtest_configs:
    tp_bps = config['tp']
    sl_bps = config['sl']

    print(f"\nBacktest with TP={tp_bps}, SL={sl_bps}...")

    # Version 1: Current logic (TP checked first)
    wins_v1 = 0
    losses_v1 = 0

    # Version 2: Realistic within-bar order
    wins_v2 = 0
    losses_v2 = 0

    for entry_time in touch_indices:
        entry_idx = test.index.get_loc(entry_time)

        if entry_idx + 10 >= len(test):
            continue

        entry_row = test.iloc[entry_idx]
        entry_price = entry_row['close']
        direction = entry_row['trend']

        future_data = test.iloc[entry_idx + 1 : entry_idx + 11]
        future_highs = future_data['high'].values
        future_lows = future_data['low'].values
        future_opens = future_data['open'].values

        # Version 1
        result_v1 = simulate_trade_v1(entry_price, direction, future_highs, future_lows, tp_bps, sl_bps, 10)
        if result_v1 == 1:
            wins_v1 += 1
        elif result_v1 == 3:
            losses_v1 += 1

        # Version 2
        result_v2 = simulate_trade_v2(entry_price, direction, future_highs, future_lows, future_opens, tp_bps, sl_bps, 10)
        if result_v2 == 1:
            wins_v2 += 1
        elif result_v2 == 3:
            losses_v2 += 1

    total_v1 = wins_v1 + losses_v1
    win_rate_v1 = wins_v1 / total_v1 * 100 if total_v1 > 0 else 0

    total_v2 = wins_v2 + losses_v2
    win_rate_v2 = wins_v2 / total_v2 * 100 if total_v2 > 0 else 0

    backtest_results[tp_bps] = {
        'v1_wins': wins_v1,
        'v1_losses': losses_v1,
        'v1_win_rate': win_rate_v1,
        'v2_wins': wins_v2,
        'v2_losses': losses_v2,
        'v2_win_rate': win_rate_v2
    }

    print(f"  V1 (TP first): {wins_v1:,} wins, {losses_v1:,} losses, {win_rate_v1:.1f}% win rate")
    print(f"  V2 (realistic): {wins_v2:,} wins, {losses_v2:,} losses, {win_rate_v2:.1f}% win rate")

# =============================================================================
# COMPARISON REPORT
# =============================================================================
print("\n" + "=" * 80)
print("INVESTIGATION RESULTS")
print("=" * 80)

print("\n### COMPARISON @ 12 BPS THRESHOLD ###")
print(f"Analysis (MFE >= 12):       {analysis_results[12]['overall_success']:.1f}% success")
print(f"Backtest V1 (TP=12, SL=12): {backtest_results[12]['v1_win_rate']:.1f}% win rate")
print(f"Backtest V2 (TP=12, SL=12): {backtest_results[12]['v2_win_rate']:.1f}% win rate")
print(f"GAP (Analysis - BT V1):     {analysis_results[12]['overall_success'] - backtest_results[12]['v1_win_rate']:.1f} pp")

print("\n### COMPARISON @ 20 BPS THRESHOLD ###")
print(f"Analysis (MFE >= 20):       {analysis_results[20]['overall_success']:.1f}% success")
print(f"Backtest V1 (TP=20, SL=20): {backtest_results[20]['v1_win_rate']:.1f}% win rate")
print(f"Backtest V2 (TP=20, SL=20): {backtest_results[20]['v2_win_rate']:.1f}% win rate")
print(f"GAP (Analysis - BT V1):     {analysis_results[20]['overall_success'] - backtest_results[20]['v1_win_rate']:.1f} pp")

print("\n### KEY FINDINGS ###")

# Finding 1: Threshold difference
gap_12 = analysis_results[12]['overall_success'] - backtest_results[12]['v1_win_rate']
gap_20 = analysis_results[20]['overall_success'] - backtest_results[20]['v1_win_rate']

print(f"\n1. THRESHOLD EFFECT:")
print(f"   - At 12 bps: Analysis {analysis_results[12]['overall_success']:.1f}% vs Backtest {backtest_results[12]['v1_win_rate']:.1f}% = {gap_12:.1f} pp gap")
print(f"   - At 20 bps: Analysis {analysis_results[20]['overall_success']:.1f}% vs Backtest {backtest_results[20]['v1_win_rate']:.1f}% = {gap_20:.1f} pp gap")
print(f"   - Conclusion: {'MAJOR ISSUE' if gap_12 > 10 else 'Minor issue'}")

# Finding 2: Within-bar order
v1_v2_diff_12 = backtest_results[12]['v1_win_rate'] - backtest_results[12]['v2_win_rate']
v1_v2_diff_20 = backtest_results[20]['v1_win_rate'] - backtest_results[20]['v2_win_rate']

print(f"\n2. WITHIN-BAR ORDER EFFECT:")
print(f"   - At 12 bps: V1 {backtest_results[12]['v1_win_rate']:.1f}% vs V2 {backtest_results[12]['v2_win_rate']:.1f}% = {v1_v2_diff_12:+.1f} pp")
print(f"   - At 20 bps: V1 {backtest_results[20]['v1_win_rate']:.1f}% vs V2 {backtest_results[20]['v2_win_rate']:.1f}% = {v1_v2_diff_20:+.1f} pp")
print(f"   - Conclusion: {'MAJOR ISSUE' if abs(v1_v2_diff_20) > 2 else 'Minor issue'}")

# Finding 3: Why the gap exists
print(f"\n3. WHY THE GAP EXISTS:")
print(f"   - Analysis measures: 'Did price REACH threshold at any point?'")
print(f"   - Backtest measures: 'Did TP hit BEFORE SL hit?'")
print(f"   - Many trades reach threshold but reverse before you can capture it")
print(f"   - This is fundamental difference, not a bug")

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
