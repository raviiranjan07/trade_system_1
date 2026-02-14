"""
All Features Extended Horizons (including H=360, H=480)

Run: .venv/Scripts/python.exe scripts/debug/test_all_features_extended.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60, 120, 240, 360, 480, 600]
SAMPLE_SIZE = 100000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ALL FEATURES - EXTENDED HORIZONS (H=3 to H=600)")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

# Calculate features
print("Calculating features...")

# RSI
delta = ohlcv['close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
ohlcv['rsi'] = 100 - (100 / (1 + rs))

# Range Position
range_high = ohlcv['high'].rolling(50).max()
range_low = ohlcv['low'].rolling(50).min()
ohlcv['range_pos'] = (ohlcv['close'] - range_low) / (range_high - range_low)

# EMA
ohlcv['ema50'] = ohlcv['close'].ewm(span=50, adjust=False).mean()

print("Features calculated.")

# Use 2025 data only
test_2025 = ohlcv[ohlcv.index >= "2025-01-01"].copy()
print(f"2025 data: {len(test_2025):,} candles")

close = test_2025['close'].values
rsi = test_2025['rsi'].values
range_pos = test_2025['range_pos'].values
ema = test_2025['ema50'].values
n = len(test_2025)

np.random.seed(42)
max_h = max(HORIZONS)
valid_start = 100
avail = n - max_h - valid_start
sample_idx = np.random.choice(range(valid_start, n - max_h), size=min(SAMPLE_SIZE, avail), replace=False)
print(f"Sample size: {len(sample_idx):,}")

# =============================================================================
# TEST RSI
# =============================================================================
print("\n" + "=" * 80)
print("RSI (Oversold <30, Overbought >70)")
print("=" * 80)

print(f"\n{'H':<6} {'Oversold->UP%':<15} {'Overbought->DOWN%':<18} {'Combined Edge':<15}")
print("-" * 55)

rsi_results = []
for H in HORIZONS:
    oversold_up, oversold_total = 0, 0
    overbought_down, overbought_total = 0, 0

    for i in sample_idx:
        r = rsi[i]
        if np.isnan(r):
            continue
        future = close[i + H]
        went_up = future > close[i]

        if r < 30:
            oversold_total += 1
            if went_up:
                oversold_up += 1
        elif r > 70:
            overbought_total += 1
            if not went_up:
                overbought_down += 1

    if oversold_total > 50 and overbought_total > 50:
        os_pct = 100 * oversold_up / oversold_total
        ob_pct = 100 * overbought_down / overbought_total
        combined = ((os_pct - 50) + (ob_pct - 50)) / 2
        print(f"H={H:<4} {os_pct:<15.1f} {ob_pct:<18.1f} {combined:>+14.2f}%")
        rsi_results.append({'H': H, 'combined': combined})

# =============================================================================
# TEST RANGE POSITION
# =============================================================================
print("\n" + "=" * 80)
print("RANGE POSITION (Near Low <0.2, Near High >0.8)")
print("=" * 80)

print(f"\n{'H':<6} {'Near Low->UP%':<15} {'Near High->DOWN%':<18} {'Combined Edge':<15}")
print("-" * 55)

range_results = []
for H in HORIZONS:
    low_up, low_total = 0, 0
    high_down, high_total = 0, 0

    for i in sample_idx:
        rp = range_pos[i]
        if np.isnan(rp):
            continue
        future = close[i + H]
        went_up = future > close[i]

        if rp < 0.2:
            low_total += 1
            if went_up:
                low_up += 1
        elif rp > 0.8:
            high_total += 1
            if not went_up:
                high_down += 1

    if low_total > 50 and high_total > 50:
        low_pct = 100 * low_up / low_total
        high_pct = 100 * high_down / high_total
        combined = ((low_pct - 50) + (high_pct - 50)) / 2
        print(f"H={H:<4} {low_pct:<15.1f} {high_pct:<18.1f} {combined:>+14.2f}%")
        range_results.append({'H': H, 'combined': combined})

# =============================================================================
# TEST EMA
# =============================================================================
print("\n" + "=" * 80)
print("EMA PROXIMITY (Within 20bp of EMA50)")
print("=" * 80)

print(f"\n{'H':<6} {'Below->UP%':<15} {'Above->DOWN%':<18} {'Combined Edge':<15}")
print("-" * 55)

ema_results = []
near_pct = 20 / 10000
for H in HORIZONS:
    below_up, below_total = 0, 0
    above_down, above_total = 0, 0

    for i in sample_idx:
        e = ema[i]
        if np.isnan(e):
            continue
        price = close[i]
        dist = (price - e) / e
        if abs(dist) > near_pct:
            continue

        future = close[i + H]
        went_up = future > price

        if dist < 0:
            below_total += 1
            if went_up:
                below_up += 1
        elif dist > 0:
            above_total += 1
            if not went_up:
                above_down += 1

    if below_total > 50 and above_total > 50:
        below_pct = 100 * below_up / below_total
        above_pct = 100 * above_down / above_total
        combined = ((below_pct - 50) + (above_pct - 50)) / 2
        print(f"H={H:<4} {below_pct:<15.1f} {above_pct:<18.1f} {combined:>+14.2f}%")
        ema_results.append({'H': H, 'combined': combined})

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY TABLE (2025 Data)")
print("=" * 80)

print(f"\n{'H':<8} {'RSI':<12} {'Range Pos':<12} {'EMA':<12}")
print("-" * 45)

for H in HORIZONS:
    rsi_edge = next((r['combined'] for r in rsi_results if r['H'] == H), None)
    range_edge = next((r['combined'] for r in range_results if r['H'] == H), None)
    ema_edge = next((r['combined'] for r in ema_results if r['H'] == H), None)

    rsi_str = f"{rsi_edge:+.2f}%" if rsi_edge else "N/A"
    range_str = f"{range_edge:+.2f}%" if range_edge else "N/A"
    ema_str = f"{ema_edge:+.2f}%" if ema_edge else "N/A"

    print(f"H={H:<5} {rsi_str:<12} {range_str:<12} {ema_str:<12}")

# =============================================================================
# MARKDOWN TABLE
# =============================================================================
print("\n" + "=" * 80)
print("MARKDOWN TABLE FOR ANALYSIS-18")
print("=" * 80)

print("\n| H | RSI | Range Pos | EMA |")
print("|---|-----|-----------|-----|")
for H in HORIZONS:
    rsi_edge = next((r['combined'] for r in rsi_results if r['H'] == H), None)
    range_edge = next((r['combined'] for r in range_results if r['H'] == H), None)
    ema_edge = next((r['combined'] for r in ema_results if r['H'] == H), None)

    rsi_str = f"{rsi_edge:+.1f}%" if rsi_edge else "N/A"
    range_str = f"{range_edge:+.1f}%" if range_edge else "N/A"
    ema_str = f"{ema_edge:+.1f}%" if ema_edge else "N/A"

    print(f"| H={H} | {rsi_str} | {range_str} | {ema_str} |")
