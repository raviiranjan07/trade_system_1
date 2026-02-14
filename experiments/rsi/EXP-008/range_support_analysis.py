"""
EXP-008: Range Support on 15-minute BTCUSDT
============================================
Test if price near range lows predicts UP movement on 15min bars.

WHAT analysis found 60.4% accuracy on 240min candles.
Does it work on 15min?

Range Position = (close - lowest_low) / (highest_high - lowest_low)
  <10% = near support -> expect UP
  >90% = near resistance -> expect DOWN

Run: python experiments/rsi/EXP-008/range_support_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/ohlcv/BTCUSDT_15m_ohlcv.parquet")

LOOKBACKS = [20, 50, 100, 200]
SUPPORT_LEVELS = [5, 10, 20, 30]       # range position < X%
RESISTANCE_LEVELS = [70, 80, 90, 95]   # range position > X%
HORIZONS = [5, 10, 20]                 # bars ahead to check direction
THRESHOLD_BPS = 12                      # minimum move to count as UP/DOWN (Rule #1)

TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"

print("=" * 70)
print("EXP-008: RANGE SUPPORT ON 15-MINUTE BTCUSDT")
print("=" * 70)

# =============================================================================
# LOAD DATA
# =============================================================================
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index).tz_localize(None)
print(f"Data: {df.index[0].date()} to {df.index[-1].date()} | {len(df)} bars")

# =============================================================================
# CALCULATE RANGE POSITION FOR ALL LOOKBACKS
# =============================================================================
for lb in LOOKBACKS:
    high_roll = df['high'].rolling(lb).max()
    low_roll = df['low'].rolling(lb).min()
    range_size = high_roll - low_roll
    # Avoid division by zero
    range_size = range_size.replace(0, np.nan)
    df[f'range_pos_{lb}'] = ((df['close'] - low_roll) / range_size) * 100

# =============================================================================
# CALCULATE FUTURE DIRECTION FOR ALL HORIZONS
# =============================================================================
for h in HORIZONS:
    # Max high and min low in next h bars
    df[f'future_max_{h}'] = df['high'].shift(-1).rolling(h).max().shift(-h+1)
    df[f'future_min_{h}'] = df['low'].shift(-1).rolling(h).min().shift(-h+1)

    # Did price go UP at least threshold bps?
    df[f'up_first_{h}'] = ((df[f'future_max_{h}'] - df['close']) / df['close'] * 10000) >= THRESHOLD_BPS
    # Did price go DOWN at least threshold bps?
    df[f'down_first_{h}'] = ((df['close'] - df[f'future_min_{h}']) / df['close'] * 10000) >= THRESHOLD_BPS

    # For direction: which hit threshold first?
    # Check bar by bar
    future_closes = []
    for shift in range(1, h + 1):
        future_closes.append(df['close'].shift(-shift))

    # Simple direction: close after h bars vs entry close
    df[f'future_close_{h}'] = df['close'].shift(-h)
    df[f'future_return_{h}'] = (df[f'future_close_{h}'] - df['close']) / df['close'] * 10000  # bps
    df[f'went_up_{h}'] = df[f'future_return_{h}'] > 0

# =============================================================================
# SPLIT DATA
# =============================================================================
train = df[df.index <= TRAIN_END].copy()
test = df[df.index >= TEST_START].copy()
print(f"Train: {len(train)} bars | Test: {len(test)} bars")

# =============================================================================
# ANALYSIS: SUPPORT ZONES (price near lows -> expect UP)
# =============================================================================
print("\n" + "=" * 70)
print("SUPPORT ANALYSIS: Price near range lows -> UP?")
print("=" * 70)

results = []

for period_name, data in [("TRAIN 2020-2023", train), ("TEST 2024-2025", test)]:
    print(f"\n{'='*50}")
    print(f"  {period_name}")
    print(f"{'='*50}")
    print(f"{'LB':>4} {'Level':>6} {'H':>4} {'Signals':>8} {'UP%':>7} {'Edge':>7} {'AvgRet':>8}")
    print("-" * 50)

    for lb in LOOKBACKS:
        for level in SUPPORT_LEVELS:
            for h in HORIZONS:
                col = f'range_pos_{lb}'
                mask = data[col] < level
                subset = data[mask].dropna(subset=[f'went_up_{h}'])

                if len(subset) < 30:
                    continue

                up_pct = subset[f'went_up_{h}'].mean() * 100
                edge = up_pct - 50.0
                avg_ret = subset[f'future_return_{h}'].mean()

                results.append({
                    'period': period_name,
                    'lookback': lb,
                    'level': f'<{level}%',
                    'horizon': h,
                    'signals': len(subset),
                    'up_pct': up_pct,
                    'edge': edge,
                    'avg_return_bps': avg_ret,
                    'type': 'support'
                })

                if edge > 2.0:
                    print(f"{lb:>4} {f'<{level}%':>6} {h:>4} {len(subset):>8} {up_pct:>6.1f}% {edge:>+6.1f}% {avg_ret:>+7.1f}")

# =============================================================================
# ANALYSIS: RESISTANCE ZONES (price near highs -> DOWN?)
# =============================================================================
print("\n" + "=" * 70)
print("RESISTANCE ANALYSIS: Price near range highs -> DOWN?")
print("=" * 70)

for period_name, data in [("TRAIN 2020-2023", train), ("TEST 2024-2025", test)]:
    print(f"\n{'='*50}")
    print(f"  {period_name}")
    print(f"{'='*50}")
    print(f"{'LB':>4} {'Level':>6} {'H':>4} {'Signals':>8} {'DOWN%':>7} {'Edge':>7} {'AvgRet':>8}")
    print("-" * 50)

    for lb in LOOKBACKS:
        for level in RESISTANCE_LEVELS:
            for h in HORIZONS:
                col = f'range_pos_{lb}'
                mask = data[col] > level
                subset = data[mask].dropna(subset=[f'went_up_{h}'])

                if len(subset) < 30:
                    continue

                down_pct = (1 - subset[f'went_up_{h}'].mean()) * 100
                edge = down_pct - 50.0
                avg_ret = subset[f'future_return_{h}'].mean()  # negative = went down

                results.append({
                    'period': period_name,
                    'lookback': lb,
                    'level': f'>{level}%',
                    'horizon': h,
                    'signals': len(subset),
                    'down_pct': down_pct,
                    'edge': edge,
                    'avg_return_bps': avg_ret,
                    'type': 'resistance'
                })

                if edge > 2.0:
                    print(f"{lb:>4} {f'>{level}%':>6} {h:>4} {len(subset):>8} {down_pct:>6.1f}% {edge:>+6.1f}% {avg_ret:>+7.1f}")

# =============================================================================
# TOP 10 SUPPORT SETUPS (TRAIN)
# =============================================================================
print("\n" + "=" * 70)
print("TOP 10 SUPPORT SETUPS (TRAIN)")
print("=" * 70)

support_train = [r for r in results if r['type'] == 'support' and r['period'] == 'TRAIN 2020-2023']
support_train.sort(key=lambda x: x['edge'], reverse=True)

print(f"{'Rank':>4} {'LB':>4} {'Level':>6} {'H':>4} {'Signals':>8} {'UP%':>7} {'Edge':>7} {'AvgRet':>8}")
print("-" * 55)
for i, r in enumerate(support_train[:10]):
    print(f"{i+1:>4} {r['lookback']:>4} {r['level']:>6} {r['horizon']:>4} {r['signals']:>8} {r['up_pct']:>6.1f}% {r['edge']:>+6.1f}% {r['avg_return_bps']:>+7.1f}")

# =============================================================================
# TOP 10 RESISTANCE SETUPS (TRAIN)
# =============================================================================
print("\n" + "=" * 70)
print("TOP 10 RESISTANCE SETUPS (TRAIN)")
print("=" * 70)

resist_train = [r for r in results if r['type'] == 'resistance' and r['period'] == 'TRAIN 2020-2023']
resist_train.sort(key=lambda x: x['edge'], reverse=True)

print(f"{'Rank':>4} {'LB':>4} {'Level':>6} {'H':>4} {'Signals':>8} {'DOWN%':>7} {'Edge':>7} {'AvgRet':>8}")
print("-" * 55)
for i, r in enumerate(resist_train[:10]):
    print(f"{i+1:>4} {r['lookback']:>4} {r['level']:>6} {r['horizon']:>4} {r['signals']:>8} {r.get('down_pct', 0):>6.1f}% {r['edge']:>+6.1f}% {r['avg_return_bps']:>+7.1f}")

# =============================================================================
# OOS VALIDATION: Do top train setups hold in test?
# =============================================================================
print("\n" + "=" * 70)
print("OOS VALIDATION: Top 5 Support setups")
print("=" * 70)

print(f"{'LB':>4} {'Level':>6} {'H':>4} {'Train UP%':>10} {'Test UP%':>10} {'Train Edge':>11} {'Test Edge':>11} {'Valid?':>7}")
print("-" * 70)

for r_train in support_train[:5]:
    # Find matching test result
    r_test = None
    for r in results:
        if (r['type'] == 'support' and
            r['period'] == 'TEST 2024-2025' and
            r['lookback'] == r_train['lookback'] and
            r['level'] == r_train['level'] and
            r['horizon'] == r_train['horizon']):
            r_test = r
            break

    if r_test:
        valid = "YES" if r_test['edge'] > 0 else "NO"
        print(f"{r_train['lookback']:>4} {r_train['level']:>6} {r_train['horizon']:>4} "
              f"{r_train['up_pct']:>9.1f}% {r_test['up_pct']:>9.1f}% "
              f"{r_train['edge']:>+10.1f}% {r_test['edge']:>+10.1f}% {valid:>7}")

print("\n" + "=" * 70)
print("OOS VALIDATION: Top 5 Resistance setups")
print("=" * 70)

print(f"{'LB':>4} {'Level':>6} {'H':>4} {'Train DN%':>10} {'Test DN%':>10} {'Train Edge':>11} {'Test Edge':>11} {'Valid?':>7}")
print("-" * 70)

for r_train in resist_train[:5]:
    r_test = None
    for r in results:
        if (r['type'] == 'resistance' and
            r['period'] == 'TEST 2024-2025' and
            r['lookback'] == r_train['lookback'] and
            r['level'] == r_train['level'] and
            r['horizon'] == r_train['horizon']):
            r_test = r
            break

    if r_test:
        valid = "YES" if r_test['edge'] > 0 else "NO"
        print(f"{r_train['lookback']:>4} {r_train['level']:>6} {r_train['horizon']:>4} "
              f"{r_train.get('down_pct', 0):>9.1f}% {r_test.get('down_pct', 0):>9.1f}% "
              f"{r_train['edge']:>+10.1f}% {r_test['edge']:>+10.1f}% {valid:>7}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Count how many setups have edge > 2% in both train and test
valid_support = 0
valid_resist = 0
for r_train in support_train:
    if r_train['edge'] <= 2.0:
        continue
    for r in results:
        if (r['type'] == 'support' and
            r['period'] == 'TEST 2024-2025' and
            r['lookback'] == r_train['lookback'] and
            r['level'] == r_train['level'] and
            r['horizon'] == r_train['horizon'] and
            r['edge'] > 2.0):
            valid_support += 1

for r_train in resist_train:
    if r_train['edge'] <= 2.0:
        continue
    for r in results:
        if (r['type'] == 'resistance' and
            r['period'] == 'TEST 2024-2025' and
            r['lookback'] == r_train['lookback'] and
            r['level'] == r_train['level'] and
            r['horizon'] == r_train['horizon'] and
            r['edge'] > 2.0):
            valid_resist += 1

print(f"Support setups with >2% edge in BOTH train and test: {valid_support}")
print(f"Resistance setups with >2% edge in BOTH train and test: {valid_resist}")

# Best overall
if support_train and support_train[0]['edge'] > 0:
    best = support_train[0]
    print(f"\nBest Support: LB{best['lookback']} {best['level']} H={best['horizon']} "
          f"-> {best['up_pct']:.1f}% UP ({best['edge']:+.1f}% edge)")

if resist_train and resist_train[0]['edge'] > 0:
    best = resist_train[0]
    print(f"Best Resistance: LB{best['lookback']} {best['level']} H={best['horizon']} "
          f"-> {best.get('down_pct', 0):.1f}% DOWN ({best['edge']:+.1f}% edge)")
