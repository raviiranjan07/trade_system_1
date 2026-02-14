"""
Test EMA Bounce Hypothesis

Hypothesis:
- Price approaching EMA from BELOW → EMA acts as RESISTANCE → Price more likely to go DOWN
- Price approaching EMA from ABOVE → EMA acts as SUPPORT → Price more likely to go UP

If this is true, we can use "price near EMA" as an entry signal.

Run: .venv/Scripts/python.exe scripts/debug/test_ema_bounce.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
EMA_PERIODS = [25, 50, 100, 200]  # EMAs to test
NEAR_THRESHOLD_BPS = [5, 10, 15, 20, 30]  # How close is "near" EMA (in bps)
HORIZONS = [3, 5, 10, 15, 30]
TARGET_BPS = 12  # Rule #1: minimum profitable move

TRAIN_END = "2023-12-31"
SAMPLE_SIZE = 200000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("TEST: EMA Support/Resistance Hypothesis")
print("=" * 70)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END].copy()
print(f"TRAIN: {len(train_ohlcv):,} candles (up to {TRAIN_END})")

# Calculate EMAs
for period in EMA_PERIODS:
    train_ohlcv[f'ema{period}'] = train_ohlcv['close'].ewm(span=period, adjust=False).mean()

close = train_ohlcv['close'].values
high = train_ohlcv['high'].values
low = train_ohlcv['low'].values
n = len(train_ohlcv)

# Sample for speed
np.random.seed(42)
max_h = max(HORIZONS)
max_ema = max(EMA_PERIODS)
valid_start = max_ema + 10  # Ensure EMAs are stable
sample_idx = np.random.choice(
    range(valid_start, n - max_h),
    size=min(SAMPLE_SIZE, n - max_h - valid_start),
    replace=False
)
print(f"Sampling {len(sample_idx):,} bars...")

target_pct = TARGET_BPS / 10000

# =============================================================================
# TEST 1: Does "price near EMA" predict direction?
# =============================================================================
print("\n" + "=" * 70)
print(f"TEST 1: When price is NEAR EMA, which direction wins? (Target: {TARGET_BPS}bp)")
print("=" * 70)

results = []

for ema_period in EMA_PERIODS:
    ema_col = f'ema{ema_period}'
    ema_values = train_ohlcv[ema_col].values

    for near_bps in NEAR_THRESHOLD_BPS:
        near_pct = near_bps / 10000

        for H in HORIZONS:
            # Count outcomes when price is near EMA
            near_from_below_up = 0
            near_from_below_down = 0
            near_from_above_up = 0
            near_from_above_down = 0

            for i in sample_idx:
                price = close[i]
                ema = ema_values[i]

                # Calculate distance from EMA
                distance_pct = (price - ema) / ema

                # Check if price is "near" EMA
                if abs(distance_pct) > near_pct:
                    continue  # Not near EMA

                # Determine if approaching from below or above
                from_below = distance_pct < 0  # Price is below EMA
                from_above = distance_pct > 0  # Price is above EMA

                # Check which direction hits first
                up_target = price * (1 + target_pct)
                down_target = price * (1 - target_pct)

                hit_up_bar = None
                hit_down_bar = None

                for j in range(i+1, min(i+1+H, n)):
                    if hit_up_bar is None and high[j] >= up_target:
                        hit_up_bar = j - i
                    if hit_down_bar is None and low[j] <= down_target:
                        hit_down_bar = j - i
                    if hit_up_bar and hit_down_bar:
                        break

                # Determine winner
                if hit_up_bar is None and hit_down_bar is None:
                    continue  # Neither hit (noise)
                elif hit_up_bar is None:
                    went_down = True
                    went_up = False
                elif hit_down_bar is None:
                    went_up = True
                    went_down = False
                elif hit_up_bar < hit_down_bar:
                    went_up = True
                    went_down = False
                else:
                    went_down = True
                    went_up = False

                # Count by approach direction
                if from_below:
                    if went_up:
                        near_from_below_up += 1
                    else:
                        near_from_below_down += 1
                elif from_above:
                    if went_up:
                        near_from_above_up += 1
                    else:
                        near_from_above_down += 1

            # Calculate percentages
            total_below = near_from_below_up + near_from_below_down
            total_above = near_from_above_up + near_from_above_down

            if total_below > 100 and total_above > 100:  # Minimum sample
                below_up_pct = 100 * near_from_below_up / total_below
                above_up_pct = 100 * near_from_above_up / total_above

                results.append({
                    'ema': ema_period,
                    'near_bps': near_bps,
                    'horizon': H,
                    'from_below_count': total_below,
                    'from_below_up_pct': below_up_pct,
                    'from_above_count': total_above,
                    'from_above_up_pct': above_up_pct,
                })

# =============================================================================
# DISPLAY RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS: Direction when price is NEAR EMA")
print("=" * 70)

print(f"\nHypothesis:")
print(f"  - From BELOW EMA: Expect DOWN first (EMA = resistance) → UP% < 50%")
print(f"  - From ABOVE EMA: Expect UP first (EMA = support) → UP% > 50%")

print(f"\n{'EMA':<6} {'Near':<6} {'H':<4} {'From Below':<20} {'From Above':<20} {'Edge?'}")
print(f"{'':6} {'(bps)':<6} {'':4} {'Count':<8} {'UP%':<10} {'Count':<8} {'UP%':<10}")
print("-" * 80)

for r in results:
    # Check if hypothesis holds
    # From below: expect DOWN (UP% < 50%)
    # From above: expect UP (UP% > 50%)
    below_edge = r['from_below_up_pct'] < 48  # Significantly below 50%
    above_edge = r['from_above_up_pct'] > 52  # Significantly above 50%

    if below_edge and above_edge:
        edge = "YES"
    elif below_edge or above_edge:
        edge = "partial"
    else:
        edge = "no"

    print(f"EMA{r['ema']:<3} {r['near_bps']:<6} H={r['horizon']:<2} "
          f"{r['from_below_count']:<8} {r['from_below_up_pct']:<10.1f} "
          f"{r['from_above_count']:<8} {r['from_above_up_pct']:<10.1f} {edge}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Find best cases
best_below = min(results, key=lambda x: x['from_below_up_pct'])
best_above = max(results, key=lambda x: x['from_above_up_pct'])

print(f"\nBest 'From Below' case (lowest UP%, expect resistance):")
print(f"  EMA{best_below['ema']}, Near={best_below['near_bps']}bp, H={best_below['horizon']}")
print(f"  UP first: {best_below['from_below_up_pct']:.1f}% (want < 50%)")
print(f"  Count: {best_below['from_below_count']}")

print(f"\nBest 'From Above' case (highest UP%, expect support):")
print(f"  EMA{best_above['ema']}, Near={best_above['near_bps']}bp, H={best_above['horizon']}")
print(f"  UP first: {best_above['from_above_up_pct']:.1f}% (want > 50%)")
print(f"  Count: {best_above['from_above_count']}")

# Overall verdict
avg_below_up = np.mean([r['from_below_up_pct'] for r in results])
avg_above_up = np.mean([r['from_above_up_pct'] for r in results])

print(f"\nOverall averages:")
print(f"  From Below EMA → UP first: {avg_below_up:.1f}% (hypothesis: < 50%)")
print(f"  From Above EMA → UP first: {avg_above_up:.1f}% (hypothesis: > 50%)")

if avg_below_up < 48 and avg_above_up > 52:
    print(f"\n*** HYPOTHESIS SUPPORTED: EMA acts as support/resistance ***")
elif avg_below_up < 49 or avg_above_up > 51:
    print(f"\n*** WEAK EVIDENCE: Small effect, may not overcome fees ***")
else:
    print(f"\n*** HYPOTHESIS NOT SUPPORTED: EMA does not predict direction ***")
