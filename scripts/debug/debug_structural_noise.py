"""
Debug script to quantify structural noise and compute expansion rates on tradeable bars only.

Run: .venv/Scripts/python.exe debug_structural_noise.py

KEY INSIGHT:
- Structural noise = bars where max_move < MWNM (price can't move enough to profit)
- These bars should be REMOVED before any expansion analysis
- Current expansion rate (~28%) is DILUTED by noise
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
MWNM_BPS = 15  # Minimum Worthwhile Net Move (fees + slippage + buffer)
MWNM_PCT = MWNM_BPS / 10000  # Convert to percentage

HORIZONS = [30, 60, 120]  # Test multiple horizons
INVALIDATION_RATIO = 0.5  # Stop = 50% of target (2:1 R:R)

TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 70)
print("STRUCTURAL NOISE ANALYSIS")
print("=" * 70)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
if not ohlcv_path.exists():
    print(f"ERROR: OHLCV file not found: {ohlcv_path}")
    exit(1)

ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")
print(f"Date range: {ohlcv.index.min()} to {ohlcv.index.max()}")

# Split into train/test
train_ohlcv = ohlcv[ohlcv.index <= TRAIN_END]
test_ohlcv = ohlcv[ohlcv.index >= TEST_START]
print(f"\nTRAIN: {len(train_ohlcv):,} candles (up to {TRAIN_END})")
print(f"TEST:  {len(test_ohlcv):,} candles (from {TEST_START})")

print(f"\nMWNM (Minimum Worthwhile Net Move): {MWNM_BPS} bps")
print(f"  - Fees: 8 bps")
print(f"  - Slippage: 2 bps")
print(f"  - Buffer: 5 bps")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_max_moves(ohlcv_df, horizon):
    """Compute max move in either direction for each bar."""
    close = ohlcv_df['close'].values
    high = ohlcv_df['high'].values
    low = ohlcv_df['low'].values
    n = len(ohlcv_df)

    max_up = np.zeros(n)
    max_down = np.zeros(n)
    max_move = np.zeros(n)

    for i in range(n - horizon):
        entry = close[i]
        future_high = np.max(high[i+1:i+1+horizon])
        future_low = np.min(low[i+1:i+1+horizon])

        max_up[i] = (future_high - entry) / entry
        max_down[i] = (entry - future_low) / entry
        max_move[i] = max(max_up[i], max_down[i])

    # Last H bars have no future data
    max_up[-horizon:] = np.nan
    max_down[-horizon:] = np.nan
    max_move[-horizon:] = np.nan

    return max_up, max_down, max_move


def compute_expansion_labels(ohlcv_df, horizon, target_pct, stop_pct):
    """Compute path-dependent expansion labels."""
    close = ohlcv_df['close'].values
    high = ohlcv_df['high'].values
    low = ohlcv_df['low'].values
    n = len(ohlcv_df)

    long_expansion = np.zeros(n)
    short_expansion = np.zeros(n)

    for i in range(n - horizon):
        entry = close[i]
        target_up = entry * (1 + target_pct)
        stop_up = entry * (1 - stop_pct)
        target_down = entry * (1 - target_pct)
        stop_down = entry * (1 + stop_pct)

        # Check long: target before stop
        for j in range(i+1, i+1+horizon):
            if low[j] <= stop_up:
                break  # Stop hit first
            if high[j] >= target_up:
                long_expansion[i] = 1
                break

        # Check short: target before stop
        for j in range(i+1, i+1+horizon):
            if high[j] >= stop_down:
                break  # Stop hit first
            if low[j] <= target_down:
                short_expansion[i] = 1
                break

    return long_expansion, short_expansion


# =============================================================================
# ANALYZE EACH HORIZON
# =============================================================================

for H in HORIZONS:
    print("\n" + "=" * 70)
    print(f"HORIZON = {H} bars ({H} minutes)")
    print("=" * 70)

    # Compute max moves on TRAIN data
    print(f"\nComputing max moves on TRAIN data...")
    max_up, max_down, max_move = compute_max_moves(train_ohlcv, H)

    # Remove NaN (last H bars)
    valid_mask = ~np.isnan(max_move)
    max_move_valid = max_move[valid_mask]

    # Compute noise statistics
    noise_mask = max_move_valid < MWNM_PCT
    noise_pct = np.mean(noise_mask) * 100
    tradeable_pct = 100 - noise_pct

    print(f"\n--- STRUCTURAL NOISE (TRAIN) ---")
    print(f"  Total bars analyzed: {len(max_move_valid):,}")
    print(f"  NOISE bars (max_move < {MWNM_BPS} bps): {np.sum(noise_mask):,} ({noise_pct:.1f}%)")
    print(f"  TRADEABLE bars: {np.sum(~noise_mask):,} ({tradeable_pct:.1f}%)")

    # Distribution of max moves
    print(f"\n--- MAX MOVE DISTRIBUTION (TRAIN) ---")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(max_move_valid, p) * 10000
        print(f"  {p}th percentile: {val:.1f} bps")

    # Compute thresholds on TRAIN (median as target)
    median_move = np.percentile(max_move_valid, 50)
    target_pct = median_move
    stop_pct = target_pct * INVALIDATION_RATIO

    print(f"\n--- THRESHOLDS (from TRAIN median) ---")
    print(f"  Target: {target_pct * 10000:.1f} bps")
    print(f"  Stop: {stop_pct * 10000:.1f} bps")
    print(f"  R:R: {target_pct / stop_pct:.1f}:1")

    # Compute expansion labels on TRAIN
    print(f"\nComputing expansion labels on TRAIN data...")
    train_long_exp, train_short_exp = compute_expansion_labels(train_ohlcv, H, target_pct, stop_pct)

    # Get valid indices (exclude last H bars)
    train_indices = np.arange(len(train_ohlcv) - H)

    # Expansion rates on ALL bars
    all_long_rate = np.mean(train_long_exp[train_indices]) * 100
    all_short_rate = np.mean(train_short_exp[train_indices]) * 100

    # Noise mask for TRAIN (using same valid indices)
    train_max_move = max_move[:len(train_ohlcv)]
    train_noise_mask = train_max_move[train_indices] < MWNM_PCT
    train_tradeable_mask = ~train_noise_mask

    # Expansion rates on TRADEABLE bars only
    if np.sum(train_tradeable_mask) > 0:
        tradeable_long_rate = np.mean(train_long_exp[train_indices][train_tradeable_mask]) * 100
        tradeable_short_rate = np.mean(train_short_exp[train_indices][train_tradeable_mask]) * 100
    else:
        tradeable_long_rate = 0
        tradeable_short_rate = 0

    print(f"\n--- EXPANSION RATES (TRAIN) ---")
    print(f"{'':30} {'LONG':>12} {'SHORT':>12}")
    print(f"{'-'*56}")
    print(f"{'ALL bars:':30} {all_long_rate:>11.1f}% {all_short_rate:>11.1f}%")
    print(f"{'TRADEABLE bars only:':30} {tradeable_long_rate:>11.1f}% {tradeable_short_rate:>11.1f}%")
    print(f"{'Improvement:':30} {tradeable_long_rate - all_long_rate:>+10.1f}pp {tradeable_short_rate - all_short_rate:>+10.1f}pp")

    # Break-even analysis
    breakeven_wr = stop_pct / (target_pct + stop_pct) * 100
    print(f"\n--- PROFITABILITY CHECK (TRAIN) ---")
    print(f"  Break-even WR: {breakeven_wr:.1f}%")
    print(f"  ALL bars gap: {all_long_rate - breakeven_wr:+.1f}pp (LONG), {all_short_rate - breakeven_wr:+.1f}pp (SHORT)")
    print(f"  TRADEABLE gap: {tradeable_long_rate - breakeven_wr:+.1f}pp (LONG), {tradeable_short_rate - breakeven_wr:+.1f}pp (SHORT)")

    if tradeable_long_rate >= breakeven_wr:
        print(f"  >>> LONG on tradeable bars CROSSES break-even!")
    if tradeable_short_rate >= breakeven_wr:
        print(f"  >>> SHORT on tradeable bars CROSSES break-even!")

    # Now TEST data
    print(f"\n--- TEST DATA VALIDATION ---")
    test_max_up, test_max_down, test_max_move = compute_max_moves(test_ohlcv, H)

    test_valid_mask = ~np.isnan(test_max_move)
    test_max_move_valid = test_max_move[test_valid_mask]

    test_noise_mask = test_max_move_valid < MWNM_PCT
    test_noise_pct = np.mean(test_noise_mask) * 100

    print(f"  TEST noise %: {test_noise_pct:.1f}% (vs TRAIN: {noise_pct:.1f}%)")

    # Expansion labels on TEST (using TRAIN thresholds)
    test_long_exp, test_short_exp = compute_expansion_labels(test_ohlcv, H, target_pct, stop_pct)
    test_indices = np.arange(len(test_ohlcv) - H)

    # All bars
    test_all_long = np.mean(test_long_exp[test_indices]) * 100
    test_all_short = np.mean(test_short_exp[test_indices]) * 100

    # Tradeable bars
    test_full_max_move = test_max_move[:len(test_ohlcv)]
    test_tradeable_mask = test_full_max_move[test_indices] >= MWNM_PCT

    if np.sum(test_tradeable_mask) > 0:
        test_tradeable_long = np.mean(test_long_exp[test_indices][test_tradeable_mask]) * 100
        test_tradeable_short = np.mean(test_short_exp[test_indices][test_tradeable_mask]) * 100
    else:
        test_tradeable_long = 0
        test_tradeable_short = 0

    print(f"\n  TEST expansion (ALL bars):      LONG={test_all_long:.1f}%, SHORT={test_all_short:.1f}%")
    print(f"  TEST expansion (TRADEABLE):     LONG={test_tradeable_long:.1f}%, SHORT={test_tradeable_short:.1f}%")
    print(f"  TEST improvement:               LONG={test_tradeable_long - test_all_long:+.1f}pp, SHORT={test_tradeable_short - test_all_short:+.1f}pp")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
KEY FINDINGS:
1. Structural noise = bars where max_move < {MWNM_BPS} bps
2. By removing noise bars, expansion rates should INCREASE
3. This is because noise bars always contribute 0% to expansion

THE MATH:
- If X% of bars are noise (always 0% expansion)
- And Y% is the expansion rate on ALL bars
- Then expansion rate on TRADEABLE bars = Y / (1 - X)

NEXT STEPS:
- If tradeable bar expansion rate > break-even: we have a path to profit!
- The realtime challenge: predict which bars are tradeable without future data
- Proxies: ATR threshold, volume, triggers, session filters
""")
