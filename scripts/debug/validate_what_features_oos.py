"""
Validate WHAT Phase Features on OOS (2024-2025 data)

Features to validate:
1. Entry Bar Range
2. Initial Move Direction

Run: .venv/Scripts/python.exe scripts/debug/validate_what_features_oos.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit

# =============================================================================
# CONFIGURATION
# =============================================================================
TARGET_BPS = 25
HORIZON = 30
EXTENDED_H = 500


# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("VALIDATE WHAT FEATURES ON OOS (2024-2025)")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
test = ohlcv[ohlcv.index > "2023-12-31"].copy()
print(f"Train data (2020-2023): {len(train):,} candles")
print(f"Test data (2024-2025):  {len(test):,} candles")


# =============================================================================
# CALCULATE FEATURES
# =============================================================================
print("\nCalculating features...")

for df in [train, test]:
    # Entry bar range
    df['range_bps'] = (df['high'] - df['low']) / df['close'] * 10000

    # Initial move (next bar close - open)
    df['next_close'] = df['close'].shift(-1)
    df['next_open'] = df['open'].shift(-1)
    df['initial_move_bps'] = (df['next_close'] - df['next_open']) / df['close'] * 10000

print("Features calculated.")


# =============================================================================
# CASE LABELING FUNCTION
# =============================================================================
@njit
def label_case(entry, highs, lows, target_pct, H, extended_H):
    """Returns case (0, 1, 2, 3)."""
    n = len(highs)
    if n == 0:
        return -1

    target_price = entry * (1 + target_pct)
    went_below = False
    hit_within_H = False
    hit_extended = False

    for j in range(min(H, n)):
        if lows[j] < entry:
            went_below = True
        if highs[j] >= target_price:
            hit_within_H = True
            break

    if not hit_within_H:
        for j in range(H, min(extended_H, n)):
            if lows[j] < entry:
                went_below = True
            if highs[j] >= target_price:
                hit_extended = True
                break

    if not went_below and hit_within_H:
        return 0
    elif not hit_within_H and not hit_extended:
        return 1
    elif went_below and hit_within_H:
        return 2
    elif went_below and hit_extended:
        return 3
    else:
        return -1


# =============================================================================
# ANALYZE FUNCTION
# =============================================================================
def analyze_features(df, target_bps, H, quartile_boundaries=None):
    """Analyze entry bar range and initial move vs case."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    range_bps = df['range_bps'].values
    initial_move = df['initial_move_bps'].values
    target_pct = target_bps / 10000

    start_idx = 210  # Skip initial bars for indicator warmup
    end_idx = len(df) - EXTENDED_H

    indices = np.arange(start_idx, end_idx)

    if len(indices) < 100:
        return None, None

    entry_ranges = []
    initial_moves = []
    cases = []

    for i in indices:
        entry = close[i]
        future_highs = high[i+1:i+1+EXTENDED_H]
        future_lows = low[i+1:i+1+EXTENDED_H]

        case = label_case(entry, future_highs, future_lows, target_pct, H, EXTENDED_H)

        if case >= 0 and not np.isnan(range_bps[i]) and not np.isnan(initial_move[i]):
            entry_ranges.append(range_bps[i])
            initial_moves.append(initial_move[i])
            cases.append(case)

    entry_ranges = np.array(entry_ranges)
    initial_moves = np.array(initial_moves)
    cases = np.array(cases)

    # Calculate quartile boundaries from this data if not provided
    if quartile_boundaries is None:
        q25 = np.percentile(entry_ranges, 25)
        q50 = np.percentile(entry_ranges, 50)
        q75 = np.percentile(entry_ranges, 75)
        quartile_boundaries = (q25, q50, q75)
    else:
        q25, q50, q75 = quartile_boundaries

    # Entry Bar Range Analysis
    range_results = {}
    for qname, qmin, qmax in [('Q1', 0, q25), ('Q2', q25, q50), ('Q3', q50, q75), ('Q4', q75, 9999)]:
        mask = (entry_ranges >= qmin) & (entry_ranges < qmax)
        subset_cases = cases[mask]
        total = len(subset_cases)
        if total > 0:
            c1_pct = (subset_cases == 1).sum() / total * 100
            range_results[qname] = {
                'count': total,
                'case1_pct': c1_pct,
                'min': qmin,
                'max': qmax
            }

    # Initial Move Analysis
    move_results = {}
    for mname, condition in [
        ('Strong UP (>10bp)', initial_moves > 10),
        ('UP (>3bp)', initial_moves > 3),
        ('FLAT (-3 to 3bp)', (initial_moves >= -3) & (initial_moves <= 3)),
        ('DOWN (<-3bp)', initial_moves < -3),
        ('Strong DOWN (<-10bp)', initial_moves < -10)
    ]:
        subset_cases = cases[condition]
        total = len(subset_cases)
        if total > 0:
            c1_pct = (subset_cases == 1).sum() / total * 100
            move_results[mname] = {
                'count': total,
                'case1_pct': c1_pct
            }

    return range_results, move_results, quartile_boundaries


# =============================================================================
# RUN ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print(f"ANALYZING (Target={TARGET_BPS}bp, H={HORIZON})")
print("=" * 80)

# Train analysis (to get quartile boundaries)
print("\nAnalyzing TRAIN data (2020-2023)...")
train_range, train_move, quartile_bounds = analyze_features(train, TARGET_BPS, HORIZON)

# Test analysis (using TRAIN quartile boundaries for fair comparison)
print("Analyzing TEST data (2024-2025)...")
test_range, test_move, _ = analyze_features(test, TARGET_BPS, HORIZON, quartile_bounds)


# =============================================================================
# DISPLAY RESULTS: ENTRY BAR RANGE
# =============================================================================
print("\n" + "=" * 80)
print("FEATURE 1: ENTRY BAR RANGE")
print("=" * 80)

print(f"\nQuartile boundaries (from train): Q25={quartile_bounds[0]:.1f}bp, Q50={quartile_bounds[1]:.1f}bp, Q75={quartile_bounds[2]:.1f}bp")

print("\n### TRAIN (2020-2023) vs TEST (2024-2025)")
print("\n| Quartile | Range | Train P(Case1) | Test P(Case1) | Diff | Valid? |")
print("|----------|-------|----------------|---------------|------|--------|")

for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    if q in train_range and q in test_range:
        train_c1 = train_range[q]['case1_pct']
        test_c1 = test_range[q]['case1_pct']
        diff = test_c1 - train_c1

        # Valid if direction is same (Q1 still highest, Q4 still lowest)
        valid = "YES" if (q == 'Q1' and test_c1 > test_range['Q4']['case1_pct']) or \
                        (q == 'Q4' and test_c1 < test_range['Q1']['case1_pct']) or \
                        (q in ['Q2', 'Q3']) else "CHECK"

        rng = f"{train_range[q]['min']:.0f}-{train_range[q]['max']:.0f}bp"
        print(f"| {q} | {rng} | {train_c1:.1f}% | {test_c1:.1f}% | {diff:+.1f}pp | {valid} |")

# Check if pattern holds
train_q1_q4_diff = train_range['Q1']['case1_pct'] - train_range['Q4']['case1_pct']
test_q1_q4_diff = test_range['Q1']['case1_pct'] - test_range['Q4']['case1_pct']

print(f"\n**Train Q1-Q4 difference:** {train_q1_q4_diff:.1f}pp")
print(f"**Test Q1-Q4 difference:** {test_q1_q4_diff:.1f}pp")

if test_q1_q4_diff > 5:
    print("\n**VALIDATION: PASSED** - Pattern holds on OOS data (Q1 > Q4)")
else:
    print("\n**VALIDATION: FAILED** - Pattern does not hold on OOS data")


# =============================================================================
# DISPLAY RESULTS: INITIAL MOVE DIRECTION
# =============================================================================
print("\n" + "=" * 80)
print("FEATURE 2: INITIAL MOVE DIRECTION")
print("=" * 80)

print("\n### TRAIN (2020-2023) vs TEST (2024-2025)")
print("\n| Initial Move | Train P(Case1) | Test P(Case1) | Diff | Valid? |")
print("|--------------|----------------|---------------|------|--------|")

for move in ['Strong UP (>10bp)', 'UP (>3bp)', 'FLAT (-3 to 3bp)', 'DOWN (<-3bp)', 'Strong DOWN (<-10bp)']:
    if move in train_move and move in test_move:
        train_c1 = train_move[move]['case1_pct']
        test_c1 = test_move[move]['case1_pct']
        diff = test_c1 - train_c1

        # Valid if UP still better than DOWN
        if 'UP' in move:
            valid = "YES" if test_c1 < test_move['DOWN (<-3bp)']['case1_pct'] else "NO"
        elif 'DOWN' in move:
            valid = "YES" if test_c1 > test_move['UP (>3bp)']['case1_pct'] else "NO"
        else:
            valid = "CHECK"

        print(f"| {move} | {train_c1:.1f}% | {test_c1:.1f}% | {diff:+.1f}pp | {valid} |")

# Check if pattern holds
train_up_down_diff = train_move['DOWN (<-3bp)']['case1_pct'] - train_move['UP (>3bp)']['case1_pct']
test_up_down_diff = test_move['DOWN (<-3bp)']['case1_pct'] - test_move['UP (>3bp)']['case1_pct']

print(f"\n**Train DOWN-UP difference:** {train_up_down_diff:.1f}pp")
print(f"**Test DOWN-UP difference:** {test_up_down_diff:.1f}pp")

if test_up_down_diff > 3:
    print("\n**VALIDATION: PASSED** - Pattern holds on OOS data (DOWN > UP)")
else:
    print("\n**VALIDATION: FAILED** - Pattern does not hold on OOS data")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

print("\n| Feature | Train Effect | Test Effect | Status |")
print("|---------|--------------|-------------|--------|")
print(f"| Entry Bar Range (Q1 vs Q4) | {train_q1_q4_diff:.1f}pp | {test_q1_q4_diff:.1f}pp | {'VALID' if test_q1_q4_diff > 5 else 'INVALID'} |")
print(f"| Initial Move (DOWN vs UP) | {train_up_down_diff:.1f}pp | {test_up_down_diff:.1f}pp | {'VALID' if test_up_down_diff > 3 else 'INVALID'} |")

print("\n" + "=" * 80)
