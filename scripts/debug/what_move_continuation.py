"""
WHAT PHASE: Move Continuation Analysis (W-EXP2)

Measure whether initial moves continue or reverse:
- Initial 1-bar move direction
- Does it continue in same direction or reverse?
- How does continuation relate to case outcomes?

Parameters:
- Horizons: 5, 10, 15, 30, 60, 120 bars
- Targets: 12, 15, 20, 25, 30, 50 bp

Run: .venv/Scripts/python.exe scripts/debug/what_move_continuation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import itertools

# =============================================================================
# CONFIGURATION
# =============================================================================
TARGETS = [12, 15, 20, 25, 30, 50]  # basis points
HORIZONS = [5, 10, 15, 30, 60, 120]  # bars
EXTENDED_H = 500

ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200


# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("WHAT PHASE: MOVE CONTINUATION ANALYSIS (W-EXP2)")
print(f"Testing {len(TARGETS)} targets x {len(HORIZONS)} horizons = {len(TARGETS)*len(HORIZONS)} combinations")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
test = ohlcv[ohlcv.index > "2023-12-31"].copy()
print(f"Train data (2020-2023): {len(train):,} candles")
print(f"Test data (2024-2025):  {len(test):,} candles")


# =============================================================================
# CALCULATE INDICATORS
# =============================================================================
print("\nCalculating indicators...")

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr

for df in [train, test]:
    df['atr'] = calculate_atr(df, ATR_PERIOD)
    df['atr_pct'] = df['atr'] / df['close'] * 100
    # Initial move = close - open of next bar (direction at entry+1)
    df['next_close'] = df['close'].shift(-1)
    df['next_open'] = df['open'].shift(-1)
    df['initial_move_bps'] = (df['next_close'] - df['next_open']) / df['close'] * 10000
    df['ema50'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

print("Indicators calculated.")


# =============================================================================
# CASE LABELING FUNCTION
# =============================================================================
@njit
def label_case_with_path(entry, highs, lows, target_pct, H, extended_H):
    """Returns case and path info."""
    n = len(highs)
    if n == 0:
        return -1, 0, 0, 0.0, 0.0

    target_price = entry * (1 + target_pct)
    went_below = False
    hit_within_H = False
    hit_extended = False

    # Track first significant move
    first_up = 0
    first_down = 0
    max_up = 0.0
    max_down = 0.0

    for j in range(min(H, n)):
        up_move = (highs[j] - entry) / entry * 10000
        down_move = (entry - lows[j]) / entry * 10000

        if up_move > max_up:
            max_up = up_move
        if down_move > max_down:
            max_down = down_move

        # First significant move (>5bp)
        if first_up == 0 and up_move >= 5:
            first_up = j + 1
        if first_down == 0 and down_move >= 5:
            first_down = j + 1

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
        case = 0
    elif not hit_within_H and not hit_extended:
        case = 1
    elif went_below and hit_within_H:
        case = 2
    elif went_below and hit_extended:
        case = 3
    else:
        case = -1

    return case, first_up, first_down, max_up, max_down


# =============================================================================
# MOVE CONTINUATION ANALYSIS FUNCTION
# =============================================================================
def analyze_move_continuation(df, target_bps, H):
    """Analyze move continuation patterns by case."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    initial_move = df['initial_move_bps'].values
    target_pct = target_bps / 10000

    indices = np.arange(EMA_SLOW + 10, len(df) - EXTENDED_H)

    if len(indices) < 100:
        return None

    results = {
        # Initial move UP
        'up_case0': 0, 'up_case1': 0, 'up_case2': 0, 'up_case3': 0,
        # Initial move DOWN
        'down_case0': 0, 'down_case1': 0, 'down_case2': 0, 'down_case3': 0,
        # Initial move FLAT (small)
        'flat_case0': 0, 'flat_case1': 0, 'flat_case2': 0, 'flat_case3': 0,
        # Strong moves
        'strong_up_case0': 0, 'strong_up_case1': 0, 'strong_up_case2': 0, 'strong_up_case3': 0,
        'strong_down_case0': 0, 'strong_down_case1': 0, 'strong_down_case2': 0, 'strong_down_case3': 0,
    }

    for i in indices:
        entry = close[i]
        future_highs = high[i+1:i+1+EXTENDED_H]
        future_lows = low[i+1:i+1+EXTENDED_H]

        case, _, _, _, _ = label_case_with_path(
            entry, future_highs, future_lows, target_pct, H, EXTENDED_H
        )

        if case < 0:
            continue

        init_move = initial_move[i]

        # Categorize initial move
        if init_move > 3:  # UP
            results[f'up_case{case}'] += 1
            if init_move > 10:  # Strong UP
                results[f'strong_up_case{case}'] += 1
        elif init_move < -3:  # DOWN
            results[f'down_case{case}'] += 1
            if init_move < -10:  # Strong DOWN
                results[f'strong_down_case{case}'] += 1
        else:  # FLAT
            results[f'flat_case{case}'] += 1

    # Calculate statistics
    total_up = sum(results[f'up_case{c}'] for c in range(4))
    total_down = sum(results[f'down_case{c}'] for c in range(4))
    total_flat = sum(results[f'flat_case{c}'] for c in range(4))
    total_strong_up = sum(results[f'strong_up_case{c}'] for c in range(4))
    total_strong_down = sum(results[f'strong_down_case{c}'] for c in range(4))

    output = {
        'total': len(indices),
        'target': target_bps,
        'horizon': H,
        'total_up': total_up,
        'total_down': total_down,
        'total_flat': total_flat,
        'total_strong_up': total_strong_up,
        'total_strong_down': total_strong_down,
    }

    # Percentages for each initial move type
    for case in [0, 1, 2, 3]:
        # Initial UP
        if total_up > 0:
            output[f'up_case{case}_pct'] = results[f'up_case{case}'] / total_up * 100
        else:
            output[f'up_case{case}_pct'] = 0

        # Initial DOWN
        if total_down > 0:
            output[f'down_case{case}_pct'] = results[f'down_case{case}'] / total_down * 100
        else:
            output[f'down_case{case}_pct'] = 0

        # Initial FLAT
        if total_flat > 0:
            output[f'flat_case{case}_pct'] = results[f'flat_case{case}'] / total_flat * 100
        else:
            output[f'flat_case{case}_pct'] = 0

        # Strong UP
        if total_strong_up > 0:
            output[f'strong_up_case{case}_pct'] = results[f'strong_up_case{case}'] / total_strong_up * 100
        else:
            output[f'strong_up_case{case}_pct'] = 0

        # Strong DOWN
        if total_strong_down > 0:
            output[f'strong_down_case{case}_pct'] = results[f'strong_down_case{case}'] / total_strong_down * 100
        else:
            output[f'strong_down_case{case}_pct'] = 0

    return output


# =============================================================================
# RUN ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING MOVE CONTINUATION ANALYSIS")
print("=" * 80)

train_results = []
test_results = []

total_combos = len(TARGETS) * len(HORIZONS)
combo_num = 0

for target, horizon in itertools.product(TARGETS, HORIZONS):
    combo_num += 1
    print(f"\rProcessing {combo_num}/{total_combos}: T={target}bp, H={horizon}...", end="", flush=True)

    # Train
    train_result = analyze_move_continuation(train, target, horizon)
    if train_result:
        train_result['dataset'] = 'train'
        train_results.append(train_result)

    # Test
    test_result = analyze_move_continuation(test, target, horizon)
    if test_result:
        test_result['dataset'] = 'test'
        test_results.append(test_result)

print("\n\nAnalysis complete.")


# =============================================================================
# DISPLAY RESULTS: CASE DISTRIBUTION BY INITIAL MOVE (T=25bp, H=30)
# =============================================================================
print("\n" + "=" * 80)
print("CASE DISTRIBUTION BY INITIAL MOVE DIRECTION (T=25bp, H=30)")
print("=" * 80)

print("\n### TRAIN DATA (2020-2023)")
print("\nInitial move = first bar after entry (close - open in bps)")
print("UP = >3bp, DOWN = <-3bp, FLAT = -3 to +3bp\n")

for r in train_results:
    if r['target'] == 25 and r['horizon'] == 30:
        print("| Initial Move | Case 0 % | Case 1 % | Case 2 % | Case 3 % | Count |")
        print("|--------------|----------|----------|----------|----------|-------|")
        print(f"| UP (>3bp)    | {r['up_case0_pct']:.1f}% | {r['up_case1_pct']:.1f}% | {r['up_case2_pct']:.1f}% | {r['up_case3_pct']:.1f}% | {r['total_up']:,} |")
        print(f"| DOWN (<-3bp) | {r['down_case0_pct']:.1f}% | {r['down_case1_pct']:.1f}% | {r['down_case2_pct']:.1f}% | {r['down_case3_pct']:.1f}% | {r['total_down']:,} |")
        print(f"| FLAT         | {r['flat_case0_pct']:.1f}% | {r['flat_case1_pct']:.1f}% | {r['flat_case2_pct']:.1f}% | {r['flat_case3_pct']:.1f}% | {r['total_flat']:,} |")
        break


# =============================================================================
# STRONG MOVES ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("STRONG MOVES ANALYSIS (T=25bp, H=30)")
print("=" * 80)

print("\nStrong UP = >10bp, Strong DOWN = <-10bp\n")

for r in train_results:
    if r['target'] == 25 and r['horizon'] == 30:
        print("| Strong Move    | Case 0 % | Case 1 % | Case 2 % | Case 3 % | Count |")
        print("|----------------|----------|----------|----------|----------|-------|")
        print(f"| Strong UP      | {r['strong_up_case0_pct']:.1f}% | {r['strong_up_case1_pct']:.1f}% | {r['strong_up_case2_pct']:.1f}% | {r['strong_up_case3_pct']:.1f}% | {r['total_strong_up']:,} |")
        print(f"| Strong DOWN    | {r['strong_down_case0_pct']:.1f}% | {r['strong_down_case1_pct']:.1f}% | {r['strong_down_case2_pct']:.1f}% | {r['strong_down_case3_pct']:.1f}% | {r['total_strong_down']:,} |")
        break


# =============================================================================
# CASE 1 RATE BY INITIAL MOVE ACROSS TARGETS
# =============================================================================
print("\n" + "=" * 80)
print("CASE 1 RATE BY INITIAL MOVE DIRECTION (H=30, TRAIN)")
print("=" * 80)

print("\n| Target | UP Case1% | DOWN Case1% | FLAT Case1% | Diff (DOWN-UP) |")
print("|--------|-----------|-------------|-------------|----------------|")

for target in TARGETS:
    match = [r for r in train_results if r['target'] == target and r['horizon'] == 30]
    if match:
        r = match[0]
        up_c1 = r['up_case1_pct']
        down_c1 = r['down_case1_pct']
        flat_c1 = r['flat_case1_pct']
        diff = down_c1 - up_c1
        print(f"| {target}bp | {up_c1:.1f}% | {down_c1:.1f}% | {flat_c1:.1f}% | {diff:+.1f}pp |")


# =============================================================================
# CASE 0 + CASE 2 (SUCCESS) BY INITIAL MOVE
# =============================================================================
print("\n" + "=" * 80)
print("SUCCESS RATE (CASE 0 + 2) BY INITIAL MOVE (H=30, TRAIN)")
print("=" * 80)

print("\nSuccess = Case 0 (clean win) + Case 2 (quick recovery)")
print("Both hit target within horizon H\n")

print("| Target | UP Success% | DOWN Success% | FLAT Success% |")
print("|--------|-------------|---------------|---------------|")

for target in TARGETS:
    match = [r for r in train_results if r['target'] == target and r['horizon'] == 30]
    if match:
        r = match[0]
        up_success = r['up_case0_pct'] + r['up_case2_pct']
        down_success = r['down_case0_pct'] + r['down_case2_pct']
        flat_success = r['flat_case0_pct'] + r['flat_case2_pct']
        print(f"| {target}bp | {up_success:.1f}% | {down_success:.1f}% | {flat_success:.1f}% |")


# =============================================================================
# TRAIN vs TEST COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("TRAIN vs TEST: CASE 1 RATE BY INITIAL MOVE (T=25bp)")
print("=" * 80)

print("\n| Horizon | Train UP Case1% | Test UP Case1% | Train DOWN Case1% | Test DOWN Case1% |")
print("|---------|-----------------|----------------|-------------------|------------------|")

for horizon in HORIZONS:
    train_match = [r for r in train_results if r['target'] == 25 and r['horizon'] == horizon]
    test_match = [r for r in test_results if r['target'] == 25 and r['horizon'] == horizon]

    if train_match and test_match:
        tr = train_match[0]
        te = test_match[0]
        print(f"| H={horizon} | {tr['up_case1_pct']:.1f}% | {te['up_case1_pct']:.1f}% | {tr['down_case1_pct']:.1f}% | {te['down_case1_pct']:.1f}% |")


# =============================================================================
# KEY INSIGHT: CONTINUATION vs REVERSAL
# =============================================================================
print("\n" + "=" * 80)
print("CONTINUATION vs REVERSAL ANALYSIS (T=25bp, H=30)")
print("=" * 80)

print("""
For LONG trades (we want price to go UP):

If initial bar moves UP:
- Case 0 + Case 2 = "Continuation" (price eventually hits target)
- Case 1 = "Failure" (price never recovers)

If initial bar moves DOWN:
- Case 0 + Case 2 = "Reversal" (price recovers and hits target)
- Case 1 = "Continuation of wrong direction" (structural failure)
""")

for r in train_results:
    if r['target'] == 25 and r['horizon'] == 30:
        up_continue = r['up_case0_pct'] + r['up_case2_pct']
        down_reverse = r['down_case0_pct'] + r['down_case2_pct']
        print(f"After UP initial move:")
        print(f"  - Continues to target: {up_continue:.1f}%")
        print(f"  - Fails structurally:  {r['up_case1_pct']:.1f}%")
        print(f"\nAfter DOWN initial move:")
        print(f"  - Reverses to target:  {down_reverse:.1f}%")
        print(f"  - Fails structurally:  {r['down_case1_pct']:.1f}%")
        break


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("MOVE CONTINUATION ANALYSIS SUMMARY")
print("=" * 80)

print("""
Key Findings:

1. Initial Move = Direction of first bar after entry
   - Measures immediate market reaction
   - UP (>3bp), DOWN (<-3bp), FLAT (-3 to +3bp)

2. Initial move direction affects case distribution:
   - UP initial move: Higher success rate?
   - DOWN initial move: Higher Case 1 rate?
   - FLAT: ?

3. Strong moves (>10bp) may have different patterns:
   - Momentum continuation
   - Exhaustion reversal

4. Implications for EXECUTE phase:
   - Should we filter by initial move direction?
   - Is strong UP a continuation signal?
   - Is strong DOWN a warning sign?

5. For LONG trades:
   - UP initial: price moving in our favor (good)
   - DOWN initial: price moving against us (bad start)
""")


# =============================================================================
# SAVE RESULTS
# =============================================================================
output_path = Path("experiments/what_move_continuation.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

all_results = train_results + test_results
df_results = pd.DataFrame(all_results)
df_results.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
