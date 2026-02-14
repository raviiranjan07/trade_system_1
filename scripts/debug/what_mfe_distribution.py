"""
WHAT PHASE: MFE Distribution Analysis

Measure Maximum Favorable Excursion (MFE) - how far price went IN OUR FAVOR
before reversal or hitting target.

Same parameters as MAE analysis:
- Targets: 12, 15, 20, 25, 30, 50 bp
- Horizons: 5, 10, 15, 30, 60, 120 bars

Run: .venv/Scripts/python.exe scripts/debug/what_mfe_distribution.py
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
print("WHAT PHASE: MFE DISTRIBUTION ANALYSIS")
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
    df['atr_percentile'] = df['atr_pct'].rank(pct=True) * 100
    df['ema50'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    df['ema_separation'] = np.abs(df['ema50'] - df['ema200']) / df['close'] * 100

print("Indicators calculated.")


# =============================================================================
# MFE ANALYSIS FUNCTION
# =============================================================================
@njit
def analyze_trade_mfe(entry, highs, lows, target_pct, H, extended_H):
    """
    Returns: (case, mfe_bps, mae_bps, hit_time)
    MFE = Maximum Favorable Excursion (max high - entry)
    """
    n = len(highs)
    if n == 0:
        return -1, 0.0, 0.0, 0

    target_price = entry * (1 + target_pct)
    went_below = False
    hit_within_H = False
    hit_extended = False
    max_favorable_bps = 0.0
    max_adverse_bps = 0.0
    hit_time = 0

    for j in range(min(H, n)):
        # Calculate MFE (max upside)
        favorable = (highs[j] - entry) / entry * 10000
        if favorable > max_favorable_bps:
            max_favorable_bps = favorable

        # Calculate MAE (max downside)
        adverse = (entry - lows[j]) / entry * 10000
        if adverse > max_adverse_bps:
            max_adverse_bps = adverse

        if lows[j] < entry:
            went_below = True

        if highs[j] >= target_price:
            hit_within_H = True
            hit_time = j + 1
            break

    # If not hit within H, continue to extended_H
    if not hit_within_H:
        for j in range(H, min(extended_H, n)):
            favorable = (highs[j] - entry) / entry * 10000
            if favorable > max_favorable_bps:
                max_favorable_bps = favorable

            adverse = (entry - lows[j]) / entry * 10000
            if adverse > max_adverse_bps:
                max_adverse_bps = adverse

            if lows[j] < entry:
                went_below = True

            if highs[j] >= target_price:
                hit_extended = True
                hit_time = j + 1
                break

    # Determine case
    if not went_below and hit_within_H:
        case = 0  # Clean win
    elif not hit_within_H and not hit_extended:
        case = 1  # Never recovered
    elif went_below and hit_within_H:
        case = 2  # Fast recovery
    elif went_below and hit_extended:
        case = 3  # Slow recovery
    else:
        case = -1

    return case, max_favorable_bps, max_adverse_bps, hit_time


def run_mfe_analysis(df, target_bps, H):
    """Run MFE analysis for single target/horizon combination."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    target_pct = target_bps / 10000

    # Sample every bar (baseline)
    indices = np.arange(EMA_SLOW + 10, len(df) - EXTENDED_H)

    if len(indices) < 100:
        return None

    results = {
        'case0_mfe': [], 'case0_mae': [],
        'case1_mfe': [], 'case1_mae': [],
        'case2_mfe': [], 'case2_mae': [],
        'case3_mfe': [], 'case3_mae': [],
    }

    for i in indices:
        entry = close[i]
        future_highs = high[i+1:i+1+EXTENDED_H]
        future_lows = low[i+1:i+1+EXTENDED_H]

        case, mfe, mae, hit_time = analyze_trade_mfe(
            entry, future_highs, future_lows, target_pct, H, EXTENDED_H
        )

        if case == 0:
            results['case0_mfe'].append(mfe)
            results['case0_mae'].append(mae)
        elif case == 1:
            results['case1_mfe'].append(mfe)
            results['case1_mae'].append(mae)
        elif case == 2:
            results['case2_mfe'].append(mfe)
            results['case2_mae'].append(mae)
        elif case == 3:
            results['case3_mfe'].append(mfe)
            results['case3_mae'].append(mae)

    # Calculate statistics
    output = {
        'total': len(indices),
        'target': target_bps,
        'horizon': H,
    }

    for case in [0, 1, 2, 3]:
        mfe_arr = np.array(results[f'case{case}_mfe'])
        mae_arr = np.array(results[f'case{case}_mae'])

        output[f'case{case}_count'] = len(mfe_arr)

        if len(mfe_arr) >= 50:
            output[f'case{case}_mfe_median'] = np.median(mfe_arr)
            output[f'case{case}_mfe_p75'] = np.percentile(mfe_arr, 75)
            output[f'case{case}_mfe_p90'] = np.percentile(mfe_arr, 90)
            output[f'case{case}_mfe_p95'] = np.percentile(mfe_arr, 95)
            output[f'case{case}_mfe_p99'] = np.percentile(mfe_arr, 99)
            output[f'case{case}_mae_p95'] = np.percentile(mae_arr, 95)
        else:
            output[f'case{case}_mfe_median'] = None
            output[f'case{case}_mfe_p75'] = None
            output[f'case{case}_mfe_p90'] = None
            output[f'case{case}_mfe_p95'] = None
            output[f'case{case}_mfe_p99'] = None
            output[f'case{case}_mae_p95'] = None

    return output


# =============================================================================
# RUN ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING MFE ANALYSIS")
print("=" * 80)

train_results = []
test_results = []

total_combos = len(TARGETS) * len(HORIZONS)
combo_num = 0

for target, horizon in itertools.product(TARGETS, HORIZONS):
    combo_num += 1
    print(f"\rProcessing {combo_num}/{total_combos}: T={target}bp, H={horizon}...", end="", flush=True)

    # Train
    train_result = run_mfe_analysis(train, target, horizon)
    if train_result:
        train_result['dataset'] = 'train'
        train_results.append(train_result)

    # Test
    test_result = run_mfe_analysis(test, target, horizon)
    if test_result:
        test_result['dataset'] = 'test'
        test_results.append(test_result)

print("\n\nAnalysis complete.")


# =============================================================================
# DISPLAY RESULTS: MFE BY CASE (T=25bp reference)
# =============================================================================
print("\n" + "=" * 80)
print("MFE DISTRIBUTION BY CASE (T=25bp, Baseline)")
print("=" * 80)

print("\n### TRAIN DATA (2020-2023)")
print("\n| Case | Count | MFE Median | MFE P75 | MFE P90 | MFE P95 | MFE P99 |")
print("|------|-------|------------|---------|---------|---------|---------|")

for case in [0, 1, 2, 3]:
    for r in train_results:
        if r['target'] == 25 and r['horizon'] == 30:
            count = r[f'case{case}_count']
            if r[f'case{case}_mfe_median'] is not None:
                med = r[f'case{case}_mfe_median']
                p75 = r[f'case{case}_mfe_p75']
                p90 = r[f'case{case}_mfe_p90']
                p95 = r[f'case{case}_mfe_p95']
                p99 = r[f'case{case}_mfe_p99']
                print(f"| Case {case} | {count:,} | {med:.1f}bp | {p75:.1f}bp | {p90:.1f}bp | {p95:.1f}bp | {p99:.1f}bp |")
            else:
                print(f"| Case {case} | {count:,} | N/A | N/A | N/A | N/A | N/A |")
            break


# =============================================================================
# DISPLAY: MFE P95 BY TARGET/HORIZON (Case 2)
# =============================================================================
print("\n" + "=" * 80)
print("CASE 2 MFE P95 BY TARGET/HORIZON (TRAIN)")
print("=" * 80)

print("\n| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |")
print("|--------|-----|------|------|------|------|-------|")

for target in TARGETS:
    row = f"| {target}bp |"
    for horizon in HORIZONS:
        match = [r for r in train_results if r['target'] == target and r['horizon'] == horizon]
        if match and match[0]['case2_mfe_p95'] is not None:
            row += f" {match[0]['case2_mfe_p95']:.0f}bp |"
        else:
            row += " N/A |"
    print(row)


# =============================================================================
# DISPLAY: MFE P95 BY TARGET/HORIZON (Case 3)
# =============================================================================
print("\n" + "=" * 80)
print("CASE 3 MFE P95 BY TARGET/HORIZON (TRAIN)")
print("=" * 80)

print("\n| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |")
print("|--------|-----|------|------|------|------|-------|")

for target in TARGETS:
    row = f"| {target}bp |"
    for horizon in HORIZONS:
        match = [r for r in train_results if r['target'] == target and r['horizon'] == horizon]
        if match and match[0]['case3_mfe_p95'] is not None:
            row += f" {match[0]['case3_mfe_p95']:.0f}bp |"
        else:
            row += " N/A |"
    print(row)


# =============================================================================
# MFE vs MAE COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("MFE vs MAE COMPARISON (T=25bp, H=30, TRAIN)")
print("=" * 80)

print("\n| Case | MFE P95 | MAE P95 | Ratio (MFE/MAE) | Interpretation |")
print("|------|---------|---------|-----------------|----------------|")

for r in train_results:
    if r['target'] == 25 and r['horizon'] == 30:
        for case in [0, 2, 3]:
            if r[f'case{case}_mfe_p95'] is not None and r[f'case{case}_mae_p95'] is not None:
                mfe = r[f'case{case}_mfe_p95']
                mae = r[f'case{case}_mae_p95']
                ratio = mfe / mae if mae > 0 else 0
                if ratio > 1:
                    interp = "More upside than downside"
                elif ratio < 1:
                    interp = "More downside than upside"
                else:
                    interp = "Equal"
                print(f"| Case {case} | {mfe:.0f}bp | {mae:.0f}bp | {ratio:.2f}x | {interp} |")
        break


# =============================================================================
# TRAIN vs TEST COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("TRAIN vs TEST: CASE 2 MFE P95 (T=25bp)")
print("=" * 80)

print("\n| Horizon | Train MFE P95 | Test MFE P95 | Diff |")
print("|---------|---------------|--------------|------|")

for horizon in HORIZONS:
    train_match = [r for r in train_results if r['target'] == 25 and r['horizon'] == horizon]
    test_match = [r for r in test_results if r['target'] == 25 and r['horizon'] == horizon]

    if train_match and test_match:
        train_mfe = train_match[0]['case2_mfe_p95']
        test_mfe = test_match[0]['case2_mfe_p95']
        if train_mfe and test_mfe:
            diff = test_mfe - train_mfe
            print(f"| H={horizon} | {train_mfe:.0f}bp | {test_mfe:.0f}bp | {diff:+.0f}bp |")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("MFE ANALYSIS SUMMARY")
print("=" * 80)

print("""
Key Findings:

1. MFE (Maximum Favorable Excursion) = Max profit available before reversal
   - Shows how much "money was left on the table"
   - Helps determine optimal take-profit levels

2. MFE vs MAE Ratio:
   - Ratio > 1: More upside potential than downside risk
   - Ratio < 1: More downside risk than upside potential
   - This informs risk/reward decisions

3. Case-specific insights:
   - Case 0: Clean wins - MFE typically >> target (could take more)
   - Case 2: Fast recovery - MFE shows profit before drawdown
   - Case 3: Slow recovery - MFE shows what was available before long wait

4. Implications for EXECUTE phase:
   - Should we take partial profits at MFE P50?
   - Should we trail stops based on MFE?
   - How much is left after hitting target?
""")


# =============================================================================
# SAVE RESULTS
# =============================================================================
output_path = Path("experiments/what_mfe_distribution.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

all_results = train_results + test_results
df_results = pd.DataFrame(all_results)
df_results.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
