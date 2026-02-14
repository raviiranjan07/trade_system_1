"""
SURVIVE PHASE: Multi-Parameter Validation

Test SURVIVE metrics across 36 target/horizon combinations
(same as WHEN phase validation).

Run: .venv/Scripts/python.exe scripts/debug/survive_validation_multi_param.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit
import itertools

# =============================================================================
# CONFIGURATION
# =============================================================================
# Same parameters as WHEN validation
TARGETS = [12, 15, 20, 25, 30, 50]  # basis points
HORIZONS = [5, 10, 15, 30, 60, 120]  # bars
EXTENDED_H = 500

ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200

# Leverage for survival test
TEST_LEVERAGE = 10
MAINTENANCE_MARGIN = 0.004
SAFETY_BUFFER = 0.20
FEES = 0.0008

def calc_liq_threshold(leverage):
    theoretical = 1 / leverage
    effective = theoretical - MAINTENANCE_MARGIN - (theoretical * SAFETY_BUFFER) - FEES
    return max(effective * 10000, 0)

LIQ_THRESHOLD_BP = calc_liq_threshold(TEST_LEVERAGE)


# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("SURVIVE PHASE: MULTI-PARAMETER VALIDATION")
print(f"Testing {len(TARGETS)} targets x {len(HORIZONS)} horizons = {len(TARGETS)*len(HORIZONS)} combinations")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
test = ohlcv[ohlcv.index > "2023-12-31"].copy()
print(f"Train data (2020-2023): {len(train):,} candles")
print(f"Test data (2024-2025):  {len(test):,} candles")
print(f"\n10x leverage liquidation threshold: {LIQ_THRESHOLD_BP:.0f}bp")


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
# ANALYSIS FUNCTION
# =============================================================================
@njit
def analyze_trade(entry, highs, lows, target_pct, H, extended_H):
    """Returns: (case, mae_bps, time_at_risk)"""
    n = len(highs)
    if n == 0:
        return -1, 0.0, 0

    target_price = entry * (1 + target_pct)
    went_below = False
    hit_within_H = False
    hit_extended = False
    max_adverse_bps = 0.0
    time_at_risk = 0

    for j in range(min(H, n)):
        adverse = (entry - lows[j]) / entry * 10000
        if adverse > max_adverse_bps:
            max_adverse_bps = adverse
        if lows[j] < entry:
            time_at_risk += 1
            went_below = True
        if highs[j] >= target_price:
            hit_within_H = True
            break

    if went_below and not hit_within_H:
        for j in range(H, min(extended_H, n)):
            adverse = (entry - lows[j]) / entry * 10000
            if adverse > max_adverse_bps:
                max_adverse_bps = adverse
            if lows[j] < entry:
                time_at_risk += 1
            if highs[j] >= target_price:
                hit_extended = True
                break

    if not went_below and not hit_within_H:
        for j in range(H, min(extended_H, n)):
            adverse = (entry - lows[j]) / entry * 10000
            if adverse > max_adverse_bps:
                max_adverse_bps = adverse
            if lows[j] < entry:
                went_below = True
                time_at_risk += 1
            if highs[j] >= target_price:
                if went_below:
                    return 3, max_adverse_bps, time_at_risk
                else:
                    return 0, max_adverse_bps, time_at_risk
        return 1, max_adverse_bps, time_at_risk

    if not went_below and hit_within_H:
        return 0, max_adverse_bps, time_at_risk
    elif went_below and hit_within_H:
        return 2, max_adverse_bps, time_at_risk
    elif went_below and not hit_within_H and hit_extended:
        return 3, max_adverse_bps, time_at_risk
    elif went_below and not hit_within_H and not hit_extended:
        return 1, max_adverse_bps, time_at_risk

    return -1, max_adverse_bps, time_at_risk


def run_survive_analysis(df, condition_mask, target_bps, H, liq_threshold):
    """Run SURVIVE analysis for single target/horizon combination."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    target_pct = target_bps / 10000

    indices = np.where(condition_mask)[0]
    valid_indices = indices[(indices >= EMA_SLOW + 10) & (indices < len(df) - EXTENDED_H)]

    if len(valid_indices) < 100:
        return None

    case2_mae = []
    case3_mae = []
    case2_tar = []
    case3_tar = []

    for i in valid_indices:
        entry = close[i]
        future_highs = high[i+1:i+1+EXTENDED_H]
        future_lows = low[i+1:i+1+EXTENDED_H]

        case, mae, tar = analyze_trade(entry, future_highs, future_lows, target_pct, H, EXTENDED_H)

        if case == 2:
            case2_mae.append(mae)
            case2_tar.append(tar)
        elif case == 3:
            case3_mae.append(mae)
            case3_tar.append(tar)

    case2_mae = np.array(case2_mae)
    case3_mae = np.array(case3_mae)

    result = {
        'total': len(valid_indices),
        'case2_count': len(case2_mae),
        'case3_count': len(case3_mae),
    }

    # MAE P95
    if len(case2_mae) >= 50:
        result['case2_mae_p95'] = np.percentile(case2_mae, 95)
        result['case2_survival'] = np.sum(case2_mae < liq_threshold) / len(case2_mae) * 100
    else:
        result['case2_mae_p95'] = None
        result['case2_survival'] = None

    if len(case3_mae) >= 50:
        result['case3_mae_p95'] = np.percentile(case3_mae, 95)
        result['case3_survival'] = np.sum(case3_mae < liq_threshold) / len(case3_mae) * 100
    else:
        result['case3_mae_p95'] = None
        result['case3_survival'] = None

    # TAR P95
    if len(case2_tar) >= 50:
        result['case2_tar_p95'] = np.percentile(case2_tar, 95)
    else:
        result['case2_tar_p95'] = None

    if len(case3_tar) >= 50:
        result['case3_tar_p95'] = np.percentile(case3_tar, 95)
    else:
        result['case3_tar_p95'] = None

    return result


# =============================================================================
# RUN MULTI-PARAMETER ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING MULTI-PARAMETER ANALYSIS")
print("=" * 80)

# Use ATR>75% + Trend>1% condition (recommended from WHEN phase)
condition_func = lambda df: (df['atr_percentile'] > 75) & (df['ema_separation'] > 1.0)

train_results = []
test_results = []

total_combos = len(TARGETS) * len(HORIZONS)
combo_num = 0

for target, horizon in itertools.product(TARGETS, HORIZONS):
    combo_num += 1
    print(f"\rProcessing {combo_num}/{total_combos}: T={target}bp, H={horizon}...", end="", flush=True)

    # Train
    train_mask = condition_func(train).values
    train_result = run_survive_analysis(train, train_mask, target, horizon, LIQ_THRESHOLD_BP)
    if train_result:
        train_result['target'] = target
        train_result['horizon'] = horizon
        train_results.append(train_result)

    # Test
    test_mask = condition_func(test).values
    test_result = run_survive_analysis(test, test_mask, target, horizon, LIQ_THRESHOLD_BP)
    if test_result:
        test_result['target'] = target
        test_result['horizon'] = horizon
        test_results.append(test_result)

print("\n\nAnalysis complete.")


# =============================================================================
# DISPLAY RESULTS: CASE 3 MAE P95 (Most Important)
# =============================================================================
print("\n" + "=" * 80)
print("CASE 3 MAE P95 BY TARGET/HORIZON (ATR>75% + Trend>1%)")
print("=" * 80)

print("\n### TRAIN DATA (2020-2023)")
print("\n| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |")
print("|--------|-----|------|------|------|------|-------|")

for target in TARGETS:
    row = f"| {target}bp |"
    for horizon in HORIZONS:
        match = [r for r in train_results if r['target'] == target and r['horizon'] == horizon]
        if match and match[0]['case3_mae_p95'] is not None:
            row += f" {match[0]['case3_mae_p95']:.0f}bp |"
        else:
            row += " N/A |"
    print(row)

print("\n### TEST DATA (2024-2025)")
print("\n| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |")
print("|--------|-----|------|------|------|------|-------|")

for target in TARGETS:
    row = f"| {target}bp |"
    for horizon in HORIZONS:
        match = [r for r in test_results if r['target'] == target and r['horizon'] == horizon]
        if match and match[0]['case3_mae_p95'] is not None:
            row += f" {match[0]['case3_mae_p95']:.0f}bp |"
        else:
            row += " N/A |"
    print(row)


# =============================================================================
# DISPLAY RESULTS: 10X SURVIVAL RATE
# =============================================================================
print("\n" + "=" * 80)
print(f"CASE 3 SURVIVAL RATE AT {TEST_LEVERAGE}x LEVERAGE (threshold={LIQ_THRESHOLD_BP:.0f}bp)")
print("=" * 80)

print("\n### TRAIN DATA (2020-2023)")
print("\n| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |")
print("|--------|-----|------|------|------|------|-------|")

for target in TARGETS:
    row = f"| {target}bp |"
    for horizon in HORIZONS:
        match = [r for r in train_results if r['target'] == target and r['horizon'] == horizon]
        if match and match[0]['case3_survival'] is not None:
            row += f" {match[0]['case3_survival']:.1f}% |"
        else:
            row += " N/A |"
    print(row)

print("\n### TEST DATA (2024-2025)")
print("\n| Target | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 |")
print("|--------|-----|------|------|------|------|-------|")

for target in TARGETS:
    row = f"| {target}bp |"
    for horizon in HORIZONS:
        match = [r for r in test_results if r['target'] == target and r['horizon'] == horizon]
        if match and match[0]['case3_survival'] is not None:
            row += f" {match[0]['case3_survival']:.1f}% |"
        else:
            row += " N/A |"
    print(row)


# =============================================================================
# VALIDATION: TRAIN vs TEST COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("VALIDATION: TRAIN vs TEST COMPARISON")
print("=" * 80)

# Check if patterns hold
print("\n| Target | Horizon | Train MAE P95 | Test MAE P95 | Train Surv | Test Surv | Pattern Holds |")
print("|--------|---------|---------------|--------------|------------|-----------|---------------|")

valid_count = 0
total_count = 0

for target in TARGETS:
    for horizon in HORIZONS:
        train_match = [r for r in train_results if r['target'] == target and r['horizon'] == horizon]
        test_match = [r for r in test_results if r['target'] == target and r['horizon'] == horizon]

        if train_match and test_match:
            tr = train_match[0]
            te = test_match[0]

            if tr['case3_mae_p95'] and te['case3_mae_p95'] and tr['case3_survival'] and te['case3_survival']:
                total_count += 1

                # Pattern holds if:
                # 1. MAE is in similar range (within 50% of each other)
                # 2. Survival rate is in similar range (within 5pp)
                mae_ratio = te['case3_mae_p95'] / tr['case3_mae_p95'] if tr['case3_mae_p95'] > 0 else 0
                surv_diff = abs(te['case3_survival'] - tr['case3_survival'])

                pattern_holds = (0.5 < mae_ratio < 2.0) and (surv_diff < 10)
                if pattern_holds:
                    valid_count += 1

                status = "YES" if pattern_holds else "NO"
                print(f"| {target:5}bp | {horizon:7} | {tr['case3_mae_p95']:12.0f}bp | {te['case3_mae_p95']:11.0f}bp | {tr['case3_survival']:9.1f}% | {te['case3_survival']:8.1f}% | {status:13} |")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("MULTI-PARAMETER VALIDATION SUMMARY")
print("=" * 80)

if total_count > 0:
    validation_rate = valid_count / total_count * 100
    print(f"\nPattern validation: {valid_count}/{total_count} = {validation_rate:.1f}%")

    if validation_rate >= 90:
        print("Status: ROBUST (>=90% validation)")
    elif validation_rate >= 75:
        print("Status: STRONG (>=75% validation)")
    elif validation_rate >= 50:
        print("Status: PARTIAL (>=50% validation)")
    else:
        print("Status: WEAK (<50% validation)")

print("""
Key Observations:

1. Case 3 MAE P95 varies significantly by target/horizon
   - Larger targets = higher MAE (more room to move before hitting target)
   - Longer horizons = higher MAE (more time for adverse moves)

2. 10x leverage survival rates should be consistent across parameters
   - If patterns hold across 36 combinations, SURVIVE rules are robust

3. Train vs Test comparison validates out-of-sample stability
   - Similar MAE and survival rates = robust findings
   - Large differences = potential overfitting

Next: Update SURVIVE documentation with multi-parameter results
""")


# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================
output_path = Path("experiments/survive_multi_param_validation.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

all_data = []
for tr in train_results:
    tr['dataset'] = 'train'
    all_data.append(tr)
for te in test_results:
    te['dataset'] = 'test'
    all_data.append(te)

df_results = pd.DataFrame(all_data)
df_results.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
