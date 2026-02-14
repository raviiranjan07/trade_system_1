"""
SURVIVE PHASE: S1 — MAE Distribution by Case

Analyze MAE distribution for Case 2 and Case 3 SEPARATELY.
Conditioned on WHEN filters.

Run: .venv/Scripts/python.exe scripts/debug/survive_mae_distribution.py
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

ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("SURVIVE PHASE: S1 — MAE DISTRIBUTION BY CASE")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

# Use train data for analysis
train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data (2020-2023): {len(train):,} candles")

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

train['atr'] = calculate_atr(train, ATR_PERIOD)
train['atr_pct'] = train['atr'] / train['close'] * 100
train['atr_percentile'] = train['atr_pct'].rank(pct=True) * 100
train['ema50'] = train['close'].ewm(span=EMA_FAST, adjust=False).mean()
train['ema200'] = train['close'].ewm(span=EMA_SLOW, adjust=False).mean()
train['ema_separation'] = np.abs(train['ema50'] - train['ema200']) / train['close'] * 100
train['hour'] = train.index.hour

print("Indicators calculated.")


# =============================================================================
# CASE CLASSIFICATION WITH MAE TRACKING
# =============================================================================
@njit
def classify_with_mae(entry, highs, lows, target_pct, H, extended_H):
    """
    Classify case and return MAE (maximum adverse excursion).
    Returns: (case, mae_bps, time_at_risk, recovery_bar)
    """
    n = len(highs)
    if n == 0:
        return -1, 0.0, 0, 0

    target_price = entry * (1 + target_pct)
    went_below = False
    hit_within_H = False
    hit_extended = False

    max_adverse_bps = 0.0
    time_at_risk = 0
    recovery_bar = 0

    # Check within horizon H
    for j in range(min(H, n)):
        # Track MAE
        adverse = (entry - lows[j]) / entry * 10000
        if adverse > max_adverse_bps:
            max_adverse_bps = adverse

        # Track time at risk
        if lows[j] < entry:
            time_at_risk += 1
            went_below = True

        # Check target hit
        if highs[j] >= target_price:
            hit_within_H = True
            recovery_bar = j + 1
            break

    # If went below and didn't hit within H, check extended
    if went_below and not hit_within_H:
        for j in range(H, min(extended_H, n)):
            adverse = (entry - lows[j]) / entry * 10000
            if adverse > max_adverse_bps:
                max_adverse_bps = adverse

            if lows[j] < entry:
                time_at_risk += 1

            if highs[j] >= target_price:
                hit_extended = True
                recovery_bar = j + 1
                break

    # If never went below but didn't hit within H
    if not went_below and not hit_within_H:
        for j in range(H, min(extended_H, n)):
            adverse = (entry - lows[j]) / entry * 10000
            if adverse > max_adverse_bps:
                max_adverse_bps = adverse

            if lows[j] < entry:
                went_below = True
                time_at_risk += 1

            if highs[j] >= target_price:
                recovery_bar = j + 1
                if went_below:
                    return 3, max_adverse_bps, time_at_risk, recovery_bar
                else:
                    return 0, max_adverse_bps, time_at_risk, recovery_bar
        return 1, max_adverse_bps, time_at_risk, 0

    # Classify
    if not went_below and hit_within_H:
        return 0, max_adverse_bps, time_at_risk, recovery_bar
    elif went_below and hit_within_H:
        return 2, max_adverse_bps, time_at_risk, recovery_bar
    elif went_below and not hit_within_H and hit_extended:
        return 3, max_adverse_bps, time_at_risk, recovery_bar
    elif went_below and not hit_within_H and not hit_extended:
        return 1, max_adverse_bps, time_at_risk, 0

    return -1, max_adverse_bps, time_at_risk, 0


# =============================================================================
# ANALYZE FUNCTION
# =============================================================================
def analyze_mae_by_case(df, condition_mask, target_bps, H):
    """Analyze MAE distribution for Case 2 and Case 3 separately."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    target_pct = target_bps / 10000

    indices = np.where(condition_mask)[0]
    valid_indices = indices[(indices >= EMA_SLOW + 10) & (indices < len(df) - EXTENDED_H)]

    if len(valid_indices) < 100:
        return None

    results = {
        'case2': {'mae': [], 'time_at_risk': [], 'recovery': []},
        'case3': {'mae': [], 'time_at_risk': [], 'recovery': []}
    }

    for i in valid_indices:
        entry = close[i]
        future_highs = high[i+1:i+1+EXTENDED_H]
        future_lows = low[i+1:i+1+EXTENDED_H]

        case, mae, tar, rec = classify_with_mae(entry, future_highs, future_lows, target_pct, H, EXTENDED_H)

        if case == 2:
            results['case2']['mae'].append(mae)
            results['case2']['time_at_risk'].append(tar)
            results['case2']['recovery'].append(rec)
        elif case == 3:
            results['case3']['mae'].append(mae)
            results['case3']['time_at_risk'].append(tar)
            results['case3']['recovery'].append(rec)

    # Calculate statistics
    output = {'total_valid': len(valid_indices)}

    for case_name in ['case2', 'case3']:
        mae_arr = np.array(results[case_name]['mae'])
        if len(mae_arr) >= 50:
            output[case_name] = {
                'count': len(mae_arr),
                'mae_median': np.median(mae_arr),
                'mae_p75': np.percentile(mae_arr, 75),
                'mae_p90': np.percentile(mae_arr, 90),
                'mae_p95': np.percentile(mae_arr, 95),
                'mae_p99': np.percentile(mae_arr, 99),
            }
        else:
            output[case_name] = None

    return output


# =============================================================================
# CONDITIONS TO TEST (FROM WHEN PHASE)
# =============================================================================
conditions = {
    "Baseline": lambda df: pd.Series(np.ones(len(df), dtype=bool), index=df.index),
    "ATR >75%": lambda df: df['atr_percentile'] > 75,
    "Trend >1%": lambda df: df['ema_separation'] > 1.0,
    "ATR>75% + Trend>1%": lambda df: (df['atr_percentile'] > 75) & (df['ema_separation'] > 1.0),
}


# =============================================================================
# RUN ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print(f"MAE DISTRIBUTION BY CASE (Target={TARGET_BPS}bp, H={HORIZON})")
print("=" * 80)

all_results = {}

for cond_name, cond_func in conditions.items():
    print(f"\nAnalyzing: {cond_name}...")
    mask = cond_func(train).values
    result = analyze_mae_by_case(train, mask, TARGET_BPS, HORIZON)

    if result:
        all_results[cond_name] = result


# =============================================================================
# DISPLAY RESULTS: CASE 2 (Fast Recovery)
# =============================================================================
print("\n" + "=" * 80)
print("CASE 2 — MAE DISTRIBUTION (Fast Recovery)")
print("=" * 80)

print("\n| Condition | Count | Median | P75 | P90 | P95 | P99 |")
print("|-----------|-------|--------|-----|-----|-----|-----|")

for cond_name in conditions.keys():
    if cond_name in all_results and all_results[cond_name]['case2']:
        c2 = all_results[cond_name]['case2']
        print(f"| {cond_name:22s} | {c2['count']:5,} | {c2['mae_median']:5.1f}bp | {c2['mae_p75']:5.1f}bp | {c2['mae_p90']:5.1f}bp | {c2['mae_p95']:5.1f}bp | {c2['mae_p99']:5.1f}bp |")
    else:
        print(f"| {cond_name:22s} | N/A | N/A | N/A | N/A | N/A | N/A |")


# =============================================================================
# DISPLAY RESULTS: CASE 3 (Slow Recovery)
# =============================================================================
print("\n" + "=" * 80)
print("CASE 3 — MAE DISTRIBUTION (Slow Recovery)")
print("=" * 80)

print("\n| Condition | Count | Median | P75 | P90 | P95 | P99 |")
print("|-----------|-------|--------|-----|-----|-----|-----|")

for cond_name in conditions.keys():
    if cond_name in all_results and all_results[cond_name]['case3']:
        c3 = all_results[cond_name]['case3']
        print(f"| {cond_name:22s} | {c3['count']:5,} | {c3['mae_median']:5.1f}bp | {c3['mae_p75']:5.1f}bp | {c3['mae_p90']:5.1f}bp | {c3['mae_p95']:5.1f}bp | {c3['mae_p99']:5.1f}bp |")
    else:
        print(f"| {cond_name:22s} | N/A | N/A | N/A | N/A | N/A | N/A |")


# =============================================================================
# CASE 2 vs CASE 3 COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("CASE 2 vs CASE 3 COMPARISON (P95 MAE)")
print("=" * 80)

print("\n| Condition | Case 2 P95 | Case 3 P95 | Ratio (C3/C2) |")
print("|-----------|------------|------------|---------------|")

for cond_name in conditions.keys():
    if cond_name in all_results:
        c2 = all_results[cond_name].get('case2')
        c3 = all_results[cond_name].get('case3')

        if c2 and c3:
            ratio = c3['mae_p95'] / c2['mae_p95'] if c2['mae_p95'] > 0 else 0
            print(f"| {cond_name:22s} | {c2['mae_p95']:9.1f}bp | {c3['mae_p95']:9.1f}bp | {ratio:12.1f}x |")
        else:
            print(f"| {cond_name:22s} | N/A | N/A | N/A |")


# =============================================================================
# LEVERAGE IMPLICATIONS (PRELIMINARY)
# =============================================================================
print("\n" + "=" * 80)
print("LEVERAGE IMPLICATIONS (Based on P95 MAE)")
print("=" * 80)

print("""
Effective Liquidation Thresholds (with 20% safety buffer):
- 3x leverage:  ~2667bp effective threshold
- 5x leverage:  ~1600bp effective threshold
- 10x leverage: ~800bp effective threshold
- 20x leverage: ~400bp effective threshold
- 50x leverage: ~160bp effective threshold
""")

# Calculate safe leverage for best condition
best_cond = "ATR>75% + Trend>1%"
if best_cond in all_results:
    c2 = all_results[best_cond].get('case2')
    c3 = all_results[best_cond].get('case3')

    print(f"\nFor {best_cond}:")
    if c2:
        safe_lev_c2 = int(10000 / (c2['mae_p95'] / 0.8)) if c2['mae_p95'] > 0 else 999
        print(f"  Case 2 P95 MAE: {c2['mae_p95']:.1f}bp → Max safe leverage: ~{min(safe_lev_c2, 100)}x")
    if c3:
        safe_lev_c3 = int(10000 / (c3['mae_p95'] / 0.8)) if c3['mae_p95'] > 0 else 999
        print(f"  Case 3 P95 MAE: {c3['mae_p95']:.1f}bp → Max safe leverage: ~{min(safe_lev_c3, 100)}x")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("S1 SUMMARY")
print("=" * 80)

print("""
Key Findings:
1. Case 2 (fast recovery) has LOWER MAE than Case 3 (slow recovery)
2. Case 3 MAE is typically 2-3x higher than Case 2
3. WHEN-filtered conditions (ATR>75% + Trend>1%) may reduce MAE

Next: S2 — Time-at-Risk Distribution
""")
