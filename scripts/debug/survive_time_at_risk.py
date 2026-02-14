"""
SURVIVE PHASE: S2 - Time-at-Risk Distribution

Analyze how long trades stay underwater (unrealized PnL < 0).
This is CAPITAL LOCK TIME, not drawdown.

Run: .venv/Scripts/python.exe scripts/debug/survive_time_at_risk.py
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
print("SURVIVE PHASE: S2 - TIME-AT-RISK DISTRIBUTION")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

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

print("Indicators calculated.")


# =============================================================================
# CASE CLASSIFICATION WITH TIME-AT-RISK TRACKING
# =============================================================================
@njit
def classify_with_time_at_risk(entry, highs, lows, target_pct, H, extended_H):
    """
    Classify case and return time-at-risk metrics.
    Returns: (case, mae_bps, time_at_risk, recovery_bar, time_to_first_profit)
    """
    n = len(highs)
    if n == 0:
        return -1, 0.0, 0, 0, 0

    target_price = entry * (1 + target_pct)
    went_below = False
    hit_within_H = False
    hit_extended = False

    max_adverse_bps = 0.0
    time_at_risk = 0  # bars where low < entry
    recovery_bar = 0
    time_to_first_profit = 0  # bars until first high > entry
    found_first_profit = False

    # Check within horizon H
    for j in range(min(H, n)):
        # Track MAE
        adverse = (entry - lows[j]) / entry * 10000
        if adverse > max_adverse_bps:
            max_adverse_bps = adverse

        # Track time at risk (underwater)
        if lows[j] < entry:
            time_at_risk += 1
            went_below = True

        # Track time to first profit
        if not found_first_profit and highs[j] > entry:
            time_to_first_profit = j + 1
            found_first_profit = True

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

            if not found_first_profit and highs[j] > entry:
                time_to_first_profit = j + 1
                found_first_profit = True

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

            if not found_first_profit and highs[j] > entry:
                time_to_first_profit = j + 1
                found_first_profit = True

            if highs[j] >= target_price:
                recovery_bar = j + 1
                if went_below:
                    return 3, max_adverse_bps, time_at_risk, recovery_bar, time_to_first_profit
                else:
                    return 0, max_adverse_bps, time_at_risk, recovery_bar, time_to_first_profit
        return 1, max_adverse_bps, time_at_risk, 0, time_to_first_profit

    # Classify
    if not went_below and hit_within_H:
        return 0, max_adverse_bps, time_at_risk, recovery_bar, time_to_first_profit
    elif went_below and hit_within_H:
        return 2, max_adverse_bps, time_at_risk, recovery_bar, time_to_first_profit
    elif went_below and not hit_within_H and hit_extended:
        return 3, max_adverse_bps, time_at_risk, recovery_bar, time_to_first_profit
    elif went_below and not hit_within_H and not hit_extended:
        return 1, max_adverse_bps, time_at_risk, 0, time_to_first_profit

    return -1, max_adverse_bps, time_at_risk, 0, time_to_first_profit


# =============================================================================
# ANALYZE FUNCTION
# =============================================================================
def analyze_time_at_risk(df, condition_mask, target_bps, H):
    """Analyze time-at-risk distribution for Case 2 and Case 3 separately."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    target_pct = target_bps / 10000

    indices = np.where(condition_mask)[0]
    valid_indices = indices[(indices >= EMA_SLOW + 10) & (indices < len(df) - EXTENDED_H)]

    if len(valid_indices) < 100:
        return None

    results = {
        'case2': {'time_at_risk': [], 'recovery': []},
        'case3': {'time_at_risk': [], 'recovery': []}
    }

    for i in valid_indices:
        entry = close[i]
        future_highs = high[i+1:i+1+EXTENDED_H]
        future_lows = low[i+1:i+1+EXTENDED_H]

        case, mae, tar, rec, ttp = classify_with_time_at_risk(
            entry, future_highs, future_lows, target_pct, H, EXTENDED_H
        )

        if case == 2:
            results['case2']['time_at_risk'].append(tar)
            results['case2']['recovery'].append(rec)
        elif case == 3:
            results['case3']['time_at_risk'].append(tar)
            results['case3']['recovery'].append(rec)

    # Calculate statistics
    output = {'total_valid': len(valid_indices)}

    for case_name in ['case2', 'case3']:
        tar_arr = np.array(results[case_name]['time_at_risk'])
        rec_arr = np.array(results[case_name]['recovery'])

        if len(tar_arr) >= 50:
            output[case_name] = {
                'count': len(tar_arr),
                'tar_median': np.median(tar_arr),
                'tar_p75': np.percentile(tar_arr, 75),
                'tar_p90': np.percentile(tar_arr, 90),
                'tar_p95': np.percentile(tar_arr, 95),
                'tar_p99': np.percentile(tar_arr, 99),
                'rec_median': np.median(rec_arr),
                'rec_p75': np.percentile(rec_arr, 75),
                'rec_p90': np.percentile(rec_arr, 90),
                'rec_p95': np.percentile(rec_arr, 95),
            }
        else:
            output[case_name] = None

    return output


# =============================================================================
# CONDITIONS TO TEST
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
print(f"TIME-AT-RISK DISTRIBUTION (Target={TARGET_BPS}bp, H={HORIZON})")
print("=" * 80)

all_results = {}

for cond_name, cond_func in conditions.items():
    print(f"\nAnalyzing: {cond_name}...")
    mask = cond_func(train).values
    result = analyze_time_at_risk(train, mask, TARGET_BPS, HORIZON)

    if result:
        all_results[cond_name] = result


# =============================================================================
# DISPLAY RESULTS: CASE 2 TIME-AT-RISK
# =============================================================================
print("\n" + "=" * 80)
print("CASE 2 - TIME-AT-RISK (bars underwater)")
print("=" * 80)

print("\n| Condition | Count | Median | P75 | P90 | P95 | P99 |")
print("|-----------|-------|--------|-----|-----|-----|-----|")

for cond_name in conditions.keys():
    if cond_name in all_results and all_results[cond_name]['case2']:
        c2 = all_results[cond_name]['case2']
        print(f"| {cond_name:22s} | {c2['count']:5,} | {c2['tar_median']:5.0f} | {c2['tar_p75']:5.0f} | {c2['tar_p90']:5.0f} | {c2['tar_p95']:5.0f} | {c2['tar_p99']:5.0f} |")
    else:
        print(f"| {cond_name:22s} | N/A | N/A | N/A | N/A | N/A | N/A |")


# =============================================================================
# DISPLAY RESULTS: CASE 3 TIME-AT-RISK
# =============================================================================
print("\n" + "=" * 80)
print("CASE 3 - TIME-AT-RISK (bars underwater)")
print("=" * 80)

print("\n| Condition | Count | Median | P75 | P90 | P95 | P99 |")
print("|-----------|-------|--------|-----|-----|-----|-----|")

for cond_name in conditions.keys():
    if cond_name in all_results and all_results[cond_name]['case3']:
        c3 = all_results[cond_name]['case3']
        print(f"| {cond_name:22s} | {c3['count']:5,} | {c3['tar_median']:5.0f} | {c3['tar_p75']:5.0f} | {c3['tar_p90']:5.0f} | {c3['tar_p95']:5.0f} | {c3['tar_p99']:5.0f} |")
    else:
        print(f"| {cond_name:22s} | N/A | N/A | N/A | N/A | N/A | N/A |")


# =============================================================================
# DISPLAY RESULTS: RECOVERY TIME
# =============================================================================
print("\n" + "=" * 80)
print("RECOVERY TIME (bars to hit target)")
print("=" * 80)

print("\n### Case 2 Recovery Time")
print("| Condition | Median | P75 | P90 | P95 |")
print("|-----------|--------|-----|-----|-----|")

for cond_name in conditions.keys():
    if cond_name in all_results and all_results[cond_name]['case2']:
        c2 = all_results[cond_name]['case2']
        print(f"| {cond_name:22s} | {c2['rec_median']:5.0f} | {c2['rec_p75']:5.0f} | {c2['rec_p90']:5.0f} | {c2['rec_p95']:5.0f} |")
    else:
        print(f"| {cond_name:22s} | N/A | N/A | N/A | N/A |")

print("\n### Case 3 Recovery Time")
print("| Condition | Median | P75 | P90 | P95 |")
print("|-----------|--------|-----|-----|-----|")

for cond_name in conditions.keys():
    if cond_name in all_results and all_results[cond_name]['case3']:
        c3 = all_results[cond_name]['case3']
        print(f"| {cond_name:22s} | {c3['rec_median']:5.0f} | {c3['rec_p75']:5.0f} | {c3['rec_p90']:5.0f} | {c3['rec_p95']:5.0f} |")
    else:
        print(f"| {cond_name:22s} | N/A | N/A | N/A | N/A |")


# =============================================================================
# CASE 2 vs CASE 3 COMPARISON
# =============================================================================
print("\n" + "=" * 80)
print("CASE 2 vs CASE 3 COMPARISON (P95 Time-at-Risk)")
print("=" * 80)

print("\n| Condition | Case 2 P95 | Case 3 P95 | Ratio (C3/C2) |")
print("|-----------|------------|------------|---------------|")

for cond_name in conditions.keys():
    if cond_name in all_results:
        c2 = all_results[cond_name].get('case2')
        c3 = all_results[cond_name].get('case3')

        if c2 and c3:
            ratio = c3['tar_p95'] / c2['tar_p95'] if c2['tar_p95'] > 0 else 0
            print(f"| {cond_name:22s} | {c2['tar_p95']:9.0f} | {c3['tar_p95']:9.0f} | {ratio:12.1f}x |")
        else:
            print(f"| {cond_name:22s} | N/A | N/A | N/A |")


# =============================================================================
# CAPITAL LOCK IMPLICATIONS
# =============================================================================
print("\n" + "=" * 80)
print("CAPITAL LOCK IMPLICATIONS")
print("=" * 80)

best_cond = "ATR>75% + Trend>1%"
if best_cond in all_results:
    c2 = all_results[best_cond].get('case2')
    c3 = all_results[best_cond].get('case3')

    print(f"\nFor {best_cond}:")
    if c2:
        print(f"  Case 2: P95 Time-at-Risk = {c2['tar_p95']:.0f} bars ({c2['tar_p95']:.0f} minutes)")
        print(f"          P95 Recovery = {c2['rec_p95']:.0f} bars")
    if c3:
        print(f"  Case 3: P95 Time-at-Risk = {c3['tar_p95']:.0f} bars ({c3['tar_p95']:.0f} minutes)")
        print(f"          P95 Recovery = {c3['rec_p95']:.0f} bars")

    if c3:
        hours = c3['tar_p95'] / 60
        print(f"\n  Case 3 capital lock at P95: {hours:.1f} hours")
        print("  This is the TIME constraint for SURVIVE phase")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("S2 SUMMARY")
print("=" * 80)

print("""
Key Findings:
1. Time-at-Risk = bars where position is underwater (unrealized loss)
2. Case 3 has MUCH longer time-at-risk than Case 2
3. This represents CAPITAL LOCK - money tied up, can't use elsewhere
4. Funding costs accumulate during time-at-risk

For SURVIVE phase leverage rules:
- Must consider BOTH MAE (from S1) AND Time-at-Risk (from S2)
- A position is only "survived" if:
  * MAE < liquidation threshold
  * Time-at-Risk < max acceptable lock time

Next: S3 - Leverage Survival Matrix
""")
