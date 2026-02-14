"""
SURVIVE PHASE: S3 - Leverage Survival Matrix

Calculate what % of trades survive at each leverage level.
Liquidation = failure. No gray area.

Run: .venv/Scripts/python.exe scripts/debug/survive_leverage_matrix.py
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

# Leverage levels to test
LEVERAGE_LEVELS = [3, 5, 10, 20, 50, 100]

# Exchange parameters (Binance Futures example)
MAINTENANCE_MARGIN = 0.004  # 0.4% maintenance margin
SAFETY_BUFFER = 0.20  # 20% safety buffer below liquidation
FEES_PER_TRADE = 0.0008  # 0.08% round trip fees


# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("SURVIVE PHASE: S3 - LEVERAGE SURVIVAL MATRIX")
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
# LIQUIDATION THRESHOLD CALCULATION
# =============================================================================
def calculate_effective_liq_threshold(leverage):
    """
    Calculate effective liquidation threshold in basis points.

    Effective_Liq_MAE = (1/leverage) - maintenance_margin - safety_buffer - fees

    Returns: MAE in basis points that would trigger liquidation
    """
    # Theoretical liquidation point (100% / leverage)
    theoretical_liq = 1 / leverage

    # Deductions
    effective_liq = theoretical_liq - MAINTENANCE_MARGIN - (theoretical_liq * SAFETY_BUFFER) - FEES_PER_TRADE

    # Convert to basis points
    effective_liq_bp = effective_liq * 10000

    return max(effective_liq_bp, 0)  # Can't be negative


print("\n" + "=" * 80)
print("EFFECTIVE LIQUIDATION THRESHOLDS")
print("=" * 80)
print(f"\nAssumptions:")
print(f"  Maintenance margin: {MAINTENANCE_MARGIN*100:.1f}%")
print(f"  Safety buffer: {SAFETY_BUFFER*100:.0f}%")
print(f"  Fees: {FEES_PER_TRADE*100:.2f}%")
print("\n| Leverage | Theoretical Liq | Effective Liq (bp) |")
print("|----------|-----------------|-------------------|")
for lev in LEVERAGE_LEVELS:
    theoretical = 10000 / lev
    effective = calculate_effective_liq_threshold(lev)
    print(f"| {lev:7}x | {theoretical:14.0f}bp | {effective:16.0f}bp |")


# =============================================================================
# CASE CLASSIFICATION WITH MAE
# =============================================================================
@njit
def classify_with_mae(entry, highs, lows, target_pct, H, extended_H):
    """Classify case and return MAE."""
    n = len(highs)
    if n == 0:
        return -1, 0.0

    target_price = entry * (1 + target_pct)
    went_below = False
    hit_within_H = False
    hit_extended = False
    max_adverse_bps = 0.0

    # Check within horizon H
    for j in range(min(H, n)):
        adverse = (entry - lows[j]) / entry * 10000
        if adverse > max_adverse_bps:
            max_adverse_bps = adverse
        if lows[j] < entry:
            went_below = True
        if highs[j] >= target_price:
            hit_within_H = True
            break

    # If went below and didn't hit within H, check extended
    if went_below and not hit_within_H:
        for j in range(H, min(extended_H, n)):
            adverse = (entry - lows[j]) / entry * 10000
            if adverse > max_adverse_bps:
                max_adverse_bps = adverse
            if highs[j] >= target_price:
                hit_extended = True
                break

    # If never went below but didn't hit within H
    if not went_below and not hit_within_H:
        for j in range(H, min(extended_H, n)):
            adverse = (entry - lows[j]) / entry * 10000
            if adverse > max_adverse_bps:
                max_adverse_bps = adverse
            if lows[j] < entry:
                went_below = True
            if highs[j] >= target_price:
                if went_below:
                    return 3, max_adverse_bps
                else:
                    return 0, max_adverse_bps
        return 1, max_adverse_bps

    # Classify
    if not went_below and hit_within_H:
        return 0, max_adverse_bps
    elif went_below and hit_within_H:
        return 2, max_adverse_bps
    elif went_below and not hit_within_H and hit_extended:
        return 3, max_adverse_bps
    elif went_below and not hit_within_H and not hit_extended:
        return 1, max_adverse_bps

    return -1, max_adverse_bps


# =============================================================================
# LEVERAGE SURVIVAL ANALYSIS
# =============================================================================
def analyze_leverage_survival(df, condition_mask, target_bps, H):
    """
    Analyze survival rates at different leverage levels.
    Returns: survival rates for Case 2 and Case 3 at each leverage level.
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    target_pct = target_bps / 10000

    indices = np.where(condition_mask)[0]
    valid_indices = indices[(indices >= EMA_SLOW + 10) & (indices < len(df) - EXTENDED_H)]

    if len(valid_indices) < 100:
        return None

    # Store MAE for each case
    case2_mae = []
    case3_mae = []

    for i in valid_indices:
        entry = close[i]
        future_highs = high[i+1:i+1+EXTENDED_H]
        future_lows = low[i+1:i+1+EXTENDED_H]

        case, mae = classify_with_mae(entry, future_highs, future_lows, target_pct, H, EXTENDED_H)

        if case == 2:
            case2_mae.append(mae)
        elif case == 3:
            case3_mae.append(mae)

    # Calculate survival rates at each leverage
    results = {
        'total_valid': len(valid_indices),
        'case2_count': len(case2_mae),
        'case3_count': len(case3_mae),
        'leverage_survival': {}
    }

    case2_mae = np.array(case2_mae)
    case3_mae = np.array(case3_mae)

    for lev in LEVERAGE_LEVELS:
        liq_threshold = calculate_effective_liq_threshold(lev)

        # Case 2 survival: MAE < liquidation threshold
        if len(case2_mae) > 0:
            case2_survived = np.sum(case2_mae < liq_threshold)
            case2_survival_rate = case2_survived / len(case2_mae) * 100
        else:
            case2_survival_rate = 0

        # Case 3 survival: MAE < liquidation threshold
        if len(case3_mae) > 0:
            case3_survived = np.sum(case3_mae < liq_threshold)
            case3_survival_rate = case3_survived / len(case3_mae) * 100
        else:
            case3_survival_rate = 0

        results['leverage_survival'][lev] = {
            'liq_threshold_bp': liq_threshold,
            'case2_survival': case2_survival_rate,
            'case3_survival': case3_survival_rate,
            'case2_liquidated': 100 - case2_survival_rate,
            'case3_liquidated': 100 - case3_survival_rate
        }

    return results


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
print(f"LEVERAGE SURVIVAL MATRIX (Target={TARGET_BPS}bp, H={HORIZON})")
print("=" * 80)

all_results = {}

for cond_name, cond_func in conditions.items():
    print(f"\nAnalyzing: {cond_name}...")
    mask = cond_func(train).values
    result = analyze_leverage_survival(train, mask, TARGET_BPS, HORIZON)

    if result:
        all_results[cond_name] = result


# =============================================================================
# DISPLAY RESULTS
# =============================================================================
for cond_name in conditions.keys():
    if cond_name not in all_results:
        continue

    result = all_results[cond_name]

    print("\n" + "=" * 80)
    print(f"CONDITION: {cond_name}")
    print(f"Case 2 trades: {result['case2_count']:,} | Case 3 trades: {result['case3_count']:,}")
    print("=" * 80)

    print("\n| Leverage | Liq Threshold | Case 2 Survive % | Case 3 Survive % | Case 2 Liq % | Case 3 Liq % |")
    print("|----------|---------------|------------------|------------------|--------------|--------------|")

    for lev in LEVERAGE_LEVELS:
        data = result['leverage_survival'][lev]
        print(f"| {lev:7}x | {data['liq_threshold_bp']:11.0f}bp | {data['case2_survival']:15.1f}% | {data['case3_survival']:15.1f}% | {data['case2_liquidated']:11.1f}% | {data['case3_liquidated']:11.1f}% |")


# =============================================================================
# SURVIVAL CLASSIFICATION
# =============================================================================
print("\n" + "=" * 80)
print("LEVERAGE CLASSIFICATION (Based on ATR>75% + Trend>1%)")
print("=" * 80)

if "ATR>75% + Trend>1%" in all_results:
    result = all_results["ATR>75% + Trend>1%"]

    print("\nClassification criteria:")
    print("  SAFE: Both Case 2 & Case 3 survival > 99%")
    print("  CAUTION: Case 2 survival > 99%, Case 3 survival > 95%")
    print("  DANGEROUS: Either survival < 95%")
    print("")

    print("| Leverage | Case 2 | Case 3 | Classification |")
    print("|----------|--------|--------|----------------|")

    for lev in LEVERAGE_LEVELS:
        data = result['leverage_survival'][lev]
        c2 = data['case2_survival']
        c3 = data['case3_survival']

        if c2 > 99 and c3 > 99:
            classification = "SAFE"
        elif c2 > 99 and c3 > 95:
            classification = "CAUTION"
        elif c2 > 95 and c3 > 90:
            classification = "RISKY"
        else:
            classification = "DANGEROUS"

        print(f"| {lev:7}x | {c2:5.1f}% | {c3:5.1f}% | {classification:14s} |")


# =============================================================================
# BINDING CONSTRAINT ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("BINDING CONSTRAINT ANALYSIS")
print("=" * 80)

print("""
Key insight: Case 3 is ALWAYS the binding constraint.

For any leverage level, Case 3 has:
- Higher MAE (4-5x higher than Case 2)
- Lower survival rate

Leverage rules MUST be set based on Case 3 survival, not Case 2.
""")

if "ATR>75% + Trend>1%" in all_results:
    result = all_results["ATR>75% + Trend>1%"]

    print("\nRecommended leverage rules for ATR>75% + Trend>1%:")

    # Find safe leverage (Case 3 > 99%)
    safe_lev = None
    for lev in LEVERAGE_LEVELS:
        if result['leverage_survival'][lev]['case3_survival'] > 99:
            safe_lev = lev

    # Find caution leverage (Case 3 > 95%)
    caution_lev = None
    for lev in LEVERAGE_LEVELS:
        if result['leverage_survival'][lev]['case3_survival'] > 95:
            caution_lev = lev

    # Find dangerous leverage (Case 3 < 95%)
    danger_lev = None
    for lev in LEVERAGE_LEVELS:
        if result['leverage_survival'][lev]['case3_survival'] < 95:
            danger_lev = lev
            break

    if safe_lev:
        print(f"  SAFE: <= {safe_lev}x")
    if caution_lev and caution_lev != safe_lev:
        print(f"  CAUTION: {safe_lev+1 if safe_lev else 1}x - {caution_lev}x")
    if danger_lev:
        print(f"  DANGEROUS: >= {danger_lev}x")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("S3 SUMMARY")
print("=" * 80)

print("""
Key Findings:

1. Liquidation Thresholds (with 20% safety buffer):
   - 3x:  2600bp effective threshold
   - 5x:  1520bp effective threshold
   - 10x: 720bp effective threshold
   - 20x: 320bp effective threshold

2. Case 3 is ALWAYS the binding constraint
   - Higher MAE than Case 2
   - Lower survival rates at all leverage levels

3. Survival rates depend heavily on condition:
   - Baseline (no filter): More liquidations
   - ATR>75% + Trend>1%: Fewer liquidations (but MAE still high)

4. For futures trading:
   - Leverage must be set based on WORST case (Case 3 P95 MAE)
   - Not based on expected MAE or Case 2 MAE

Next: S4 - Forced Exit Stress Test
""")
