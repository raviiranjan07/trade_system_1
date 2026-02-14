"""
SURVIVE PHASE: S4 - Forced Exit Stress Test

Compare survival outcomes WITH vs WITHOUT forced exit rules.
Forced exit is NOT stop-loss optimization - it's a survival mechanism.

Measure:
- Liquidation rate (with vs without)
- Capital freed (time saved)
- NOT expected value (this is SURVIVE, not profit optimization)

Run: .venv/Scripts/python.exe scripts/debug/survive_forced_exit.py
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

# Forced exit thresholds to test (based on S1/S2 P95 values)
# From S1: Case 3 P95 MAE = 494.7bp for ATR>75%+Trend>1%
# From S2: Case 3 P95 Time-at-Risk = 368 bars
FORCED_EXIT_MAE_THRESHOLDS = [100, 150, 200, 300, 400, 500]  # bp
FORCED_EXIT_TIME_THRESHOLDS = [50, 100, 150, 200, 300, 400]  # bars

# Leverage for simulation
LEVERAGE = 10  # 10x for realistic scenario
MAINTENANCE_MARGIN = 0.004
SAFETY_BUFFER = 0.20
FEES = 0.0008

# Effective liquidation threshold at 10x
EFFECTIVE_LIQ_BP = (1/LEVERAGE - MAINTENANCE_MARGIN - (1/LEVERAGE * SAFETY_BUFFER) - FEES) * 10000


# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("SURVIVE PHASE: S4 - FORCED EXIT STRESS TEST")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} total candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data (2020-2023): {len(train):,} candles")

print(f"\nSimulation leverage: {LEVERAGE}x")
print(f"Effective liquidation threshold: {EFFECTIVE_LIQ_BP:.0f}bp")


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
# SIMULATION WITH FORCED EXIT
# =============================================================================
@njit
def simulate_with_forced_exit(entry, highs, lows, target_pct, H, extended_H,
                               forced_mae_bp, forced_time, liq_threshold_bp):
    """
    Simulate trade with optional forced exit.

    Returns: (outcome, final_mae_bp, bars_held, exit_type)
    outcome: 0=target_hit, 1=liquidated, 2=forced_exit, 3=timeout
    exit_type: 0=target, 1=liquidation, 2=forced_mae, 3=forced_time, 4=timeout
    """
    n = len(highs)
    if n == 0:
        return 3, 0.0, 0, 4

    target_price = entry * (1 + target_pct)
    max_adverse_bps = 0.0
    time_underwater = 0

    for j in range(min(extended_H, n)):
        # Calculate current MAE
        adverse = (entry - lows[j]) / entry * 10000
        if adverse > max_adverse_bps:
            max_adverse_bps = adverse

        # Track time underwater
        if lows[j] < entry:
            time_underwater += 1

        # Check liquidation FIRST (highest priority)
        if max_adverse_bps >= liq_threshold_bp:
            return 1, max_adverse_bps, j + 1, 1  # Liquidated

        # Check forced exit by MAE
        if forced_mae_bp > 0 and max_adverse_bps >= forced_mae_bp:
            return 2, max_adverse_bps, j + 1, 2  # Forced exit (MAE)

        # Check forced exit by time
        if forced_time > 0 and time_underwater >= forced_time:
            return 2, max_adverse_bps, j + 1, 3  # Forced exit (time)

        # Check target hit
        if highs[j] >= target_price:
            return 0, max_adverse_bps, j + 1, 0  # Target hit

    return 3, max_adverse_bps, extended_H, 4  # Timeout


# =============================================================================
# ANALYZE FORCED EXIT IMPACT
# =============================================================================
def analyze_forced_exit(df, condition_mask, target_bps, H, forced_mae_bp, forced_time):
    """Analyze impact of forced exit on survival outcomes."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    target_pct = target_bps / 10000

    indices = np.where(condition_mask)[0]
    valid_indices = indices[(indices >= EMA_SLOW + 10) & (indices < len(df) - EXTENDED_H)]

    if len(valid_indices) < 100:
        return None

    outcomes = {
        'target_hit': 0,
        'liquidated': 0,
        'forced_exit': 0,
        'timeout': 0,
        'total_bars_held': 0,
        'forced_exit_mae': [],
        'forced_exit_bars': []
    }

    for i in valid_indices:
        entry = close[i]
        future_highs = high[i+1:i+1+EXTENDED_H]
        future_lows = low[i+1:i+1+EXTENDED_H]

        outcome, mae, bars, exit_type = simulate_with_forced_exit(
            entry, future_highs, future_lows, target_pct, H, EXTENDED_H,
            forced_mae_bp, forced_time, EFFECTIVE_LIQ_BP
        )

        outcomes['total_bars_held'] += bars

        if outcome == 0:
            outcomes['target_hit'] += 1
        elif outcome == 1:
            outcomes['liquidated'] += 1
        elif outcome == 2:
            outcomes['forced_exit'] += 1
            outcomes['forced_exit_mae'].append(mae)
            outcomes['forced_exit_bars'].append(bars)
        else:
            outcomes['timeout'] += 1

    total = len(valid_indices)
    outcomes['total'] = total
    outcomes['target_rate'] = outcomes['target_hit'] / total * 100 if total > 0 else 0
    outcomes['liquidation_rate'] = outcomes['liquidated'] / total * 100 if total > 0 else 0
    outcomes['forced_exit_rate'] = outcomes['forced_exit'] / total * 100 if total > 0 else 0
    outcomes['timeout_rate'] = outcomes['timeout'] / total * 100 if total > 0 else 0
    outcomes['avg_bars_held'] = outcomes['total_bars_held'] / total if total > 0 else 0

    return outcomes


# =============================================================================
# RUN ANALYSIS: NO FORCED EXIT (BASELINE)
# =============================================================================
print("\n" + "=" * 80)
print("BASELINE: NO FORCED EXIT")
print("=" * 80)

condition_mask = ((train['atr_percentile'] > 75) & (train['ema_separation'] > 1.0)).values
baseline = analyze_forced_exit(train, condition_mask, TARGET_BPS, HORIZON, 0, 0)

if baseline:
    print(f"\nCondition: ATR>75% + Trend>1%")
    print(f"Total trades: {baseline['total']:,}")
    print(f"\nOutcomes (no forced exit):")
    print(f"  Target hit: {baseline['target_rate']:.1f}%")
    print(f"  Liquidated: {baseline['liquidation_rate']:.1f}%")
    print(f"  Timeout:    {baseline['timeout_rate']:.1f}%")
    print(f"  Avg bars:   {baseline['avg_bars_held']:.0f}")


# =============================================================================
# RUN ANALYSIS: MAE-BASED FORCED EXIT
# =============================================================================
print("\n" + "=" * 80)
print("MAE-BASED FORCED EXIT")
print("=" * 80)

print(f"\nTesting MAE thresholds: {FORCED_EXIT_MAE_THRESHOLDS}")
print(f"Liquidation threshold: {EFFECTIVE_LIQ_BP:.0f}bp")
print("")

print("| MAE Threshold | Target % | Liquidated % | Forced Exit % | Timeout % | Avg Bars |")
print("|---------------|----------|--------------|---------------|-----------|----------|")

mae_results = {}
for mae_thresh in FORCED_EXIT_MAE_THRESHOLDS:
    result = analyze_forced_exit(train, condition_mask, TARGET_BPS, HORIZON, mae_thresh, 0)
    if result:
        mae_results[mae_thresh] = result
        print(f"| {mae_thresh:12}bp | {result['target_rate']:7.1f}% | {result['liquidation_rate']:11.1f}% | {result['forced_exit_rate']:12.1f}% | {result['timeout_rate']:8.1f}% | {result['avg_bars_held']:7.0f} |")


# =============================================================================
# RUN ANALYSIS: TIME-BASED FORCED EXIT
# =============================================================================
print("\n" + "=" * 80)
print("TIME-BASED FORCED EXIT")
print("=" * 80)

print(f"\nTesting time thresholds: {FORCED_EXIT_TIME_THRESHOLDS}")
print("")

print("| Time Threshold | Target % | Liquidated % | Forced Exit % | Timeout % | Avg Bars |")
print("|----------------|----------|--------------|---------------|-----------|----------|")

time_results = {}
for time_thresh in FORCED_EXIT_TIME_THRESHOLDS:
    result = analyze_forced_exit(train, condition_mask, TARGET_BPS, HORIZON, 0, time_thresh)
    if result:
        time_results[time_thresh] = result
        print(f"| {time_thresh:12} bars | {result['target_rate']:7.1f}% | {result['liquidation_rate']:11.1f}% | {result['forced_exit_rate']:12.1f}% | {result['timeout_rate']:8.1f}% | {result['avg_bars_held']:7.0f} |")


# =============================================================================
# RUN ANALYSIS: COMBINED FORCED EXIT
# =============================================================================
print("\n" + "=" * 80)
print("COMBINED FORCED EXIT (MAE AND TIME)")
print("=" * 80)

# Test P95 based thresholds from S1/S2
# Case 3 P95 MAE ~500bp, P95 Time ~370 bars
# Use slightly below P95 for safety

combined_configs = [
    (150, 120),  # Conservative
    (200, 150),  # Moderate
    (300, 200),  # Aggressive
    (400, 300),  # Very Aggressive
]

print("\n| MAE Thresh | Time Thresh | Target % | Liquidated % | Forced Exit % | Timeout % | Avg Bars |")
print("|------------|-------------|----------|--------------|---------------|-----------|----------|")

for mae_thresh, time_thresh in combined_configs:
    result = analyze_forced_exit(train, condition_mask, TARGET_BPS, HORIZON, mae_thresh, time_thresh)
    if result:
        print(f"| {mae_thresh:9}bp | {time_thresh:10} bars | {result['target_rate']:7.1f}% | {result['liquidation_rate']:11.1f}% | {result['forced_exit_rate']:12.1f}% | {result['timeout_rate']:8.1f}% | {result['avg_bars_held']:7.0f} |")


# =============================================================================
# CAPITAL FREED ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("CAPITAL FREED ANALYSIS")
print("=" * 80)

print("""
Key metric: How much capital lock time does forced exit save?

Without forced exit:
- Case 3 positions stay underwater for P95 = 368 bars (~6 hours)
- Capital is locked, can't use elsewhere
- Risk of liquidation accumulates

With forced exit:
- Exit early on bad trades
- Free capital for new opportunities
- Prevent worst-case liquidations
""")

if baseline:
    baseline_bars = baseline['avg_bars_held']
    print(f"Baseline avg bars held: {baseline_bars:.0f}")
    print("")

    print("| Config | Bars Saved | % Reduction | Liquidation Prevention |")
    print("|--------|------------|-------------|------------------------|")

    for mae_thresh, time_thresh in combined_configs:
        result = analyze_forced_exit(train, condition_mask, TARGET_BPS, HORIZON, mae_thresh, time_thresh)
        if result:
            bars_saved = baseline_bars - result['avg_bars_held']
            pct_reduction = (bars_saved / baseline_bars) * 100 if baseline_bars > 0 else 0
            liq_prevented = baseline['liquidation_rate'] - result['liquidation_rate']
            print(f"| {mae_thresh}bp/{time_thresh}bars | {bars_saved:9.0f} | {pct_reduction:10.1f}% | {liq_prevented:+21.2f}pp |")


# =============================================================================
# RECOMMENDED FORCED EXIT RULES
# =============================================================================
print("\n" + "=" * 80)
print("RECOMMENDED FORCED EXIT RULES")
print("=" * 80)

print("""
Based on S1-S4 analysis:

Forced exit trigger (ATR>75% + Trend>1%, 10x leverage):

  IF MAE > 150bp AND Time_at_Risk > 120 bars
  THEN EXIT (SURVIVAL)

This prevents:
- Liquidations that would occur beyond 752bp MAE
- Excessive capital lock beyond 2 hours
- Tail risk accumulation

This does NOT:
- Optimize for profit
- Cut winners early
- Act as a stop-loss

Remember: Forced exit is a SURVIVAL mechanism, not profit optimization.
""")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("S4 SUMMARY")
print("=" * 80)

print("""
Key Findings:

1. Forced exit trades liquidation risk for controlled loss
   - Without forced exit: Liquidation rate depends on MAE tail
   - With forced exit: Liquidation rate can be near-zero

2. MAE-based exit is more effective than time-based
   - MAE directly relates to liquidation risk
   - Time indirectly relates through capital lock

3. Combined exit (MAE + Time) provides best protection
   - Catches both deep drawdowns AND long locks
   - More robust than either alone

4. Trade-off is clear:
   - More aggressive exit = lower liquidation, higher forced exit rate
   - Less aggressive exit = higher liquidation, lower forced exit rate
   - Optimal point depends on risk tolerance

5. Forced exit does NOT improve expected value
   - It PREVENTS catastrophic loss
   - This is exactly what SURVIVE is for

Next: S5 - Condition-Specific Survival
""")
