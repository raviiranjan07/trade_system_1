"""
ANALYSIS-12 RE-RUN: EMA Support/Resistance on Multiple Timeframes

Tests: When price is NEAR EMA, does it predict direction?
- Price ABOVE EMA -> EMA acts as support -> expect UP
- Price BELOW EMA -> EMA acts as resistance -> expect DOWN

Timeframes: 3, 5, 10, 15, 30, 60, 120, 240, 480 minutes

Run: python scripts/debug/analysis12_multi_timeframe.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
TIMEFRAMES_MINUTES = [3, 5, 10, 15, 30, 60, 120, 240, 480]
EMA_PERIODS = [20, 50, 100, 200]
NEAR_THRESHOLDS_BPS = [10, 15, 20, 30]  # How close to EMA to count as "near"
HORIZONS_BARS = [5, 10, 20]  # Bars to measure direction
SAMPLE_SIZE = 50000

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS-12 RE-RUN: EMA SUPPORT/RESISTANCE ON MULTIPLE TIMEFRAMES")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv_1m = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv_1m):,} 1-minute candles")

train_1m = ohlcv_1m[ohlcv_1m.index <= "2023-12-31"].copy()
print(f"Train data: {len(train_1m):,} 1-minute candles")


# =============================================================================
# RESAMPLE
# =============================================================================
def resample_ohlcv(df, minutes):
    tf_str = f'{minutes}min'
    return df.resample(tf_str).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


print("\nResampling...")
timeframe_data = {}
for tf_min in TIMEFRAMES_MINUTES:
    timeframe_data[tf_min] = resample_ohlcv(train_1m, tf_min)
    print(f"  {tf_min}-minute: {len(timeframe_data[tf_min]):,} candles")


# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================
def analyze_ema_support_resistance(df, ema_period, near_threshold_bps, H_bars):
    """
    Analyze EMA support/resistance pattern.

    Returns:
    - from_above_up_pct: % of bars FROM ABOVE EMA that went UP
    - from_below_up_pct: % of bars FROM BELOW EMA that went UP
    - combined_edge: from_above_up_pct - from_below_up_pct (should be positive if EMA works)
    """
    close = df['close'].values
    n = len(df)

    # Calculate EMA on this timeframe
    ema = df['close'].ewm(span=ema_period, adjust=False).mean().values

    near_pct = near_threshold_bps / 10000

    from_above_up = 0
    from_above_total = 0
    from_below_up = 0
    from_below_total = 0

    valid_start = ema_period + 10
    max_end = n - H_bars - 5

    # Sample for efficiency
    np.random.seed(42)
    available = list(range(valid_start, max_end))
    sample_size = min(SAMPLE_SIZE, len(available))
    sample_idx = np.random.choice(available, size=sample_size, replace=False)

    for i in sample_idx:
        ema_val = ema[i]
        close_val = close[i]

        # Calculate distance from EMA
        distance_pct = (close_val - ema_val) / ema_val

        # Only analyze if NEAR EMA
        if abs(distance_pct) > near_pct:
            continue

        # Future direction
        future_close = close[i + H_bars]
        went_up = future_close > close_val

        if distance_pct > 0:  # Price ABOVE EMA (support test)
            from_above_total += 1
            if went_up:
                from_above_up += 1
        else:  # Price BELOW EMA (resistance test)
            from_below_total += 1
            if went_up:
                from_below_up += 1

    # Calculate percentages
    from_above_up_pct = from_above_up / from_above_total * 100 if from_above_total > 0 else 50
    from_below_up_pct = from_below_up / from_below_total * 100 if from_below_total > 0 else 50

    # Combined edge: If EMA works as support/resistance:
    # - From above should go UP more often (support)
    # - From below should go DOWN more often (resistance)
    # So combined_edge = from_above_up_pct - from_below_up_pct should be POSITIVE
    combined_edge = from_above_up_pct - from_below_up_pct

    return {
        'from_above_total': from_above_total,
        'from_below_total': from_below_total,
        'from_above_up_pct': from_above_up_pct,
        'from_below_up_pct': from_below_up_pct,
        'combined_edge': combined_edge
    }


# =============================================================================
# RUN ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("RUNNING ANALYSIS ON ALL TIMEFRAMES...")
print("=" * 80)

all_results = []

for tf_min in TIMEFRAMES_MINUTES:
    print(f"\nProcessing {tf_min}-minute timeframe...")
    df = timeframe_data[tf_min]

    for ema_period in EMA_PERIODS:
        for near_bps in NEAR_THRESHOLDS_BPS:
            for H_bars in HORIZONS_BARS:
                H_time = H_bars * tf_min  # Real time in minutes

                result = analyze_ema_support_resistance(df, ema_period, near_bps, H_bars)

                total_samples = result['from_above_total'] + result['from_below_total']

                if total_samples >= 100:  # Need enough samples
                    all_results.append({
                        'tf_min': tf_min,
                        'ema': ema_period,
                        'near_bps': near_bps,
                        'H_bars': H_bars,
                        'H_time': H_time,
                        'from_above_total': result['from_above_total'],
                        'from_below_total': result['from_below_total'],
                        'from_above_up_pct': result['from_above_up_pct'],
                        'from_below_up_pct': result['from_below_up_pct'],
                        'combined_edge': result['combined_edge']
                    })


# =============================================================================
# RESULTS BY TIMEFRAME
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS BY TIMEFRAME")
print("=" * 80)

for tf_min in TIMEFRAMES_MINUTES:
    tf_results = [r for r in all_results if r['tf_min'] == tf_min]
    if not tf_results:
        continue

    print(f"\n### {tf_min}-MINUTE TIMEFRAME")
    print(f"EMA calculated on {tf_min}-min candles")

    # Sort by combined edge
    tf_results.sort(key=lambda x: x['combined_edge'], reverse=True)

    print("\nTop 5 combinations:")
    print("| EMA | Near | H(bars) | H(time) | Above-UP% | Below-UP% | Combined Edge |")
    print("|-----|------|---------|---------|-----------|-----------|---------------|")

    for r in tf_results[:5]:
        sign = "+" if r['combined_edge'] > 0 else ""
        print(f"| EMA{r['ema']} | {r['near_bps']}bp | {r['H_bars']}b | {r['H_time']}min | "
              f"{r['from_above_up_pct']:.1f}% | {r['from_below_up_pct']:.1f}% | "
              f"{sign}{r['combined_edge']:.1f}% |")

    # Average edge for this timeframe
    avg_edge = np.mean([r['combined_edge'] for r in tf_results])
    max_edge = max([r['combined_edge'] for r in tf_results])
    positive_count = sum(1 for r in tf_results if r['combined_edge'] > 0)

    print(f"\nTimeframe Summary:")
    print(f"  Average combined edge: {avg_edge:+.2f}%")
    print(f"  Max combined edge: {max_edge:+.2f}%")
    print(f"  Combinations with positive edge: {positive_count}/{len(tf_results)}")


# =============================================================================
# SUMMARY: BEST BY TIMEFRAME
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: BEST RESULT PER TIMEFRAME")
print("=" * 80)

print("\n| Timeframe | EMA Period | Best Edge | Above->UP | Below->UP | Significant? |")
print("|-----------|------------|-----------|----------|----------|--------------|")

for tf_min in TIMEFRAMES_MINUTES:
    tf_results = [r for r in all_results if r['tf_min'] == tf_min]
    if tf_results:
        tf_results.sort(key=lambda x: x['combined_edge'], reverse=True)
        best = tf_results[0]

        # Significant if edge > 2% (arbitrary threshold)
        significant = "YES" if best['combined_edge'] > 2 else "NO"

        print(f"| {tf_min}min | EMA{best['ema']} | {best['combined_edge']:+.1f}% | "
              f"{best['from_above_up_pct']:.1f}% | {best['from_below_up_pct']:.1f}% | {significant} |")


# =============================================================================
# COMPARE TO RANDOM (50/50)
# =============================================================================
print("\n" + "=" * 80)
print("EDGE INTERPRETATION")
print("=" * 80)

print("""
If EMA works as SUPPORT/RESISTANCE:
- Price ABOVE EMA -> should bounce UP (support) -> Above->UP% > 50%
- Price BELOW EMA -> should bounce DOWN (resistance) -> Below->UP% < 50%
- Combined Edge = Above->UP% - Below->UP% should be POSITIVE

Example of WORKING support/resistance:
- Above EMA -> 55% go UP (support works)
- Below EMA -> 45% go UP (resistance works)
- Combined Edge = 55% - 45% = +10%

Example of NO support/resistance:
- Above EMA -> 50% go UP (random)
- Below EMA -> 50% go UP (random)
- Combined Edge = 50% - 50% = 0%
""")


# =============================================================================
# KEY FINDINGS
# =============================================================================
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Overall statistics
avg_edge_all = np.mean([r['combined_edge'] for r in all_results])
max_edge_all = max([r['combined_edge'] for r in all_results])
significant_count = sum(1 for r in all_results if r['combined_edge'] > 2)

print(f"""
1. TIMEFRAMES TESTED: {TIMEFRAMES_MINUTES}
   - EMA calculated on EACH timeframe's candles
   - Total combinations: {len(all_results)}

2. OVERALL RESULTS:
   - Average combined edge: {avg_edge_all:+.2f}%
   - Max combined edge: {max_edge_all:+.2f}%
   - Combinations with >2% edge: {significant_count} ({significant_count/len(all_results)*100:.1f}%)

3. INTERPRETATION:
   - If avg edge ~ 0%: EMA does NOT predict direction
   - If avg edge > 2%: EMA has SOME predictive power
   - If avg edge > 5%: EMA has MEANINGFUL edge

4. CONCLUSION:
""")

if avg_edge_all > 2:
    print(f"   EMA support/resistance DOES show edge ({avg_edge_all:+.2f}% average)")
    print("   But need to verify if tradeable after fees")
else:
    print(f"   EMA support/resistance has NEGLIGIBLE edge ({avg_edge_all:+.2f}% average)")
    print("   Cannot rely on EMA for directional prediction")
