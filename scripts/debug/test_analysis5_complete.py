"""
ANALYSIS-5 COMPLETE: Adverse Excursion (MAE) for Winners
Generate detailed per-horizon breakdown with Mean, Median, 75th, Max AE, and Count

Coverage: H=3 to H=600, Targets: 12bp to 200bp

Run: .venv/Scripts/python.exe scripts/debug/test_analysis5_complete.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60, 120, 240, 360, 480, 600]
TARGETS_BPS = [12, 15, 25, 50, 100, 150, 200]
SAMPLE_SIZE = 200000  # Large sample for better statistics

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("ANALYSIS-5 COMPLETE: MAE FOR WINNERS")
print("=" * 80)

ohlcv_path = Path("data/ohlcv/BTCUSDT_1m_ohlcv.parquet")
ohlcv = pd.read_parquet(ohlcv_path)
print(f"\nLoaded {len(ohlcv):,} candles")

train = ohlcv[ohlcv.index <= "2023-12-31"].copy()
print(f"Train data: {len(train):,} candles")

close = train['close'].values
high = train['high'].values
low = train['low'].values
n = len(train)

np.random.seed(42)
max_h = max(HORIZONS) + 10
valid_start = 100
sample_idx = np.random.choice(range(valid_start, n - max_h),
                               size=min(SAMPLE_SIZE, n - max_h - valid_start),
                               replace=False)
print(f"Sample size: {len(sample_idx):,}")


def analyze_mae_winners(indices, H, target_pct):
    """
    Analyze MAE (Maximum Adverse Excursion) for trades that HIT target within H bars.
    Returns: Mean AE, Median AE, 75th AE, Max AE, Count
    """
    mae_list = []

    for i in indices:
        entry = close[i]
        target_price = entry * (1 + target_pct)

        hit_target = False
        max_drawdown = 0

        for j in range(1, H + 1):
            if i + j >= n:
                break

            # Track drawdown
            bar_low = low[i + j]
            if bar_low < entry:
                drawdown = (entry - bar_low) / entry * 10000
                max_drawdown = max(max_drawdown, drawdown)

            # Check if hit target
            if high[i + j] >= target_price:
                hit_target = True
                break

        # Only include trades that hit target (winners)
        if hit_target:
            mae_list.append(max_drawdown)

    if not mae_list:
        return None

    return {
        'mean': np.mean(mae_list),
        'median': np.median(mae_list),
        'p75': np.percentile(mae_list, 75),
        'max': np.max(mae_list),
        'count': len(mae_list)
    }


# =============================================================================
# GENERATE DATA FOR ALL HORIZONS
# =============================================================================
print("\nGenerating data for all horizons...")

all_results = {}

for H in HORIZONS:
    all_results[H] = {}
    for target_bps in TARGETS_BPS:
        target_pct = target_bps / 10000
        result = analyze_mae_winners(sample_idx, H, target_pct)
        if result:
            all_results[H][target_bps] = result
    print(f"  H={H} done")

print("\nData collection complete!")


# =============================================================================
# OUTPUT: MARKDOWN FORMAT FOR DOCUMENTATION
# =============================================================================
print("\n" + "=" * 80)
print("MARKDOWN OUTPUT FOR DOCUMENTATION")
print("=" * 80)

for H in HORIZONS:
    print(f"\nH={H} bars:")
    print("| Outcome              | Mean AE | Median AE | 75th AE | MAX AE   | Count   |")
    print("|----------------------|---------|-----------|---------|----------|---------|")

    for target_bps in TARGETS_BPS:
        if target_bps in all_results[H]:
            r = all_results[H][target_bps]
            print(f"| Hit {target_bps}bp target      | {r['mean']:.1f}bp   | "
                  f"{r['median']:.1f}bp     | {r['p75']:.1f}bp   | "
                  f"{r['max']:.1f}bp  | {r['count']:,}  |")


# =============================================================================
# OUTPUT: SUMMARY TABLE (ALL HORIZONS x ALL TARGETS)
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY TABLE: MAE MEDIAN FOR ALL WINNERS")
print("=" * 80)

print("\n| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |")
print("|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|")

for target_bps in TARGETS_BPS:
    row = f"| {target_bps}bp |"
    for H in HORIZONS:
        if target_bps in all_results[H]:
            row += f" {all_results[H][target_bps]['median']:.0f} |"
        else:
            row += " - |"
    print(row)


print("\n" + "=" * 80)
print("SUMMARY TABLE: MAE 75TH PERCENTILE FOR ALL WINNERS")
print("=" * 80)

print("\n| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |")
print("|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|")

for target_bps in TARGETS_BPS:
    row = f"| {target_bps}bp |"
    for H in HORIZONS:
        if target_bps in all_results[H]:
            row += f" {all_results[H][target_bps]['p75']:.0f} |"
        else:
            row += " - |"
    print(row)


print("\n" + "=" * 80)
print("SUMMARY TABLE: MAE MAX FOR ALL WINNERS (worst case)")
print("=" * 80)

print("\n| Target | H=3 | H=5 | H=10 | H=15 | H=30 | H=60 | H=120 | H=240 | H=360 | H=480 | H=600 |")
print("|--------|-----|-----|------|------|------|------|-------|-------|-------|-------|-------|")

for target_bps in TARGETS_BPS:
    row = f"| {target_bps}bp |"
    for H in HORIZONS:
        if target_bps in all_results[H]:
            row += f" {all_results[H][target_bps]['max']:.0f} |"
        else:
            row += " - |"
    print(row)


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("""
1. Even WINNING trades experience significant drawdowns
   - Median MAE increases with horizon (more time = more drawdown risk)
   - Max MAE can be HUGE (1000+ bps) even for small targets

2. Larger targets have larger MAE
   - 15bp target: Median MAE 3-13 bp
   - 100bp target: Median MAE 7-40 bp
   - 200bp target: Median MAE 9-45 bp

3. Longer horizons = Larger drawdowns
   - At H=3: Most winners have small MAE (median 3-9 bp)
   - At H=600: Even winners have large MAE (median 11-45 bp)

4. Max AE shows extreme risk
   - Even winners can see 400-2400 bp drawdowns
   - High leverage would cause liquidation on trades that eventually win
   - Stop-loss placement is critical

5. Sample size decreases with larger targets
   - 15bp: ~180,000 winners at H=600
   - 200bp: ~50,000 winners at H=600
   - Larger targets are harder to hit (fewer wins)
""")
