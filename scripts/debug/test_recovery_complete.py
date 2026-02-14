"""
ANALYSIS-8/9/10 COMPLETE: Recovery Cases with All Horizons and Targets

- Case 1: Wrong Direction (never hit target even with extended time)
- Case 2: Quick Recovery (hit target within H bars)
- Case 3: Slow Recovery (didn't hit within H, but hit later)

Run: .venv/Scripts/python.exe scripts/debug/test_recovery_complete.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
HORIZONS = [3, 5, 10, 15, 30, 60, 120, 240, 360, 480, 600]
TARGETS_BPS = [8, 12, 15, 25, 50, 100, 150, 200]
SAMPLE_SIZE = 50000
EXTENDED_H = 600  # How far to look for eventual recovery

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("RECOVERY ANALYSIS COMPLETE: All Horizons and Targets")
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
max_h = EXTENDED_H + 100
valid_start = 100
sample_idx = np.random.choice(range(valid_start, n - max_h), size=min(SAMPLE_SIZE, n - max_h - valid_start), replace=False)
print(f"Sample size: {len(sample_idx):,}")


def analyze_recovery(indices, H, target_pct, extended_H=600):
    """
    Complete recovery analysis:
    - Case 0: Clean Win (hit target without going below entry)
    - Case 1: Wrong Direction (went below, never hit target even with extended time)
    - Case 2: Quick Recovery (went below, hit target within H bars)
    - Case 3: Slow Recovery (went below, didn't hit within H, but hit later)
    """
    case0_count = 0  # Clean Win
    case1_count = 0  # Wrong Direction
    case2_count = 0  # Quick Recovery
    case3_count = 0  # Slow Recovery

    # MAE values by case
    case2_mae = []
    case3_mae = []
    case1_mae = []

    # Recovery time for Case 3 (bars AFTER H)
    case3_recovery_time = []
    case3_total_time = []  # Total from entry

    for i in indices:
        entry = close[i]
        target_price = entry * (1 + target_pct)

        hit_within_H = False
        hit_extended = False
        ever_below = False
        max_drawdown = 0
        hit_bar = None

        # Check within H bars
        for j in range(1, H + 1):
            if i + j >= n:
                break

            bar_low = low[i + j]
            if bar_low < entry:
                ever_below = True
                drawdown = (entry - bar_low) / entry * 10000
                max_drawdown = max(max_drawdown, drawdown)

            if high[i + j] >= target_price:
                hit_within_H = True
                hit_bar = j
                break

        # If didn't hit within H, check extended
        if not hit_within_H:
            for j in range(H + 1, extended_H + 1):
                if i + j >= n:
                    break

                bar_low = low[i + j]
                if bar_low < entry:
                    ever_below = True
                    drawdown = (entry - bar_low) / entry * 10000
                    max_drawdown = max(max_drawdown, drawdown)

                if high[i + j] >= target_price:
                    hit_extended = True
                    hit_bar = j
                    break

        # Classify
        if hit_within_H:
            if ever_below:
                case2_count += 1  # Quick Recovery
                case2_mae.append(max_drawdown)
            else:
                case0_count += 1  # Clean Win
        elif hit_extended:
            case3_count += 1  # Slow Recovery
            case3_mae.append(max_drawdown)
            case3_recovery_time.append(hit_bar - H)  # Bars after H
            case3_total_time.append(hit_bar)  # Total from entry
        else:
            if ever_below:
                case1_count += 1  # Wrong Direction
                case1_mae.append(max_drawdown)
            else:
                case1_count += 1  # Never went below but also never hit (rare)

    total = case0_count + case1_count + case2_count + case3_count

    result = {
        'case0_pct': 100 * case0_count / total if total > 0 else 0,  # Clean Win
        'case1_pct': 100 * case1_count / total if total > 0 else 0,  # Wrong Dir
        'case2_pct': 100 * case2_count / total if total > 0 else 0,  # Quick Rec
        'case3_pct': 100 * case3_count / total if total > 0 else 0,  # Slow Rec
        'recovery_rate': 100 * (case0_count + case2_count + case3_count) / total if total > 0 else 0,
    }

    # MAE stats
    if case2_mae:
        result['case2_mae_median'] = np.median(case2_mae)
        result['case2_mae_75th'] = np.percentile(case2_mae, 75)
    else:
        result['case2_mae_median'] = result['case2_mae_75th'] = 0

    if case3_mae:
        result['case3_mae_median'] = np.median(case3_mae)
        result['case3_mae_75th'] = np.percentile(case3_mae, 75)
        result['case3_mae_max'] = np.max(case3_mae)
    else:
        result['case3_mae_median'] = result['case3_mae_75th'] = result['case3_mae_max'] = 0

    if case1_mae:
        result['case1_mae_median'] = np.median(case1_mae)
    else:
        result['case1_mae_median'] = 0

    # Recovery time for Case 3
    if case3_recovery_time:
        result['case3_rec_median'] = np.median(case3_recovery_time)
        result['case3_rec_75th'] = np.percentile(case3_recovery_time, 75)
        result['case3_rec_90th'] = np.percentile(case3_recovery_time, 90)
        result['case3_total_median'] = np.median(case3_total_time)
        result['case3_total_90th'] = np.percentile(case3_total_time, 90)
    else:
        result['case3_rec_median'] = result['case3_rec_75th'] = result['case3_rec_90th'] = 0
        result['case3_total_median'] = result['case3_total_90th'] = 0

    return result


# =============================================================================
# COLLECT ALL DATA
# =============================================================================
print("\nCollecting data for all combinations...")
all_results = []

for H in HORIZONS:
    for target_bps in TARGETS_BPS:
        target_pct = target_bps / 10000
        r = analyze_recovery(sample_idx, H, target_pct, EXTENDED_H)
        r['H'] = H
        r['target'] = target_bps
        all_results.append(r)
    print(f"  H={H} done")

print("\nData collection complete!")


# =============================================================================
# TABLE A1: Case 3 % (Slow Recovery - hit target AFTER H)
# =============================================================================
print("\n" + "=" * 100)
print("TABLE A1: CASE 3 % (Slow Recovery - eventually hit target but NOT within H)")
print("=" * 100)

header = f"{'Target':<10}"
for H in HORIZONS:
    header += f"H={H:<6}"
print(header)
print("-" * 100)

for target_bps in TARGETS_BPS:
    row = f"{target_bps}bp{'':<6}"
    for H in HORIZONS:
        for r in all_results:
            if r['H'] == H and r['target'] == target_bps:
                row += f"{r['case3_pct']:<8.1f}"
                break
    print(row)


# =============================================================================
# TABLE A2: Case 1 % (Wrong Direction - never recovered)
# =============================================================================
print("\n" + "=" * 100)
print("TABLE A2: CASE 1 % (Wrong Direction - never hit target even with extended time)")
print("=" * 100)

header = f"{'Target':<10}"
for H in HORIZONS:
    header += f"H={H:<6}"
print(header)
print("-" * 100)

for target_bps in TARGETS_BPS:
    row = f"{target_bps}bp{'':<6}"
    for H in HORIZONS:
        for r in all_results:
            if r['H'] == H and r['target'] == target_bps:
                row += f"{r['case1_pct']:<8.1f}"
                break
    print(row)


# =============================================================================
# TABLE A3: Recovery Rate % (Case 0 + Case 2 + Case 3)
# =============================================================================
print("\n" + "=" * 100)
print("TABLE A3: RECOVERY RATE % (eventually hit target - Case 0 + 2 + 3)")
print("=" * 100)

header = f"{'Target':<10}"
for H in HORIZONS:
    header += f"H={H:<6}"
print(header)
print("-" * 100)

for target_bps in TARGETS_BPS:
    row = f"{target_bps}bp{'':<6}"
    for H in HORIZONS:
        for r in all_results:
            if r['H'] == H and r['target'] == target_bps:
                row += f"{r['recovery_rate']:<8.1f}"
                break
    print(row)


# =============================================================================
# TABLE B1: Case 3 MAE Median (drawdown before eventually hitting target)
# =============================================================================
print("\n" + "=" * 100)
print("TABLE B1: CASE 3 MAE MEDIAN (bp drawdown before eventually hitting target)")
print("=" * 100)

header = f"{'Target':<10}"
for H in HORIZONS:
    header += f"H={H:<6}"
print(header)
print("-" * 100)

for target_bps in TARGETS_BPS:
    row = f"{target_bps}bp{'':<6}"
    for H in HORIZONS:
        for r in all_results:
            if r['H'] == H and r['target'] == target_bps:
                row += f"{r['case3_mae_median']:<8.1f}"
                break
    print(row)


# =============================================================================
# TABLE B2: Case 3 MAE 75th (worse 25% of slow recovery trades)
# =============================================================================
print("\n" + "=" * 100)
print("TABLE B2: CASE 3 MAE 75th PERCENTILE (bp)")
print("=" * 100)

header = f"{'Target':<10}"
for H in HORIZONS:
    header += f"H={H:<6}"
print(header)
print("-" * 100)

for target_bps in TARGETS_BPS:
    row = f"{target_bps}bp{'':<6}"
    for H in HORIZONS:
        for r in all_results:
            if r['H'] == H and r['target'] == target_bps:
                row += f"{r['case3_mae_75th']:<8.1f}"
                break
    print(row)


# =============================================================================
# TABLE B3: Case 3 MAE MAX
# =============================================================================
print("\n" + "=" * 100)
print("TABLE B3: CASE 3 MAE MAX (bp - worst case drawdown)")
print("=" * 100)

header = f"{'Target':<10}"
for H in HORIZONS:
    header += f"H={H:<6}"
print(header)
print("-" * 100)

for target_bps in TARGETS_BPS:
    row = f"{target_bps}bp{'':<6}"
    for H in HORIZONS:
        for r in all_results:
            if r['H'] == H and r['target'] == target_bps:
                row += f"{r['case3_mae_max']:<8.0f}"
                break
    print(row)


# =============================================================================
# TABLE C1: Case 3 Total Recovery Time - Median (bars from entry)
# =============================================================================
print("\n" + "=" * 100)
print("TABLE C1: CASE 3 TOTAL RECOVERY TIME - MEDIAN (bars from entry to hitting target)")
print("=" * 100)

header = f"{'Target':<10}"
for H in HORIZONS:
    header += f"H={H:<6}"
print(header)
print("-" * 100)

for target_bps in TARGETS_BPS:
    row = f"{target_bps}bp{'':<6}"
    for H in HORIZONS:
        for r in all_results:
            if r['H'] == H and r['target'] == target_bps:
                row += f"{r['case3_total_median']:<8.0f}"
                break
    print(row)


# =============================================================================
# TABLE C2: Case 3 Total Recovery Time - 90th Percentile (worst 10%)
# =============================================================================
print("\n" + "=" * 100)
print("TABLE C2: CASE 3 TOTAL RECOVERY TIME - 90th PERCENTILE (bars from entry)")
print("=" * 100)

header = f"{'Target':<10}"
for H in HORIZONS:
    header += f"H={H:<6}"
print(header)
print("-" * 100)

for target_bps in TARGETS_BPS:
    row = f"{target_bps}bp{'':<6}"
    for H in HORIZONS:
        for r in all_results:
            if r['H'] == H and r['target'] == target_bps:
                row += f"{r['case3_total_90th']:<8.0f}"
                break
    print(row)


# =============================================================================
# MARKDOWN TABLES
# =============================================================================
print("\n" + "=" * 100)
print("MARKDOWN TABLES FOR ANALYSIS-8/9/10 UPDATE")
print("=" * 100)

# Case 3 %
print("\n### Case 3 % (Slow Recovery - hit target AFTER H bars)")
print("| Target |", end="")
for H in HORIZONS:
    print(f" H={H} |", end="")
print()
print("|--------|", end="")
for H in HORIZONS:
    print("------|", end="")
print()

for target_bps in TARGETS_BPS:
    print(f"| {target_bps}bp |", end="")
    for H in HORIZONS:
        for r in all_results:
            if r['H'] == H and r['target'] == target_bps:
                print(f" {r['case3_pct']:.1f}% |", end="")
                break
    print()

# Case 1 % (Wrong Direction)
print("\n### Case 1 % (Wrong Direction - never recovered)")
print("| Target |", end="")
for H in HORIZONS:
    print(f" H={H} |", end="")
print()
print("|--------|", end="")
for H in HORIZONS:
    print("------|", end="")
print()

for target_bps in TARGETS_BPS:
    print(f"| {target_bps}bp |", end="")
    for H in HORIZONS:
        for r in all_results:
            if r['H'] == H and r['target'] == target_bps:
                print(f" {r['case1_pct']:.1f}% |", end="")
                break
    print()

# Recovery Rate
print("\n### Recovery Rate % (eventually hit target)")
print("| Target |", end="")
for H in HORIZONS:
    print(f" H={H} |", end="")
print()
print("|--------|", end="")
for H in HORIZONS:
    print("------|", end="")
print()

for target_bps in TARGETS_BPS:
    print(f"| {target_bps}bp |", end="")
    for H in HORIZONS:
        for r in all_results:
            if r['H'] == H and r['target'] == target_bps:
                print(f" {r['recovery_rate']:.1f}% |", end="")
                break
    print()

# Case 3 MAE Median
print("\n### Case 3 MAE Median (bp drawdown for slow recovery trades)")
print("| Target |", end="")
for H in HORIZONS:
    print(f" H={H} |", end="")
print()
print("|--------|", end="")
for H in HORIZONS:
    print("------|", end="")
print()

for target_bps in TARGETS_BPS:
    print(f"| {target_bps}bp |", end="")
    for H in HORIZONS:
        for r in all_results:
            if r['H'] == H and r['target'] == target_bps:
                print(f" {r['case3_mae_median']:.0f} |", end="")
                break
    print()

# Case 3 Total Time Median
print("\n### Case 3 Total Recovery Time - Median (bars from entry)")
print("| Target |", end="")
for H in HORIZONS:
    print(f" H={H} |", end="")
print()
print("|--------|", end="")
for H in HORIZONS:
    print("------|", end="")
print()

for target_bps in TARGETS_BPS:
    print(f"| {target_bps}bp |", end="")
    for H in HORIZONS:
        for r in all_results:
            if r['H'] == H and r['target'] == target_bps:
                print(f" {r['case3_total_median']:.0f} |", end="")
                break
    print()

# Case 3 Total Time 90th
print("\n### Case 3 Total Recovery Time - 90th Percentile (worst 10%)")
print("| Target |", end="")
for H in HORIZONS:
    print(f" H={H} |", end="")
print()
print("|--------|", end="")
for H in HORIZONS:
    print("------|", end="")
print()

for target_bps in TARGETS_BPS:
    print(f"| {target_bps}bp |", end="")
    for H in HORIZONS:
        for r in all_results:
            if r['H'] == H and r['target'] == target_bps:
                print(f" {r['case3_total_90th']:.0f} |", end="")
                break
    print()


# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "=" * 100)
print("KEY INSIGHTS")
print("=" * 100)

print("""
1. CASE 3 (Slow Recovery) decreases with longer H:
   - At H=3, most "Never Hit" trades are actually Case 3 (would have won with more time)
   - At H=600, fewer Case 3 because more time already given

2. CASE 1 (Wrong Direction) is CONSTANT across horizons:
   - ~5% for 8bp target
   - ~8% for 12bp target
   - ~10% for 15bp target
   - ~16% for 25bp target
   - ~29% for 50bp target
   - ~50% for 100bp target
   - ~65% for 150bp target
   - ~75% for 200bp target

3. Case 3 MAE is HIGHER than Case 2 MAE:
   - Quick Recovery: Small drawdown (3-12bp)
   - Slow Recovery: Larger drawdown (16-100bp+)

4. Case 3 Total Time can be VERY LONG:
   - Median: 20-300 bars depending on target
   - 90th percentile: 200-500+ bars (3-8+ hours!)

5. LARGER TARGETS have more Wrong Direction cases:
   - 8bp: 5% wrong, 95% eventually recover
   - 50bp: 29% wrong, 71% eventually recover
   - 100bp: 50% wrong, 50% eventually recover
   - 200bp: 75% wrong, only 25% eventually recover
""")
