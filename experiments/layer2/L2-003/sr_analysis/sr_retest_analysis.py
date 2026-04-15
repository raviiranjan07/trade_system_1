"""S/R Retest Analysis - Data-First Investigation

User's definition of S/R:
  "If a candle has a high, and another candle breaks the previous candle's high,
   and if it again comes back to the previous candle high, it acts as a resistance.
   The chances of bouncing is very high."

This script analyzes:
1. How often do retests happen on 15min BTC data?
2. When price retests a previous high (resistance) or low (support), what happens?
3. Does it bounce or break through?
4. What is the H8 direction outcome (which +/-15bps hits first)?
5. How does this combine with existing features (roc, rsi, range_position)?

Definitions:
  RESISTANCE RETEST:
    - Bar A has a high
    - Bar B (later) breaks above Bar A's high
    - Bar C (later) comes back down to Bar A's high level
    => Bar C is a resistance retest. Does price bounce DOWN or break UP?

  SUPPORT RETEST:
    - Bar A has a low
    - Bar B (later) breaks below Bar A's low
    - Bar C (later) comes back up to Bar A's low level
    => Bar C is a support retest. Does price bounce UP or break DOWN?

Run: PYTHONPATH=src python experiments/layer2/L2-003/sr_retest_analysis.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OHLCV_PATH = ROOT / "data" / "ohlcv" / "BTCUSDT_15m_ohlcv.parquet"
LABELS_PATH = Path(__file__).parent / "labels.parquet"

# How close price needs to be to a level to count as "retest" (in bps)
RETEST_TOLERANCE_BPS = 5  # within 5bps of the level
# How many bars back to look for S/R levels
LOOKBACK_BARS = 20
# Minimum bars between level creation and retest
MIN_GAP = 2  # at least 2 bars between the break and the retest

print("=" * 70)
print("S/R RETEST ANALYSIS")
print("=" * 70)

# Load data
print("\nLoading data...")
ohlcv = pd.read_parquet(OHLCV_PATH)
labels = pd.read_parquet(LABELS_PATH)
print(f"  OHLCV: {ohlcv.shape[0]} bars, {ohlcv.index.min()} to {ohlcv.index.max()}")
print(f"  Labels: {labels.shape[0]} bars")

# Make labels timezone-naive to match
labels.index = labels.index.tz_localize(None) if labels.index.tz is None else labels.index
ohlcv.index = ohlcv.index.tz_localize(None) if ohlcv.index.tz is not None else ohlcv.index

# Extract arrays for speed
high = ohlcv["high"].values
low = ohlcv["low"].values
close = ohlcv["close"].values
open_ = ohlcv["open"].values
dates = ohlcv.index
N = len(ohlcv)

print(f"\n{'='*70}")
print("PART 1: RESISTANCE RETEST (Previous High)")
print(f"{'='*70}")
print("""
Pattern:
  1. Bar A has high = H
  2. Later bar breaks ABOVE H (price went higher)
  3. Even later, price comes BACK DOWN to H
  => Retest! Old resistance. Does it bounce down or break through up?
""")

# For each bar, check if it's retesting a recent high level
resistance_retests = []
support_retests = []

print("Scanning for retests...")
for i in range(LOOKBACK_BARS + MIN_GAP, N):
    price = close[i]
    bar_low = low[i]
    bar_high = high[i]

    # --- RESISTANCE RETEST ---
    # Look at bars in [i-LOOKBACK_BARS, i-MIN_GAP] for highs that were broken and now retested
    for j in range(i - LOOKBACK_BARS, i - MIN_GAP):
        if j < 0:
            continue
        level = high[j]  # previous bar's high
        tol = level * RETEST_TOLERANCE_BPS / 10000

        # Check: was this high broken (some bar between j+1 and i went above it)?
        # And is current bar retesting it from above (bar's low touches the level)?
        slice_highs = high[j+1:i]
        if len(slice_highs) == 0:
            continue

        # Was the level broken? (some bar between j+1 and i-1 went above it)
        max_between = np.max(slice_highs)
        if max_between <= level:
            continue  # level was never broken above, skip

        # Is current bar retesting from above?
        # Price is near the level (bar's low is within tolerance of the level)
        if abs(bar_low - level) <= tol:
            # This is a resistance retest!
            bars_since_level = i - j
            resistance_retests.append({
                "date": dates[i],
                "bar_idx": i,
                "level": level,
                "level_bar": j,
                "bars_since": bars_since_level,
                "close": price,
                "low": bar_low,
                "high": bar_high,
                "dist_bps": (bar_low - level) / level * 10000,
            })
            break  # one retest per bar (take the most recent level)

    # --- SUPPORT RETEST ---
    for j in range(i - LOOKBACK_BARS, i - MIN_GAP):
        if j < 0:
            continue
        level = low[j]  # previous bar's low
        tol = level * RETEST_TOLERANCE_BPS / 10000

        slice_lows = low[j+1:i]
        if len(slice_lows) == 0:
            continue

        # Was the level broken below?
        min_between = np.min(slice_lows)
        if min_between >= level:
            continue  # level was never broken below, skip

        # Is current bar retesting from below?
        if abs(bar_high - level) <= tol:
            bars_since_level = i - j
            support_retests.append({
                "date": dates[i],
                "bar_idx": i,
                "level": level,
                "level_bar": j,
                "bars_since": bars_since_level,
                "close": price,
                "low": bar_low,
                "high": bar_high,
                "dist_bps": (bar_high - level) / level * 10000,
            })
            break

    if i % 50000 == 0:
        print(f"  {i}/{N} bars scanned... ({len(resistance_retests)} res, {len(support_retests)} sup retests)")

print(f"\nFound {len(resistance_retests)} resistance retests")
print(f"Found {len(support_retests)} support retests")
print(f"Total bars: {N}")
print(f"Resistance retest rate: {len(resistance_retests)/N*100:.1f}%")
print(f"Support retest rate: {len(support_retests)/N*100:.1f}%")

# Convert to DataFrames
res_df = pd.DataFrame(resistance_retests)
sup_df = pd.DataFrame(support_retests)

# Join with labels (H8 direction)
label_map = {0: "LONG", 1: "SHORT", 2: "BOTH", 3: "SKIP"}

print(f"\n{'='*70}")
print("PART 2: DIRECTION OUTCOME AFTER RETEST")
print(f"{'='*70}")

for name, df in [("RESISTANCE", res_df), ("SUPPORT", sup_df)]:
    if len(df) == 0:
        print(f"\n{name}: No retests found!")
        continue

    print(f"\n--- {name} RETESTS ---")
    print(f"Total: {len(df)}")
    print(f"Bars since level: mean={df['bars_since'].mean():.1f}, median={df['bars_since'].median():.0f}")

    # Match with labels
    df_with_labels = df.set_index("date").join(labels[["direction", "direction_h8"]], how="inner")
    matched = len(df_with_labels)
    print(f"Matched with labels: {matched}/{len(df)}")

    if matched == 0:
        continue

    # H8 direction distribution
    print(f"\nH8 Direction (which +/-15bps hits first within 8 bars):")
    for val, label in label_map.items():
        count = (df_with_labels["direction_h8"] == val).sum()
        pct = count / matched * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    # H96 direction distribution
    print(f"\nH96 Direction (which +/-15bps hits first within 96 bars):")
    for val, label in label_map.items():
        count = (df_with_labels["direction"] == val).sum()
        pct = count / matched * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    # For resistance: expectation is bounce DOWN (SHORT)
    # For support: expectation is bounce UP (LONG)
    if name == "RESISTANCE":
        expected = "SHORT"
        expected_val = 1
        print(f"\n  => If resistance holds, expect SHORT (bounce down)")
    else:
        expected = "LONG"
        expected_val = 0
        print(f"\n  => If support holds, expect LONG (bounce up)")

    # Filter SKIP and BOTH
    valid_h8 = df_with_labels[df_with_labels["direction_h8"].isin([0, 1])]
    if len(valid_h8) > 0:
        expected_rate = (valid_h8["direction_h8"] == expected_val).mean() * 100
        print(f"\n  H8 {expected} rate (excl SKIP/BOTH): {expected_rate:.1f}% ({len(valid_h8)} bars)")

    valid_h96 = df_with_labels[df_with_labels["direction"].isin([0, 1])]
    if len(valid_h96) > 0:
        expected_rate = (valid_h96["direction"] == expected_val).mean() * 100
        print(f"  H96 {expected} rate (excl SKIP/BOTH): {expected_rate:.1f}% ({len(valid_h96)} bars)")


print(f"\n{'='*70}")
print("PART 3: TRAIN vs TEST SPLIT")
print(f"{'='*70}")

for name, df in [("RESISTANCE", res_df), ("SUPPORT", sup_df)]:
    if len(df) == 0:
        continue
    print(f"\n--- {name} ---")

    df_with_labels = df.set_index("date").join(labels[["direction", "direction_h8"]], how="inner")

    for period, start, end in [("TRAIN 2020-2023", "2020-01-01", "2023-12-31"),
                                ("TEST 2024-2025", "2024-01-01", "2025-12-31")]:
        subset = df_with_labels.loc[start:end]
        valid = subset[subset["direction_h8"].isin([0, 1])]
        if len(valid) == 0:
            print(f"  {period}: no data")
            continue

        if name == "RESISTANCE":
            bounce_rate = (valid["direction_h8"] == 1).mean() * 100  # SHORT = bounce down
            label = "SHORT (bounce down)"
        else:
            bounce_rate = (valid["direction_h8"] == 0).mean() * 100  # LONG = bounce up
            label = "LONG (bounce up)"

        print(f"  {period}: {len(valid)} retests, {label} = {bounce_rate:.1f}%")


print(f"\n{'='*70}")
print("PART 4: RETEST QUALITY - DOES TOUCH COUNT MATTER?")
print(f"{'='*70}")
print("""
Check: If a level has been tested multiple times, is it stronger?
We count how many times the level was approached before the current retest.
""")

# For resistance: count how many bars in the lookback window had their low near the level
for name, df in [("RESISTANCE", res_df), ("SUPPORT", sup_df)]:
    if len(df) == 0:
        continue
    print(f"\n--- {name} ---")

    touch_counts = []
    for _, row in df.iterrows():
        level = row["level"]
        bar_idx = int(row["bar_idx"])
        level_bar = int(row["level_bar"])
        tol = level * 10 / 10000  # 10bps tolerance for counting touches

        count = 0
        for k in range(level_bar + 1, bar_idx):
            if name == "RESISTANCE":
                # How many bars tested this level from below (high near level)
                if abs(high[k] - level) <= tol or abs(low[k] - level) <= tol:
                    count += 1
            else:
                if abs(low[k] - level) <= tol or abs(high[k] - level) <= tol:
                    count += 1
        touch_counts.append(count)

    df_copy = df.copy()
    df_copy["touches"] = touch_counts

    # Join with labels
    df_wl = df_copy.set_index("date").join(labels[["direction_h8"]], how="inner")
    valid = df_wl[df_wl["direction_h8"].isin([0, 1])]

    if len(valid) == 0:
        continue

    expected_val = 1 if name == "RESISTANCE" else 0
    expected_label = "SHORT" if name == "RESISTANCE" else "LONG"

    # Bucket by touch count
    for touches_min, touches_max, label in [(0, 0, "0 touches"), (1, 2, "1-2 touches"), (3, 99, "3+ touches")]:
        subset = valid[(valid["touches"] >= touches_min) & (valid["touches"] <= touches_max)]
        if len(subset) < 10:
            continue
        rate = (subset["direction_h8"] == expected_val).mean() * 100
        print(f"  {label}: {len(subset)} retests, {expected_label} rate = {rate:.1f}%")


print(f"\n{'='*70}")
print("PART 5: DISTANCE FROM LEVEL - DOES PRECISION MATTER?")
print(f"{'='*70}")

for name, df in [("RESISTANCE", res_df), ("SUPPORT", sup_df)]:
    if len(df) == 0:
        continue
    print(f"\n--- {name} ---")

    df_wl = df.set_index("date").join(labels[["direction_h8"]], how="inner")
    valid = df_wl[df_wl["direction_h8"].isin([0, 1])]

    if len(valid) == 0:
        continue

    expected_val = 1 if name == "RESISTANCE" else 0
    expected_label = "SHORT" if name == "RESISTANCE" else "LONG"

    # Bucket by distance
    abs_dist = valid["dist_bps"].abs()
    for dist_max, label in [(1, "<1 bps"), (3, "<3 bps"), (5, "<5 bps")]:
        subset = valid[abs_dist <= dist_max]
        if len(subset) < 10:
            continue
        rate = (subset["direction_h8"] == expected_val).mean() * 100
        print(f"  {label} from level: {len(subset)} retests, {expected_label} rate = {rate:.1f}%")


print(f"\n{'='*70}")
print("PART 6: BASELINE COMPARISON")
print(f"{'='*70}")
print("""
Baseline: On ALL bars (no S/R filter), what is the H8 direction distribution?
This tells us if S/R retests have ANY edge over random bars.
""")

all_valid = labels[labels["direction_h8"].isin([0, 1])]
long_rate = (all_valid["direction_h8"] == 0).mean() * 100
short_rate = (all_valid["direction_h8"] == 1).mean() * 100
print(f"All bars (no filter): {len(all_valid)} bars, LONG={long_rate:.1f}%, SHORT={short_rate:.1f}%")

# Train
all_train = all_valid.loc["2020-01-01":"2023-12-31"]
if len(all_train) > 0:
    print(f"Train 2020-2023: LONG={((all_train['direction_h8']==0).mean()*100):.1f}%, SHORT={((all_train['direction_h8']==1).mean()*100):.1f}%")

all_test = all_valid.loc["2024-01-01":"2025-12-31"]
if len(all_test) > 0:
    print(f"Test 2024-2025:  LONG={((all_test['direction_h8']==0).mean()*100):.1f}%, SHORT={((all_test['direction_h8']==1).mean()*100):.1f}%")


print(f"\n{'='*70}")
print("PART 7: MFE MAGNITUDE AT RETESTS")
print(f"{'='*70}")
print("How much does price move after a retest? (in the bounce direction)")

for name, df in [("RESISTANCE", res_df), ("SUPPORT", sup_df)]:
    if len(df) == 0:
        continue
    print(f"\n--- {name} ---")

    df_wl = df.set_index("date").join(
        labels[["mfe_up_8", "mfe_down_8", "mfe_up_32", "mfe_down_32", "direction_h8"]],
        how="inner"
    )

    if name == "RESISTANCE":
        # Bounce = SHORT = down move
        print(f"  MFE down (bounce direction) H8:  mean={df_wl['mfe_down_8'].mean():.1f} bps, median={df_wl['mfe_down_8'].median():.1f} bps")
        print(f"  MFE up (break direction) H8:     mean={df_wl['mfe_up_8'].mean():.1f} bps, median={df_wl['mfe_up_8'].median():.1f} bps")
        print(f"  MFE down (bounce) H32:           mean={df_wl['mfe_down_32'].mean():.1f} bps, median={df_wl['mfe_down_32'].median():.1f} bps")
        print(f"  MFE up (break) H32:              mean={df_wl['mfe_up_32'].mean():.1f} bps, median={df_wl['mfe_up_32'].median():.1f} bps")
    else:
        print(f"  MFE up (bounce direction) H8:    mean={df_wl['mfe_up_8'].mean():.1f} bps, median={df_wl['mfe_up_8'].median():.1f} bps")
        print(f"  MFE down (break direction) H8:   mean={df_wl['mfe_down_8'].mean():.1f} bps, median={df_wl['mfe_down_8'].median():.1f} bps")
        print(f"  MFE up (bounce) H32:             mean={df_wl['mfe_up_32'].mean():.1f} bps, median={df_wl['mfe_up_32'].median():.1f} bps")
        print(f"  MFE down (break) H32:            mean={df_wl['mfe_down_32'].mean():.1f} bps, median={df_wl['mfe_down_32'].median():.1f} bps")


print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
