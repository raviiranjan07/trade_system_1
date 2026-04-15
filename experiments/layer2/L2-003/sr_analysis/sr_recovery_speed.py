"""S/R Recovery Speed Analysis

When support/resistance breaks and price recovers back to the zone:
Does the SPEED of recovery predict what happens next?

Fast recovery (1 bar) vs slow recovery (2-3+ bars) -- is there a difference
in bounce rate, break-through rate, and move size?

Run: PYTHONPATH=src python experiments/layer2/L2-003/sr_recovery_speed.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OHLCV_PATH = ROOT / "data" / "ohlcv" / "BTCUSDT_15m_ohlcv.parquet"

LOOKBACK = 7

print("=" * 70)
print("S/R RECOVERY SPEED ANALYSIS")
print("=" * 70)

ohlcv = pd.read_parquet(OHLCV_PATH)
ohlcv.index = ohlcv.index.tz_localize(None) if ohlcv.index.tz is not None else ohlcv.index

high = ohlcv["high"].values
low = ohlcv["low"].values
close = ohlcv["close"].values
N = len(ohlcv)

print("\nComputing S/R zones, breaks, and recovery speed...")

support_rows = []
resistance_rows = []

for t in range(LOOKBACK + 1, N - 14):
    bar_high = high[t]
    bar_low = low[t]
    bar_close = close[t]
    prev_close = close[t - 1]

    best_support = None
    best_resistance = None

    for a in range(t - LOOKBACK, t):
        zone_high = high[a]
        zone_low = low[a]

        # SUPPORT ZONE
        broken_up = False
        highest_after = zone_high
        for k in range(a + 1, t):
            if high[k] > zone_high:
                broken_up = True
                if high[k] > highest_after:
                    highest_after = high[k]

        if broken_up and bar_low <= zone_high and bar_high >= zone_low and prev_close > zone_high:
            touches = 0
            for k in range(a + 1, t):
                if low[k] <= zone_high and high[k] >= zone_low:
                    touches += 1
            dist = abs(bar_close - zone_high) / zone_high * 10000
            if best_support is None or dist < best_support["dist"]:
                best_support = {
                    "zone_high": zone_high, "zone_low": zone_low,
                    "touches": touches, "dist": dist,
                    "recent_high": highest_after,
                }

        # RESISTANCE ZONE
        broken_down = False
        lowest_after = zone_low
        for k in range(a + 1, t):
            if low[k] < zone_low:
                broken_down = True
                if low[k] < lowest_after:
                    lowest_after = low[k]

        if broken_down and bar_high >= zone_low and bar_low <= zone_high and prev_close < zone_low:
            touches = 0
            for k in range(a + 1, t):
                if low[k] <= zone_high and high[k] >= zone_low:
                    touches += 1
            dist = abs(bar_close - zone_low) / zone_low * 10000
            if best_resistance is None or dist < best_resistance["dist"]:
                best_resistance = {
                    "zone_high": zone_high, "zone_low": zone_low,
                    "touches": touches, "dist": dist,
                    "recent_low": lowest_after,
                }

    # SUPPORT BREAK + RECOVERY
    if best_support:
        zh = best_support["zone_high"]
        zl = best_support["zone_low"]

        if close[t + 1] < zl:  # support broke
            # Find recovery bar
            recovery_bar = None
            for b in range(t + 2, t + 11):
                if high[b] >= zl:
                    recovery_bar = b
                    break

            if recovery_bar is not None:
                recovery_speed = recovery_bar - (t + 1)  # bars to recover
                rb1 = recovery_bar + 1
                if rb1 + 3 < N:
                    post_highs = high[rb1:rb1 + 3]
                    post_lows = low[rb1:rb1 + 3]

                    broke_through_up = np.max(post_highs) > zh
                    rejected_down = np.min(post_lows) < zl
                    max_up = (np.max(post_highs) - zh) / zh * 10000
                    max_down = max((zl - np.min(post_lows)) / zl * 10000, 0)

                    # Break depth
                    min_before = np.min(low[t + 1:recovery_bar + 1])
                    break_depth = (zl - min_before) / zl * 10000

                    support_rows.append({
                        "date": ohlcv.index[t],
                        "recovery_speed": recovery_speed,
                        "break_depth": break_depth,
                        "zone_width": (zh - zl) / zl * 10000,
                        "touches": best_support["touches"],
                        "broke_through_up": broke_through_up,
                        "rejected_down": rejected_down,
                        "max_up": max_up,
                        "max_down": max_down,
                    })

    # RESISTANCE BREAK + RECOVERY
    if best_resistance:
        zh = best_resistance["zone_high"]
        zl = best_resistance["zone_low"]

        if close[t + 1] > zh:  # resistance broke
            recovery_bar = None
            for b in range(t + 2, t + 11):
                if low[b] <= zh:
                    recovery_bar = b
                    break

            if recovery_bar is not None:
                recovery_speed = recovery_bar - (t + 1)
                rb1 = recovery_bar + 1
                if rb1 + 3 < N:
                    post_highs = high[rb1:rb1 + 3]
                    post_lows = low[rb1:rb1 + 3]

                    broke_through_down = np.min(post_lows) < zl
                    rejected_up = np.max(post_highs) > zh
                    max_down = max((zl - np.min(post_lows)) / zl * 10000, 0)
                    max_up = max((np.max(post_highs) - zh) / zh * 10000, 0)

                    max_before = np.max(high[t + 1:recovery_bar + 1])
                    break_height = (max_before - zh) / zh * 10000

                    resistance_rows.append({
                        "date": ohlcv.index[t],
                        "recovery_speed": recovery_speed,
                        "break_depth": break_height,
                        "zone_width": (zh - zl) / zl * 10000,
                        "touches": best_resistance["touches"],
                        "broke_through_down": broke_through_down,
                        "rejected_up": rejected_up,
                        "max_down": max_down,
                        "max_up": max_up,
                    })

    if t % 50000 == 0:
        print(f"  {t}/{N}... sup={len(support_rows)}, res={len(resistance_rows)}")

sup_df = pd.DataFrame(support_rows)
res_df = pd.DataFrame(resistance_rows)

# =========================================================================
# SUPPORT: Recovery speed analysis
# =========================================================================
print(f"\n{'='*70}")
print("SUPPORT: Does recovery speed predict outcome?")
print(f"{'='*70}")

print(f"\nTotal support breaks with recovery: {len(sup_df)}")
print(f"\nRecovery speed distribution:")
for speed in sorted(sup_df["recovery_speed"].unique()):
    count = (sup_df["recovery_speed"] == speed).sum()
    print(f"  {speed} bar(s): {count} ({count/len(sup_df)*100:.1f}%)")

print(f"\nOutcome by recovery speed:")
for speed_min, speed_max, label in [(1, 1, "1 bar (fast)"), (2, 2, "2 bars"), (3, 3, "3 bars"), (4, 9, "4+ bars (slow)")]:
    mask = (sup_df["recovery_speed"] >= speed_min) & (sup_df["recovery_speed"] <= speed_max)
    sub = sup_df[mask]
    if len(sub) < 20:
        continue

    through = sub["broke_through_up"].mean() * 100
    rejected = sub["rejected_down"].mean() * 100
    clean_through = ((sub["broke_through_up"]) & (~sub["rejected_down"])).mean() * 100
    clean_rejected = ((~sub["broke_through_up"]) & (sub["rejected_down"])).mean() * 100
    avg_up = sub["max_up"].mean()
    avg_down = sub["max_down"].mean()
    avg_depth = sub["break_depth"].mean()

    print(f"\n  {label}: {len(sub)} bars")
    print(f"    Break depth before recovery: mean {avg_depth:.1f} bps")
    print(f"    Clean break through up: {clean_through:.1f}%")
    print(f"    Clean rejected down:    {clean_rejected:.1f}%")
    print(f"    Volatile (both ways):   {through - clean_through:.1f}%")
    print(f"    Avg move up after:  {avg_up:.1f} bps")
    print(f"    Avg move down after: {avg_down:.1f} bps")

# =========================================================================
# RESISTANCE: Recovery speed analysis
# =========================================================================
print(f"\n{'='*70}")
print("RESISTANCE: Does recovery speed predict outcome?")
print(f"{'='*70}")

print(f"\nTotal resistance breaks with recovery: {len(res_df)}")
print(f"\nRecovery speed distribution:")
for speed in sorted(res_df["recovery_speed"].unique()):
    count = (res_df["recovery_speed"] == speed).sum()
    print(f"  {speed} bar(s): {count} ({count/len(res_df)*100:.1f}%)")

print(f"\nOutcome by recovery speed:")
for speed_min, speed_max, label in [(1, 1, "1 bar (fast)"), (2, 2, "2 bars"), (3, 3, "3 bars"), (4, 9, "4+ bars (slow)")]:
    mask = (res_df["recovery_speed"] >= speed_min) & (res_df["recovery_speed"] <= speed_max)
    sub = res_df[mask]
    if len(sub) < 20:
        continue

    through = sub["broke_through_down"].mean() * 100
    rejected = sub["rejected_up"].mean() * 100
    clean_through = ((sub["broke_through_down"]) & (~sub["rejected_up"])).mean() * 100
    clean_rejected = ((~sub["broke_through_down"]) & (sub["rejected_up"])).mean() * 100
    avg_down = sub["max_down"].mean()
    avg_up = sub["max_up"].mean()
    avg_depth = sub["break_depth"].mean()

    print(f"\n  {label}: {len(sub)} bars")
    print(f"    Break height before recovery: mean {avg_depth:.1f} bps")
    print(f"    Clean break through down: {clean_through:.1f}%")
    print(f"    Clean rejected up:        {clean_rejected:.1f}%")
    print(f"    Volatile (both ways):     {through - clean_through:.1f}%")
    print(f"    Avg move down after: {avg_down:.1f} bps")
    print(f"    Avg move up after:   {avg_up:.1f} bps")

# =========================================================================
# Combined: recovery speed + zone width
# =========================================================================
print(f"\n{'='*70}")
print("COMBINED: Recovery speed + Zone width")
print(f"{'='*70}")

for name, df, through_col in [("SUPPORT", sup_df, "broke_through_up"), ("RESISTANCE", res_df, "broke_through_down")]:
    print(f"\n--- {name} ---")
    median_width = df["zone_width"].median()

    for speed_label, s_min, s_max in [("Fast (1 bar)", 1, 1), ("Slow (3+ bars)", 3, 9)]:
        for width_label, w_cond in [("Narrow (<median)", df["zone_width"] < median_width), ("Wide (>=median)", df["zone_width"] >= median_width)]:
            s_mask = (df["recovery_speed"] >= s_min) & (df["recovery_speed"] <= s_max)
            combined = s_mask & w_cond
            sub = df[combined]
            if len(sub) < 20:
                continue
            through = sub[through_col].mean() * 100
            print(f"  {speed_label} + {width_label}: {len(sub)} bars, break through = {through:.1f}%")

# =========================================================================
# Train vs Test
# =========================================================================
print(f"\n{'='*70}")
print("TRAIN vs TEST by recovery speed")
print(f"{'='*70}")

for name, df, through_col in [("SUPPORT", sup_df, "broke_through_up"), ("RESISTANCE", res_df, "broke_through_down")]:
    print(f"\n--- {name} ---")
    df_dated = df.set_index("date")

    for period, start, end in [("TRAIN 2020-2023", "2020-01-01", "2023-12-31"), ("TEST 2024-2025", "2024-01-01", "2025-12-31")]:
        p_mask = (df_dated.index >= start) & (df_dated.index <= end)
        sub = df_dated[p_mask]
        if len(sub) == 0:
            continue

        for speed_label, s_min, s_max in [("Fast (1 bar)", 1, 1), ("Slow (3+ bars)", 3, 9)]:
            s_mask = (sub["recovery_speed"] >= s_min) & (sub["recovery_speed"] <= s_max)
            ss = sub[s_mask]
            if len(ss) < 20:
                continue
            through = ss[through_col].mean() * 100
            rejected_col = "rejected_down" if name == "SUPPORT" else "rejected_up"
            rejected = ss[rejected_col].mean() * 100
            print(f"  {period} | {speed_label}: {len(ss)} bars | through={through:.1f}% | rejected={rejected:.1f}%")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
