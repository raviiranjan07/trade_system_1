"""S/R Feature Test V5 - Confirmed support/resistance (2+ touches) + window extremes

Rules:
  1. Support = cluster of lows with 2+ members (confirmed)
  2. Resistance = cluster of highs with 2+ members (confirmed)
  3. If lowest cluster has only 1 low, skip to next cluster with 2+
  4. If no cluster has 2+, carry forward from previous snapshot
  5. Window low/high = absolute extremes (separate features)

11 Features:
  1. zone_width (bps)
  2. support_range (confirmed, 2+ touches)
  3. resistance_range (confirmed, 2+ touches)
  4. support_retest (count)
  5. resistance_retest (count)
  6. distance_to_support (bps)
  7. distance_to_resistance (bps)
  8. recovery_up (bps/bar)
  9. recovery_down (bps/bar)
  10. window_low (absolute low)
  11. window_high (absolute high)

Run: PYTHONPATH=src python experiments/layer2/L2-003/sr_feature_test_v5.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OHLCV_PATH = ROOT / "data" / "ohlcv" / "BTCUSDT_15m_ohlcv.parquet"

LOOKBACK = 7

print("=" * 70)
print("S/R FEATURE TEST V5 - Confirmed (2+) + Window Extremes")
print("=" * 70)

ohlcv = pd.read_parquet(OHLCV_PATH)
ohlcv.index = ohlcv.index.tz_localize(None) if ohlcv.index.tz is not None else ohlcv.index

high = ohlcv["high"].values
low = ohlcv["low"].values
close = ohlcv["close"].values
N = len(ohlcv)

print(f"Loaded: {N} bars")


def find_confirmed_zones(highs, lows):
    """Find confirmed support and resistance using natural gaps.

    Support = lowest cluster with 2+ lows.
    Resistance = highest cluster with 2+ highs.
    If lowest/highest cluster has only 1, skip to next with 2+.
    """
    # --- SUPPORT ---
    sorted_lows = np.sort(lows)
    sup_low, sup_high = None, None

    # Try splitting at each gap from biggest to smallest
    # Find all gaps and try biggest first
    gaps = np.diff(sorted_lows)
    if len(gaps) > 0:
        # Start with biggest gap
        gap_order = np.argsort(gaps)[::-1]  # indices sorted by gap size descending

        for gap_idx in gap_order:
            # Bottom cluster = everything at or below this gap
            bottom = sorted_lows[:gap_idx + 1]
            if len(bottom) >= 2:
                sup_low = bottom[0]
                sup_high = bottom[-1]
                break

    # If no gap gave us 2+ cluster, check if all lows are close (one big cluster)
    if sup_low is None and len(sorted_lows) >= 2:
        # All lows might be in one cluster — check total range
        total_range = sorted_lows[-1] - sorted_lows[0]
        ref = sorted_lows[0]
        # If all lows span less than 20 bps, they're one cluster
        if total_range / ref * 10000 < 20:
            sup_low = sorted_lows[0]
            sup_high = sorted_lows[1]  # take lowest 2

    # --- RESISTANCE ---
    sorted_highs = np.sort(highs)
    res_low, res_high = None, None

    gaps_h = np.diff(sorted_highs)
    if len(gaps_h) > 0:
        gap_order_h = np.argsort(gaps_h)[::-1]

        for gap_idx in gap_order_h:
            # Top cluster = everything above this gap
            top = sorted_highs[gap_idx + 1:]
            if len(top) >= 2:
                res_low = top[0]
                res_high = top[-1]
                break

    if res_low is None and len(sorted_highs) >= 2:
        total_range = sorted_highs[-1] - sorted_highs[0]
        ref = sorted_highs[0]
        if total_range / ref * 10000 < 20:
            res_low = sorted_highs[-2]
            res_high = sorted_highs[-1]

    # Zone valid only if resistance above support
    if sup_low is not None and res_low is not None and res_low <= sup_high:
        return None, None, None, None

    return sup_low, sup_high, res_low, res_high


def count_bar_touches(highs, lows, range_low, range_high):
    """Count how many bars touch a price range."""
    count = 0
    for i in range(len(highs)):
        if lows[i] <= range_high and highs[i] >= range_low:
            count += 1
    return count


def compute_recovery_speeds(highs, lows, closes, sup_low, sup_high, res_low, res_high):
    """Compute recovery_up and recovery_down speeds."""
    n = len(highs)
    recovery_up = 0.0
    recovery_down = 0.0

    if sup_low is not None and sup_high is not None:
        for i in range(n - 1):
            if lows[i] <= sup_high and lows[i] >= sup_low:
                for j in range(i + 1, n):
                    if closes[j] > sup_high:
                        bars = j - i
                        move = (closes[j] - sup_high) / sup_high * 10000
                        speed = move / bars
                        if speed > recovery_up:
                            recovery_up = speed
                        break

    if res_low is not None and res_high is not None:
        for i in range(n - 1):
            if highs[i] >= res_low and highs[i] <= res_high:
                for j in range(i + 1, n):
                    if closes[j] < res_low:
                        bars = j - i
                        move = (res_low - closes[j]) / res_low * 10000
                        speed = move / bars
                        if speed > recovery_down:
                            recovery_down = speed
                        break

    return recovery_up, recovery_down


# =========================================================================
# Compute features
# =========================================================================
print("\nComputing features...")

results = []

# Carry-forward state
prev_sup_low = None
prev_sup_high = None
prev_res_low = None
prev_res_high = None

# Stats
found_both = 0
found_sup_only = 0
found_res_only = 0
found_none = 0
carried_sup = 0
carried_res = 0

for t in range(LOOKBACK + 1, N - 3):
    w_high = high[t-LOOKBACK:t+1]
    w_low = low[t-LOOKBACK:t+1]
    w_close = close[t-LOOKBACK:t+1]
    cur_price = close[t]

    # Window extremes (always available)
    w_low_val = np.min(w_low)
    w_high_val = np.max(w_high)

    # Find confirmed zones
    sup_low, sup_high, res_low, res_high = find_confirmed_zones(w_high, w_low)

    has_new_sup = sup_low is not None
    has_new_res = res_low is not None

    src_sup = "new"
    src_res = "new"

    if has_new_sup and has_new_res:
        found_both += 1
    elif has_new_sup:
        found_sup_only += 1
    elif has_new_res:
        found_res_only += 1
    else:
        found_none += 1

    # Carry forward if not found
    if not has_new_sup and prev_sup_low is not None:
        sup_low = prev_sup_low
        sup_high = prev_sup_high
        src_sup = "carried"
        carried_sup += 1
    if not has_new_res and prev_res_low is not None:
        res_low = prev_res_low
        res_high = prev_res_high
        src_res = "carried"
        carried_res += 1

    # Update carry-forward
    if has_new_sup:
        prev_sup_low = sup_low
        prev_sup_high = sup_high
    if has_new_res:
        prev_res_low = res_low
        prev_res_high = res_high

    has_support = sup_low is not None
    has_resistance = res_low is not None
    has_zone = has_support and has_resistance and res_low > sup_high

    row = {
        "date": ohlcv.index[t],
        "current_price": cur_price,
        "has_zone": has_zone,
        "has_support": has_support,
        "has_resistance": has_resistance,
        "src_sup": src_sup,
        "src_res": src_res,
        "window_low": w_low_val,
        "window_high": w_high_val,
    }

    if has_zone:
        row["zone_width"] = (res_low - sup_high) / sup_high * 10000
        row["support_range_low"] = sup_low
        row["support_range_high"] = sup_high
        row["resistance_range_low"] = res_low
        row["resistance_range_high"] = res_high
        row["support_retest"] = count_bar_touches(w_high, w_low, sup_low, sup_high)
        row["resistance_retest"] = count_bar_touches(w_high, w_low, res_low, res_high)
        row["distance_to_support"] = (cur_price - sup_high) / sup_high * 10000
        row["distance_to_resistance"] = (res_low - cur_price) / res_low * 10000
        row["dist_to_window_low"] = (cur_price - w_low_val) / w_low_val * 10000
        row["dist_to_window_high"] = (w_high_val - cur_price) / w_high_val * 10000

        rec_up, rec_down = compute_recovery_speeds(w_high, w_low, w_close, sup_low, sup_high, res_low, res_high)
        row["recovery_up"] = rec_up
        row["recovery_down"] = rec_down

        # Price position
        if cur_price < sup_low:
            row["position"] = "below_support"
        elif cur_price <= sup_high + sup_high * 5 / 10000:
            row["position"] = "at_support"
        elif cur_price >= res_low - res_low * 5 / 10000:
            row["position"] = "at_resistance"
        elif cur_price > res_high:
            row["position"] = "above_resistance"
        else:
            row["position"] = "in_middle"
    else:
        row["zone_width"] = np.nan
        row["support_range_low"] = sup_low
        row["support_range_high"] = sup_high
        row["resistance_range_low"] = res_low
        row["resistance_range_high"] = res_high
        row["support_retest"] = count_bar_touches(w_high, w_low, sup_low, sup_high) if has_support else 0
        row["resistance_retest"] = count_bar_touches(w_high, w_low, res_low, res_high) if has_resistance else 0
        row["distance_to_support"] = (cur_price - sup_high) / sup_high * 10000 if has_support else np.nan
        row["distance_to_resistance"] = (res_low - cur_price) / res_low * 10000 if has_resistance else np.nan
        row["dist_to_window_low"] = (cur_price - w_low_val) / w_low_val * 10000
        row["dist_to_window_high"] = (w_high_val - cur_price) / w_high_val * 10000
        row["recovery_up"] = 0.0
        row["recovery_down"] = 0.0
        row["position"] = "no_zone"

    # Outcomes
    row["next1_close"] = close[t+1]
    row["next3_close"] = close[t+3]
    row["next1_high"] = high[t+1]
    row["next1_low"] = low[t+1]

    results.append(row)

    if t % 50000 == 0:
        z = sum(1 for r in results if r["has_zone"])
        print(f"  {t}/{N}... {len(results)} bars, {z} zones ({z/len(results)*100:.1f}%)")

df = pd.DataFrame(results)
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

total = len(df)
print(f"\nTotal bars: {total}")
print(f"  Has confirmed support: {df['has_support'].sum()} ({df['has_support'].mean()*100:.1f}%)")
print(f"  Has confirmed resistance: {df['has_resistance'].sum()} ({df['has_resistance'].mean()*100:.1f}%)")
print(f"  Has full zone: {df['has_zone'].sum()} ({df['has_zone'].mean()*100:.1f}%)")

print(f"\nSource breakdown:")
print(f"  Found both: {found_both} ({found_both/total*100:.1f}%)")
print(f"  Found sup only: {found_sup_only} ({found_sup_only/total*100:.1f}%)")
print(f"  Found res only: {found_res_only} ({found_res_only/total*100:.1f}%)")
print(f"  Found none: {found_none} ({found_none/total*100:.1f}%)")
print(f"  Carried sup: {carried_sup} ({carried_sup/total*100:.1f}%)")
print(f"  Carried res: {carried_res} ({carried_res/total*100:.1f}%)")

# =========================================================================
# Verification with the example bar
# =========================================================================
print(f"\n{'='*70}")
print("VERIFICATION: Example at t=2005")
print(f"{'='*70}")

t_check = 2005
if t_check - LOOKBACK - 1 >= 0:
    w_h = high[t_check-7:t_check+1]
    w_l = low[t_check-7:t_check+1]
    w_c = close[t_check-7:t_check+1]
    w_o = ohlcv["open"].values[t_check-7:t_check+1]

    print(f"\n  8-bar window at {ohlcv.index[t_check]}:")
    for i in range(8):
        print(f"    Bar {i-7}: open={w_o[i]:.2f}  high={w_h[i]:.2f}  low={w_l[i]:.2f}  close={w_c[i]:.2f}")

    sl, sh, rl, rh = find_confirmed_zones(w_h, w_l)
    print(f"\n  Confirmed support: {sl:.2f} - {sh:.2f}" if sl else "\n  Confirmed support: NONE")
    print(f"  Confirmed resistance: {rl:.2f} - {rh:.2f}" if rl else "  Confirmed resistance: NONE")
    print(f"  Window low: {np.min(w_l):.2f}")
    print(f"  Window high: {np.max(w_h):.2f}")

    if sl and rl:
        st = count_bar_touches(w_h, w_l, sl, sh)
        rt = count_bar_touches(w_h, w_l, rl, rh)
        print(f"  Support touches: {st}")
        print(f"  Resistance touches: {rt}")
        print(f"  Zone width: {(rl - sh) / sh * 10000:.1f} bps")

    # Show the clusters
    sorted_lows = np.sort(w_l)
    print(f"\n  Sorted lows: {sorted_lows}")
    print(f"  Low gaps: {np.diff(sorted_lows)}")

    sorted_highs = np.sort(w_h)
    print(f"  Sorted highs: {sorted_highs}")
    print(f"  High gaps: {np.diff(sorted_highs)}")

# =========================================================================
# Outcome analysis
# =========================================================================
print(f"\n{'='*70}")
print("OUTCOME: Does price respect confirmed S/R?")
print(f"{'='*70}")

zdf = df[df["has_zone"]].copy()
print(f"\nBars with full zone: {len(zdf)} ({len(zdf)/total*100:.1f}%)")

if len(zdf) > 0:
    positions = zdf["position"].value_counts()
    print(f"\nPrice positions:")
    for pos in ["at_support", "at_resistance", "in_middle", "below_support", "above_resistance"]:
        if pos in positions:
            print(f"  {pos}: {positions[pos]} ({positions[pos]/len(zdf)*100:.1f}%)")

    for pos, expected_dir, dir_label in [
        ("at_support", "up", "UP"),
        ("at_resistance", "down", "DOWN"),
        ("in_middle", "up", "UP"),
        ("below_support", "up", "UP"),
        ("above_resistance", "down", "DOWN"),
    ]:
        subset = zdf[zdf["position"] == pos]
        if len(subset) < 20:
            continue
        if expected_dir == "up":
            bounce1 = (subset["next1_close"] > subset["current_price"]).mean() * 100
            bounce3 = (subset["next3_close"] > subset["current_price"]).mean() * 100
        else:
            bounce1 = (subset["next1_close"] < subset["current_price"]).mean() * 100
            bounce3 = (subset["next3_close"] < subset["current_price"]).mean() * 100

        print(f"\n  {pos} ({len(subset)} bars):")
        print(f"    Next bar {dir_label}: {bounce1:.1f}%")
        print(f"    3 bars {dir_label}:   {bounce3:.1f}%")

    # By support retest count (at support)
    print(f"\n--- Support retest count (at support) ---")
    sup_bars = zdf[zdf["position"] == "at_support"]
    if len(sup_bars) > 0:
        for tc in sorted(sup_bars["support_retest"].unique()):
            sub = sup_bars[sup_bars["support_retest"] == tc]
            if len(sub) < 20:
                continue
            bounce = (sub["next1_close"] > sub["current_price"]).mean() * 100
            print(f"  {int(tc)} touches: {len(sub)} bars, bounce UP = {bounce:.1f}%")

    # By resistance retest count (at resistance)
    print(f"\n--- Resistance retest count (at resistance) ---")
    res_bars = zdf[zdf["position"] == "at_resistance"]
    if len(res_bars) > 0:
        for tc in sorted(res_bars["resistance_retest"].unique()):
            sub = res_bars[res_bars["resistance_retest"] == tc]
            if len(sub) < 20:
                continue
            bounce = (sub["next1_close"] < sub["current_price"]).mean() * 100
            print(f"  {int(tc)} touches: {len(sub)} bars, bounce DOWN = {bounce:.1f}%")

    # By zone width (at support)
    print(f"\n--- Zone width (at support) ---")
    if len(sup_bars) > 0:
        for q_low, q_high, label in [(0, 0.25, "Q1 narrow"), (0.25, 0.5, "Q2"), (0.5, 0.75, "Q3"), (0.75, 1.01, "Q4 wide")]:
            low_val = sup_bars["zone_width"].quantile(q_low)
            high_val = sup_bars["zone_width"].quantile(min(q_high, 1.0))
            mask = (sup_bars["zone_width"] >= low_val) & (sup_bars["zone_width"] < high_val)
            sub = sup_bars[mask]
            if len(sub) < 20:
                continue
            bounce = (sub["next1_close"] > sub["current_price"]).mean() * 100
            print(f"  {label} ({low_val:.0f}-{high_val:.0f} bps): {len(sub)} bars, bounce UP = {bounce:.1f}%")

    # By recovery_up (at support)
    print(f"\n--- Recovery up speed (at support) ---")
    if len(sup_bars) > 0:
        no_rec = sup_bars[sup_bars["recovery_up"] == 0]
        has_rec = sup_bars[sup_bars["recovery_up"] > 0]
        if len(no_rec) >= 20:
            bounce = (no_rec["next1_close"] > no_rec["current_price"]).mean() * 100
            print(f"  No recovery: {len(no_rec)} bars, bounce UP = {bounce:.1f}%")
        if len(has_rec) >= 40:
            med = has_rec["recovery_up"].median()
            slow = has_rec[has_rec["recovery_up"] <= med]
            fast = has_rec[has_rec["recovery_up"] > med]
            if len(slow) >= 20:
                bounce = (slow["next1_close"] > slow["current_price"]).mean() * 100
                print(f"  Slow (<={med:.1f}): {len(slow)} bars, bounce UP = {bounce:.1f}%")
            if len(fast) >= 20:
                bounce = (fast["next1_close"] > fast["current_price"]).mean() * 100
                print(f"  Fast (>{med:.1f}): {len(fast)} bars, bounce UP = {bounce:.1f}%")

    # New vs Carried
    print(f"\n--- New vs Carried S/R ---")
    new_sr = zdf[(zdf["src_sup"] == "new") & (zdf["src_res"] == "new")]
    carried_sr = zdf[(zdf["src_sup"] == "carried") | (zdf["src_res"] == "carried")]

    for label, subset in [("New S/R", new_sr), ("Carried S/R", carried_sr)]:
        if len(subset) == 0:
            continue
        at_s = subset[subset["position"] == "at_support"]
        at_r = subset[subset["position"] == "at_resistance"]
        s_b = (at_s["next1_close"] > at_s["current_price"]).mean() * 100 if len(at_s) > 20 else float("nan")
        r_b = (at_r["next1_close"] < at_r["current_price"]).mean() * 100 if len(at_r) > 20 else float("nan")
        print(f"  {label}: sup {len(at_s)} bars bounce {s_b:.1f}% | res {len(at_r)} bars bounce {r_b:.1f}%")

# =========================================================================
# Train vs Test
# =========================================================================
print(f"\n{'='*70}")
print("TRAIN vs TEST")
print(f"{'='*70}")

for period, start, end in [("TRAIN 2020-2023", "2020-01-01", "2023-12-31"),
                            ("TEST 2024-2025", "2024-01-01", "2025-12-31")]:
    sub = zdf[(zdf.index >= start) & (zdf.index <= end)]
    if len(sub) == 0:
        continue
    at_s = sub[sub["position"] == "at_support"]
    at_r = sub[sub["position"] == "at_resistance"]
    s_b = (at_s["next1_close"] > at_s["current_price"]).mean() * 100 if len(at_s) > 0 else 0
    r_b = (at_r["next1_close"] < at_r["current_price"]).mean() * 100 if len(at_r) > 0 else 0
    print(f"  {period}: sup {len(at_s)} bars bounce {s_b:.1f}% | res {len(at_r)} bars bounce {r_b:.1f}%")

# Baseline
print(f"\n{'='*70}")
print("BASELINE")
print(f"{'='*70}")
all_up = (df["next1_close"] > df["current_price"]).mean() * 100
print(f"  All bars: next bar UP = {all_up:.1f}%")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
