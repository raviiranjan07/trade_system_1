"""Brain Pipeline -- Step 3: Label Generation

Compute labels for each bar:
  - direction: which +/-threshold_bps hits first within horizon bars (LONG=0, SHORT=1, BOTH=2, SKIP=3)
  - mfe_up: max favorable excursion upward at each horizon (bps)
  - mfe_down: max favorable excursion downward at each horizon (bps)
  - bounce_break: 4-class S/R outcome label
      BOUNCE_SUPPORT=0 (at support, price goes UP)
      BREAK_SUPPORT=1 (at support, price goes DOWN)
      BOUNCE_RESISTANCE=2 (at resistance, price goes DOWN)
      BREAK_RESISTANCE=3 (at resistance, price goes UP)
      SKIP=4 (not at S/R or neither threshold hit)
"""

import numpy as np
import pandas as pd

LONG = 0
SHORT = 1
BOTH = 2
SKIP = 3

BOUNCE_SUPPORT = 0
BREAK_SUPPORT = 1
BOUNCE_RESISTANCE = 2
BREAK_RESISTANCE = 3
BB_SKIP = 4


def compute_labels(df: pd.DataFrame, config: dict, features: pd.DataFrame = None) -> pd.DataFrame:
    """Compute all labels for every bar.

    Input: OHLCV dataframe
    Output: Label dataframe with direction + MFE columns
    """
    label_cfg = config["labels"]
    high = df["high"].values
    low = df["low"].values
    open_arr = df["open"].values
    N = len(df)

    labels = pd.DataFrame(index=df.index)

    # --- Direction ---
    if label_cfg.get("direction", {}).get("enabled", True):
        horizon = label_cfg["direction"]["horizon"]
        threshold = label_cfg["direction"]["threshold_bps"]
        print(f"  Computing direction labels (H{horizon}, +/-{threshold}bps)...")

        direction = np.full(N, SKIP, dtype=np.int8)

        for i in range(N - horizon):
            entry = open_arr[i + 1]  # next bar open
            thresh_up = entry * (1 + threshold / 10000)
            thresh_down = entry * (1 - threshold / 10000)

            first_up = 0
            first_down = 0

            for j in range(1, horizon + 1):
                if i + j >= N:
                    break
                if first_up == 0 and high[i + j] >= thresh_up:
                    first_up = j
                if first_down == 0 and low[i + j] <= thresh_down:
                    first_down = j

            if first_up > 0 and first_down > 0:
                if first_up < first_down:
                    direction[i] = LONG
                elif first_down < first_up:
                    direction[i] = SHORT
                else:
                    direction[i] = BOTH
            elif first_up > 0:
                direction[i] = LONG
            elif first_down > 0:
                direction[i] = SHORT
            else:
                direction[i] = SKIP

            if i % 50000 == 0:
                print(f"    Direction: {i}/{N} bars...")

        labels["direction"] = direction

        # Print distribution
        counts = pd.Series(direction).value_counts().sort_index()
        label_map = {0: "LONG", 1: "SHORT", 2: "BOTH", 3: "SKIP"}
        print(f"  Direction distribution:")
        for k, v in counts.items():
            print(f"    {label_map.get(k, k)}: {v} ({v/N*100:.1f}%)")

    # --- MFE ---
    if label_cfg.get("mfe", {}).get("enabled", True):
        horizons = label_cfg["mfe"]["horizons"]
        print(f"  Computing MFE labels (horizons: {horizons})...")

        for H in horizons:
            mfe_up = np.zeros(N, dtype=np.float32)
            mfe_down = np.zeros(N, dtype=np.float32)

            for i in range(N - H):
                entry = open_arr[i + 1]
                if entry <= 0:
                    continue
                window_high = high[i + 1:i + H + 1]
                window_low = low[i + 1:i + H + 1]
                up = (np.max(window_high) - entry) / entry * 10000
                down = (entry - np.min(window_low)) / entry * 10000
                mfe_up[i] = max(up, 0.0)
                mfe_down[i] = max(down, 0.0)

            labels[f"mfe_up_{H}"] = mfe_up
            labels[f"mfe_down_{H}"] = mfe_down

        print(f"  MFE columns: {len(horizons) * 2}")

    # --- Bounce/Break (4-class S/R outcome) ---
    bb_cfg = label_cfg.get("bounce_break", {})
    if bb_cfg.get("enabled", False) and features is not None:
        dist_thresh = bb_cfg["distance_threshold"]
        move_thresh = bb_cfg["move_threshold"]
        horizon = bb_cfg["horizon"]
        print(f"  Computing bounce/break labels (dist<={dist_thresh}bps, move>={move_thresh}bps, H{horizon})...")

        bb_label = np.full(N, BB_SKIP, dtype=np.int8)

        # Get S/R distances from features (vectorized)
        dist_sup = np.full(N, 999.0)
        dist_res = np.full(N, 999.0)

        if "distance_to_support" in features.columns:
            aligned = features["distance_to_support"].reindex(df.index)
            valid = aligned.notna()
            dist_sup[valid] = aligned[valid].values

        if "distance_to_resistance" in features.columns:
            aligned = features["distance_to_resistance"].reindex(df.index)
            valid = aligned.notna()
            dist_res[valid] = aligned[valid].values

        counts = {
            "bounce_support": 0,
            "break_support": 0,
            "bounce_resistance": 0,
            "break_resistance": 0,
        }
        at_support = 0
        at_resistance = 0

        for i in range(N - horizon):
            entry = open_arr[i + 1]
            thresh_up = entry * (1 + move_thresh / 10000)
            thresh_down = entry * (1 - move_thresh / 10000)

            is_at_support = 0 <= dist_sup[i] <= dist_thresh
            is_at_resistance = 0 <= dist_res[i] <= dist_thresh

            if not is_at_support and not is_at_resistance:
                continue

            # Find which threshold hits first
            first_up = 0
            first_down = 0
            for j in range(1, horizon + 1):
                if i + j >= N:
                    break
                if first_up == 0 and high[i + j] >= thresh_up:
                    first_up = j
                if first_down == 0 and low[i + j] <= thresh_down:
                    first_down = j

            # Determine direction of move
            if first_up > 0 and first_down > 0 and first_up == first_down:
                continue  # both hit same bar, skip
            went_up = first_up > 0 and (first_down == 0 or first_up < first_down)
            went_down = first_down > 0 and (first_up == 0 or first_down < first_up)

            if not went_up and not went_down:
                continue  # neither hit threshold

            if is_at_support:
                at_support += 1
                if went_up:
                    bb_label[i] = BOUNCE_SUPPORT
                    counts["bounce_support"] += 1
                else:
                    bb_label[i] = BREAK_SUPPORT
                    counts["break_support"] += 1
            elif is_at_resistance:
                at_resistance += 1
                if went_down:
                    bb_label[i] = BOUNCE_RESISTANCE
                    counts["bounce_resistance"] += 1
                else:
                    bb_label[i] = BREAK_RESISTANCE
                    counts["break_resistance"] += 1

        labels["bounce_break"] = bb_label

        skip_count = (bb_label == BB_SKIP).sum()
        labeled = sum(counts.values())
        print(f"  At support: {at_support}, At resistance: {at_resistance}")
        print(f"  BOUNCE_SUPPORT:     {counts['bounce_support']}")
        print(f"  BREAK_SUPPORT:      {counts['break_support']}")
        print(f"  BOUNCE_RESISTANCE:  {counts['bounce_resistance']}")
        print(f"  BREAK_RESISTANCE:   {counts['break_resistance']}")
        print(f"  SKIP: {skip_count}")
        print(f"  Total labeled: {labeled}")

    print(f"  Labels computed: {len(labels.columns)} columns, {len(labels)} bars")

    return labels
