"""Stage 9: Redesigned S/R Dataset Builder

Builds the new Stage 9 dataset with:
  - 5 static features (price_position, bounce_ratio, touch_count,
    strong_bounce_share, strong_break_share)
  - 11 dynamic features × 25 bars (unchanged from existing pipeline)
  - Binary labels only (BOUNCE=1, BREAK=0); CHOP samples dropped
  - Strict touch definition (price_position ≤ 0.05 or ≥ 0.95)
  - Last N=10 strict-touch history per zone (separate from existing Bayesian memory)

Reuses from existing pipeline:
  - load_ohlcv (brain.ingestion)
  - find_sr_zones, count_bar_touches, compute_recovery_speed (brain.processing)
  - ZoneRegistry (brain.zone_registry) — used only for matching/drift/role-flip

Stage 9-specific tracking lives in this script (zone_history dict), NOT in Zone class.

Usage:
  PYTHONPATH=src python experiments/brain/SR/build_dataset_stage9.py \
      --config experiments/brain/SR/config.yaml
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from brain.config import load_config
from brain.ingestion import load_ohlcv
from brain.processing import find_sr_zones, count_bar_touches, compute_recovery_speed
from brain.zone_registry import ZoneRegistry


# Stage 9 constants (locked decisions from PLAN.md Stage 9)
STRICT_SUPPORT_THRESHOLD = 0.10    # price_position ≤ this = support touch
STRICT_RESISTANCE_THRESHOLD = 0.90  # price_position ≥ this = resistance touch
HISTORY_N = 10                      # last N strict touches per zone
STRONG_FRACTION = 0.5               # MFE > this × zone_width = "strong"
TOUCH_COUNT_NORMALIZER = 10.0       # touch_count / this → ~[0,1]

STAGE9_STATIC_FEATURES = [
    "price_position",
    "bounce_ratio",
    "touch_count_norm",
    "strong_bounce_share",
    "strong_break_share",
]

STAGE9_DYNAMIC_FEATURES = [
    "dist_to_zone_pct", "support_width_pct", "res_width_pct",
    "support_retest", "resistance_retest", "zone_width",
    "recovery_up_pct", "recovery_down_pct",
    "speed_short", "speed_mid", "speed_long",
]


def load_sr_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_price_position(close: float, sh: float, rl: float) -> float:
    """price_position ∈ [0, 1]: 0=at support, 1=at resistance."""
    gap = rl - sh
    if gap <= 0:
        return 0.5  # degenerate; should not happen if sl<sh<rl<rh enforced
    pp = (close - sh) / gap
    return float(np.clip(pp, 0.0, 1.0))


def compute_stage9_static(price_position: float, history: deque) -> list:
    """Compute the 5 Stage 9 static features for a touch.

    history: deque of dicts {outcome: 'bounce'|'break', strong: bool}
             (CHOP samples are NOT added to history)
    """
    n = len(history)
    if n == 0:
        return [price_position, 0.5, 0.0, 0.0, 0.0]

    bounces = [h for h in history if h["outcome"] == "bounce"]
    breaks = [h for h in history if h["outcome"] == "break"]

    bounce_ratio = len(bounces) / n
    touch_count_norm = min(n / TOUCH_COUNT_NORMALIZER, 1.0)
    strong_bnc_share = (sum(1 for h in bounces if h["strong"]) / len(bounces)) if bounces else 0.0
    strong_brk_share = (sum(1 for h in breaks if h["strong"]) / len(breaks)) if breaks else 0.0

    return [price_position, bounce_ratio, touch_count_norm, strong_bnc_share, strong_brk_share]


def build_dataset(config_path: str):
    sr_cfg = load_sr_config(config_path)
    base_cfg = load_config("configs/base.yaml")

    print("=" * 70)
    print("STAGE 9: Redesigned S/R Dataset Builder")
    print("=" * 70)

    # --- Load OHLCV ---
    df, _ = load_ohlcv(base_cfg)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    open_arr = df["open"].values
    N = len(df)
    dates = df.index

    # --- KDE per bar (REUSED from sr_dataset.py) ---
    lookback = sr_cfg["sr_detection"]["lookback"]
    min_touches = sr_cfg["sr_detection"].get("min_touches", 2)
    sr_method = sr_cfg["sr_detection"].get("method", "hybrid_kde")
    sr_bandwidth = sr_cfg["sr_detection"].get("bandwidth", 0.03)
    sr_sup_weight = sr_cfg["sr_detection"].get("support_weight", "reaction")
    sr_res_weight = sr_cfg["sr_detection"].get("resistance_weight", "recency")
    sr_peak_thresh = sr_cfg["sr_detection"].get("peak_threshold", 0.2)

    print(f"\nComputing S/R zones ({lookback}-bar lookback, {sr_method})...")
    sr_data = [None] * N

    for t in range(lookback, N):
        w_high = high[t - lookback + 1:t + 1]
        w_low = low[t - lookback + 1:t + 1]

        sl, sh, rl, rh = find_sr_zones(
            w_high, w_low, min_touches,
            method=sr_method, bandwidth=sr_bandwidth,
            support_weight=sr_sup_weight,
            resistance_weight=sr_res_weight,
            peak_threshold=sr_peak_thresh,
        )

        if sl is not None and rl is not None and rl > sh:
            zone_width_bps = (rl - sh) / sh * 10000
            zone_width_price = rl - sh
            dist_sup = (close[t] - sh) / sh * 10000
            dist_res = (rl - close[t]) / rl * 10000
            sup_retest = count_bar_touches(w_high, w_low, sl, sh)
            res_retest = count_bar_touches(w_high, w_low, rl, rh)
            rec_up = compute_recovery_speed(w_high[-min(lookback, len(w_high)):],
                                            w_low[-min(lookback, len(w_low)):],
                                            close[t - lookback + 1:t + 1], sl, sh, "up")
            rec_down = compute_recovery_speed(w_high[-min(lookback, len(w_high)):],
                                              w_low[-min(lookback, len(w_low)):],
                                              close[t - lookback + 1:t + 1], rl, rh, "down")

            sr_data[t] = {
                "sl": sl, "sh": sh, "rl": rl, "rh": rh,
                "zone_width_bps": zone_width_bps,
                "zone_width_price": zone_width_price,
                "dist_sup": dist_sup, "dist_res": dist_res,
                "sup_retest": sup_retest, "res_retest": res_retest,
                "rec_up": rec_up, "rec_down": rec_down,
                "sup_width": sh - sl, "res_width": rh - rl,
                "price_position": compute_price_position(close[t], sh, rl),
            }

        if t % 50000 == 0:
            print(f"  S/R: {t}/{N} bars...")

    # --- Pre-compute 11 dynamic features per bar (REUSED LOGIC) ---
    print("\nComputing dynamic features per bar...")
    bar_features = np.zeros((N, len(STAGE9_DYNAMIC_FEATURES)), dtype=np.float32)
    clip_range = sr_cfg["features"]["clip_range"]

    for t in range(N):
        sr = sr_data[t]
        if sr is None:
            continue
        zw_bps = sr["zone_width_bps"]
        zw_price = sr["zone_width_price"]
        safe_zw_bps = max(zw_bps, 1.0)
        safe_zw_price = max(zw_price, 1.0)

        if sr["dist_sup"] >= 0 and sr["dist_res"] >= 0:
            dist_to_zone = min(sr["dist_sup"], sr["dist_res"])
        elif sr["dist_sup"] >= 0:
            dist_to_zone = sr["dist_sup"]
        elif sr["dist_res"] >= 0:
            dist_to_zone = sr["dist_res"]
        else:
            dist_to_zone = min(abs(sr["dist_sup"]), abs(sr["dist_res"]))

        bar_features[t, 0] = dist_to_zone / safe_zw_bps
        bar_features[t, 1] = sr["sup_width"] / safe_zw_price
        bar_features[t, 2] = sr["res_width"] / safe_zw_price
        bar_features[t, 3] = sr["sup_retest"]
        bar_features[t, 4] = sr["res_retest"]
        bar_features[t, 5] = np.log1p(zw_bps)
        bar_features[t, 6] = sr["rec_up"] / safe_zw_bps
        bar_features[t, 7] = sr["rec_down"] / safe_zw_bps

    print("Computing speed features...")
    for t in range(3, N):
        bar_features[t, 8] = bar_features[t - 3, 0] - bar_features[t, 0]
    for t in range(10, N):
        bar_features[t, 9] = bar_features[t - 10, 0] - bar_features[t, 0]
    for t in range(lookback, N):
        bar_features[t, 10] = bar_features[t - lookback, 0] - bar_features[t, 0]

    clip_idx = [0, 1, 2, 6, 7, 8, 9, 10]
    bar_features[:, clip_idx] = np.clip(bar_features[:, clip_idx], -clip_range, clip_range)

    # --- Sequential touch loop ---
    print("\nProcessing strict touches...")
    registry = ZoneRegistry(sr_cfg)
    horizon = sr_cfg["label"]["horizon"]
    bounce_thresh = sr_cfg["label"]["bounce_threshold"]
    min_move = sr_cfg["label"].get("min_move_bps", 15)
    snapshot_count = sr_cfg["features"]["snapshot_count"]

    # Stage 9: per-zone strict-touch history (separate from Zone.history)
    # zone_id -> deque of {outcome, strong}
    zone_history = {}

    pending = []        # events awaiting label resolution
    resolved_events = []

    for t in range(lookback, N):
        sr = sr_data[t]
        if sr is None:
            continue

        price = close[t]
        zw_price = sr["zone_width_price"]
        price_position = sr["price_position"]

        # Process zones via registry (matching, drift, role flip)
        sup_zone = registry.process_zone(sr["sl"], sr["sh"], "support", t, price)
        res_zone = registry.process_zone(sr["rl"], sr["rh"], "resistance", t, price)
        registry.check_role_flip(sup_zone, price, zw_price, price)
        registry.check_role_flip(res_zone, price, zw_price, price)

        # --- Resolve pending events whose horizon has passed ---
        new_pending = []
        for event in pending:
            if t < event["entry_bar"] + horizon:
                new_pending.append(event)
                continue

            eb = event["entry_bar"]
            if eb + 1 >= N or eb + horizon + 1 > N:
                continue
            entry = open_arr[eb + 1]
            if entry <= 0:
                continue

            window_high = high[eb + 1:eb + horizon + 1]
            window_low = low[eb + 1:eb + horizon + 1]
            mfe_up = max((np.max(window_high) - entry) / entry * 10000, 0)
            mae_down = max((entry - np.min(window_low)) / entry * 10000, 0)

            if event["role"] == "support":
                favorable, adverse = mfe_up, mae_down
            else:
                favorable, adverse = mae_down, mfe_up

            total = favorable + adverse
            if total == 0:
                outcome = "chop"
            else:
                fav_pct = favorable / total
                adv_pct = adverse / total
                if fav_pct > bounce_thresh and favorable >= min_move:
                    outcome = "bounce"
                elif adv_pct > bounce_thresh and adverse >= min_move:
                    outcome = "break"
                else:
                    outcome = "chop"

            # Update Stage 9 zone history (CHOP samples skipped from history)
            if outcome != "chop":
                mfe_used = favorable if outcome == "bounce" else adverse
                strong = mfe_used > (STRONG_FRACTION * event["zone_width_bps"])
                zid = event["zone_id"]
                if zid not in zone_history:
                    zone_history[zid] = deque(maxlen=HISTORY_N)
                zone_history[zid].append({"outcome": outcome, "strong": strong})

                # Keep this resolved event for the dataset (binary only)
                event["label"] = 1 if outcome == "bounce" else 0
                event["outcome"] = outcome
                resolved_events.append(event)
            # CHOP: dropped entirely (no label, not in dataset, not in history)

        pending = new_pending

        # --- Detect strict touches (Stage 9 definition) ---
        for zone, role in [(sup_zone, "support"), (res_zone, "resistance")]:
            if not zone.active:
                continue

            if role == "support":
                is_strict = price_position <= STRICT_SUPPORT_THRESHOLD
            else:
                is_strict = price_position >= STRICT_RESISTANCE_THRESHOLD

            if not is_strict:
                continue
            if t < snapshot_count:
                continue

            # Snapshot static features (using history BEFORE this touch)
            history = zone_history.get(zone.id, deque(maxlen=HISTORY_N))
            static_vec = compute_stage9_static(price_position, history)
            dynamic_snapshot = bar_features[t - snapshot_count + 1:t + 1, :].copy()

            pending.append({
                "entry_bar": t,
                "zone_id": zone.id,
                "role": role,
                "static": static_vec,
                "dynamic": dynamic_snapshot,
                "zone_width_bps": sr["zone_width_bps"],
                "date": dates[t],
            })
            zone.last_touch_bar = t

        if t % 50000 == 0:
            print(f"  Processing: {t}/{N}, zones={len(registry.zones)}, "
                  f"pending={len(pending)}, resolved={len(resolved_events)}")

    print(f"\nResolved bounce/break events: {len(resolved_events)}")

    # --- Build arrays ---
    X_static = np.array([e["static"] for e in resolved_events], dtype=np.float32)
    X_dynamic = np.array([e["dynamic"] for e in resolved_events], dtype=np.float32)
    Y = np.array([e["label"] for e in resolved_events], dtype=np.int64)
    event_dates = np.array([e["date"] for e in resolved_events])
    event_bars = np.array([e["entry_bar"] for e in resolved_events], dtype=np.int64)

    print(f"  X_static: {X_static.shape}")
    print(f"  X_dynamic: {X_dynamic.shape}")
    print(f"  Y: {Y.shape}")
    print(f"  Bounce: {(Y == 1).sum()} ({(Y == 1).mean() * 100:.1f}%)")
    print(f"  Break:  {(Y == 0).sum()} ({(Y == 0).mean() * 100:.1f}%)")

    # --- Date split (REUSED) ---
    data_cfg = sr_cfg["data"]
    train_end = pd.Timestamp(data_cfg["train"][1])
    val_start = pd.Timestamp(data_cfg["val"][0])
    val_end = pd.Timestamp(data_cfg["val"][1])
    test_start = pd.Timestamp(data_cfg["test"][0])

    train_mask = event_dates <= train_end
    val_mask = (event_dates >= val_start) & (event_dates <= val_end)
    test_mask = event_dates >= test_start

    print(f"\n  Train: {train_mask.sum()}")
    print(f"  Val:   {val_mask.sum()}")
    print(f"  Test:  {test_mask.sum()}")

    # --- Save ---
    out_dir = Path(config_path).parent / "datasets_stage9"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        np.savez_compressed(
            out_dir / f"{name}.npz",
            X_static=X_static[mask],
            X_dynamic=X_dynamic[mask],
            Y=Y[mask],
            bars=event_bars[mask],
        )

    meta = {
        "stage": 9,
        "static_features": STAGE9_STATIC_FEATURES,
        "dynamic_features": STAGE9_DYNAMIC_FEATURES,
        "snapshot_count": snapshot_count,
        "history_n": HISTORY_N,
        "strict_support_threshold": STRICT_SUPPORT_THRESHOLD,
        "strict_resistance_threshold": STRICT_RESISTANCE_THRESHOLD,
        "strong_fraction": STRONG_FRACTION,
        "touch_count_normalizer": TOUCH_COUNT_NORMALIZER,
        "label_map": {"0": "BREAK", "1": "BOUNCE"},
        "total_samples": len(Y),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {out_dir}/")

    # --- Validation ---
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    print("\nLabel distribution per split:")
    for split, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        if mask.sum() == 0:
            continue
        bnc = (Y[mask] == 1).sum()
        brk = (Y[mask] == 0).sum()
        print(f"  {split}: bounce={bnc} ({bnc / mask.sum() * 100:.1f}%) "
              f"break={brk} ({brk / mask.sum() * 100:.1f}%)")

    print("\nStatic feature stats (train):")
    for i, name in enumerate(STAGE9_STATIC_FEATURES):
        v = X_static[train_mask, i]
        print(f"  {name:25s} mean={v.mean():.3f} std={v.std():.3f} "
              f"min={v.min():.3f} max={v.max():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/brain/SR/config.yaml")
    args = parser.parse_args()
    build_dataset(args.config)
