"""Stage 9A: Static-memory dataset builder.

Builds a binary bounce/break dataset for static-memory feature testing:
  - price_position
  - bounce_ratio
  - touch_count_scaled
  - recent_bounce_ratio
  - pressure
  - bars_since_touch
  - last_outcome
  - bounce_streak
  - chop_ratio
  - speed_short
  - speed_mid
  - speed_long

Usage:
  PYTHONPATH=src python experiments/brain/SR/build_dataset_stage9a_static.py \
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
from brain.processing import find_sr_zones
from brain.zone_registry import ZoneRegistry


STRICT_SUPPORT_THRESHOLD = 0.10
STRICT_RESISTANCE_THRESHOLD = 0.90
HISTORY_N = 10
RECENT_HISTORY_N = 5
PRESSURE_MIN_TOUCHES = 3

STATIC_FEATURES = [
    "price_position",
    "bounce_ratio",
    "touch_count_scaled",
    "recent_bounce_ratio",
    "pressure",
    "bars_since_touch",
    "last_outcome",
    "bounce_streak",
    "chop_ratio",
]
SPEED_FEATURES = ["speed_short", "speed_mid", "speed_long"]


def load_sr_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_price_position(close: float, sh: float, rl: float) -> float:
    gap = rl - sh
    if gap <= 0:
        return 0.5
    return float(np.clip((close - sh) / gap, 0.0, 1.0))


def compute_static_features(
    price_position: float, history: deque, bars_since_touch: int, stats: dict
) -> list[float]:
    n = len(history)
    if n == 0:
        chop_ratio = (stats["chop"] / stats["total"]) if stats["total"] > 0 else 0.0
        return [price_position, 0.5, 0.0, 0.5, 0.0, np.log1p(max(bars_since_touch, 0)), 0.5, 0.0, chop_ratio]

    outcomes = [h["outcome"] for h in history]
    bounces = sum(1 for o in outcomes if o == "bounce")
    recent = outcomes[-RECENT_HISTORY_N:]

    bounce_ratio = bounces / n
    touch_count_scaled = float(np.log1p(stats["total"]))
    recent_bounce_ratio = sum(1 for o in recent if o == "bounce") / len(recent)
    bars_since_touch_log = float(np.log1p(max(bars_since_touch, 0)))
    last_outcome = 1.0 if outcomes[-1] == "bounce" else 0.0

    bounce_streak = 0
    for outcome in reversed(outcomes):
        if outcome == "bounce":
            bounce_streak += 1
        else:
            break
    chop_ratio = (stats["chop"] / stats["total"]) if stats["total"] > 0 else 0.0

    if n < PRESSURE_MIN_TOUCHES:
        pressure = 0.0
    else:
        pressure_count = 0
        for outcome in reversed(outcomes):
            if outcome == "break":
                pressure_count += 1
            else:
                break
        pressure = float(pressure_count)

    return [
        price_position,
        bounce_ratio,
        touch_count_scaled,
        recent_bounce_ratio,
        pressure,
        bars_since_touch_log,
        last_outcome,
        float(bounce_streak),
        float(chop_ratio),
    ]


def build_dataset(config_path: str):
    sr_cfg = load_sr_config(config_path)
    base_cfg = load_config("configs/base.yaml")

    print("=" * 70)
    print("STAGE 9A: Minimal Static-Memory Dataset Builder")
    print("=" * 70)

    df, _ = load_ohlcv(base_cfg)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    open_arr = df["open"].values
    dates = df.index
    n_bars = len(df)

    lookback = sr_cfg["sr_detection"]["lookback"]
    min_touches = sr_cfg["sr_detection"].get("min_touches", 2)
    sr_method = sr_cfg["sr_detection"].get("method", "hybrid_kde")
    sr_bandwidth = sr_cfg["sr_detection"].get("bandwidth", 0.03)
    sr_sup_weight = sr_cfg["sr_detection"].get("support_weight", "reaction")
    sr_res_weight = sr_cfg["sr_detection"].get("resistance_weight", "recency")
    sr_peak_thresh = sr_cfg["sr_detection"].get("peak_threshold", 0.2)

    print(f"\nComputing S/R zones ({lookback}-bar lookback, {sr_method})...")
    sr_data = [None] * n_bars
    dist_norm = np.zeros(n_bars, dtype=np.float32)
    for t in range(lookback, n_bars):
        w_high = high[t - lookback + 1:t + 1]
        w_low = low[t - lookback + 1:t + 1]

        sl, sh, rl, rh = find_sr_zones(
            w_high,
            w_low,
            min_touches,
            method=sr_method,
            bandwidth=sr_bandwidth,
            support_weight=sr_sup_weight,
            resistance_weight=sr_res_weight,
            peak_threshold=sr_peak_thresh,
        )

        if sl is not None and rl is not None and rl > sh:
            zone_width_bps = (rl - sh) / sh * 10000
            dist_sup = (close[t] - sh) / sh * 10000
            dist_res = (rl - close[t]) / rl * 10000
            safe_zw = max(zone_width_bps, 1.0)
            if dist_sup >= 0 and dist_res >= 0:
                dist_to_zone = min(dist_sup, dist_res)
            elif dist_sup >= 0:
                dist_to_zone = dist_sup
            elif dist_res >= 0:
                dist_to_zone = dist_res
            else:
                dist_to_zone = min(abs(dist_sup), abs(dist_res))
            dist_norm[t] = float(np.clip(dist_to_zone / safe_zw, -3.0, 3.0))
            sr_data[t] = {
                "sl": sl,
                "sh": sh,
                "rl": rl,
                "rh": rh,
                "zone_width_bps": zone_width_bps,
                "zone_width_price": rl - sh,
                "price_position": compute_price_position(close[t], sh, rl),
                "dist_sup": dist_sup,
                "dist_res": dist_res,
            }

        if t % 50000 == 0:
            print(f"  S/R: {t}/{n_bars} bars...")

    registry = ZoneRegistry(sr_cfg)
    horizon = sr_cfg["label"]["horizon"]
    bounce_thresh = sr_cfg["label"]["bounce_threshold"]
    min_move = sr_cfg["label"].get("min_move_bps", 15)

    zone_history: dict[int, deque] = {}
    zone_stats: dict[int, dict] = {}
    pending = []
    resolved = []

    print("\nProcessing strict touch events...")
    for t in range(lookback, n_bars):
        sr = sr_data[t]
        if sr is None:
            continue

        price = close[t]
        price_position = sr["price_position"]
        zw_price = sr["zone_width_price"]

        sup_zone = registry.process_zone(sr["sl"], sr["sh"], "support", t, price)
        res_zone = registry.process_zone(sr["rl"], sr["rh"], "resistance", t, price)
        registry.check_role_flip(sup_zone, price, zw_price, price)
        registry.check_role_flip(res_zone, price, zw_price, price)

        new_pending = []
        for event in pending:
            if t < event["entry_bar"] + horizon:
                new_pending.append(event)
                continue

            eb = event["entry_bar"]
            if eb + 1 >= n_bars or eb + horizon + 1 > n_bars:
                continue

            entry = open_arr[eb + 1]
            if entry <= 0:
                continue

            window_high = high[eb + 1:eb + horizon + 1]
            window_low = low[eb + 1:eb + horizon + 1]
            mfe_up = max((np.max(window_high) - entry) / entry * 10000, 0.0)
            mae_down = max((entry - np.min(window_low)) / entry * 10000, 0.0)

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

            zid = event["zone_id"]
            if zid not in zone_stats:
                zone_stats[zid] = {"total": 0, "bounce": 0, "break": 0, "chop": 0}

            zone_stats[zid]["total"] += 1
            zone_stats[zid][outcome] += 1

            if outcome != "chop":
                if zid not in zone_history:
                    zone_history[zid] = deque(maxlen=HISTORY_N)
                zone_history[zid].append({"outcome": outcome})

                event["label"] = 1 if outcome == "bounce" else 0
                resolved.append(event)

        pending = new_pending

        for zone, role in [(sup_zone, "support"), (res_zone, "resistance")]:
            if not zone.active:
                continue

            if role == "support":
                is_strict = price_position <= STRICT_SUPPORT_THRESHOLD
            else:
                is_strict = price_position >= STRICT_RESISTANCE_THRESHOLD

            if not is_strict:
                continue

            history = zone_history.get(zone.id, deque(maxlen=HISTORY_N))
            stats = zone_stats.get(zone.id, {"total": 0, "bounce": 0, "break": 0, "chop": 0})
            bars_since_touch = t - zone.last_touch_bar
            static_vec = compute_static_features(price_position, history, bars_since_touch, stats)
            speed_short = dist_norm[max(t - 3, 0)] - dist_norm[t] if t >= 3 else 0.0
            speed_mid = dist_norm[max(t - 10, 0)] - dist_norm[t] if t >= 10 else 0.0
            speed_long = dist_norm[max(t - lookback, 0)] - dist_norm[t] if t >= lookback else 0.0
            speed_vec = [
                float(np.clip(speed_short, -3.0, 3.0)),
                float(np.clip(speed_mid, -3.0, 3.0)),
                float(np.clip(speed_long, -3.0, 3.0)),
            ]
            pending.append(
                {
                    "entry_bar": t,
                    "zone_id": zone.id,
                    "role": role,
                    "static": static_vec,
                    "speed": speed_vec,
                    "date": dates[t],
                }
            )
            zone.last_touch_bar = t

        if t % 50000 == 0:
            print(f"  Processing: {t}/{n_bars}, zones={len(registry.zones)}, resolved={len(resolved)}")

    x_static = np.array([e["static"] for e in resolved], dtype=np.float32)
    x_speed = np.array([e["speed"] for e in resolved], dtype=np.float32)
    y = np.array([e["label"] for e in resolved], dtype=np.int64)
    event_dates = np.array([e["date"] for e in resolved])
    event_bars = np.array([e["entry_bar"] for e in resolved], dtype=np.int64)

    print(f"\nResolved samples: {len(y)}")
    print(f"Bounce: {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)")
    print(f"Break:  {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)")

    data_cfg = sr_cfg["data"]
    train_end = pd.Timestamp(data_cfg["train"][1])
    val_start = pd.Timestamp(data_cfg["val"][0])
    val_end = pd.Timestamp(data_cfg["val"][1])
    test_start = pd.Timestamp(data_cfg["test"][0])

    train_mask = event_dates <= train_end
    val_mask = (event_dates >= val_start) & (event_dates <= val_end)
    test_mask = event_dates >= test_start

    out_dir = Path(config_path).parent / "datasets_stage9a_static"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        np.savez_compressed(
            out_dir / f"{name}.npz",
            X_static=x_static[mask],
            X_speed=x_speed[mask],
            Y=y[mask],
            bars=event_bars[mask],
        )

    meta = {
        "stage": "9A",
        "static_features": STATIC_FEATURES,
        "speed_features": SPEED_FEATURES,
        "strict_support_threshold": STRICT_SUPPORT_THRESHOLD,
        "strict_resistance_threshold": STRICT_RESISTANCE_THRESHOLD,
        "history_n": HISTORY_N,
        "recent_history_n": RECENT_HISTORY_N,
        "pressure_min_touches": PRESSURE_MIN_TOUCHES,
        "label_map": {"0": "BREAK", "1": "BOUNCE"},
        "total_samples": int(len(y)),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/brain/SR/config.yaml")
    args = parser.parse_args()
    build_dataset(args.config)
