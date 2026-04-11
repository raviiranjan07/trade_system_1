"""Fast enrichment: add X_speed to datasets_stage9a_static from datasets_stage9.

This avoids rerunning KDE + registry + label resolution just to add speed features.

Requirements:
- Stage9 and Stage9A must use the same event rows per split (bars + Y).

Run:
  PYTHONPATH=src python experiments/brain/SR/enrich_stage9a_speed_from_stage9.py
"""

import json
from pathlib import Path

import numpy as np


STAGE9_DIR = Path("experiments/brain/SR/datasets_stage9")
STAGE9A_DIR = Path("experiments/brain/SR/datasets_stage9a_static")
SPEED_FEATURES = ["speed_short", "speed_mid", "speed_long"]


def process_split(split: str):
    d9 = np.load(STAGE9_DIR / f"{split}.npz")
    d9a = np.load(STAGE9A_DIR / f"{split}.npz")

    bars_9 = d9["bars"]
    bars_9a = d9a["bars"]
    y_9 = d9["Y"]
    y_9a = d9a["Y"]

    if len(bars_9) != len(bars_9a):
        raise ValueError(f"{split}: row count mismatch ({len(bars_9)} vs {len(bars_9a)})")
    if not np.array_equal(bars_9, bars_9a):
        raise ValueError(f"{split}: bars mismatch; cannot safely transfer speed features.")
    if not np.array_equal(y_9, y_9a):
        raise ValueError(f"{split}: labels mismatch; cannot safely transfer speed features.")

    # Stage9 dynamic order:
    # [dist_to_zone_pct, support_width_pct, res_width_pct, support_retest, resistance_retest,
    #  zone_width, recovery_up_pct, recovery_down_pct, speed_short, speed_mid, speed_long]
    x_speed = d9["X_dynamic"][:, -1, 8:11].astype(np.float32)

    np.savez_compressed(
        STAGE9A_DIR / f"{split}.npz",
        X_static=d9a["X_static"],
        X_speed=x_speed,
        Y=y_9a,
        bars=bars_9a,
    )
    return len(x_speed)


def main():
    print("=" * 70)
    print("ENRICH STAGE9A WITH SPEED FROM STAGE9")
    print("=" * 70)

    for required in [STAGE9_DIR, STAGE9A_DIR]:
        if not required.exists():
            raise FileNotFoundError(f"Missing directory: {required}")

    counts = {}
    for split in ["train", "val", "test"]:
        counts[split] = process_split(split)
        print(f"{split}: wrote X_speed for {counts[split]} rows")

    meta_path = STAGE9A_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    meta["speed_features"] = SPEED_FEATURES
    meta["speed_source"] = "datasets_stage9 X_dynamic last frame indices 8:11"

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nUpdated metadata with speed_features.")
    print("Done.")


if __name__ == "__main__":
    main()

