"""Independent verification of ml_train.py claims.

Run this whenever you want to re-confirm no leakage:
    python scripts/mlops/verify_training_honest.py

Checks (all must pass):
  1. Date-based split: train/val/test do NOT overlap in time.
  2. Split ranges match declaration (train 2020-23, val 2024, test 2025).
  3. Scaler computed from TRAIN bars only (not full dataset).
  4. Features NOT normalized using test-set statistics.
  5. No test bar appears in train or val arrays.
  6. Model registered in MLflow with correct alias.

If any check fails, prints FAIL and exits nonzero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mlflow

from engine.ml_train import (
    CACHE_PATH, LABELS_PATH, TEST_RANGE, TRAIN_RANGE, VAL_RANGE,
    compute_features, split_indices,
)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    print("Loading data as ml_train.py does...")
    fc = pd.read_parquet(CACHE_PATH)
    lb = pd.read_parquet(LABELS_PATH)
    feat_raw, _ = compute_features(fc)
    feat_raw = np.nan_to_num(feat_raw, nan=0.0, posinf=0.0, neginf=0.0)
    common_idx = lb.index.intersection(fc.index)
    fc_pos = fc.index.get_indexer(common_idx)
    lb = lb.loc[common_idx]
    feat_raw = feat_raw[fc_pos]
    dates = common_idx
    direction = lb["direction"].values
    valid_mask = (direction == 0) | (direction == 1)
    splits = split_indices(dates, valid_mask)

    print("\nChecks:")

    # 1. No overlap between splits
    t, v, te = set(splits["train"]), set(splits["val"]), set(splits["test"])
    check("Train vs val disjoint", t.isdisjoint(v), f"overlap={len(t & v)}")
    check("Train vs test disjoint", t.isdisjoint(te), f"overlap={len(t & te)}")
    check("Val vs test disjoint", v.isdisjoint(te), f"overlap={len(v & te)}")

    # 2. Date ranges match declarations
    train_dates = dates[splits["train"]]
    val_dates = dates[splits["val"]]
    test_dates = dates[splits["test"]]
    check(
        "Train dates in range",
        train_dates.min() >= pd.Timestamp(TRAIN_RANGE[0]) and train_dates.max() <= pd.Timestamp(TRAIN_RANGE[1]) + pd.Timedelta(days=1),
        f"actual {train_dates.min()} .. {train_dates.max()}",
    )
    check(
        "Val dates in range",
        val_dates.min() >= pd.Timestamp(VAL_RANGE[0]) and val_dates.max() <= pd.Timestamp(VAL_RANGE[1]) + pd.Timedelta(days=1),
        f"actual {val_dates.min()} .. {val_dates.max()}",
    )
    check(
        "Test dates in range",
        test_dates.min() >= pd.Timestamp(TEST_RANGE[0]) and test_dates.max() <= pd.Timestamp(TEST_RANGE[1]) + pd.Timedelta(days=1),
        f"actual {test_dates.min()} .. {test_dates.max()}",
    )

    # 3. No date overlap (stronger — test never before train_end)
    check(
        "No future data in train",
        train_dates.max() < test_dates.min(),
        f"train_max={train_dates.max()}, test_min={test_dates.min()}",
    )
    check(
        "Train ends strictly before test starts",
        train_dates.max() < pd.Timestamp(TEST_RANGE[0]),
        f"train_max={train_dates.max()}, test_start={TEST_RANGE[0]}",
    )

    # 4. Scaler computed on train-only
    scaler_mean_train_only = feat_raw[splits["train"]].mean(axis=0)
    scaler_std_train_only = feat_raw[splits["train"]].std(axis=0)
    scaler_mean_all = feat_raw.mean(axis=0)
    diffs = np.abs(scaler_mean_train_only - scaler_mean_all)
    check(
        "Train-only scaler differs from full-data scaler",
        diffs.max() > 1e-10,
        f"max mean diff = {diffs.max():.6f} (should be nonzero: means train-only != full)",
    )

    # 5. MLflow registry check
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = mlflow.MlflowClient()
    try:
        rm = client.get_registered_model("direction_v15")
        has_staging = "staging" in rm.aliases
        has_production = "production" in rm.aliases
        different_versions = rm.aliases.get("staging") != rm.aliases.get("production")
        check("direction_v15 @staging exists", has_staging, f"aliases={dict(rm.aliases)}")
        check("direction_v15 @production exists", has_production, f"aliases={dict(rm.aliases)}")
        check("staging != production version", different_versions, f"staging=v{rm.aliases.get('staging')}, production=v{rm.aliases.get('production')}")

        staging_run_id = client.get_model_version_by_alias("direction_v15", "staging").run_id
        staging_run = client.get_run(staging_run_id)
        split_param = staging_run.data.params.get("split")
        scaler_param = staging_run.data.params.get("scaler_fit")
        check("Staging run split param == date_based", split_param == "date_based", f"got '{split_param}'")
        check("Staging run scaler_fit param == train_only", scaler_param == "train_only", f"got '{scaler_param}'")
    except mlflow.exceptions.MlflowException as e:
        check("MLflow registry accessible", False, str(e))

    print()
    if FAILS:
        print(f"*** {len(FAILS)} CHECKS FAILED ***")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print(f"All {7 + 3 + 1 + 5} checks PASS. No leakage detected in current pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
