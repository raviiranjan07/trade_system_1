"""XGBoost training for S/R bounce/break prediction.

Trains a gradient boosted tree model to predict 4-class S/R outcomes:
  0: BOUNCE_SUPPORT     (at support, price goes UP)
  1: BREAK_SUPPORT      (at support, price goes DOWN)
  2: BOUNCE_RESISTANCE  (at resistance, price goes DOWN)
  3: BREAK_RESISTANCE   (at resistance, price goes UP)

Usage:
  PYTHONPATH=src python -m brain.train_xgboost --data data/archive/brain_datasets/sr_bounce_break
"""

import argparse
import json
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


LABEL_NAMES = ["BOUNCE_SUP", "BREAK_SUP", "BOUNCE_RES", "BREAK_RES"]


def load_data(data_dir: str):
    """Load train/val/test from .npz files."""
    data_dir = Path(data_dir)

    train = np.load(data_dir / "train.npz")
    val = np.load(data_dir / "val.npz")
    test = np.load(data_dir / "test.npz")

    meta = json.load(open(data_dir / "metadata.json"))
    feature_names = meta["feature_names"]

    return {
        "train_X": train["X"], "train_Y": train["Y"].ravel().astype(int),
        "val_X": val["X"], "val_Y": val["Y"].ravel().astype(int),
        "test_X": test["X"], "test_Y": test["Y"].ravel().astype(int),
        "feature_names": feature_names,
    }


def train_xgboost(data_dir: str, output_dir: str, sr_features_only: bool = True):
    """Train XGBoost on bounce/break labels."""
    print("=" * 70)
    print("XGBOOST TRAINING: S/R Bounce/Break")
    print("=" * 70)

    # --- Load data ---
    data = load_data(data_dir)
    feature_names = data["feature_names"]

    train_X = data["train_X"]
    train_Y = data["train_Y"]
    val_X = data["val_X"]
    val_Y = data["val_Y"]
    test_X = data["test_X"]
    test_Y = data["test_Y"]

    # --- Select S/R features only ---
    sr_feature_names = [
        "support_range_low", "support_range_high",
        "resistance_range_low", "resistance_range_high",
        "zone_width", "support_retest", "resistance_retest",
        "distance_to_support", "distance_to_resistance",
        "recovery_up", "recovery_down",
        "window_low", "window_high",
    ]

    if sr_features_only:
        sr_indices = [i for i, f in enumerate(feature_names) if f in sr_feature_names]
        train_X = train_X[:, sr_indices]
        val_X = val_X[:, sr_indices]
        test_X = test_X[:, sr_indices]
        used_features = [feature_names[i] for i in sr_indices]
        print(f"\nUsing S/R features only: {len(used_features)}")
    else:
        used_features = feature_names
        print(f"\nUsing all features: {len(used_features)}")

    for f in used_features:
        print(f"  - {f}")

    print(f"\nTrain: {len(train_X)}, Val: {len(val_X)}, Test: {len(test_X)}")
    print(f"\nLabel distribution:")
    for split_name, y in [("Train", train_Y), ("Val", val_Y), ("Test", test_Y)]:
        counts = np.bincount(y, minlength=4)
        print(f"  {split_name}: {dict(zip(LABEL_NAMES, counts))}")

    # --- Train ---
    print(f"\nTraining XGBoost...")
    dtrain = xgb.DMatrix(train_X, label=train_Y, feature_names=used_features)
    dval = xgb.DMatrix(val_X, label=val_Y, feature_names=used_features)
    dtest = xgb.DMatrix(test_X, label=test_Y, feature_names=used_features)

    params = {
        "objective": "multi:softprob",
        "num_class": 4,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "eval_metric": "mlogloss",
        "seed": 42,
        "verbosity": 1,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    # --- Evaluate ---
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    for split_name, dset, y_true in [("Train", dtrain, train_Y),
                                      ("Val", dval, val_Y),
                                      ("Test", dtest, test_Y)]:
        y_prob = model.predict(dset)
        y_pred = y_prob.argmax(axis=1)
        acc = accuracy_score(y_true, y_pred)
        print(f"\n{split_name} accuracy: {acc*100:.1f}%")
        print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=3))

    # --- Feature importance ---
    print(f"\nFeature Importance (gain):")
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for fname, score in sorted_imp:
        print(f"  {fname}: {score:.1f}")

    # --- Confusion matrix (test) ---
    y_prob_test = model.predict(dtest)
    y_pred_test = y_prob_test.argmax(axis=1)
    cm = confusion_matrix(test_Y, y_pred_test)
    print(f"\nConfusion Matrix (test):")
    print(f"{'':>15} {'BOUNCE_SUP':>12} {'BREAK_SUP':>12} {'BOUNCE_RES':>12} {'BREAK_RES':>12}")
    for i, label in enumerate(LABEL_NAMES):
        row = "  ".join(f"{cm[i][j]:>10}" for j in range(4))
        print(f"  {label:>13} {row}")

    # --- Save ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(output_dir / "model.json"))

    metrics = {
        "train_acc": float(accuracy_score(train_Y, model.predict(dtrain).argmax(axis=1))),
        "val_acc": float(accuracy_score(val_Y, model.predict(dval).argmax(axis=1))),
        "test_acc": float(accuracy_score(test_Y, y_pred_test)),
        "features_used": used_features,
        "num_features": len(used_features),
        "train_samples": len(train_X),
        "val_samples": len(val_X),
        "test_samples": len(test_X),
        "best_iteration": model.best_iteration,
        "params": params,
        "feature_importance": dict(sorted_imp),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print(f"  model.json, metrics.json")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="experiments/brain/EXP-B002")
    parser.add_argument("--all-features", action="store_true")
    args = parser.parse_args()

    train_xgboost(args.data, args.output, sr_features_only=not args.all_features)
