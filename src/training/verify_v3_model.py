"""Stage 4b: Verify V3 model meets minimum quality gates.

Checks that the trained model is not degenerate before running backtest.
Pipeline FAILS if any gate check fails.

Run: PYTHONPATH=src python -m training.verify_v3_model
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort

from engine.signals import feature_lib

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models/ML_V3_staging"
FEATURES_PATH = REPO_ROOT / "data/features/direction_prediction/feature_cache.parquet"
LABELS_PATH = REPO_ROOT / "data/features/direction_prediction/exit_aware_labels.parquet"
OUTPUT_PATH = REPO_ROOT / "data/reports/v3_model_verification.json"

LONG, SHORT, SKIP = 0, 1, 2
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    logger.info("Loading model from %s", MODEL_DIR)
    onnx_path = MODEL_DIR / "v3_model.onnx"
    scaler_path = MODEL_DIR / "scaler.npz"

    if not onnx_path.exists():
        logger.error("Model not found: %s", onnx_path)
        sys.exit(1)

    sess = ort.InferenceSession(str(onnx_path))
    scaler = np.load(scaler_path)
    scaler_mean, scaler_std = scaler["mean"], scaler["std"]

    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    logger.info("ONNX inputs: %s | outputs: %s", input_names, output_names)

    # Load features and labels
    logger.info("Loading features and labels...")
    fc = pd.read_parquet(FEATURES_PATH)
    lb = pd.read_parquet(LABELS_PATH)

    # Compute diff features (same as training)
    close = fc["close"].values.astype(np.float64)
    rsi7 = fc["rsi7"].values.astype(np.float64)
    rp = fc["range_position"].values.astype(np.float64)
    sma200 = fc["sma200_dist_pct"].values.astype(np.float64)
    atr_pctl = fc["atr_percentile"].values.astype(np.float64) if "atr_percentile" in fc.columns else np.full(len(close), 50.0)

    lookbacks = [1, 2, 3, 4, 5, 6, 7, 8]
    diff_raw = feature_lib.diff_features(close, rsi7, rp, sma200, lookbacks)
    diff_raw = np.nan_to_num(diff_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Snapshot features (position)
    snap_raw = feature_lib.snapshot_features(rsi7, rp, sma200, atr_pctl)
    snap_raw = np.nan_to_num(snap_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale diffs
    diff = (diff_raw - scaler_mean) / scaler_std
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale snapshot
    snap_mean = scaler["snap_mean"] if "snap_mean" in scaler else snap_raw.mean(axis=0)
    snap_std = scaler["snap_std"] if "snap_std" in scaler else snap_raw.std(axis=0)
    snap_std[snap_std < 1e-8] = 1.0
    snap = (snap_raw - snap_mean) / snap_std
    snap = np.nan_to_num(snap, nan=0.0, posinf=0.0, neginf=0.0)

    # Test set
    common_idx = lb.index.intersection(fc.index)
    fc_pos = fc.index.get_indexer(common_idx)
    lb_aligned = lb.loc[common_idx]
    diff_aligned = diff[fc_pos]
    snap_aligned = snap[fc_pos]

    test_mask = (common_idx >= TEST_START) & (common_idx <= TEST_END)
    test_mask &= lb_aligned["direction"].isin([LONG, SHORT, SKIP]).values

    X_test = diff_aligned[test_mask].reshape(-1, 8, 4)
    X_snap_test = snap_aligned[test_mask].astype(np.float32)
    y_test = lb_aligned[test_mask]["direction"].values
    n_test = len(X_test)
    logger.info("Test set: %d bars", n_test)

    # Run inference in batches
    logger.info("Running inference...")
    all_outputs = [[] for _ in output_names]
    batch_size = 4096
    for i in range(0, n_test, batch_size):
        feed = {
            input_names[0]: X_test[i:i+batch_size],
            input_names[1]: X_snap_test[i:i+batch_size],
        }
        outs = sess.run(output_names, feed)
        for j, o in enumerate(outs):
            all_outputs[j].append(o)
    outputs = [np.concatenate(parts) for parts in all_outputs]

    # Parse outputs based on what's available
    if len(outputs) == 1:
        dir_logits = outputs[0]
    elif len(outputs) == 3:
        long_pnl_pred, short_pnl_pred, dir_logits = outputs
    else:
        dir_logits = outputs[-1]

    # Direction predictions
    if dir_logits.ndim == 2 and dir_logits.shape[1] == 3:
        # Softmax
        exp_l = np.exp(dir_logits - dir_logits.max(axis=1, keepdims=True))
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        pred_class = probs.argmax(axis=1)
    elif dir_logits.ndim == 1 or dir_logits.shape[1] == 1:
        # Binary fallback
        logits_flat = dir_logits.flatten()
        probs = 1 / (1 + np.exp(-logits_flat))
        pred_class = (probs > 0.5).astype(int)
    else:
        logger.error("Unexpected output shape: %s", dir_logits.shape)
        sys.exit(1)

    # === GATE CHECKS ===
    checks = {}
    all_pass = True

    # 1. Overall 3-class accuracy
    acc_3class = float((pred_class == y_test).mean())
    checks["overall_accuracy_3class"] = {"value": round(acc_3class, 4), "threshold": 0.34, "pass": acc_3class >= 0.34}
    logger.info("  Overall accuracy (3-class): %.1f%% (gate >= 34%%)", acc_3class * 100)

    # 2-4. Per-class recall
    for cls, name, val in [(LONG, "long", LONG), (SHORT, "short", SHORT), (SKIP, "skip", SKIP)]:
        mask = y_test == val
        if mask.sum() > 0:
            recall = float((pred_class[mask] == val).mean())
        else:
            recall = 0.0
        checks[f"{name}_recall"] = {"value": round(recall, 4), "threshold": 0.10, "pass": recall >= 0.10}
        logger.info("  %s recall: %.1f%% (gate >= 10%%)", name.upper(), recall * 100)

    # 5. N confident signals (using params thresholds)
    import yaml
    with open(REPO_ROOT / "configs/params.yaml") as f:
        params = yaml.safe_load(f)
    conf_long = params["ml_v3"]["inference"]["conf_long"]
    conf_short = params["ml_v3"]["inference"]["conf_short"]

    if dir_logits.ndim == 2 and dir_logits.shape[1] == 3:
        n_conf_long = int((probs[:, LONG] >= conf_long).sum())
        n_conf_short = int((probs[:, SHORT] >= conf_short).sum())
    else:
        n_conf_long = int((pred_class == LONG).sum())
        n_conf_short = int((pred_class == SHORT).sum())

    n_conf = n_conf_long + n_conf_short
    checks["n_confident_signals"] = {"value": n_conf, "threshold": 100, "pass": n_conf >= 100}
    logger.info("  Confident signals: %d (%dL + %dS) (gate >= 100)", n_conf, n_conf_long, n_conf_short)

    # 6. Prediction distribution (not a gate, informational)
    pred_dist = {
        "LONG": int((pred_class == LONG).sum()),
        "SHORT": int((pred_class == SHORT).sum()),
        "SKIP": int((pred_class == SKIP).sum()),
    }
    logger.info("  Prediction distribution: %s", pred_dist)

    # Check all gates
    for name, check in checks.items():
        if not check["pass"]:
            all_pass = False
            logger.error("  GATE FAIL: %s = %s (required %s)", name, check["value"], check["threshold"])

    # Save report
    report = {
        "status": "PASS" if all_pass else "FAIL",
        "n_test": n_test,
        "checks": checks,
        "prediction_distribution": pred_dist,
        "conf_long": conf_long,
        "conf_short": conf_short,
        "n_confident_long": n_conf_long,
        "n_confident_short": n_conf_short,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("\n%s: %s", "PIPELINE GATE", report["status"])
    logger.info("Saved to %s", OUTPUT_PATH)

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
