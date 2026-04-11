"""Brain Pipeline -- Step 6: Evaluation

Comprehensive evaluation of trained model.
Generates standard report with direction, S/R, MFE, consistency metrics.

Usage:
  PYTHONPATH=src python -m brain.evaluation --data_dir experiments/brain/datasets --model_dir experiments/brain/EXP-B001 --config configs/base.yaml
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


def evaluate_model(config, data_dir, model_dir, output_dir=None):
    """Full evaluation on all splits."""
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    if output_dir is None:
        output_dir = model_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load data ---
    print("Loading data...")
    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)

    num_features = meta["snapshot_shape"][1]
    snapshot_count = meta["snapshot_shape"][0]
    feature_names = meta["feature_names"]
    label_names = meta["label_names"]

    # --- Load model ---
    print("Loading model...")
    try:
        from brain.models.lstm import build_model
        model = build_model(config, num_features, snapshot_count)
    except ImportError:
        from brain.training import build_model_from_config
        model = build_model_from_config(config, num_features, snapshot_count)

    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()

    # --- Evaluate each split ---
    report = {
        "config": str(config.get("training", {}).get("architecture", "unknown")),
        "features": len(feature_names),
        "feature_names": feature_names,
    }

    for split_name in ["train", "val", "test"]:
        print(f"\n--- {split_name.upper()} ---")
        data = np.load(data_dir / f"{split_name}.npz")
        X = data["X"]
        Y = data["Y"]

        direction = Y[:, 0].astype(np.int64)
        mfe_true = Y[:, 1:].astype(np.float32)

        # Predict
        loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(direction), torch.tensor(mfe_true)),
            batch_size=512, shuffle=False,
        )

        all_probs = []
        all_dirs = []
        all_mfe_pred = []
        all_mfe_true = []

        with torch.no_grad():
            for X_b, dir_b, mfe_b in loader:
                X_b = X_b.to(device)
                dir_pred, mfe_pred = model(X_b)
                probs = torch.softmax(dir_pred, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_dirs.append(dir_b.numpy())
                all_mfe_pred.append(mfe_pred.cpu().numpy())
                all_mfe_true.append(mfe_b.numpy())

        probs = np.concatenate(all_probs)
        dirs = np.concatenate(all_dirs)
        mfe_pred = np.concatenate(all_mfe_pred)
        mfe_true = np.concatenate(all_mfe_true)

        pred_class = np.argmax(probs, axis=1)
        max_prob = np.max(probs, axis=1)

        split_report = {}

        # --- Direction accuracy ---
        overall_acc = (pred_class == dirs).mean() * 100
        split_report["overall_accuracy"] = round(overall_acc, 1)
        print(f"  Overall accuracy: {overall_acc:.1f}% ({len(dirs)} bars)")

        # Per class
        long_mask = dirs == 0
        short_mask = dirs == 1
        if long_mask.sum() > 0:
            long_acc = (pred_class[long_mask] == 0).mean() * 100
            split_report["long_accuracy"] = round(long_acc, 1)
            print(f"  LONG accuracy:  {long_acc:.1f}% ({long_mask.sum()} bars)")
        if short_mask.sum() > 0:
            short_acc = (pred_class[short_mask] == 1).mean() * 100
            split_report["short_accuracy"] = round(short_acc, 1)
            print(f"  SHORT accuracy: {short_acc:.1f}% ({short_mask.sum()} bars)")

        # Confident accuracy
        for thresh in config.get("evaluation", {}).get("confidence_thresholds", [0.55, 0.60, 0.65, 0.70]):
            mask = max_prob >= thresh
            if mask.sum() > 0:
                acc = (pred_class[mask] == dirs[mask]).mean() * 100
                split_report[f"conf_{int(thresh*100)}_acc"] = round(acc, 1)
                split_report[f"conf_{int(thresh*100)}_count"] = int(mask.sum())
                print(f"  Confident (>{thresh}): {acc:.1f}% ({mask.sum()} bars, {mask.sum()/len(dirs)*100:.1f}%)")

        # --- MFE ---
        mfe_mae = np.mean(np.abs(mfe_pred - mfe_true))
        split_report["mfe_mae"] = round(float(mfe_mae), 2)
        print(f"  MFE MAE: {mfe_mae:.2f} bps")

        # MFE direction accuracy: does model predict higher mfe_up vs mfe_down?
        # mfe columns: mfe_up_1, mfe_down_1, mfe_up_2, mfe_down_2, ...
        num_horizons = mfe_pred.shape[1] // 2
        mfe_up_pred = mfe_pred[:, 0::2]    # even indices
        mfe_down_pred = mfe_pred[:, 1::2]  # odd indices
        # Use max horizon for direction
        mfe_dir_pred = (mfe_up_pred[:, -1] > mfe_down_pred[:, -1]).astype(int)
        # 0 = up bigger = LONG, 1 = down bigger = SHORT
        mfe_dir_pred = 1 - mfe_dir_pred  # flip: 0=LONG, 1=SHORT
        mfe_dir_acc = (mfe_dir_pred == dirs).mean() * 100
        split_report["mfe_direction_accuracy"] = round(mfe_dir_acc, 1)
        print(f"  MFE-derived direction: {mfe_dir_acc:.1f}%")

        # --- S/R specific (using features from X) ---
        # Feature indices for distance_to_support and distance_to_resistance
        if "distance_to_support" in feature_names and "distance_to_resistance" in feature_names:
            sup_idx = feature_names.index("distance_to_support")
            res_idx = feature_names.index("distance_to_resistance")

            # Use last snapshot (current bar)
            dist_sup = X[:, -1, sup_idx]
            dist_res = X[:, -1, res_idx]

            # Load scaler to un-normalize
            scaler = np.load(data_dir / "scaler.npz")
            sup_mean, sup_std = scaler["mean"][sup_idx], scaler["std"][sup_idx]
            res_mean, res_std = scaler["mean"][res_idx], scaler["std"][res_idx]

            dist_sup_raw = dist_sup * sup_std + sup_mean
            dist_res_raw = dist_res * res_std + res_mean

            # At support: distance_to_support small and positive
            sr_method = config.get("evaluation", {}).get("sr_position", {}).get("method", "fixed")
            if sr_method == "fixed":
                thresh_bps = config.get("evaluation", {}).get("sr_position", {}).get("fixed_threshold_bps", 5)
            else:
                thresh_bps = 10  # data-driven default

            at_sup = (dist_sup_raw >= 0) & (dist_sup_raw <= thresh_bps)
            at_res = (dist_res_raw >= 0) & (dist_res_raw <= thresh_bps)

            if at_sup.sum() > 20:
                # At support, check if model predicts LONG more often
                sup_pred_long = (pred_class[at_sup] == 0).mean() * 100
                sup_actual_long = (dirs[at_sup] == 0).mean() * 100
                sup_acc = (pred_class[at_sup] == dirs[at_sup]).mean() * 100
                split_report["at_support_count"] = int(at_sup.sum())
                split_report["at_support_accuracy"] = round(sup_acc, 1)
                split_report["at_support_pred_long"] = round(sup_pred_long, 1)
                print(f"  At support ({at_sup.sum()} bars): acc={sup_acc:.1f}%, pred LONG={sup_pred_long:.1f}%, actual LONG={sup_actual_long:.1f}%")

            if at_res.sum() > 20:
                res_pred_short = (pred_class[at_res] == 1).mean() * 100
                res_actual_short = (dirs[at_res] == 1).mean() * 100
                res_acc = (pred_class[at_res] == dirs[at_res]).mean() * 100
                split_report["at_resistance_count"] = int(at_res.sum())
                split_report["at_resistance_accuracy"] = round(res_acc, 1)
                split_report["at_resistance_pred_short"] = round(res_pred_short, 1)
                print(f"  At resistance ({at_res.sum()} bars): acc={res_acc:.1f}%, pred SHORT={res_pred_short:.1f}%, actual SHORT={res_actual_short:.1f}%")

        report[split_name] = split_report

    # --- Consistency ---
    if "train" in report and "test" in report:
        gap = report["train"]["overall_accuracy"] - report["test"]["overall_accuracy"]
        report["train_test_gap"] = round(gap, 1)
        print(f"\n--- CONSISTENCY ---")
        print(f"  Train-Test gap: {gap:.1f}%")

    # --- Comparison vs baselines ---
    baselines = config.get("evaluation", {}).get("comparison", {}).get("baselines", [])
    if baselines and "test" in report:
        print(f"\n--- vs BASELINES ---")
        test_acc = report["test"]["overall_accuracy"]
        print(f"  | {'Model':25s} | {'Accuracy':>8s} | {'vs This':>8s} |")
        print(f"  |{'-'*27}|{'-'*10}|{'-'*10}|")
        for b in baselines:
            diff = test_acc - b["accuracy"]
            print(f"  | {b['name']:25s} | {b['accuracy']:7.1f}% | {diff:+7.1f}% |")
        print(f"  | {'THIS MODEL':25s} | {test_acc:7.1f}% | {'--':>8s} |")

    # --- Save report ---
    with open(output_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {output_dir}/evaluation_report.json")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Evaluation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    try:
        from brain.config import load_config
        config = load_config(args.config)
    except ImportError:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)

    evaluate_model(config, args.data_dir, args.model_dir, args.output_dir)
