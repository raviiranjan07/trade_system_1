"""Brain Pipeline -- End-to-end runner

Runs Steps 1-4 locally (data processing).
Exports dataset for Colab training (Step 5).

Usage:
  PYTHONPATH=src python -m brain.pipeline --config configs/base.yaml
"""

import argparse
import time
from pathlib import Path

from brain.config import load_config, get_enabled_features, validate_config
from brain.ingestion import load_ohlcv
from brain.processing import compute_all_features
from brain.labels import compute_labels
from brain.dataset import create_dataset, save_dataset


def run_pipeline(config_path: str):
    """Run the full data pipeline (Steps 1-4)."""
    start_time = time.time()

    print("=" * 70)
    print("BRAIN PIPELINE")
    print("=" * 70)

    # --- Load config ---
    print(f"\nConfig: {config_path}")
    config = load_config(config_path)

    warnings = validate_config(config)
    if warnings:
        print(f"  Warnings:")
        for w in warnings:
            print(f"    - {w}")

    features_list = get_enabled_features(config)
    print(f"  Enabled features ({len(features_list)}): {features_list}")
    print(f"  Architecture: {config['training']['architecture']}")
    print(f"  Snapshots: {config['dataset']['snapshot_count']}")

    # --- Step 1: Data Ingestion ---
    print(f"\n{'='*70}")
    print("STEP 1: DATA INGESTION")
    print(f"{'='*70}")
    df, ing_report = load_ohlcv(config)

    # --- Step 2: Data Processing ---
    print(f"\n{'='*70}")
    print("STEP 2: DATA PROCESSING")
    print(f"{'='*70}")
    features = compute_all_features(df, config)

    # --- Step 3: Label Generation ---
    print(f"\n{'='*70}")
    print("STEP 3: LABEL GENERATION")
    print(f"{'='*70}")
    labels = compute_labels(df, config, features=features)

    # --- Step 4: Dataset Creation ---
    print(f"\n{'='*70}")
    print("STEP 4: DATASET CREATION")
    print(f"{'='*70}")
    data = create_dataset(features, labels, config)

    # --- Export for Colab ---
    tracking = config.get("tracking", {})
    exp_dir = tracking.get("experiment_dir", "experiments/brain/")
    export_path = Path(exp_dir) / "datasets"

    print(f"\n{'='*70}")
    print("EXPORT")
    print(f"{'='*70}")
    save_dataset(data, str(export_path))

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  1. Upload {export_path}/ to Google Colab")
    print(f"  2. Run training (Step 5) on Colab T4 GPU")
    print(f"  3. Download model weights")
    print(f"  4. Run evaluation (Step 6) locally")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    run_pipeline(args.config)
