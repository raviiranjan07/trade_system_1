"""Register current production models in the MLflow Model Registry.

One-time bootstrap: wraps the files under models/ML_V1/ and
models/direction_attention/ as MLflow runs, then registers each as a
named model with @production alias.

Run from repo root:
    PYTHONPATH=src python scripts/mlops/register_models.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import mlflow
import yaml

from mlops import tracking

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "direction_prediction"


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def dvc_hash(dvc_file: Path) -> str:
    """Read the md5 hash from a .dvc pointer file."""
    with open(dvc_file) as f:
        meta = yaml.safe_load(f)
    return meta["outs"][0]["md5"]


def register(
    name: str,
    run_name: str,
    artifacts: list[Path],
    metrics: dict[str, float],
    params: dict[str, Any],
    notes: str,
) -> str:
    """Create a run, log artifacts/metrics, register as model, alias @production.

    Returns the new model version (e.g. '1').
    """
    run_id = tracking.start_run(EXPERIMENT, run_name=run_name)

    tracking.log_params(params)
    tracking.log_metrics(metrics)
    tracking.set_tags(
        {
            "git_commit": git_commit(),
            "data_15m_md5": dvc_hash(REPO_ROOT / "data/raw/BTCUSDT_15m_ohlcv.parquet.dvc"),
            "data_1m_md5": dvc_hash(REPO_ROOT / "data/raw/BTCUSDT_1m_ohlcv.parquet.dvc"),
            "source": "imported_from_production_files",
            "notes": notes,
        }
    )
    for path in artifacts:
        tracking.log_artifact(path, artifact_path="model")

    tracking.end_run("FINISHED")

    client = mlflow.MlflowClient()
    try:
        client.create_registered_model(name)
    except mlflow.exceptions.RestException:
        pass  # already exists

    model_uri = f"runs:/{run_id}/model"
    mv = client.create_model_version(name=name, source=model_uri, run_id=run_id)
    client.set_registered_model_alias(name=name, alias="production", version=mv.version)
    return mv.version


def main() -> None:
    tracking.init()

    # ML_V1 — MLP, deployed in production as ML V1.5
    v15_files = sorted((REPO_ROOT / "models/ML_V1").glob("*"))
    v15_version = register(
        name="ML_V1",
        run_name="direction_v15_v1_imported",
        artifacts=v15_files,
        metrics={
            "test_accuracy": 0.575,
            "test_confident_accuracy": 0.575,
            "oos_total_bps": 18207,
            "oos_n_trades": 1767,
            "oos_win_pct": 53.3,
        },
        params={
            "model_type": "MLP",
            "architecture": "10-128-128-1",
            "features": "roc1-8+range_position+rsi7",
            "horizon": 8,
            "label": "direction_h8",
        },
        notes="Imported from existing production files. Trained 2026-04-12 (run 15f73d0c).",
    )
    print(f"  Registered ML_V1 version {v15_version}")

    # direction_attention — V2 attention model
    attn_files = sorted((REPO_ROOT / "models/direction_attention").glob("*"))
    attn_version = register(
        name="direction_attention",
        run_name="direction_attention_v1_imported",
        artifacts=attn_files,
        metrics={
            "test_accuracy": 0.514,
            "test_confident_accuracy": 0.594,
            "test_n_confident": 834,
            "backtest_win_rate": 47.7,
            "backtest_total_bps": 7271,
            "backtest_profit_factor": 1.69,
        },
        params={
            "model_type": "LSTMAttention_temp0.5",
            "features": "roc_diff+rsi_diff+rp_diff+sma200_diff x 8 = 32",
            "horizon": 8,
            "label": "direction_h8",
            "threshold_long": 0.60,
            "threshold_short": 0.40,
            "temperature": 0.5,
        },
        notes="Imported from existing production files. Trained 2026-04-12 on Colab T4 (run 7b6386f0).",
    )
    print(f"  Registered direction_attention version {attn_version}")

    print("\nDone. Verify with:")
    print("  mlflow models list")
    print("  Or in Python: mlflow.MlflowClient().get_model_version_by_alias('ML_V1', 'production')")


if __name__ == "__main__":
    main()
