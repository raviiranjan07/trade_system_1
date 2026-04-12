"""Thin wrapper around MLflow tracking API.

This is the ONLY file in the mlops package that imports mlflow directly.
If MLflow is ever swapped for another tracker, only this file needs to change.

Module-level functions mirror mlflow.* to keep the call sites simple.
"""

from pathlib import Path
from typing import Any

import mlflow


# Default backend: SQLite (file-store is deprecated as of MLflow 3.x, Feb 2026)
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def init(tracking_uri: str = DEFAULT_TRACKING_URI) -> None:
    """Point MLflow at a tracking store. Idempotent.

    Default: sqlite:///mlflow.db in the current working directory.
    Pass a different URI to override (e.g., for tests).
    """
    mlflow.set_tracking_uri(tracking_uri)


def start_run(experiment_name: str, run_name: str | None = None) -> str:
    """Create or get the experiment, then start a run under it.

    Returns the MLflow run_id (32-char hex).
    """
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    return run.info.run_id


def end_run(status: str = "FINISHED") -> None:
    """Finalize the currently active run.

    status: one of 'FINISHED', 'FAILED', 'KILLED', 'SCHEDULED', 'RUNNING'
    """
    mlflow.end_run(status=status)


def active_run_id() -> str | None:
    """Return the run_id of the currently active run, or None if no active run."""
    run = mlflow.active_run()
    return run.info.run_id if run else None


# ---------- Params ----------

def _flatten_params(params: dict, prefix: str = "") -> dict[str, str]:
    """Flatten nested dicts and coerce values to strings.

    {"model": {"lr": 0.001}} becomes {"model.lr": "0.001"}
    """
    out: dict[str, str] = {}
    for k, v in params.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_params(v, prefix=f"{key}."))
        else:
            out[key] = str(v)
    return out


def log_params(params: dict) -> None:
    """Log hyperparameters. Nested dicts are flattened with dot notation."""
    flat = _flatten_params(params)
    if flat:
        mlflow.log_params(flat)


# ---------- Metrics ----------

def log_metric(key: str, value: float, step: int | None = None) -> None:
    """Log a single numeric metric."""
    mlflow.log_metric(key, float(value), step=step)


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log multiple metrics at once.

    Values that can't be coerced to float are silently skipped
    (e.g., nested objects, confusion matrices — those should be logged as artifacts).
    """
    numeric: dict[str, float] = {}
    for k, v in metrics.items():
        try:
            numeric[k] = float(v)
        except (TypeError, ValueError):
            continue
    if numeric:
        mlflow.log_metrics(numeric, step=step)


# ---------- Artifacts ----------

def log_artifact(local_path: str | Path, artifact_path: str | None = None) -> None:
    """Upload a file to the run's artifact store.

    artifact_path: optional subfolder within the run's artifacts.
    """
    mlflow.log_artifact(str(local_path), artifact_path=artifact_path)


# ---------- Tags ----------

def set_tag(key: str, value: Any) -> None:
    """Set a metadata tag on the current run."""
    mlflow.set_tag(key, str(value))


def set_tags(tags: dict) -> None:
    """Set multiple tags at once."""
    mlflow.set_tags({k: str(v) for k, v in tags.items()})
