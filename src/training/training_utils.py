"""Shared training scaffolding — the ritual every trainer performs, once.

Extracted from ml_train.py / train_attention.py / train_v3.py, which each
carried a private copy (~270 lines x3, audit item M1). Everything here is
model-agnostic: data loading/alignment, date splits, train-only scalers,
the early-stopping fit loop, ONNX export, manifest writing, MLflow
registration, git/dvc provenance.

Model-specific things (architecture, loss, targets, eval) live in
train.py's TASKS and architectures.py — NOT here.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "data/features/direction_prediction/feature_cache.parquet"
DATA_15M_DVC = REPO_ROOT / "data/raw/BTCUSDT_15m_ohlcv.parquet.dvc"


# ------------------------------------------------------------------ config

def load_params(model_key: str) -> dict:
    """Read one model's block from configs/params.yaml."""
    with open(REPO_ROOT / "configs/params.yaml") as f:
        params = yaml.safe_load(f)
    if model_key not in params:
        raise KeyError(f"No {model_key!r} block in configs/params.yaml")
    return params[model_key]


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_ranges(split_cfg: dict) -> dict[str, tuple[str, str]]:
    return {
        "train": (split_cfg["train_start"], split_cfg["train_end"]),
        "val": (split_cfg["val_start"], split_cfg["val_end"]),
        "test": (split_cfg["test_start"], split_cfg["test_end"]),
    }


# ------------------------------------------------------------------ data

def load_cache_and_labels(labels_path: Path, cache_path: Path = CACHE_PATH):
    fc = pd.read_parquet(cache_path)
    lb = pd.read_parquet(labels_path)
    return fc, lb


def align_to_labels(fc: pd.DataFrame, lb: pd.DataFrame, *arrays: np.ndarray):
    """Intersect cache/label indexes; slice feature arrays to the common rows.

    Returns (lb_aligned, dates, sliced_arrays...).
    """
    common_idx = lb.index.intersection(fc.index)
    fc_pos = fc.index.get_indexer(common_idx)
    lb = lb.loc[common_idx]
    sliced = tuple(a[fc_pos] for a in arrays)
    return (lb, common_idx, *sliced)


def split_by_dates(dates, valid_mask: np.ndarray,
                   ranges: dict[str, tuple[str, str]]) -> dict[str, np.ndarray]:
    """Date-based train/val/test indexes (the honest-split discipline)."""
    def in_range(lo: str, hi: str) -> np.ndarray:
        return (dates >= lo) & (dates <= hi) & valid_mask

    return {name: np.where(in_range(*rng))[0] for name, rng in ranges.items()}


def fit_scaler(arr: np.ndarray, train_idx: np.ndarray):
    """Mean/std fit on TRAIN ONLY (no leakage); dead columns get std=1."""
    mean = arr[train_idx].mean(axis=0)
    std = arr[train_idx].std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_scaler(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (arr - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------------------------ fit loop

def fit(
    model: torch.nn.Module,
    train_loader,
    train_loss_fn: Callable,      # (model, batch) -> loss tensor
    val_loss_fn: Callable,        # (model) -> float (full-val loss)
    optimizer,
    max_epochs: int,
    patience: int,
    improve_eps: float = 0.0,
    scheduler=None,
    log_fn: Callable[[str], None] = print,
    log_every: int = 10,
):
    """The one early-stopping loop (was copied into all three trainers).

    Per epoch: train batches (grad-clip 1.0), full-val loss, LR schedule,
    best-state tracking with patience. Restores best weights before return.
    Returns (model, best_val_loss, epochs_used).
    """
    best_vl = float("inf")
    best_state = None
    patience_ctr = 0
    epochs_used = 0

    for epoch in range(1, max_epochs + 1):
        epochs_used = epoch
        model.train()
        tr_sum = 0.0
        for batch in train_loader:
            loss = train_loss_fn(model, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_sum += loss.item()

        model.eval()
        vl = val_loss_fn(model)
        if scheduler is not None:
            scheduler.step(vl)

        if vl < best_vl - improve_eps:
            best_vl = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            if epoch % log_every == 0 or epoch <= 3:
                log_fn(f"  epoch {epoch:3d} train={tr_sum / len(train_loader):.4f} val={vl:.4f} ***")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log_fn(f"  early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, best_vl, epochs_used


# ------------------------------------------------------------------ export

def export_model(
    out_dir: Path,
    export_module: torch.nn.Module,
    full_state_dict: dict,
    dummy_inputs,                     # tensor or tuple of tensors
    onnx_name: str,
    pt_name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    opset_version: int,
    checkpoint_key: str = "model",
    scaler_arrays: Optional[dict] = None,
):
    """Save .pt checkpoint, ONNX export, and scaler.npz."""
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({checkpoint_key: full_state_dict}, out_dir / pt_name)

    export_module.eval()
    dynamic_axes = {n: {0: "batch"} for n in (*input_names, *output_names)}
    torch.onnx.export(
        export_module, dummy_inputs, str(out_dir / onnx_name),
        input_names=list(input_names), output_names=list(output_names),
        dynamic_axes=dynamic_axes, opset_version=opset_version,
    )

    if scaler_arrays:
        np.savez(out_dir / "scaler.npz", **scaler_arrays)


# ------------------------------------------------------------------ provenance

def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def dvc_hash(dvc_file: Path = DATA_15M_DVC) -> str:
    with open(dvc_file) as f:
        return yaml.safe_load(f)["outs"][0]["md5"]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    out_dir: Path,
    model_name: str,
    labels_path: Path,
    ranges: dict[str, tuple[str, str]],
    compute_fn: str,
    label_desc: str,
    metrics: dict,
    mlflow_info: dict,
    cache_path: Path = CACHE_PATH,
    extras: Optional[dict] = None,
) -> Path:
    """training_manifest.json — the contract read by src/mlops/verify.py."""
    import json
    from datetime import datetime as _dt

    manifest = {
        "schema_version": 1,
        "model_name": model_name,
        "trained_at": _dt.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "data": {
            "feature_cache_path": str(cache_path.relative_to(REPO_ROOT)),
            "feature_cache_md5": md5_of(cache_path),
            "labels_path": str(labels_path.relative_to(REPO_ROOT)),
            "labels_md5": md5_of(labels_path),
        },
        "split": {
            "method": "date_based",
            "train": list(ranges["train"]),
            "val": list(ranges["val"]),
            "test": list(ranges["test"]),
        },
        "scaler": {"fit_on": "train_only"},
        "feature_recipe": {
            "compute_fn": compute_fn,
            "source_parquet": str(cache_path.relative_to(REPO_ROOT)),
        },
        "label": label_desc,
        "metrics": metrics,
        "mlflow": mlflow_info,
    }
    if extras:
        manifest.update(extras)
    path = out_dir / "training_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def register_staging(model_name: str, run_id: str, source: str) -> str:
    """Create/point the MLflow registered model's @staging alias."""
    import mlflow

    client = mlflow.MlflowClient()
    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(model_name)
    mv = client.create_model_version(name=model_name, source=source, run_id=run_id)
    client.set_registered_model_alias(name=model_name, alias="staging", version=mv.version)
    return mv.version
