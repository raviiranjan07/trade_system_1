"""Train ML_V1 MLP with honest date-based split and auto-register in MLflow.

- Train:  2020-2023
- Val:    2024  (early stopping)
- Test:   2025  (true OOS — never seen during training)
- Scaler fit on train only (no leakage)

Artifacts written to models/ML_V1_staging/ so the live bot
(which reads models/ML_V1/) is unaffected. Promotion is manual:
a separate script copies staging files over production + swaps the
MLflow alias.

MLflow:
- Logs run into experiment "direction_prediction" with honest metrics
- Registers model as ML_V1 with alias @staging
- Tags capture git commit, data hash (from .dvc), train date

Run:
    python src/training/ml_train.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mlflow

from mlops import tracking
from mlops.runner import run_experiment
from mlops.evaluation import evaluate_direction_prediction
from engine.signals import feature_lib

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "data/features/direction_prediction/feature_cache.parquet"
LABELS_PATH = REPO_ROOT / "data/features/direction_prediction/labels.parquet"
OUT_DIR = REPO_ROOT / "models/ML_V1_staging"
EXPERIMENT = "direction_prediction"
MODEL_NAME = "ML_V1"

# Load tunable parameters from configs/params.yaml — DVC tracks changes here.
with open(REPO_ROOT / "configs/params.yaml") as _f:
    _params = yaml.safe_load(_f)

_model_cfg = _params["ml_v1"]

# Input spec — must match what signals/ml_v1.py reads (same params.yaml block).
_feat = _model_cfg.get("features", {})
ROC_PERIODS = list(_feat.get("roc_periods", range(1, 9)))
RSI_PERIOD = int(_feat.get("rsi_period", 7))
RANGE_WINDOW = int(_feat.get("range_position_window", 50))

_split = _model_cfg["split"]
TRAIN_RANGE = (_split["train_start"], _split["train_end"])
VAL_RANGE = (_split["val_start"], _split["val_end"])
TEST_RANGE = (_split["test_start"], _split["test_end"])

_t = _model_cfg["training"]
LR = _t["lr"]
WEIGHT_DECAY = _t["weight_decay"]
BATCH_SIZE = _t["batch_size"]
MAX_EPOCHS = _t["max_epochs"]
PATIENCE = _t["patience"]
HIDDEN = _t["hidden"]
SEED = _t["seed"]
# Thresholds for confident_accuracy metric — read from inference section so
# the reported metric matches what the live bot sees at runtime (single source
# of truth). Semantics: conf_long = required P(LONG), conf_short = required
# P(SHORT); so SHORT fires when P(LONG) < 1 - conf_short.
_i = _model_cfg["inference"]
CONF_LONG = _i["conf_long"]
CONF_SHORT = _i["conf_short"]


class MLPBinaryDir(nn.Module):
    def __init__(self, input_size: int = 10, hidden: int = HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.out(h).squeeze(-1)


class ArrayDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def dvc_hash(dvc_file: Path) -> str:
    with open(dvc_file) as f:
        return yaml.safe_load(f)["outs"][0]["md5"]


def compute_features(fc: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Compute the 10 features used by MLP: roc1-8, range_position_50, rsi7.

    Computed fresh from OHLC (not read from cache) to keep training
    self-contained and match the legacy production feature set.
    """
    close = fc["close"].values.astype(np.float64)
    high = fc["high"].values.astype(np.float64)
    low = fc["low"].values.astype(np.float64)

    for n in ROC_PERIODS:
        fc[f"roc{n}"] = feature_lib.roc_bps(close, n)

    fc["rsi7_mlp"] = feature_lib.rsi_rolling(pd.Series(close), RSI_PERIOD).values

    fc["range_position_50_mlp"] = feature_lib.range_position_inclusive(
        high, low, close, RANGE_WINDOW)

    feat_cols = [f"roc{n}" for n in ROC_PERIODS] + ["range_position_50_mlp", "rsi7_mlp"]
    return fc[feat_cols].values.astype(np.float32), feat_cols


def split_indices(dates: pd.DatetimeIndex, valid_mask: np.ndarray) -> dict[str, np.ndarray]:
    def in_range(lo: str, hi: str) -> np.ndarray:
        return (dates >= lo) & (dates <= hi) & valid_mask

    return {
        "train": np.where(in_range(*TRAIN_RANGE))[0],
        "val": np.where(in_range(*VAL_RANGE))[0],
        "test": np.where(in_range(*TEST_RANGE))[0],
    }


def compute_metrics(probs: np.ndarray, y: np.ndarray, conf_long: float, conf_short: float) -> dict:
    pred = (probs > 0.5).astype(int)
    acc = float((pred == y.astype(int)).mean())

    conf_mask = (probs >= conf_long) | (probs <= conf_short)
    if conf_mask.sum() > 0:
        conf_pred = (probs[conf_mask] > 0.5).astype(int)
        conf_acc = float((conf_pred == y[conf_mask].astype(int)).mean())
        n_conf = int(conf_mask.sum())
    else:
        conf_acc = 0.0
        n_conf = 0

    return {
        "accuracy": round(acc, 4),
        "confident_accuracy": round(conf_acc, 4),
        "n_confident": n_conf,
        "confident_fraction": round(n_conf / len(y), 4) if len(y) else 0.0,
    }


def train_once(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[MLPBinaryDir, float, int]:
    model = MLPBinaryDir()
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_loader = DataLoader(ArrayDS(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ArrayDS(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    epochs_used = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_used = epoch
        model.train()
        tr_sum = 0.0
        for Xb, yb in train_loader:
            loss = bce(model(Xb), yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_sum += loss.item()

        model.eval()
        vl_sum = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                vl_sum += bce(model(Xb), yb).item()
        vl = vl_sum / len(val_loader)
        scheduler.step(vl)

        if vl < best_val - 1e-5:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            if epoch % 10 == 0 or epoch <= 3:
                print(f"  epoch {epoch:3d} train={tr_sum/len(train_loader):.4f} val={vl:.4f} ***")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, best_val, epochs_used


def export(model: MLPBinaryDir, scaler_mean: np.ndarray, scaler_std: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = out_dir / "direction_model.pt"
    torch.save({"model": model.state_dict()}, pt_path)

    onnx_path = out_dir / "direction_model.onnx"
    dummy = torch.randn(1, 10)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["features"], output_names=["logit"],
        dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=18,
    )

    scaler_path = out_dir / "scaler.npz"
    np.savez(scaler_path, mean=scaler_mean, std=scaler_std)


def register_staging(run_id: str, artifact_subdir: str) -> str:
    client = mlflow.MlflowClient()
    try:
        client.get_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(MODEL_NAME)
    mv = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{run_id}/{artifact_subdir}",
        run_id=run_id,
    )
    client.set_registered_model_alias(name=MODEL_NAME, alias="staging", version=mv.version)
    return mv.version


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading features + labels...")
    fc = pd.read_parquet(CACHE_PATH)
    lb = pd.read_parquet(LABELS_PATH)

    feat_arr_raw, feat_cols = compute_features(fc)
    feat_arr_raw = np.nan_to_num(feat_arr_raw, nan=0.0, posinf=0.0, neginf=0.0)

    common_idx = lb.index.intersection(fc.index)
    fc_pos = fc.index.get_indexer(common_idx)
    lb = lb.loc[common_idx]

    feat_arr_raw = feat_arr_raw[fc_pos]
    dates = common_idx
    direction = lb["direction"].values
    valid_mask = (direction == 0) | (direction == 1)
    y_binary = np.zeros(len(direction), dtype=np.float32)
    y_binary[direction == 0] = 1.0  # LONG=1, SHORT=0

    splits = split_indices(dates, valid_mask)
    print(f"  train {len(splits['train'])} | val {len(splits['val'])} | test {len(splits['test'])}")

    # Scaler fit on TRAIN only (no leakage)
    scaler_mean = feat_arr_raw[splits["train"]].mean(axis=0)
    scaler_std = feat_arr_raw[splits["train"]].std(axis=0)
    scaler_std[scaler_std < 1e-8] = 1.0

    X = (feat_arr_raw - scaler_mean) / scaler_std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, y_train = X[splits["train"]], y_binary[splits["train"]]
    X_val,   y_val   = X[splits["val"]],   y_binary[splits["val"]]
    X_test,  y_test  = X[splits["test"]],  y_binary[splits["test"]]

    print("\nTraining...")
    model, best_val_loss, epochs_used = train_once(X_train, y_train, X_val, y_val)

    print("\nEvaluating...")
    model.eval()
    with torch.no_grad():
        probs_train = torch.sigmoid(model(torch.from_numpy(X_train))).numpy()
        probs_val   = torch.sigmoid(model(torch.from_numpy(X_val))).numpy()
        probs_test  = torch.sigmoid(model(torch.from_numpy(X_test))).numpy()

    # MFE arrays for test-period magnitude metrics (used by evaluate_direction_prediction)
    mfe_up_test = lb["mfe_up_96"].values[splits["test"]]
    mfe_down_test = lb["mfe_down_96"].values[splits["test"]]

    print(f"\nExporting to {OUT_DIR.relative_to(REPO_ROOT)}...")
    export(model, scaler_mean, scaler_std, OUT_DIR)

    # Protocol-enforced run: runner validates all required metrics present on exit
    print("\nLogging to MLflow via protocol-enforced runner...")
    tracking.init()
    run_id = None
    with run_experiment(
        experiment_name=EXPERIMENT,
        protocol_name="direction_prediction_v1",
        params={
            "model_type": "MLP",
            "architecture": "10-128-128-1",
            "features": ",".join(feat_cols),
            "label": "direction (H96)",
            "split_method": "date_based",
            "train_range": f"{TRAIN_RANGE[0]}..{TRAIN_RANGE[1]}",
            "val_range": f"{VAL_RANGE[0]}..{VAL_RANGE[1]}",
            "test_range": f"{TEST_RANGE[0]}..{TEST_RANGE[1]}",
            "scaler_fit": "train_only",
            "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "patience": PATIENCE, "hidden": HIDDEN,
            "conf_long": CONF_LONG, "conf_short": CONF_SHORT,
            "seed": SEED, "epochs_used": epochs_used,
        },
        primary_metric="test_confident_accuracy",
        model_type="MLP",
        dataset_version="feature_cache_v2",
        notes=f"Honest retrain {date.today().isoformat()} — date-split, train-only scaler, V1 exits ready.",
    ) as run:
        run_id = run.mlflow_run_id

        # Protocol uses conf_threshold_short = "prob below this = SHORT".
        # Our CONF_SHORT is symmetric with CONF_LONG (required conf for SHORT class),
        # so the eval's conf_threshold_short = 1 - CONF_SHORT.
        conf_threshold_short_for_eval = 1 - CONF_SHORT

        m_train = evaluate_direction_prediction(
            probs_train, y_train.astype(int),
            conf_threshold_long=CONF_LONG,
            conf_threshold_short=conf_threshold_short_for_eval,
            split_name="train",
        )
        m_val = evaluate_direction_prediction(
            probs_val, y_val.astype(int),
            conf_threshold_long=CONF_LONG,
            conf_threshold_short=conf_threshold_short_for_eval,
            split_name="val",
        )
        m_test = evaluate_direction_prediction(
            probs_test, y_test.astype(int),
            mfe_up_bps=mfe_up_test, mfe_down_bps=mfe_down_test,
            conf_threshold_long=CONF_LONG,
            conf_threshold_short=conf_threshold_short_for_eval,
            split_name="test",
        )
        run.log_metrics({**m_train, **m_val, **m_test})
        run.log_metric("best_val_loss", round(best_val_loss, 6))

        run.set_tag("git_commit", git_commit())
        run.set_tag("data_15m_md5", dvc_hash(REPO_ROOT / "data/raw/BTCUSDT_15m_ohlcv.parquet.dvc"),)
        run.set_tag("source", "honest_retrain")

        for f in sorted(OUT_DIR.glob("*")):
            run.log_artifact(str(f), artifact_subdir="model")

        print(f"  train acc {m_train['train_accuracy']:.4f} conf_acc {m_train['train_confident_accuracy']:.4f}")
        print(f"  val   acc {m_val['val_accuracy']:.4f} conf_acc {m_val['val_confident_accuracy']:.4f}")
        print(f"  TEST  acc {m_test['test_accuracy']:.4f} conf_acc {m_test['test_confident_accuracy']:.4f}")

    # Runner validated against protocol. Now register the MLflow model version.
    version = register_staging(run_id, "model")
    print(f"\nRegistered {MODEL_NAME} v{version} @staging.")
    print(f"MLflow run: {run_id}")

    # Training manifest — contract read by src/mlops/verify.py
    import hashlib, json
    from datetime import datetime as _dt

    def _md5_of(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "trained_at": _dt.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "data": {
            "feature_cache_path": str(CACHE_PATH.relative_to(REPO_ROOT)),
            "feature_cache_md5": _md5_of(CACHE_PATH),
            "labels_path": str(LABELS_PATH.relative_to(REPO_ROOT)),
            "labels_md5": _md5_of(LABELS_PATH),
        },
        "split": {
            "method": "date_based",
            "train": list(TRAIN_RANGE),
            "val": list(VAL_RANGE),
            "test": list(TEST_RANGE),
        },
        "scaler": {"fit_on": "train_only"},
        "feature_recipe": {
            "compute_fn": "training.ml_train.compute_features",
            "source_parquet": str(CACHE_PATH.relative_to(REPO_ROOT)),
        },
        "features": feat_cols,
        "label": "direction (H96)",
        "metrics": {
            "train_accuracy": m_train.get("train_accuracy"),
            "train_confident_accuracy": m_train.get("train_confident_accuracy"),
            "val_accuracy": m_val.get("val_accuracy"),
            "val_confident_accuracy": m_val.get("val_confident_accuracy"),
            "test_accuracy": m_test.get("test_accuracy"),
            "test_confident_accuracy": m_test.get("test_confident_accuracy"),
            "test_f1": m_test.get("test_f1"),
            "test_precision_long": m_test.get("test_precision_long"),
            "test_recall_long": m_test.get("test_recall_long"),
        },
        "mlflow": {
            "run_id": run_id,
            "registered_version": int(version),
            "alias": "staging",
        },
    }
    (OUT_DIR / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {(OUT_DIR / 'training_manifest.json').relative_to(REPO_ROOT)}")

    print(f"\nTo promote: copy models/{OUT_DIR.name}/* to models/ML_V1/ and swap alias @staging -> @production")


if __name__ == "__main__":
    main()
