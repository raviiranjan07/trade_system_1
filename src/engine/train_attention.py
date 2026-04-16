"""Train ML_V2_ATTENTION (LSTM+Attention) with honest date-based split.

Same architecture as scripts/colab/train_attention_production.py but:
- Runs locally (CPU) instead of Colab GPU
- Date-based split: train 2020-23 / val 2024 / test 2025
- Scaler fit on TRAIN ONLY (no leakage)
- Auto-registers as ML_V2_ATTENTION @staging in MLflow
- Writes training_manifest.json for the generic verifier

Run: python src/engine/train_attention.py
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

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "data/features/direction_prediction/feature_cache.parquet"
LABELS_PATH = REPO_ROOT / "data/features/direction_prediction/labels.parquet"
OUT_DIR = REPO_ROOT / "models/ML_V2_ATTENTION_staging"
EXPERIMENT = "direction_prediction"
MODEL_NAME = "ML_V2_ATTENTION"

# Load params
with open(REPO_ROOT / "configs/params.yaml") as _f:
    _params = yaml.safe_load(_f)
_cfg = _params.get("ml_v2_attention", {})

_split = _cfg.get("split", _params.get("ml_v1", {}).get("split", {}))
TRAIN_RANGE = (_split["train_start"], _split["train_end"])
VAL_RANGE = (_split["val_start"], _split["val_end"])
TEST_RANGE = (_split["test_start"], _split["test_end"])

_t = _cfg.get("training", {})
LR = _t.get("lr", 0.001)
BATCH_SIZE = _t.get("batch_size", 512)  # smaller than Colab's 2048 for CPU
MAX_EPOCHS = _t.get("max_epochs", 50)   # reduced for CPU — early stop usually hits well before
PATIENCE = _t.get("patience", 10)
HIDDEN = _t.get("hidden", 128)
DROPOUT = _t.get("dropout", 0.5)
TEMPERATURE = _t.get("temperature", 0.5)
SEED = _t.get("seed", 42)
# Confident-accuracy thresholds — read from INFERENCE section (single source
# of truth with the signal generator + live bot). Consistent semantics:
# conf_long = required P(LONG), conf_short = required P(SHORT).
_i = _cfg.get("inference", {})
CONF_LONG = _i.get("conf_long", 0.60)
CONF_SHORT = _i.get("conf_short", 0.60)

LOOKBACKS = [1, 2, 3, 4, 5, 6, 7, 8]
MFE_HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8]


class SeqDS(Dataset):
    def __init__(self, X: np.ndarray, y_mu: np.ndarray, y_md: np.ndarray, y_dir: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y_mu = torch.from_numpy(y_mu)
        self.y_md = torch.from_numpy(y_md)
        self.y_dir = torch.from_numpy(y_dir)

    def __len__(self) -> int:
        return len(self.y_dir)

    def __getitem__(self, i: int):
        return self.X[i], self.y_mu[i], self.y_md[i], self.y_dir[i]


class LSTMAttention(nn.Module):
    def __init__(self, input_size=4, hidden=128, dropout=0.5, temperature=0.5):
        super().__init__()
        self.hidden = hidden
        self.temperature = temperature
        self.lstm = nn.LSTM(input_size, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_score = nn.Linear(hidden, 1)
        self.h_mfe_up = nn.Linear(hidden, 8)
        self.h_mfe_down = nn.Linear(hidden, 8)
        self.h_dir = nn.Linear(hidden + 8 + 8, 1)

    def forward(self, x):
        all_h, _ = self.lstm(x)
        scores = self.attn_score(all_h).squeeze(-1)
        attn_weights = torch.softmax(scores / self.temperature, dim=1)
        attended = torch.bmm(attn_weights.unsqueeze(1), all_h).squeeze(1)
        attended = self.dropout(attended)
        p_mu = self.h_mfe_up(attended)
        p_md = self.h_mfe_down(attended)
        dir_input = torch.cat([attended, p_mu, p_md], dim=1)
        p_dir = self.h_dir(dir_input).squeeze(-1)
        return p_mu, p_md, p_dir


class AttentionDirectionOnly(nn.Module):
    """ONNX-export wrapper — outputs only direction logit."""
    def __init__(self, full_model):
        super().__init__()
        self.model = full_model

    def forward(self, x):
        _, _, p_dir = self.model(x)
        return p_dir


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def dvc_hash(dvc_file: Path) -> str:
    with open(dvc_file) as f:
        return yaml.safe_load(f)["outs"][0]["md5"]


def compute_features(fc: pd.DataFrame):
    """Compute 32 diff features as [N, 32] array. Rows are per-bar features.
    Sequence reshape to [N, 8, 4] happens inside the dataset iterator.
    """
    close = fc["close"].values.astype(np.float64)
    rsi7 = fc["rsi7"].values.astype(np.float64)
    rp = fc["range_position"].values.astype(np.float64)
    sma200 = fc["sma200_dist_pct"].values.astype(np.float64)

    diff_list = []
    for n in LOOKBACKS:
        roc_d = np.zeros(len(close), dtype=np.float32)
        roc_d[n:] = ((close[n:] - close[:-n]) / close[:-n] * 10000).astype(np.float32)
        rsi_d = np.zeros(len(close), dtype=np.float32)
        rsi_d[n:] = (rsi7[n:] - rsi7[:-n]).astype(np.float32)
        rp_d = np.zeros(len(close), dtype=np.float32)
        rp_d[n:] = (rp[n:] - rp[:-n]).astype(np.float32)
        sma_d = np.zeros(len(close), dtype=np.float32)
        sma_d[n:] = (sma200[n:] - sma200[:-n]).astype(np.float32)
        diff_list.extend([roc_d, rsi_d, rp_d, sma_d])

    return np.column_stack(diff_list).astype(np.float32)


def split_indices(dates, valid_mask):
    def in_range(lo, hi):
        return (dates >= lo) & (dates <= hi) & valid_mask
    return {
        "train": np.where(in_range(*TRAIN_RANGE))[0],
        "val": np.where(in_range(*VAL_RANGE))[0],
        "test": np.where(in_range(*TEST_RANGE))[0],
    }


def compute_metrics(probs, y, conf_long, conf_short):
    pred = (probs > 0.5).astype(int)
    acc = float((pred == y.astype(int)).mean())
    conf_mask = (probs >= conf_long) | (probs <= (1 - conf_short))
    if conf_mask.sum() > 0:
        conf_pred = (probs[conf_mask] > 0.5).astype(int)
        conf_acc = float((conf_pred == y[conf_mask].astype(int)).mean())
        n_conf = int(conf_mask.sum())
    else:
        conf_acc = 0.0
        n_conf = 0
    return {"accuracy": round(acc, 4), "confident_accuracy": round(conf_acc, 4), "n_confident": n_conf}


def train_once(X_train, y_mu_train, y_md_train, y_dir_train,
               X_val, y_mu_val, y_md_val, y_dir_val):
    model = LSTMAttention(input_size=4, hidden=HIDDEN, dropout=DROPOUT, temperature=TEMPERATURE)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_loader = DataLoader(SeqDS(X_train, y_mu_train, y_md_train, y_dir_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SeqDS(X_val, y_mu_val, y_md_val, y_dir_val),
                            batch_size=BATCH_SIZE, shuffle=False)

    best_vl = float("inf")
    best_state = None
    patience_ctr = 0
    epochs_used = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_used = epoch
        model.train()
        tr_sum = 0.0
        for xb, ymu, ymd, ydir in train_loader:
            p_mu, p_md, p_dir = model(xb)
            loss = mse(p_mu, ymu) + mse(p_md, ymd) + 0.5 * bce(p_dir, ydir)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_sum += loss.item()

        model.eval()
        vl_sum = 0.0
        with torch.no_grad():
            for xb, ymu, ymd, ydir in val_loader:
                p_mu, p_md, p_dir = model(xb)
                vl_sum += (mse(p_mu, ymu) + mse(p_md, ymd) + 0.5 * bce(p_dir, ydir)).item()
        vl = vl_sum / len(val_loader)
        scheduler.step(vl)

        if vl < best_vl - 1e-5:
            best_vl = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            if epoch % 5 == 0 or epoch <= 3:
                print(f"  epoch {epoch:3d}  train={tr_sum/len(train_loader):.4f}  val={vl:.4f} ***")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, best_vl, epochs_used


def export(model: LSTMAttention, scaler_mean, scaler_std, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = out_dir / "attention_model.pt"
    torch.save({"model": model.state_dict()}, pt_path)

    # Export direction-only for production (bot only needs direction)
    onnx_path = out_dir / "attention_model.onnx"
    direction_only = AttentionDirectionOnly(model)
    direction_only.eval()
    dummy = torch.randn(1, 8, 4)
    torch.onnx.export(
        direction_only, dummy, str(onnx_path),
        input_names=["features"], output_names=["logit"],
        dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=18,
    )

    np.savez(out_dir / "scaler.npz", mean=scaler_mean, std=scaler_std)


def register_staging(run_id: str, artifact_subdir: str) -> str:
    client = mlflow.MlflowClient()
    try:
        client.get_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(MODEL_NAME)
    mv = client.create_model_version(name=MODEL_NAME, source=f"runs:/{run_id}/{artifact_subdir}", run_id=run_id)
    client.set_registered_model_alias(name=MODEL_NAME, alias="staging", version=mv.version)
    return mv.version


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading data...")
    fc = pd.read_parquet(CACHE_PATH)
    lb = pd.read_parquet(LABELS_PATH)

    print("Computing 32 diff features (4 x 8 lookbacks)...")
    diff_raw = compute_features(fc)
    diff_raw = np.nan_to_num(diff_raw, nan=0.0, posinf=0.0, neginf=0.0)

    common_idx = lb.index.intersection(fc.index)
    fc_pos = fc.index.get_indexer(common_idx)
    lb = lb.loc[common_idx]
    diff_raw = diff_raw[fc_pos]
    dates = common_idx

    direction_h8 = lb["direction_h8"].values
    valid_mask = (direction_h8 == 0) | (direction_h8 == 1)
    y_dir = np.zeros(len(direction_h8), dtype=np.float32)
    y_dir[direction_h8 == 0] = 1.0  # LONG=1

    y_mfe_up = lb[[f"mfe_up_{H}" for H in MFE_HORIZONS]].values.astype(np.float32) / 100.0
    y_mfe_down = lb[[f"mfe_down_{H}" for H in MFE_HORIZONS]].values.astype(np.float32) / 100.0

    splits = split_indices(dates, valid_mask)
    print(f"  train {len(splits['train'])} | val {len(splits['val'])} | test {len(splits['test'])}")

    # Scaler fit on TRAIN only
    scaler_mean = diff_raw[splits["train"]].mean(axis=0)
    scaler_std = diff_raw[splits["train"]].std(axis=0)
    scaler_std[scaler_std < 1e-8] = 1.0

    diff = (diff_raw - scaler_mean) / scaler_std
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

    # Reshape to [N, 8, 4] for LSTM
    def seq_split(idx):
        return diff[idx].reshape(-1, 8, 4), y_mfe_up[idx], y_mfe_down[idx], y_dir[idx]

    X_tr, ymu_tr, ymd_tr, ydir_tr = seq_split(splits["train"])
    X_va, ymu_va, ymd_va, ydir_va = seq_split(splits["val"])
    X_te, ymu_te, ymd_te, ydir_te = seq_split(splits["test"])

    print(f"\nTraining LSTMAttention (CPU)...")
    model, best_val_loss, epochs_used = train_once(
        X_tr, ymu_tr, ymd_tr, ydir_tr,
        X_va, ymu_va, ymd_va, ydir_va,
    )

    print("\nEvaluating...")
    model.eval()
    with torch.no_grad():
        _, _, logits_tr = model(torch.from_numpy(X_tr))
        _, _, logits_va = model(torch.from_numpy(X_va))
        _, _, logits_te = model(torch.from_numpy(X_te))
    probs_tr = torch.sigmoid(logits_tr).numpy()
    probs_va = torch.sigmoid(logits_va).numpy()
    probs_te = torch.sigmoid(logits_te).numpy()

    # MFE arrays for test-period magnitude metrics (protocol requires)
    # Attention uses H8 labels, so use mfe_up_8 / mfe_down_8 (not H96 like MLP)
    mfe_up_test = lb["mfe_up_8"].values[splits["test"]]
    mfe_down_test = lb["mfe_down_8"].values[splits["test"]]

    print(f"\nExporting to {OUT_DIR.relative_to(REPO_ROOT)}...")
    export(model, scaler_mean, scaler_std, OUT_DIR)

    print("\nLogging to MLflow via protocol-enforced runner...")
    tracking.init()
    run_id = None
    with run_experiment(
        experiment_name=EXPERIMENT,
        protocol_name="direction_prediction_v1",
        params={
            "model_type": "LSTMAttention",
            "architecture": f"LSTM({HIDDEN}) + attention(temp={TEMPERATURE}) + MFE heads + dir head",
            "features": "32 diffs (roc/rsi/rp/sma200) x 8 lookbacks",
            "label": "direction_h8 (H8)",
            "split_method": "date_based",
            "train_range": f"{TRAIN_RANGE[0]}..{TRAIN_RANGE[1]}",
            "val_range": f"{VAL_RANGE[0]}..{VAL_RANGE[1]}",
            "test_range": f"{TEST_RANGE[0]}..{TEST_RANGE[1]}",
            "scaler_fit": "train_only",
            "lr": LR, "batch_size": BATCH_SIZE, "hidden": HIDDEN, "dropout": DROPOUT,
            "temperature": TEMPERATURE, "patience": PATIENCE, "seed": SEED,
            "conf_long": CONF_LONG, "conf_short": CONF_SHORT,
            "epochs_used": epochs_used,
        },
        primary_metric="test_confident_accuracy",
        model_type="LSTMAttention",
        dataset_version="feature_cache_v2",
        notes=f"Honest attention retrain {date.today().isoformat()} — replaces leaky Colab version.",
    ) as run:
        run_id = run.mlflow_run_id

        conf_threshold_short_for_eval = 1 - CONF_SHORT

        m_tr = evaluate_direction_prediction(
            probs_tr, ydir_tr.astype(int),
            conf_threshold_long=CONF_LONG,
            conf_threshold_short=conf_threshold_short_for_eval,
            split_name="train",
        )
        m_va = evaluate_direction_prediction(
            probs_va, ydir_va.astype(int),
            conf_threshold_long=CONF_LONG,
            conf_threshold_short=conf_threshold_short_for_eval,
            split_name="val",
        )
        m_te = evaluate_direction_prediction(
            probs_te, ydir_te.astype(int),
            mfe_up_bps=mfe_up_test, mfe_down_bps=mfe_down_test,
            conf_threshold_long=CONF_LONG,
            conf_threshold_short=conf_threshold_short_for_eval,
            split_name="test",
        )
        run.log_metrics({**m_tr, **m_va, **m_te})
        run.log_metric("best_val_loss", round(best_val_loss, 6))

        run.set_tag("git_commit", git_commit())
        run.set_tag("data_15m_md5", dvc_hash(REPO_ROOT / "data/raw/BTCUSDT_15m_ohlcv.parquet.dvc"))
        run.set_tag("source", "honest_retrain_local_cpu")

        for f in sorted(OUT_DIR.glob("*")):
            run.log_artifact(str(f), artifact_subdir="model")

        print(f"  train acc {m_tr['train_accuracy']:.4f} conf_acc {m_tr['train_confident_accuracy']:.4f}")
        print(f"  val   acc {m_va['val_accuracy']:.4f} conf_acc {m_va['val_confident_accuracy']:.4f}")
        print(f"  TEST  acc {m_te['test_accuracy']:.4f} conf_acc {m_te['test_confident_accuracy']:.4f}")

    version = register_staging(run_id, "model")
    print(f"\nRegistered {MODEL_NAME} v{version} @staging.")

    # Training manifest for generic verifier
    import hashlib, json
    from datetime import datetime as _dt

    def _md5_of(path):
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
            "train": list(TRAIN_RANGE), "val": list(VAL_RANGE), "test": list(TEST_RANGE),
        },
        "scaler": {"fit_on": "train_only"},
        "feature_recipe": {
            "compute_fn": "engine.train_attention.compute_features",
            "source_parquet": str(CACHE_PATH.relative_to(REPO_ROOT)),
        },
        "label": "direction_h8 (H8)",
        "metrics": {
            "train_accuracy": m_tr.get("train_accuracy"),
            "val_accuracy": m_va.get("val_accuracy"),
            "test_accuracy": m_te.get("test_accuracy"),
            "test_confident_accuracy": m_te.get("test_confident_accuracy"),
            "test_f1": m_te.get("test_f1"),
            "test_precision_long": m_te.get("test_precision_long"),
            "test_recall_long": m_te.get("test_recall_long"),
        },
        "mlflow": {"run_id": run_id, "registered_version": int(version), "alias": "staging"},
    }
    (OUT_DIR / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {(OUT_DIR / 'training_manifest.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
