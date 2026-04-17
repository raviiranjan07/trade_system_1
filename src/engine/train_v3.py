"""Stage 4: Train ML V3 — Exit-Aware Direction Prediction.

Architecture: LSTM(4→hidden) + Attention + PnL heads + 3-class direction
Labels: exit-aware (simulated trade P&L with V2 exit rules on 1m ticks)
Loss: MSE(long_pnl) + MSE(short_pnl) + CrossEntropy(direction)

Run: PYTHONPATH=src python -m engine.train_v3
"""

import json
import logging
import subprocess
import sys
import time
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

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "data/features/direction_prediction/feature_cache.parquet"
LABELS_PATH = REPO_ROOT / "data/features/direction_prediction/exit_aware_labels.parquet"
OUT_DIR = REPO_ROOT / "models/ML_V3_staging"
METRICS_PATH = REPO_ROOT / "data/reports/v3_train_metrics.json"

logger = logging.getLogger(__name__)

# Load hyperparams from params.yaml
with open(REPO_ROOT / "configs/params.yaml") as _f:
    _cfg = yaml.safe_load(_f).get("ml_v3", {})
_t = _cfg.get("training", {})
_i = _cfg.get("inference", {})
_s = _cfg.get("split", {})

HIDDEN = _t.get("hidden", 128)
DROPOUT = _t.get("dropout", 0.5)
TEMPERATURE = _t.get("temperature", 0.5)
LR = _t.get("lr", 0.001)
BATCH_SIZE = _t.get("batch_size", 2048)
MAX_EPOCHS = _t.get("max_epochs", 100)
PATIENCE = _t.get("patience", 10)
SEED = _t.get("seed", 42)
LOSS_W_PNL = _t.get("loss_weight_pnl", 1.0)
LOSS_W_DIR = _t.get("loss_weight_dir", 1.0)
CONF_LONG = _i.get("conf_long", 0.50)
CONF_SHORT = _i.get("conf_short", 0.50)
TRAIN_RANGE = (_s.get("train_start", "2020-01-01"), _s.get("train_end", "2023-12-31"))
VAL_RANGE = (_s.get("val_start", "2024-01-01"), _s.get("val_end", "2024-12-31"))
TEST_RANGE = (_s.get("test_start", "2025-01-01"), _s.get("test_end", "2025-12-31"))

LOOKBACKS = [1, 2, 3, 4, 5, 6, 7, 8]
LONG, SHORT, SKIP = 0, 1, 2
N_CLASSES = 3
MODEL_NAME = "ML_V3"
EXPERIMENT = "ml_v3_exit_aware"


# =====================================================================
# MODEL
# =====================================================================

class LSTMAttentionV3(nn.Module):
    """LSTM + Attention + PnL heads + 3-class direction."""

    def __init__(self, input_size=4, hidden=128, dropout=0.5, temperature=0.5):
        super().__init__()
        self.hidden = hidden
        self.temperature = temperature
        self.lstm = nn.LSTM(input_size, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_score = nn.Linear(hidden, 1)
        self.h_long_pnl = nn.Linear(hidden, 1)
        self.h_short_pnl = nn.Linear(hidden, 1)
        self.h_dir = nn.Linear(hidden + 2, N_CLASSES)

    def forward(self, x):
        all_h, _ = self.lstm(x)
        scores = self.attn_score(all_h).squeeze(-1)
        attn_w = torch.softmax(scores / self.temperature, dim=1)
        attended = torch.bmm(attn_w.unsqueeze(1), all_h).squeeze(1)
        attended = self.dropout(attended)

        p_long = self.h_long_pnl(attended).squeeze(-1)
        p_short = self.h_short_pnl(attended).squeeze(-1)

        dir_input = torch.cat([attended, p_long.unsqueeze(1), p_short.unsqueeze(1)], dim=1)
        p_dir = self.h_dir(dir_input)

        return p_long, p_short, p_dir


class V3DirectionOnly(nn.Module):
    """ONNX export wrapper — outputs all 3 heads."""

    def __init__(self, full_model):
        super().__init__()
        self.model = full_model

    def forward(self, x):
        p_long, p_short, p_dir = self.model(x)
        return p_long, p_short, p_dir


# =====================================================================
# DATASET
# =====================================================================

class V3Dataset(Dataset):
    def __init__(self, X, y_long_pnl, y_short_pnl, y_dir):
        self.X = torch.from_numpy(X)
        self.y_long = torch.from_numpy(y_long_pnl)
        self.y_short = torch.from_numpy(y_short_pnl)
        self.y_dir = torch.from_numpy(y_dir)

    def __len__(self):
        return len(self.y_dir)

    def __getitem__(self, i):
        return self.X[i], self.y_long[i], self.y_short[i], self.y_dir[i]


# =====================================================================
# FEATURES
# =====================================================================

def compute_features(fc):
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


# =====================================================================
# TRAINING
# =====================================================================

def train_once(X_tr, yl_tr, ys_tr, yd_tr, X_va, yl_va, ys_va, yd_va):
    model = LSTMAttentionV3(input_size=4, hidden=HIDDEN, dropout=DROPOUT, temperature=TEMPERATURE)
    mse = nn.MSELoss()
    # Class weights: penalize SKIP over-prediction. SKIP is ~35% of labels
    # but model defaults to predicting it ~58%. Upweight LONG/SHORT to force
    # the model to learn directional patterns instead of the easy SKIP path.
    dir_counts = np.bincount(yd_tr.astype(int), minlength=3).astype(np.float32)
    dir_weights = 1.0 / (dir_counts + 1)
    dir_weights = dir_weights / dir_weights.min()  # normalize so smallest = 1.0
    logger.info("  Class weights: LONG=%.2f SHORT=%.2f SKIP=%.2f", *dir_weights)
    ce = nn.CrossEntropyLoss(weight=torch.from_numpy(dir_weights))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    loader = DataLoader(V3Dataset(X_tr, yl_tr, ys_tr, yd_tr),
                        batch_size=BATCH_SIZE, shuffle=True)

    best_vl = float("inf")
    best_state = None
    patience_ctr = 0
    epochs_used = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_used = epoch
        model.train()
        tr_sum = 0.0
        for xb, yl, ys, yd in loader:
            p_long, p_short, p_dir = model(xb)
            loss = (LOSS_W_PNL * (mse(p_long, yl) + mse(p_short, ys))
                    + LOSS_W_DIR * ce(p_dir, yd.long()))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_sum += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            p_l_v, p_s_v, p_d_v = model(torch.from_numpy(X_va))
            vl = (LOSS_W_PNL * (mse(p_l_v, torch.from_numpy(yl_va))
                                + mse(p_s_v, torch.from_numpy(ys_va)))
                  + LOSS_W_DIR * ce(p_d_v, torch.from_numpy(yd_va).long())).item()

        scheduler.step(vl)

        if vl < best_vl:
            best_vl = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, epochs_used, best_vl


# =====================================================================
# EVALUATION
# =====================================================================

def evaluate(model, X, y_long, y_short, y_dir, prefix="test"):
    model.eval()
    # Batch inference to avoid OOM on large datasets
    batch_size = 4096
    all_long, all_short, all_dir = [], [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size])
            pl, ps, pd = model(xb)
            all_long.append(pl)
            all_short.append(ps)
            all_dir.append(pd)
    p_long = torch.cat(all_long)
    p_short = torch.cat(all_short)
    p_dir = torch.cat(all_dir)

    # Direction metrics
    probs = torch.softmax(p_dir, dim=1).numpy()
    pred = probs.argmax(axis=1)
    y = y_dir.astype(int)
    acc = float((pred == y).mean())

    # Per-class recall
    metrics = {f"{prefix}_accuracy_3class": round(acc, 4)}
    for cls, name in [(LONG, "long"), (SHORT, "short"), (SKIP, "skip")]:
        mask = y == cls
        if mask.sum() > 0:
            recall = float((pred[mask] == cls).mean())
            precision = float((y[pred == cls] == cls).mean()) if (pred == cls).sum() > 0 else 0.0
        else:
            recall, precision = 0.0, 0.0
        metrics[f"{prefix}_{name}_recall"] = round(recall, 4)
        metrics[f"{prefix}_{name}_precision"] = round(precision, 4)

    # Confident signals
    n_conf_long = int((probs[:, LONG] >= CONF_LONG).sum())
    n_conf_short = int((probs[:, SHORT] >= CONF_SHORT).sum())
    metrics[f"{prefix}_n_confident"] = n_conf_long + n_conf_short
    metrics[f"{prefix}_n_confident_long"] = n_conf_long
    metrics[f"{prefix}_n_confident_short"] = n_conf_short

    # Confident accuracy
    conf_mask = (probs[:, LONG] >= CONF_LONG) | (probs[:, SHORT] >= CONF_SHORT)
    if conf_mask.sum() > 0:
        conf_acc = float((pred[conf_mask] == y[conf_mask]).mean())
        metrics[f"{prefix}_confident_accuracy"] = round(conf_acc, 4)
    else:
        metrics[f"{prefix}_confident_accuracy"] = 0.0

    # PnL regression MSE
    pnl_mse_long = float(nn.MSELoss()(p_long, torch.from_numpy(y_long)).item())
    pnl_mse_short = float(nn.MSELoss()(p_short, torch.from_numpy(y_short)).item())
    metrics[f"{prefix}_pnl_mse_long"] = round(pnl_mse_long, 4)
    metrics[f"{prefix}_pnl_mse_short"] = round(pnl_mse_short, 4)

    return metrics


# =====================================================================
# EXPORT
# =====================================================================

def export(model, scaler_mean, scaler_std, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save torch checkpoint
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "v3_model.pt")

    # ONNX export with all 3 outputs
    model.eval()
    wrapper = V3DirectionOnly(model)
    wrapper.eval()
    dummy = torch.randn(1, 8, 4)
    torch.onnx.export(
        wrapper, dummy, str(out_dir / "v3_model.onnx"),
        input_names=["features"],
        output_names=["long_pnl", "short_pnl", "direction"],
        dynamic_axes={"features": {0: "batch"}, "long_pnl": {0: "batch"},
                       "short_pnl": {0: "batch"}, "direction": {0: "batch"}},
        opset_version=14,
    )

    # Scaler
    np.savez(out_dir / "scaler.npz", mean=scaler_mean, std=scaler_std)
    logger.info("Exported to %s", out_dir)


# =====================================================================
# MAIN
# =====================================================================

def main():
    t0 = time.time()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logger.info("Loading data...")
    fc = pd.read_parquet(CACHE_PATH)
    lb = pd.read_parquet(LABELS_PATH)

    logger.info("Computing features...")
    diff_raw = compute_features(fc)
    diff_raw = np.nan_to_num(diff_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Align features and labels
    common_idx = lb.index.intersection(fc.index)
    fc_pos = fc.index.get_indexer(common_idx)
    lb = lb.loc[common_idx]
    diff_raw = diff_raw[fc_pos]
    dates = common_idx

    # Labels
    y_dir = lb["direction"].values.astype(np.int64)
    valid_mask = np.isin(y_dir, [LONG, SHORT, SKIP])

    # PnL targets (scale to reasonable range for MSE)
    y_long_pnl = lb["long_net_bps"].values.astype(np.float32) / 100.0
    y_short_pnl = lb["short_net_bps"].values.astype(np.float32) / 100.0
    y_long_pnl = np.nan_to_num(y_long_pnl, nan=0.0)
    y_short_pnl = np.nan_to_num(y_short_pnl, nan=0.0)

    # Splits
    def in_range(lo, hi):
        return (dates >= lo) & (dates <= hi) & valid_mask
    splits = {
        "train": np.where(in_range(*TRAIN_RANGE))[0],
        "val": np.where(in_range(*VAL_RANGE))[0],
        "test": np.where(in_range(*TEST_RANGE))[0],
    }
    logger.info("  train %d | val %d | test %d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))

    # Scaler (train only)
    scaler_mean = diff_raw[splits["train"]].mean(axis=0)
    scaler_std = diff_raw[splits["train"]].std(axis=0)
    scaler_std[scaler_std < 1e-8] = 1.0
    diff = (diff_raw - scaler_mean) / scaler_std
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

    def get_split(idx):
        return (diff[idx].reshape(-1, 8, 4),
                y_long_pnl[idx], y_short_pnl[idx], y_dir[idx])

    X_tr, yl_tr, ys_tr, yd_tr = get_split(splits["train"])
    X_va, yl_va, ys_va, yd_va = get_split(splits["val"])
    X_te, yl_te, ys_te, yd_te = get_split(splits["test"])

    # Label distribution
    for name, idx in splits.items():
        d = y_dir[idx]
        logger.info("  %s: LONG=%d SHORT=%d SKIP=%d", name,
                    (d == LONG).sum(), (d == SHORT).sum(), (d == SKIP).sum())

    # Train
    logger.info("\nTraining LSTMAttentionV3 (hidden=%d, temp=%.1f, loss_pnl=%.1f, loss_dir=%.1f)...",
                HIDDEN, TEMPERATURE, LOSS_W_PNL, LOSS_W_DIR)
    model, epochs_used, val_loss = train_once(
        X_tr, yl_tr, ys_tr, yd_tr,
        X_va, yl_va, ys_va, yd_va,
    )
    logger.info("  Best val loss: %.4f after %d epochs", val_loss, epochs_used)

    # Evaluate
    logger.info("\nEvaluating...")
    m_train = evaluate(model, X_tr, yl_tr, ys_tr, yd_tr, "train")
    m_val = evaluate(model, X_va, yl_va, ys_va, yd_va, "val")
    m_test = evaluate(model, X_te, yl_te, ys_te, yd_te, "test")

    logger.info("  Train: acc=%.1f%% | conf_acc=%.1f%% (%d signals)",
                m_train["train_accuracy_3class"] * 100,
                m_train["train_confident_accuracy"] * 100,
                m_train["train_n_confident"])
    logger.info("  Val:   acc=%.1f%% | conf_acc=%.1f%% (%d signals)",
                m_val["val_accuracy_3class"] * 100,
                m_val["val_confident_accuracy"] * 100,
                m_val["val_n_confident"])
    logger.info("  Test:  acc=%.1f%% | conf_acc=%.1f%% (%d signals)",
                m_test["test_accuracy_3class"] * 100,
                m_test["test_confident_accuracy"] * 100,
                m_test["test_n_confident"])

    # Export
    logger.info("\nExporting to %s", OUT_DIR.relative_to(REPO_ROOT))
    export(model, scaler_mean, scaler_std, OUT_DIR)

    # MLflow logging
    logger.info("\nLogging to MLflow...")
    tracking.init()

    all_metrics = {**m_train, **m_val, **m_test,
                   "epochs": epochs_used, "val_loss": round(val_loss, 4)}

    with mlflow.start_run(run_name=f"v3_{date.today().isoformat()}"):
        mlflow.set_experiment(EXPERIMENT)
        mlflow.log_params({
            "model_type": "LSTMAttentionV3",
            "architecture": f"LSTM({HIDDEN}) + attention(temp={TEMPERATURE}) + PnL heads + 3-class dir",
            "features": "32 diffs (roc/rsi/rp/sma200) x 8 lookbacks",
            "labels": "exit_aware (V2 rules, 1m tick simulation)",
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "temperature": TEMPERATURE,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "loss_weight_pnl": LOSS_W_PNL,
            "loss_weight_dir": LOSS_W_DIR,
            "conf_long": CONF_LONG,
            "conf_short": CONF_SHORT,
            "seed": SEED,
        })
        mlflow.log_metrics(all_metrics)
        mlflow.log_artifacts(str(OUT_DIR), artifact_path="model")

    # Save metrics JSON for DVC
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    elapsed = time.time() - t0
    logger.info("\nDone. Total time: %.1f min", elapsed / 60)
    logger.info("Metrics saved to %s", METRICS_PATH)


if __name__ == "__main__":
    main()
