"""
COLAB: Train Attention temp=0.5 for PRODUCTION deployment.

Trains on ALL data 2020-2025 (random 90/10 split).
Exports: ONNX model + PyTorch weights + scaler
Saves to Google Drive for download.

Label: direction_h8 (H8, binary LONG/SHORT)
Features: 4 diffs × 8 lookbacks = 32 as [8,4] sequence
Architecture: LSTMAttention(temp=0.5) + connected MFE heads

Usage: paste into Colab cell, T4 GPU runtime, run.
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
CACHE_PATH  = "/content/drive/MyDrive/L2-003/feature_cache.parquet"
LABELS_PATH = "/content/drive/MyDrive/L2-003/labels.parquet"

# Output — save to Drive so you can download
OUT_DIR = Path("/content/drive/MyDrive/L2-003/models/attention_temp05")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACKS = [1, 2, 3, 4, 5, 6, 7, 8]
MFE_HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8]

BATCH_SIZE = 2048
MAX_EPOCHS = 100
PATIENCE   = 10
LR         = 0.001
HIDDEN     = 128
DROPOUT    = 0.5
TEMPERATURE = 0.5
SEED       = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading data...")
fc = pd.read_parquet(CACHE_PATH)
lb = pd.read_parquet(LABELS_PATH)
print(f"  feature_cache: {fc.shape}")
print(f"  labels: {lb.shape}")

close = fc["close"].values.astype(np.float64)
rsi7 = fc["rsi7"].values.astype(np.float64)
rp = fc["range_position"].values.astype(np.float64)
sma200 = fc["sma200_dist_pct"].values.astype(np.float64)

print("Computing diff features...")
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

diff_arr_raw = np.column_stack(diff_list).astype(np.float32)

common_idx = lb.index.intersection(fc.index)
lb = lb.loc[common_idx]
label_pos = np.array([fc.index.get_loc(dt) for dt in common_idx], dtype=np.int32)

# Scaler on ALL data (production — same approach as V1.5)
diff_arr_raw = np.nan_to_num(diff_arr_raw, nan=0.0, posinf=0.0, neginf=0.0)
scaler_mean = diff_arr_raw.mean(axis=0)
scaler_std = diff_arr_raw.std(axis=0)
scaler_std[scaler_std < 1e-8] = 1.0
diff_arr = (diff_arr_raw - scaler_mean) / scaler_std
diff_aligned = diff_arr[label_pos]

# Labels
y_mfe_up = lb[[f"mfe_up_{H}" for H in MFE_HORIZONS]].values.astype(np.float32) / 100.0
y_mfe_down = lb[[f"mfe_down_{H}" for H in MFE_HORIZONS]].values.astype(np.float32) / 100.0

direction_h8 = lb["direction_h8"].values
valid_mask = (direction_h8 == 0) | (direction_h8 == 1)
y_dir = np.zeros(len(direction_h8), dtype=np.float32)
y_dir[direction_h8 == 0] = 1.0

# Random 90/10 split on ALL valid data (same as V1.5 production)
valid_indices = np.where(valid_mask)[0]
np.random.seed(SEED)
np.random.shuffle(valid_indices)
split_point = int(len(valid_indices) * 0.90)
train_idx = valid_indices[:split_point]
val_idx = valid_indices[split_point:]

print(f"  Total valid: {len(valid_indices)} | Train: {len(train_idx)} (90%) | Val: {len(val_idx)} (10%)")

# =============================================================================
# DATASET + MODEL
# =============================================================================
class SeqDS(Dataset):
    def __init__(self, idx_array):
        self.idx = idx_array
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        n = self.idx[i]
        seq = diff_aligned[n].reshape(len(LOOKBACKS), 4)
        return (torch.from_numpy(seq),
                torch.from_numpy(y_mfe_up[n]),
                torch.from_numpy(y_mfe_down[n]),
                torch.tensor(y_dir[n]))

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

# Direction-only wrapper for ONNX export (no MFE heads in output)
class AttentionDirectionOnly(nn.Module):
    """Wraps LSTMAttention to output only direction logit for ONNX export."""
    def __init__(self, full_model):
        super().__init__()
        self.model = full_model

    def forward(self, x):
        _, _, p_dir = self.model(x)
        return p_dir

# =============================================================================
# TRAIN
# =============================================================================
print("\nTraining...")
torch.manual_seed(SEED)
np.random.seed(SEED)

model = LSTMAttention(input_size=4, hidden=HIDDEN, dropout=DROPOUT, temperature=TEMPERATURE).to(DEVICE)
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

mse_fn = nn.MSELoss()
bce_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

train_loader = DataLoader(SeqDS(train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(SeqDS(val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

best_vl = float("inf")
best_state = None
pat = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    for x, mu, md, d in train_loader:
        x, mu, md, d = x.to(DEVICE), mu.to(DEVICE), md.to(DEVICE), d.to(DEVICE)
        p_mu, p_md, p_dir = model(x)
        loss = mse_fn(p_mu, mu) + mse_fn(p_md, md) + 5.0 * bce_fn(p_dir, d)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item()
    tr_loss /= len(train_loader)

    model.eval()
    vl = 0.0
    with torch.no_grad():
        for x, mu, md, d in val_loader:
            x, mu, md, d = x.to(DEVICE), mu.to(DEVICE), md.to(DEVICE), d.to(DEVICE)
            p_mu, p_md, p_dir = model(x)
            vl += (mse_fn(p_mu, mu) + mse_fn(p_md, md) + 5.0 * bce_fn(p_dir, d)).item()
    vl /= len(val_loader)
    scheduler.step(vl)

    if vl < best_vl - 1e-5:
        best_vl = vl
        pat = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch <= 3:
            print(f"  Epoch {epoch:3d}: train={tr_loss:.4f} val={vl:.4f} ***")
    else:
        pat += 1
        if pat >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
model.eval()
model = model.cpu()  # move to CPU for export

# =============================================================================
# QUICK VALIDATION
# =============================================================================
print("\nQuick validation on val set...")
all_dir = []
all_y = []
with torch.no_grad():
    for x, mu, md, d in val_loader:
        _, _, p_dir = model(x)
        all_dir.append(p_dir.numpy())
        all_y.append(d.numpy())

p_dir = np.concatenate(all_dir)
y = np.concatenate(all_y)
probs = 1 / (1 + np.exp(-p_dir))

acc = ((probs > 0.5).astype(int) == y).mean()
conf = (probs >= 0.60) | (probs <= 0.40)
n_conf = conf.sum()
conf_acc = (np.where(probs[conf] > 0.5, 1, 0) == y[conf]).mean() if n_conf > 0 else 0
conf_long = (probs >= 0.60).sum()
conf_short = (probs <= 0.40).sum()

print(f"  Val accuracy: {acc*100:.1f}%")
print(f"  Val confident accuracy: {conf_acc*100:.1f}% ({n_conf} bars: {conf_long}L + {conf_short}S)")

# =============================================================================
# SAVE PyTorch weights
# =============================================================================
pt_path = OUT_DIR / "attention_model.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "config": {
        "input_size": 4,
        "hidden": HIDDEN,
        "dropout": DROPOUT,
        "temperature": TEMPERATURE,
        "lookbacks": LOOKBACKS,
        "mfe_horizons": MFE_HORIZONS,
        "label": "direction_h8",
        "features": ["roc_diff", "rsi_diff", "rp_diff", "sma200_diff"],
        "scaler": "all_data_2020_2025",
    },
    "val_loss": best_vl,
}, pt_path)
print(f"\n  PyTorch saved → {pt_path}")

# =============================================================================
# SAVE scaler
# =============================================================================
scaler_path = OUT_DIR / "scaler.npz"
np.savez(scaler_path, mean=scaler_mean, std=scaler_std)
print(f"  Scaler saved → {scaler_path}")

# =============================================================================
# EXPORT to ONNX (direction-only output for production inference)
# =============================================================================
onnx_path = OUT_DIR / "attention_model.onnx"

export_model = AttentionDirectionOnly(model)
export_model.eval()

dummy_input = torch.randn(1, 8, 4)  # [batch, 8 steps, 4 features]

torch.onnx.export(
    export_model,
    dummy_input,
    str(onnx_path),
    input_names=["features"],
    output_names=["logit"],
    dynamic_axes={
        "features": {0: "batch"},
        "logit": {0: "batch"},
    },
    opset_version=18,
)
print(f"  ONNX exported → {onnx_path}")

# Verify ONNX works
import onnxruntime as ort
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
test_input = np.random.randn(1, 8, 4).astype(np.float32)
test_output = sess.run(["logit"], {"features": test_input})[0]
print(f"  ONNX verification: input shape {test_input.shape} → output shape {test_output.shape}, value={test_output.item():.4f}")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*70}")
print(f"PRODUCTION MODEL SAVED")
print(f"{'='*70}")
print(f"  Location: {OUT_DIR}")
print(f"  Files:")
print(f"    attention_model.pt    — PyTorch weights + config")
print(f"    attention_model.onnx  — ONNX for production inference")
print(f"    scaler.npz            — normalization params (32 features)")
print(f"")
print(f"  Architecture: LSTMAttention(4→128, temp=0.5) + connected MFE heads")
print(f"  Input: [batch, 8, 4] — 4 diff features × 8 lookback steps")
print(f"  Output: single logit → sigmoid → probability")
print(f"  Label: direction_h8 (H8)")
print(f"  Training: ALL data 2020-2025, random 90/10 split")
print(f"  Scaler: fit on ALL data")
print(f"")
print(f"  Val results: {acc*100:.1f}% overall, {conf_acc*100:.1f}% confident ({n_conf} bars)")
print(f"")
print(f"  Download these 3 files to: models/direction_attention/")
print(f"  Then update bot to load the second model.")
