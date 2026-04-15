"""L2-003: Test MLP model as filter for V1.4 trades

Runs V1.4 backtest, then for each trade's entry bar:
1. Compute 10 features (roc1-8 + range_position + rsi7)
2. Get model prediction (probability of LONG)
3. Check if model agrees with V1.4's direction
4. Compare outcomes: model agrees vs disagrees

Run: PYTHONPATH=src python experiments/layer2/L2-003/L2_003_v14_filter_test.py
"""

import sys
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# V1.4 imports
from engine.config.loader import load_config
from engine.backtest import run_backtest

logging.basicConfig(level=logging.WARNING)

# =============================================================================
# CONFIG
# =============================================================================
BASE = Path("experiments/layer2/L2-003")
CACHE_PATH = BASE / "feature_cache.parquet"
MODEL_PATH = BASE / "mlp_model_best.pt"  # we'll save best model here

# =============================================================================
# STEP 1: Run V1.4 backtest to get all trades
# =============================================================================
print("="*70)
print("STEP 1: Running V1.4 backtest (2024-2025)")
print("="*70)

config = load_config()
trades = run_backtest(config, start="2024-01-01", end="2025-12-31")
tdf = pd.DataFrame([asdict(t) for t in trades])

print(f"\n  Total trades: {len(tdf)}")
print(f"  Win rate: {(tdf['net_profit_bps'] > 0).mean()*100:.1f}%")
print(f"  Total bps: {tdf['net_profit_bps'].sum():.0f}")
print(f"  Signal types: {tdf['signal_type'].value_counts().to_dict()}")

# =============================================================================
# STEP 2: Load feature cache and compute ROC features
# =============================================================================
print(f"\n{'='*70}")
print("STEP 2: Computing features at V1.4 entry bars")
print("="*70)

fc = pd.read_parquet(CACHE_PATH)
close = fc["close"].values.astype(np.float64)

# Compute roc1-roc8
for n in range(1, 9):
    roc = np.zeros(len(close), dtype=np.float32)
    roc[n:] = ((close[n:] - close[:-n]) / close[:-n] * 10000).astype(np.float32)
    fc[f"roc{n}"] = roc

feat_cols = [f"roc{n}" for n in range(1, 9)] + ["range_position", "rsi7"]
F = len(feat_cols)

# Normalize using 2020-2023 training stats (same as model was trained)
feat_arr = fc[feat_cols].values.astype(np.float32)
train_mask = (fc.index >= "2020-01-01") & (fc.index <= "2023-12-31")
scaler_mean = feat_arr[train_mask].mean(axis=0)
scaler_std = feat_arr[train_mask].std(axis=0)
scaler_std[scaler_std < 1e-8] = 1.0
feat_arr_norm = (feat_arr - scaler_mean) / scaler_std
feat_arr_norm = np.nan_to_num(feat_arr_norm, nan=0.0, posinf=0.0, neginf=0.0)

print(f"  Features: {feat_cols}")

# =============================================================================
# STEP 3: Train model (same as best config: 2020-2023 train)
# =============================================================================
print(f"\n{'='*70}")
print("STEP 3: Training MLP model (2020-2023)")
print("="*70)

# Load labels for training
lb = pd.read_parquet(BASE / "labels.parquet")
common_idx = lb.index.intersection(fc.index)
lb = lb.loc[common_idx]

direction = lb["direction"].values
valid_mask = (direction == 0) | (direction == 1)
y_binary = np.zeros(len(direction), dtype=np.float32)
y_binary[direction == 0] = 1.0
label_dates = np.array(common_idx)

# Aligned feature array
fc_aligned_idx = [fc.index.get_loc(dt) for dt in common_idx]
feat_aligned = feat_arr_norm[fc_aligned_idx]

train_split = (label_dates >= np.datetime64("2020-01-01")) & (label_dates <= np.datetime64("2023-12-31")) & valid_mask
val_split = (label_dates >= np.datetime64("2024-01-01")) & (label_dates <= np.datetime64("2024-06-30")) & valid_mask

print(f"  Train: {train_split.sum()} | Val: {val_split.sum()}")

# Quick model definition and training
class MLPBinaryDir(nn.Module):
    def __init__(self, input_size, hidden=128, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        h = torch.relu(self.fc2(h))
        h = self.dropout(h)
        return self.out(h).squeeze(-1)

from torch.utils.data import Dataset, DataLoader

class SimpleDS(Dataset):
    def __init__(self, idx):
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        n = self.idx[i]
        return torch.from_numpy(feat_aligned[n]), torch.tensor(y_binary[n])

DEVICE = torch.device("cpu")
model = MLPBinaryDir(input_size=F).to(DEVICE)
bce = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

train_loader = DataLoader(SimpleDS(np.where(train_split)[0]), batch_size=512, shuffle=True)
val_loader = DataLoader(SimpleDS(np.where(val_split)[0]), batch_size=512, shuffle=False)

best_val_loss = float("inf")
patience_ctr = 0

for epoch in range(1, 101):
    model.train()
    for batch in train_loader:
        x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
        loss = bce(model(x), y)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    vl = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            vl += bce(model(x), y).item()
    vl /= len(val_loader)
    scheduler.step(vl)

    if vl < best_val_loss - 1e-5:
        best_val_loss = vl
        patience_ctr = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_ctr += 1
        if patience_ctr >= 10:
            print(f"  Trained {epoch} epochs, best val_loss: {best_val_loss:.4f}")
            break

model.load_state_dict(best_state)
model.eval()
print(f"  Model ready")

# =============================================================================
# STEP 4: Score every V1.4 trade with model
# =============================================================================
print(f"\n{'='*70}")
print("STEP 4: Scoring V1.4 trades with model")
print("="*70)

# For each trade, find the signal bar in feature_cache and get prediction
results = []
for _, trade in tdf.iterrows():
    # Signal time is when V1.4 decided to trade
    signal_time = pd.Timestamp(trade['signal_time'])
    entry_time = pd.Timestamp(trade['entry_time'])

    # Find closest bar in feature cache to signal time
    if signal_time in fc.index:
        bar_idx = fc.index.get_loc(signal_time)
    else:
        # Find nearest bar
        diffs = np.abs(fc.index - signal_time)
        bar_idx = diffs.argmin()

    # Get normalized features
    features = feat_arr_norm[bar_idx]
    x = torch.from_numpy(features).unsqueeze(0)

    with torch.no_grad():
        logit = model(x).item()
    prob_long = 1 / (1 + np.exp(-logit))

    # V1.4 direction
    v14_dir = trade['direction']  # "LONG" or "SHORT"
    v14_is_long = v14_dir == "LONG"

    # Model agrees?
    if v14_is_long:
        model_agrees = prob_long > 0.50
        model_confidence = prob_long
    else:
        model_agrees = prob_long < 0.50
        model_confidence = 1 - prob_long

    results.append({
        'signal_time': signal_time,
        'entry_time': entry_time,
        'direction': v14_dir,
        'signal_type': trade['signal_type'],
        'net_profit_bps': trade['net_profit_bps'],
        'mfe_bps': trade['mfe_bps'],
        'exit_reason': trade['exit_reason'],
        'prob_long': prob_long,
        'model_agrees': model_agrees,
        'model_confidence': model_confidence,
        'is_winner': trade['net_profit_bps'] > 0,
    })

rdf = pd.DataFrame(results)

# =============================================================================
# STEP 5: Compare model agrees vs disagrees
# =============================================================================
print(f"\n{'='*70}")
print("STEP 5: Model agrees vs disagrees — trade outcomes")
print("="*70)

agrees = rdf[rdf['model_agrees']]
disagrees = rdf[~rdf['model_agrees']]

print(f"\n  ALL V1.4 trades: {len(rdf)}")
print(f"    Win rate: {rdf['is_winner'].mean()*100:.1f}%")
print(f"    Total bps: {rdf['net_profit_bps'].sum():.0f}")
print(f"    Avg bps/trade: {rdf['net_profit_bps'].mean():.1f}")

print(f"\n  Model AGREES ({len(agrees)} trades, {len(agrees)/len(rdf)*100:.0f}%):")
print(f"    Win rate: {agrees['is_winner'].mean()*100:.1f}%")
print(f"    Total bps: {agrees['net_profit_bps'].sum():.0f}")
print(f"    Avg bps/trade: {agrees['net_profit_bps'].mean():.1f}")

print(f"\n  Model DISAGREES ({len(disagrees)} trades, {len(disagrees)/len(rdf)*100:.0f}%):")
if len(disagrees) > 0:
    print(f"    Win rate: {disagrees['is_winner'].mean()*100:.1f}%")
    print(f"    Total bps: {disagrees['net_profit_bps'].sum():.0f}")
    print(f"    Avg bps/trade: {disagrees['net_profit_bps'].mean():.1f}")
else:
    print(f"    No trades — model always agrees")

# =============================================================================
# STEP 6: By confidence level
# =============================================================================
print(f"\n{'='*70}")
print("STEP 6: Trade outcomes by model confidence")
print("="*70)

print(f"\n  {'Confidence':<15s} | {'N trades':>8s} {'Win%':>6s} {'Total bps':>10s} {'Avg bps':>8s}")
print(f"  {'-'*15}-+-{'-'*8}-{'-'*6}-{'-'*10}-{'-'*8}")

conf_buckets = [(0.50, 0.52), (0.52, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.0)]
for lo, hi in conf_buckets:
    mask = (rdf['model_confidence'] >= lo) & (rdf['model_confidence'] < hi)
    n = mask.sum()
    if n > 0:
        win = rdf.loc[mask, 'is_winner'].mean() * 100
        total = rdf.loc[mask, 'net_profit_bps'].sum()
        avg = rdf.loc[mask, 'net_profit_bps'].mean()
        print(f"  conf [{lo:.2f}-{hi:.2f}) | {n:8d} {win:5.1f}% {total:10.0f} {avg:8.1f}")

# =============================================================================
# STEP 7: By signal type
# =============================================================================
print(f"\n{'='*70}")
print("STEP 7: Model agreement by V1.4 signal type")
print("="*70)

for sig_type in rdf['signal_type'].unique():
    st_df = rdf[rdf['signal_type'] == sig_type]
    st_agrees = st_df[st_df['model_agrees']]
    st_disagrees = st_df[~st_df['model_agrees']]

    print(f"\n  {sig_type}: {len(st_df)} trades")
    print(f"    All:       win={st_df['is_winner'].mean()*100:.1f}%  bps={st_df['net_profit_bps'].sum():.0f}")
    if len(st_agrees) > 0:
        print(f"    Agrees:    win={st_agrees['is_winner'].mean()*100:.1f}%  bps={st_agrees['net_profit_bps'].sum():.0f}  n={len(st_agrees)}")
    if len(st_disagrees) > 0:
        print(f"    Disagrees: win={st_disagrees['is_winner'].mean()*100:.1f}%  bps={st_disagrees['net_profit_bps'].sum():.0f}  n={len(st_disagrees)}")

# =============================================================================
# STEP 8: What if we filtered?
# =============================================================================
print(f"\n{'='*70}")
print("STEP 8: Hypothetical — what if we skipped model-disagree trades?")
print("="*70)

for threshold in [0.50, 0.52, 0.55, 0.60]:
    filtered = rdf[rdf['model_confidence'] >= threshold]
    if len(filtered) > 0:
        win = filtered['is_winner'].mean() * 100
        total = filtered['net_profit_bps'].sum()
        avg = filtered['net_profit_bps'].mean()
        print(f"  conf>={threshold:.2f}: {len(filtered):3d} trades, win={win:.1f}%, total={total:.0f} bps, avg={avg:.1f} bps/trade")
