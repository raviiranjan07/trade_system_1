"""Stage 6: End-to-end S/R + Base Features — Training

Architecture:
  S/R(15) → Dense(8) → Dense(3) → zone_context
  base(3) + zone_context(3) → Dense(32) → Dense(4) → LONG/SHORT/BOTH/SKIP

Run: PYTHONPATH=src python experiments/brain/SR/stage6_train.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === Load data ===
print("=" * 60)
print("STAGE 6: End-to-End Training")
print("=" * 60)

print("\nLoading data...")
train = np.load("experiments/brain/SR/datasets_every_bar/train_stage6.npz")
val = np.load("experiments/brain/SR/datasets_every_bar/val_stage6.npz")
test = np.load("experiments/brain/SR/datasets_every_bar/test_stage6.npz")

def load_split(data):
    sr_dyn = data["X_sr_dynamic"]   # (N, 11)
    sr_sta = data["X_sr_static"]    # (N, 6)
    base = data["X_base"]           # (N, 3)
    Y = data["Y_h25"]              # (N,)

    sr = np.hstack([sr_dyn, sr_sta])  # (N, 17) but we use 15 (drop 2 redundant)
    return sr, base, Y

sr_train, base_train, Y_train = load_split(train)
sr_val, base_val, Y_val = load_split(val)
sr_test, base_test, Y_test = load_split(test)

print(f"  Train: {len(Y_train)} | Val: {len(Y_val)} | Test: {len(Y_test)}")
print(f"  S/R features: {sr_train.shape[1]} | Base features: {base_train.shape[1]}")

# === Normalize ===
print("\nNormalizing (z-score, fit on train)...")

# S/R features
sr_mean = np.nanmean(sr_train, axis=0)
sr_std = np.nanstd(sr_train, axis=0)
sr_std[sr_std == 0] = 1.0

sr_train = (sr_train - sr_mean) / sr_std
sr_val = (sr_val - sr_mean) / sr_std
sr_test = (sr_test - sr_mean) / sr_std

# Base features
base_mean = np.nanmean(base_train, axis=0)
base_std = np.nanstd(base_train, axis=0)
base_std[base_std == 0] = 1.0

base_train = (base_train - base_mean) / base_std
base_val = (base_val - base_mean) / base_std
base_test = (base_test - base_mean) / base_std

# Replace NaN/inf
for arr in [sr_train, sr_val, sr_test, base_train, base_val, base_test]:
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

print("  S/R mean range: [%.2f, %.2f]" % (sr_mean.min(), sr_mean.max()))
print("  Base mean range: [%.2f, %.2f]" % (base_mean.min(), base_mean.max()))

# === Feature separation ===
print("\n" + "=" * 60)
print("FEATURE SEPARATION (H25 label)")
print("=" * 60)

sr_names = ["dist_to_zone_pct", "support_width_pct", "res_width_pct",
            "support_retest", "resistance_retest", "zone_width",
            "recovery_up_pct", "recovery_down_pct",
            "speed_short", "speed_mid", "speed_long",
            "bounce_ratio", "recent_bounce_ratio", "pressure",
            "bars_since_touch", "touch_count_scaled", "level_type_binary"]
base_names = ["roc", "rsi7", "range_position"]

all_X = np.hstack([sr_train, base_train])
all_names = sr_names + base_names

print("\n%-25s | %10s %10s %10s %10s | %8s" % ("Feature", "LONG", "SHORT", "BOTH", "SKIP", "MaxDiff%"))
print("-" * 85)
for i in range(all_X.shape[1]):
    name = all_names[i] if i < len(all_names) else f"feat_{i}"
    vals = {}
    for label, lname in [(0, "LONG"), (1, "SHORT"), (2, "BOTH"), (3, "SKIP")]:
        mask = Y_train == label
        if mask.sum() > 0:
            vals[lname] = all_X[mask, i].mean()
        else:
            vals[lname] = 0

    mx = max(abs(v) for v in vals.values() if v != 0) or 0.001
    diffs = []
    for a in vals.values():
        for b in vals.values():
            diffs.append(abs(a - b))
    max_diff = max(diffs) / mx * 100

    marker = " ***" if max_diff > 20 else " **" if max_diff > 10 else ""
    print("%-25s | %10.3f %10.3f %10.3f %10.3f | %7.1f%%%s" % (
        name, vals.get("LONG", 0), vals.get("SHORT", 0), vals.get("BOTH", 0), vals.get("SKIP", 0), max_diff, marker))

# === Model ===
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

class EndToEndModel(nn.Module):
    def __init__(self, sr_size=17, base_size=3):
        super().__init__()
        # S/R encoder: compress 15+ S/R features into zone_context
        self.sr_encoder = nn.Sequential(
            nn.Linear(sr_size, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.ReLU(),
        )
        # Brain: base features + zone_context → direction
        self.brain = nn.Sequential(
            nn.Linear(base_size + 3, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4),
        )

    def forward(self, sr_features, base_features):
        zone_context = self.sr_encoder(sr_features)
        combined = torch.cat([base_features, zone_context], dim=1)
        return self.brain(combined)

sr_size = sr_train.shape[1]
base_size = base_train.shape[1]

model = EndToEndModel(sr_size, base_size)
print(f"\nModel params: {sum(p.numel() for p in model.parameters())}")
print(f"  S/R encoder: {sr_size} -> 8 -> 3")
print(f"  Brain: {base_size + 3} -> 32 -> 4")

# Convert to tensors
tr_sr = torch.FloatTensor(sr_train)
tr_base = torch.FloatTensor(base_train)
tr_Y = torch.LongTensor(Y_train)
va_sr = torch.FloatTensor(sr_val)
va_base = torch.FloatTensor(base_val)
va_Y = torch.LongTensor(Y_val)
te_sr = torch.FloatTensor(sr_test)
te_base = torch.FloatTensor(base_test)
te_Y = torch.LongTensor(Y_test)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(TensorDataset(tr_sr, tr_base, tr_Y), batch_size=256, shuffle=True)

print("\n%-6s | %-10s %-10s | %-10s %-10s | %-8s %-8s %-8s %-8s" %
      ("Epoch", "Tr_Acc", "Tr_Loss", "Va_Acc", "Va_Loss", "Va_LONG", "Va_SHORT", "Va_BOTH", "Va_SKIP"))

best_val_loss = float("inf")
patience_counter = 0
best_state = None

for epoch in range(100):
    model.train()
    total_loss = 0; correct = 0; total = 0
    for sr_b, base_b, y_b in train_loader:
        optimizer.zero_grad()
        pred = model(sr_b, base_b)
        loss = criterion(pred, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
        correct += (pred.argmax(1) == y_b).sum().item()
        total += len(y_b)

    model.eval()
    with torch.no_grad():
        va_pred = model(va_sr, va_base)
        val_loss = criterion(va_pred, va_Y).item()
        va_cls = va_pred.argmax(1)
        val_acc = (va_cls == va_Y).float().mean().item() * 100

        accs = {}
        for label, lname in [(0, "LONG"), (1, "SHORT"), (2, "BOTH"), (3, "SKIP")]:
            mask = va_Y == label
            if mask.sum() > 0:
                accs[lname] = (va_cls[mask] == label).float().mean().item() * 100
            else:
                accs[lname] = 0

    if epoch % 5 == 0 or epoch < 5:
        print("%-6d | %-10.1f %-10.4f | %-10.1f %-10.4f | %-8.1f %-8.1f %-8.1f %-8.1f" %
              (epoch, correct/total*100, total_loss/total, val_acc, val_loss,
               accs["LONG"], accs["SHORT"], accs["BOTH"], accs["SKIP"]))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch}")
            break

# === Test ===
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    te_pred = model(te_sr, te_base)
    te_cls = te_pred.argmax(1)
    te_acc = (te_cls == te_Y).float().mean().item() * 100

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)
print(f"  Overall: {te_acc:.1f}%")
for label, lname in [(0, "LONG"), (1, "SHORT"), (2, "BOTH"), (3, "SKIP")]:
    mask = te_Y == label
    if mask.sum() > 0:
        acc = (te_cls[mask] == label).float().mean().item() * 100
        print(f"  {lname}: {acc:.1f}% ({mask.sum()} samples)")

# === Also test base-only for comparison ===
print("\n" + "=" * 60)
print("BASELINE: Base features only (MLP)")
print("=" * 60)

class BaseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4),
        )
    def forward(self, x):
        return self.net(x)

base_model = BaseMLP()
base_opt = torch.optim.Adam(base_model.parameters(), lr=0.001, weight_decay=0.0001)
base_loader = DataLoader(TensorDataset(tr_base, tr_Y), batch_size=256, shuffle=True)

best_base_loss = float("inf")
best_base_state = None
patience2 = 0

for epoch in range(100):
    base_model.train()
    for xb, yb in base_loader:
        base_opt.zero_grad()
        pred = base_model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        base_opt.step()

    base_model.eval()
    with torch.no_grad():
        va_pred = base_model(va_base)
        vl = criterion(va_pred, va_Y).item()

    if vl < best_base_loss:
        best_base_loss = vl
        patience2 = 0
        best_base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    else:
        patience2 += 1
        if patience2 >= 15:
            break

base_model.load_state_dict(best_base_state)
base_model.eval()
with torch.no_grad():
    te_pred_base = base_model(te_base)
    te_cls_base = te_pred_base.argmax(1)
    base_acc = (te_cls_base == te_Y).float().mean().item() * 100

print(f"  Base-only accuracy: {base_acc:.1f}%")
for label, lname in [(0, "LONG"), (1, "SHORT"), (2, "BOTH"), (3, "SKIP")]:
    mask = te_Y == label
    if mask.sum() > 0:
        acc = (te_cls_base[mask] == label).float().mean().item() * 100
        print(f"  {lname}: {acc:.1f}% ({mask.sum()} samples)")

print(f"\n{'='*60}")
print(f"COMPARISON:")
print(f"  End-to-end (S/R + base): {te_acc:.1f}%")
print(f"  Base only:               {base_acc:.1f}%")
print(f"  Difference:              {te_acc - base_acc:+.1f}%")
print(f"  Random (4-class):        25.0%")
