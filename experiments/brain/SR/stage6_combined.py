"""Stage 6: S/R + Base Features Combined — Test 1 (MLP, no lookback)

Run: PYTHONPATH=src python experiments/brain/SR/stage6_combined.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === Step 1: Load S/R dataset + base features ===
print("=" * 60)
print("STAGE 6: S/R + Base Features Combined")
print("=" * 60)

# Load S/R datasets (has bar numbers)
print("\nLoading S/R datasets...")
train = np.load("data/features/sr_bounce_break/every_bar/train.npz")
val = np.load("data/features/sr_bounce_break/every_bar/val.npz")
test = np.load("data/features/sr_bounce_break/every_bar/test.npz")

# Load base features from feature cache
print("Loading base features from feature_cache.parquet...")
fc = pd.read_parquet("data/features/direction_prediction/feature_cache.parquet")
fc.index = fc.index.tz_localize(None) if fc.index.tz is not None else fc.index

# Get base features
base_cols = ["roc5", "rsi7", "range_position"]  # roc5 is closest to single-bar roc
# Check what roc columns exist
roc_cols = [c for c in fc.columns if c.startswith("roc")]
print("  Available roc columns:", roc_cols[:5])
# Use roc5 as our roc feature
base_features = fc[base_cols].values.astype(np.float32)
print("  Base features shape:", base_features.shape)

# === Step 2: Combine features ===
print("\nCombining features...")

def combine(split_data, name):
    Xd = split_data["X_dynamic"][:, -1, :]  # last snapshot, dynamic
    Xs = split_data["X_static"]
    bars = split_data["bars"]
    Y = split_data["Y"]

    # Get base features for these bars
    base = base_features[bars]

    # Handle NaN
    base = np.nan_to_num(base, nan=0.0)

    # Combine: S/R dynamic (9-11) + S/R static (6) + base (3)
    X_combined = np.hstack([Xd, Xs, base])

    print(f"  {name}: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
    return X_combined, Y

X_train, Y_train = combine(train, "Train")
X_val, Y_val = combine(val, "Val")
X_test, Y_test = combine(test, "Test")

# Feature names
dyn_names = ["dist_to_zone_pct", "support_width_pct", "res_width_pct",
             "support_retest", "resistance_retest", "zone_width",
             "recovery_up_pct", "recovery_down_pct",
             "speed_short", "speed_mid", "speed_long"]
sta_names = ["bounce_ratio", "recent_bounce_ratio", "pressure",
             "bars_since_touch", "touch_count_scaled", "level_type_binary"]
base_names = ["roc", "rsi7", "range_position"]
all_names = dyn_names + sta_names + base_names

num_features = X_train.shape[1]
print(f"\nTotal features: {num_features}")
print(f"  S/R dynamic: {len(dyn_names)}")
print(f"  S/R static: {len(sta_names)}")
print(f"  Base: {len(base_names)}")

# === Step 3: Feature separation ===
print("\n" + "=" * 60)
print("FEATURE SEPARATION")
print("=" * 60)

print("\n%-25s | %10s %10s %10s | %8s" % ("Feature", "Bounce", "Break", "Chop", "MaxDiff%"))
print("-" * 75)
for i in range(num_features):
    name = all_names[i] if i < len(all_names) else f"feat_{i}"
    b = X_train[Y_train == 1, i].mean()
    k = X_train[Y_train == 0, i].mean()
    c = X_train[Y_train == 2, i].mean()
    mx = max(abs(b), abs(k), abs(c), 0.001)
    diff = max(abs(b - k), abs(b - c), abs(k - c)) / mx * 100
    marker = " ***" if diff > 10 else " **" if diff > 5 else ""
    print("%-25s | %10.3f %10.3f %10.3f | %7.1f%%%s" % (name, b, k, c, diff, marker))

# === Step 4: Train MLP ===
print("\n" + "=" * 60)
print("TEST 1: MLP (no lookback)")
print("=" * 60)

tr_X = torch.FloatTensor(X_train)
va_X = torch.FloatTensor(X_val)
te_X = torch.FloatTensor(X_test)
tr_Y = torch.LongTensor(Y_train)
va_Y = torch.LongTensor(Y_val)
te_Y = torch.LongTensor(Y_test)

print(f"\nTrain: {len(tr_Y)}  Val: {len(va_Y)}  Test: {len(te_Y)}")

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),
        )
    def forward(self, x):
        return self.net(x)

model = MLP(num_features)
print(f"Params: {sum(p.numel() for p in model.parameters())}")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(TensorDataset(tr_X, tr_Y), batch_size=256, shuffle=True)

print("\n%-6s | %-10s %-10s | %-10s %-10s | %-10s %-10s %-10s" %
      ("Epoch", "Tr_Acc", "Tr_Loss", "Va_Acc", "Va_Loss", "Va_Bounce", "Va_Break", "Va_Chop"))

best_val_loss = float("inf")
patience_counter = 0
best_state = None

for epoch in range(100):
    model.train()
    total_loss = 0; correct = 0; total = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct += (pred.argmax(1) == yb).sum().item()
        total += len(yb)

    model.eval()
    with torch.no_grad():
        va_pred = model(va_X)
        val_loss = criterion(va_pred, va_Y).item()
        va_cls = va_pred.argmax(1)
        val_acc = (va_cls == va_Y).float().mean().item() * 100
        b_acc = (va_cls[va_Y == 1] == 1).float().mean().item() * 100 if (va_Y == 1).sum() > 0 else 0
        k_acc = (va_cls[va_Y == 0] == 0).float().mean().item() * 100 if (va_Y == 0).sum() > 0 else 0
        c_acc = (va_cls[va_Y == 2] == 2).float().mean().item() * 100 if (va_Y == 2).sum() > 0 else 0

    if epoch % 5 == 0 or epoch < 5:
        print("%-6d | %-10.1f %-10.4f | %-10.1f %-10.4f | %-10.1f %-10.1f %-10.1f" %
              (epoch, correct / total * 100, total_loss / total, val_acc, val_loss, b_acc, k_acc, c_acc))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    te_pred = model(te_X)
    te_cls = te_pred.argmax(1)
    te_acc = (te_cls == te_Y).float().mean().item() * 100
    b_acc = (te_cls[te_Y == 1] == 1).float().mean().item() * 100
    k_acc = (te_cls[te_Y == 0] == 0).float().mean().item() * 100
    c_acc = (te_cls[te_Y == 2] == 2).float().mean().item() * 100

print("\nTEST RESULTS:")
print(f"  Overall: {te_acc:.1f}%")
print(f"  BOUNCE: {b_acc:.1f}% ({(te_Y == 1).sum()} samples)")
print(f"  BREAK:  {k_acc:.1f}% ({(te_Y == 0).sum()} samples)")
print(f"  CHOP:   {c_acc:.1f}% ({(te_Y == 2).sum()} samples)")
print(f"\nBaseline (random 3-class): 33.3%")
print(f"Previous S/R only: ~33% (failed)")
