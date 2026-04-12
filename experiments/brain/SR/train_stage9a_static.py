"""Stage 9A: Train minimal static-memory model on datasets_stage9a_static.

Run:
  PYTHONPATH=src python experiments/brain/SR/train_stage9a_static.py
  PYTHONPATH=src python experiments/brain/SR/train_stage9a_static.py --feature-set position
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path("data/features/sr_bounce_break/stage9a_static")


def load_split(name: str):
    data = np.load(DATA_DIR / f"{name}.npz")
    x_speed = data["X_speed"] if "X_speed" in data.files else None
    return data["X_static"], x_speed, data["Y"]


def normalize(train_x, val_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1.0
    return (
        np.nan_to_num((train_x - mean) / std),
        np.nan_to_num((val_x - mean) / std),
        np.nan_to_num((test_x - mean) / std),
    )


class PositionOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.net(x)


class MemoryOnlyModel(nn.Module):
    def __init__(self, memory_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(memory_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.net(x)


class Stage9AStaticModel(nn.Module):
    def __init__(self, memory_dim: int):
        super().__init__()
        self.memory = nn.Sequential(
            nn.Linear(memory_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(1 + 4, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        position = x[:, :1]
        memory_input = x[:, 1:]
        memory_state = self.memory(memory_input)
        fused = torch.cat([position, memory_state], dim=1)
        return self.head(fused)


class MemorySpeedModel(nn.Module):
    def __init__(self, memory_dim: int):
        super().__init__()
        self.memory = nn.Sequential(
            nn.Linear(memory_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )
        self.speed = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(7, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        memory_input = x[:, :-3]
        speed_input = x[:, -3:]
        mem_state = self.memory(memory_input)
        spd_state = self.speed(speed_input)
        fused = torch.cat([mem_state, spd_state], dim=1)
        return self.head(fused)


def metrics_from_logits(logits, y_true):
    pred = logits.argmax(dim=1)
    acc = (pred == y_true).float().mean().item() * 100
    bounce_mask = y_true == 1
    break_mask = y_true == 0
    bounce_acc = (pred[bounce_mask] == 1).float().mean().item() * 100 if bounce_mask.sum() > 0 else 0.0
    break_acc = (pred[break_mask] == 0).float().mean().item() * 100 if break_mask.sum() > 0 else 0.0
    pred_bounce_rate = (pred == 1).float().mean().item() * 100
    tp_bounce = int(((pred == 1) & (y_true == 1)).sum().item())
    fp_bounce = int(((pred == 1) & (y_true == 0)).sum().item())
    fn_bounce = int(((pred == 0) & (y_true == 1)).sum().item())
    tn_bounce = int(((pred == 0) & (y_true == 0)).sum().item())
    return acc, bounce_acc, break_acc, pred_bounce_rate, pred, (tn_bounce, fp_bounce, fn_bounce, tp_bounce)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-set",
        choices=["position", "memory", "position_memory", "memory_speed"],
        default="position_memory",
    )
    parser.add_argument(
        "--memory-features",
        type=str,
        default="bounce_ratio,touch_count_scaled,recent_bounce_ratio,pressure",
        help="Comma-separated feature names for memory runs.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 9A: Train Minimal Static-Memory Model")
    print("=" * 70)
    print(f"Feature set: {args.feature_set}")

    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    static_features = meta["static_features"]
    feature_index = {name: i for i, name in enumerate(static_features)}
    print(f"Features: {static_features}")
    print(f"Thresholds: {meta['strict_support_threshold']:.2f} / {meta['strict_resistance_threshold']:.2f}")

    x_tr, s_tr, y_tr = load_split("train")
    x_va, s_va, y_va = load_split("val")
    x_te, s_te, y_te = load_split("test")

    requested_memory_features = [f.strip() for f in args.memory_features.split(",") if f.strip()]
    if args.feature_set != "position":
        missing = [f for f in requested_memory_features if f not in feature_index]
        if missing:
            raise ValueError(f"Unknown memory features: {missing}")
        memory_indices = [feature_index[f] for f in requested_memory_features]
    else:
        memory_indices = []

    if args.feature_set == "position":
        pos_idx = feature_index.get("price_position", 0)
        x_tr = x_tr[:, [pos_idx]]
        x_va = x_va[:, [pos_idx]]
        x_te = x_te[:, [pos_idx]]
        model = PositionOnlyModel()
    elif args.feature_set == "memory":
        x_tr = x_tr[:, memory_indices]
        x_va = x_va[:, memory_indices]
        x_te = x_te[:, memory_indices]
        model = MemoryOnlyModel(memory_dim=len(memory_indices))
    elif args.feature_set == "memory_speed":
        if s_tr is None or s_va is None or s_te is None:
            raise ValueError("X_speed not found in dataset. Rebuild Stage9A dataset first.")
        x_tr = np.concatenate([x_tr[:, memory_indices], s_tr], axis=1)
        x_va = np.concatenate([x_va[:, memory_indices], s_va], axis=1)
        x_te = np.concatenate([x_te[:, memory_indices], s_te], axis=1)
        model = MemorySpeedModel(memory_dim=len(memory_indices))
    else:
        pos_idx = feature_index.get("price_position", 0)
        x_tr = np.concatenate([x_tr[:, [pos_idx]], x_tr[:, memory_indices]], axis=1)
        x_va = np.concatenate([x_va[:, [pos_idx]], x_va[:, memory_indices]], axis=1)
        x_te = np.concatenate([x_te[:, [pos_idx]], x_te[:, memory_indices]], axis=1)
        model = Stage9AStaticModel(memory_dim=len(memory_indices))

    print(f"Train: {len(y_tr)} | Val: {len(y_va)} | Test: {len(y_te)}")
    print(f"Train bounce rate: {(y_tr == 1).mean() * 100:.1f}%")
    if args.feature_set != "position":
        print(f"Memory features used: {requested_memory_features}")

    x_tr, x_va, x_te = normalize(x_tr, x_va, x_te)

    tr_x = torch.FloatTensor(x_tr)
    va_x = torch.FloatTensor(x_va)
    te_x = torch.FloatTensor(x_te)
    tr_y = torch.LongTensor(y_tr)
    va_y = torch.LongTensor(y_va)
    te_y = torch.LongTensor(y_te)

    print(f"Params: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=args.batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience = 0

    print("\n%-6s | %-8s %-8s | %-8s %-8s | %-8s %-8s %-8s" %
          ("Epoch", "Tr_Acc", "Tr_Loss", "Va_Acc", "Va_Loss", "Va_Bnc", "Va_Brk", "Va_PBnc"))

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_count += len(yb)

        model.eval()
        with torch.no_grad():
            val_logits = model(va_x)
            val_loss = criterion(val_logits, va_y).item()
            val_acc, val_bnc, val_brk, val_pred_bnc, _, _ = metrics_from_logits(val_logits, va_y)

        if epoch % 10 == 0 or epoch < 5:
            print("%-6d | %-8.1f %-8.4f | %-8.1f %-8.4f | %-8.1f %-8.1f %-8.1f" % (
                epoch,
                total_correct / total_count * 100,
                total_loss / total_count,
                val_acc,
                val_loss,
                val_bnc,
                val_brk,
                val_pred_bnc,
            ))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(te_x)
        test_acc, test_bnc, test_brk, test_pred_bnc, test_pred, cm = metrics_from_logits(test_logits, te_y)
        probs = torch.softmax(test_logits, dim=1).max(dim=1).values

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Overall: {test_acc:.1f}%")
    print(f"Bounce acc: {test_bnc:.1f}%")
    print(f"Break acc:  {test_brk:.1f}%")
    print(f"Pred bounce rate: {test_pred_bnc:.1f}%")
    print(f"Confusion matrix [TN FP; FN TP]: [{cm[0]} {cm[1]}; {cm[2]} {cm[3]}]")

    for cutoff in [0.55, 0.60, 0.65]:
        mask = probs >= cutoff
        if mask.sum() > 0:
            conf_acc = (test_pred[mask] == te_y[mask]).float().mean().item() * 100
            print(f"Confident >{cutoff:.2f}: {conf_acc:.1f}% ({mask.sum().item()} bars, {mask.float().mean().item() * 100:.1f}%)")


if __name__ == "__main__":
    main()
