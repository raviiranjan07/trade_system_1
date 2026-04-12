"""Stage 9: Train redesigned S/R advisor on datasets_stage9.

Supports two trajectory pooling modes:
  - avg: original Stage 9 style average pooling
  - avg_last: upgraded version with AvgPool + LastStep fusion

Run:
  PYTHONPATH=src python experiments/brain/SR/train_stage9.py
  PYTHONPATH=src python experiments/brain/SR/train_stage9.py --pooling avg
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path("data/features/sr_bounce_break/stage9")


def load_split(name: str):
    data = np.load(DATA_DIR / f"{name}.npz")
    return data["X_static"], data["X_dynamic"], data["Y"]


def normalize_static(train_x, val_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1.0

    return (
        np.nan_to_num((train_x - mean) / std),
        np.nan_to_num((val_x - mean) / std),
        np.nan_to_num((test_x - mean) / std),
    )


def normalize_dynamic(train_x, val_x, test_x):
    mean = train_x.mean(axis=(0, 1))
    std = train_x.std(axis=(0, 1))
    std[std == 0] = 1.0

    return (
        np.nan_to_num((train_x - mean) / std),
        np.nan_to_num((val_x - mean) / std),
        np.nan_to_num((test_x - mean) / std),
    )


class Stage9Model(nn.Module):
    def __init__(self, static_dim: int, dynamic_dim: int, pooling: str):
        super().__init__()
        self.pooling = pooling

        self.history = nn.Sequential(
            nn.Linear(static_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.Conv1d(dynamic_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        traj_dim = 8 if pooling == "avg" else 16
        self.traj_proj = nn.Sequential(
            nn.Linear(traj_dim, 8),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 2),
        )

    def forward(self, x_static, x_dynamic):
        history_state = self.history(x_static)

        seq = x_dynamic.transpose(1, 2)
        seq = self.conv(seq)

        avg_state = seq.mean(dim=2)
        if self.pooling == "avg":
            traj_state = self.traj_proj(avg_state)
        else:
            last_state = seq[:, :, -1]
            traj_state = self.traj_proj(torch.cat([avg_state, last_state], dim=1))

        fused = torch.cat([history_state, traj_state], dim=1)
        return self.head(fused)


def metrics_from_logits(logits, y_true):
    pred = logits.argmax(dim=1)
    acc = (pred == y_true).float().mean().item() * 100
    bounce_mask = y_true == 1
    break_mask = y_true == 0
    bounce_acc = (pred[bounce_mask] == 1).float().mean().item() * 100 if bounce_mask.sum() > 0 else 0.0
    break_acc = (pred[break_mask] == 0).float().mean().item() * 100 if break_mask.sum() > 0 else 0.0
    return acc, bounce_acc, break_acc, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling", choices=["avg", "avg_last"], default="avg_last")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 9: Train Redesigned S/R Advisor")
    print("=" * 70)
    print(f"Pooling: {args.pooling}")

    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    print(f"Dataset thresholds: {meta['strict_support_threshold']:.2f} / {meta['strict_resistance_threshold']:.2f}")

    s_tr, d_tr, y_tr = load_split("train")
    s_va, d_va, y_va = load_split("val")
    s_te, d_te, y_te = load_split("test")

    print(f"Train: {len(y_tr)} | Val: {len(y_va)} | Test: {len(y_te)}")
    print(f"Static: {s_tr.shape[1]} | Dynamic: {d_tr.shape[1]} x {d_tr.shape[2]}")
    print(f"Train bounce rate: {(y_tr == 1).mean() * 100:.1f}%")

    s_tr, s_va, s_te = normalize_static(s_tr, s_va, s_te)
    d_tr, d_va, d_te = normalize_dynamic(d_tr, d_va, d_te)

    tr_static = torch.FloatTensor(s_tr)
    va_static = torch.FloatTensor(s_va)
    te_static = torch.FloatTensor(s_te)
    tr_dynamic = torch.FloatTensor(d_tr)
    va_dynamic = torch.FloatTensor(d_va)
    te_dynamic = torch.FloatTensor(d_te)
    tr_y = torch.LongTensor(y_tr)
    va_y = torch.LongTensor(y_va)
    te_y = torch.LongTensor(y_te)

    model = Stage9Model(static_dim=s_tr.shape[1], dynamic_dim=d_tr.shape[2], pooling=args.pooling)
    print(f"Params: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(tr_static, tr_dynamic, tr_y), batch_size=args.batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience = 0

    print("\n%-6s | %-8s %-8s | %-8s %-8s | %-8s %-8s" %
          ("Epoch", "Tr_Acc", "Tr_Loss", "Va_Acc", "Va_Loss", "Va_Bnc", "Va_Brk"))

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for xb_static, xb_dynamic, yb in loader:
            optimizer.zero_grad()
            logits = model(xb_static, xb_dynamic)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_count += len(yb)

        model.eval()
        with torch.no_grad():
            val_logits = model(va_static, va_dynamic)
            val_loss = criterion(val_logits, va_y).item()
            val_acc, val_bounce_acc, val_break_acc, _ = metrics_from_logits(val_logits, va_y)

        if epoch % 10 == 0 or epoch < 5:
            print("%-6d | %-8.1f %-8.4f | %-8.1f %-8.4f | %-8.1f %-8.1f" % (
                epoch,
                total_correct / total_count * 100,
                total_loss / total_count,
                val_acc,
                val_loss,
                val_bounce_acc,
                val_break_acc,
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
        test_logits = model(te_static, te_dynamic)
        test_acc, test_bounce_acc, test_break_acc, test_pred = metrics_from_logits(test_logits, te_y)
        probs = torch.softmax(test_logits, dim=1).max(dim=1).values

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Overall: {test_acc:.1f}%")
    print(f"Bounce acc: {test_bounce_acc:.1f}%")
    print(f"Break acc:  {test_break_acc:.1f}%")

    for cutoff in [0.55, 0.60, 0.65]:
        mask = probs >= cutoff
        if mask.sum() > 0:
            conf_acc = (test_pred[mask] == te_y[mask]).float().mean().item() * 100
            print(f"Confident >{cutoff:.2f}: {conf_acc:.1f}% ({mask.sum().item()} bars, {mask.float().mean().item() * 100:.1f}%)")


if __name__ == "__main__":
    main()
