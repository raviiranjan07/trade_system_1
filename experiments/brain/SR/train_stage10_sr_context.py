"""Stage 10: Base direction model with optional S/R context branch.

Uses data/archive/brain_datasets/{train,val,test}.npz:
  X: [N, 8, 23]
  Y[:,0]: binary direction label (0/1)

Runs:
1) Base-only (LSTM over base features)
2) Base + S/R context (LSTM + MLP context fusion)

Usage:
  PYTHONPATH=src python experiments/brain/SR/train_stage10_sr_context.py
  PYTHONPATH=src python experiments/brain/SR/train_stage10_sr_context.py --second-touch-gate
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path("data/archive/brain_datasets")
META_PATH = DATA_DIR / "metadata.json"
SCALER_PATH = DATA_DIR / "scaler.npz"


def load_split(name: str):
    data = np.load(DATA_DIR / f"{name}.npz")
    x = data["X"].astype(np.float32)              # [N, 8, 23]
    y = data["Y"][:, 0].astype(np.int64)          # direction
    return x, y


def normalize_seq(train_x, val_x, test_x):
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0
    return (
        np.nan_to_num((train_x - mean) / std),
        np.nan_to_num((val_x - mean) / std),
        np.nan_to_num((test_x - mean) / std),
    )


def filter_min_zone_width_bps(x: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray, sh_idx: int, rl_idx: int, min_bps: float):
    """Filter rows by raw zone width in bps reconstructed from standardized columns."""
    sh_raw = x[:, -1, sh_idx] * std[sh_idx] + mean[sh_idx]
    rl_raw = x[:, -1, rl_idx] * std[rl_idx] + mean[rl_idx]
    valid = (sh_raw > 0) & (rl_raw > sh_raw)
    width_bps = np.zeros_like(sh_raw)
    width_bps[valid] = (rl_raw[valid] - sh_raw[valid]) / sh_raw[valid] * 10000.0
    keep = valid & (width_bps >= min_bps)
    return x[keep], y[keep], keep.sum(), len(keep)


def summarize_metrics(logits, y_true):
    pred = logits.argmax(dim=1)
    acc = (pred == y_true).float().mean().item() * 100
    cls0 = y_true == 0
    cls1 = y_true == 1
    acc0 = (pred[cls0] == 0).float().mean().item() * 100 if cls0.sum() > 0 else 0.0
    acc1 = (pred[cls1] == 1).float().mean().item() * 100 if cls1.sum() > 0 else 0.0
    pred1 = (pred == 1).float().mean().item() * 100
    probs = torch.softmax(logits, dim=1).max(dim=1).values
    return acc, acc0, acc1, pred1, pred, probs


def parse_csv_cols(value: str):
    return [c.strip() for c in value.split(",") if c.strip()]


class BaseOnlyLSTM(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=32, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 2),
        )

    def forward(self, x_base_seq):
        _, (h, _) = self.lstm(x_base_seq)
        base_state = h[-1]
        return self.head(base_state)


class BaseSrFusion(nn.Module):
    def __init__(self, base_size: int, sr_size: int):
        super().__init__()
        self.base_lstm = nn.LSTM(input_size=base_size, hidden_size=32, num_layers=1, batch_first=True)
        self.base_proj = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.sr_ctx = nn.Sequential(
            nn.Linear(sr_size, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 2),
        )

    def forward(self, x_base_seq, x_sr_ctx):
        _, (h, _) = self.base_lstm(x_base_seq)
        base_state = self.base_proj(h[-1])
        sr_state = self.sr_ctx(x_sr_ctx)
        fused = torch.cat([base_state, sr_state], dim=1)
        return self.head(fused)


def train_eval_base_only(xtr, ytr, xva, yva, xte, yte, epochs=120, patience=15):
    model = BaseOnlyLSTM(input_size=xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.FloatTensor(xtr), torch.LongTensor(ytr)), batch_size=256, shuffle=True)
    va_x = torch.FloatTensor(xva)
    va_y = torch.LongTensor(yva)
    te_x = torch.FloatTensor(xte)
    te_y = torch.LongTensor(yte)

    best_loss = float("inf")
    best_state = None
    bad = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(va_x)
            v_loss = ce(v_logits, va_y).item()
        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        t_logits = model(te_x)
    return t_logits, sum(p.numel() for p in model.parameters())


def train_eval_base_sr(xtr_b, xtr_sr, ytr, xva_b, xva_sr, yva, xte_b, xte_sr, yte, epochs=120, patience=15):
    model = BaseSrFusion(base_size=xtr_b.shape[2], sr_size=xtr_sr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(xtr_b), torch.FloatTensor(xtr_sr), torch.LongTensor(ytr)),
        batch_size=256,
        shuffle=True,
    )
    va_b = torch.FloatTensor(xva_b)
    va_sr = torch.FloatTensor(xva_sr)
    va_y = torch.LongTensor(yva)
    te_b = torch.FloatTensor(xte_b)
    te_sr = torch.FloatTensor(xte_sr)
    te_y = torch.LongTensor(yte)

    best_loss = float("inf")
    best_state = None
    bad = 0

    for _ in range(epochs):
        model.train()
        for xb, xsr, yb in loader:
            opt.zero_grad()
            logits = model(xb, xsr)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(va_b, va_sr)
            v_loss = ce(v_logits, va_y).item()
        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        t_logits = model(te_b, te_sr)
    return t_logits, sum(p.numel() for p in model.parameters())


def print_result(name: str, logits, y_true, n_params: int):
    y_t = torch.LongTensor(y_true)
    acc, acc0, acc1, pred1, pred, probs = summarize_metrics(logits, y_t)
    tn = int(((pred == 0) & (y_t == 0)).sum().item())
    fp = int(((pred == 1) & (y_t == 0)).sum().item())
    fn = int(((pred == 0) & (y_t == 1)).sum().item())
    tp = int(((pred == 1) & (y_t == 1)).sum().item())

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Params: {n_params}")
    print(f"Overall: {acc:.2f}%")
    print(f"Class0 acc: {acc0:.2f}%")
    print(f"Class1 acc: {acc1:.2f}%")
    print(f"Pred class1 rate: {pred1:.2f}%")
    print(f"Confusion [TN FP; FN TP]: [{tn} {fp}; {fn} {tp}]")
    for c in [0.55, 0.60, 0.65]:
        m = probs >= c
        if m.sum() > 0:
            ca = (pred[m] == y_t[m]).float().mean().item() * 100
            cov = m.float().mean().item() * 100
            print(f"Conf >{c:.2f}: {ca:.2f}% ({int(m.sum().item())} rows, {cov:.2f}%)")


def print_context_diagnostics(logits_base, logits_fuse, y_true, xte_gate):
    y_t = torch.LongTensor(y_true)
    pred_base = logits_base.argmax(dim=1)
    pred_fuse = logits_fuse.argmax(dim=1)
    p1_base = torch.softmax(logits_base, dim=1)[:, 1]
    p1_fuse = torch.softmax(logits_fuse, dim=1)[:, 1]

    disagree = pred_base != pred_fuse
    abs_shift = (p1_fuse - p1_base).abs()

    support_retest = xte_gate[:, 0]
    resistance_retest = xte_gate[:, 1]
    dist_support = xte_gate[:, 2]
    dist_resistance = xte_gate[:, 3]

    near_support = dist_support <= 0.0
    near_resistance = dist_resistance <= 0.0
    second_support = support_retest >= 0.5
    second_resistance = resistance_retest >= 0.5
    strong_setup = (near_support & second_support) | (near_resistance & second_resistance)

    def subset_report(mask_np, name):
        mask = torch.from_numpy(mask_np.astype(np.bool_))
        n = int(mask.sum().item())
        if n == 0:
            print(f"{name}: no rows")
            return
        acc_b = (pred_base[mask] == y_t[mask]).float().mean().item() * 100
        acc_f = (pred_fuse[mask] == y_t[mask]).float().mean().item() * 100
        lift = acc_f - acc_b
        dis = (pred_base[mask] != pred_fuse[mask]).float().mean().item() * 100
        sh = abs_shift[mask].mean().item()
        print(f"{name}: n={n}, base={acc_b:.2f}%, base+sr={acc_f:.2f}%, lift={lift:+.2f}pp, disagree={dis:.2f}%, |Δp1|={sh:.4f}")

    print("\n" + "=" * 70)
    print("CONTEXT LEARNING DIAGNOSTICS")
    print("=" * 70)
    print(f"Overall prediction disagreement: {disagree.float().mean().item()*100:.2f}%")
    print(f"Mean |Δprob(class1)| due to SR branch: {abs_shift.mean().item():.4f}")
    subset_report(near_support, "Near support")
    subset_report(near_resistance, "Near resistance")
    subset_report(strong_setup, "Strong second-touch setup")


def apply_second_touch_gate(
    logits: torch.Tensor,
    support_retest: np.ndarray,
    resistance_retest: np.ndarray,
    distance_to_support: np.ndarray,
    distance_to_resistance: np.ndarray,
    near_threshold: float,
    retest_threshold: float,
    gate_strength: float,
):
    """Apply post-logit rule gate for second-touch near support/resistance.

    Label mapping in this dataset:
      class 0 = LONG
      class 1 = SHORT
    """
    gated = logits.clone()

    near_support = distance_to_support <= near_threshold
    near_resistance = distance_to_resistance <= near_threshold
    second_support = support_retest >= retest_threshold
    second_resistance = resistance_retest >= retest_threshold

    long_gate = torch.from_numpy((near_support & second_support).astype(np.bool_))
    short_gate = torch.from_numpy((near_resistance & second_resistance).astype(np.bool_))

    # Favor LONG on second support touch near support.
    gated[long_gate, 0] += gate_strength
    gated[long_gate, 1] -= gate_strength

    # Favor SHORT on second resistance touch near resistance.
    gated[short_gate, 1] += gate_strength
    gated[short_gate, 0] -= gate_strength

    gate_meta = {
        "long_gate_count": int(long_gate.sum().item()),
        "short_gate_count": int(short_gate.sum().item()),
        "near_threshold": float(near_threshold),
        "retest_threshold": float(retest_threshold),
        "gate_strength": float(gate_strength),
    }
    return gated, gate_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--second-touch-gate", action="store_true", help="Apply post-logit second-touch gate.")
    parser.add_argument(
        "--gate-near-threshold",
        type=float,
        default=0.0,
        help="Near-zone threshold on normalized distance features (<= threshold).",
    )
    parser.add_argument(
        "--gate-retest-threshold",
        type=float,
        default=0.5,
        help="Second-touch proxy threshold on normalized retest features (>= threshold).",
    )
    parser.add_argument("--gate-strength", type=float, default=0.35, help="Additive logit boost for gate.")
    parser.add_argument("--min-zone-bps", type=float, default=15.0, help="Minimum zone width filter in bps.")
    parser.add_argument(
        "--base-cols",
        type=str,
        default="roc,rsi7,range_position,volume_ratio",
        help="Comma-separated base feature columns from data/archive/brain_datasets/metadata.json",
    )
    args = parser.parse_args()

    with open(META_PATH) as f:
        meta = json.load(f)
    names = meta["feature_names"]
    idx = {n: i for i, n in enumerate(names)}
    scaler = np.load(SCALER_PATH)
    mean = scaler["mean"]
    std = scaler["std"]

    base_cols = parse_csv_cols(args.base_cols)
    missing_base = [c for c in base_cols if c not in idx]
    if missing_base:
        raise ValueError(f"Missing base feature(s): {missing_base}")
    # SR context branch features.
    sr_cols = [
        "support_range_low",
        "support_range_high",
        "resistance_range_low",
        "resistance_range_high",
        "zone_width",
        "support_retest",
        "resistance_retest",
    ]
    gate_cols = [
        "support_retest",
        "resistance_retest",
        "distance_to_support",
        "distance_to_resistance",
    ]
    base_idx = [idx[c] for c in base_cols]
    sr_idx = [idx[c] for c in sr_cols]
    gate_idx = [idx[c] for c in gate_cols]

    print("=" * 70)
    print("STAGE 10: Base + S/R Context")
    print("=" * 70)
    print("Base cols:", base_cols)
    print("SR cols:", sr_cols)

    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    xte, yte = load_split("test")

    sh_idx = idx["support_range_high"]
    rl_idx = idx["resistance_range_low"]
    if args.min_zone_bps > 0:
        xtr, ytr, ktr, ntr = filter_min_zone_width_bps(xtr, ytr, mean, std, sh_idx, rl_idx, args.min_zone_bps)
        xva, yva, kva, nva = filter_min_zone_width_bps(xva, yva, mean, std, sh_idx, rl_idx, args.min_zone_bps)
        xte, yte, kte, nte = filter_min_zone_width_bps(xte, yte, mean, std, sh_idx, rl_idx, args.min_zone_bps)
        print(f"Zone width filter >= {args.min_zone_bps:.1f} bps")
        print(f"Train kept: {ktr}/{ntr} ({ktr/ntr*100:.1f}%)")
        print(f"Val kept:   {kva}/{nva} ({kva/nva*100:.1f}%)")
        print(f"Test kept:  {kte}/{nte} ({kte/nte*100:.1f}%)")

    xtr_b = xtr[:, :, base_idx]
    xva_b = xva[:, :, base_idx]
    xte_b = xte[:, :, base_idx]

    # Use last snapshot for SR context branch.
    xtr_sr = xtr[:, -1, sr_idx]
    xva_sr = xva[:, -1, sr_idx]
    xte_sr = xte[:, -1, sr_idx]
    xte_gate = xte[:, -1, gate_idx]

    xtr_b, xva_b, xte_b = normalize_seq(xtr_b, xva_b, xte_b)
    xtr_sr, xva_sr, xte_sr = normalize_seq(xtr_sr[:, None, :], xva_sr[:, None, :], xte_sr[:, None, :])
    xtr_sr = xtr_sr[:, 0, :]
    xva_sr = xva_sr[:, 0, :]
    xte_sr = xte_sr[:, 0, :]

    logits_base, params_base = train_eval_base_only(xtr_b, ytr, xva_b, yva, xte_b, yte)
    logits_fuse, params_fuse = train_eval_base_sr(
        xtr_b, xtr_sr, ytr, xva_b, xva_sr, yva, xte_b, xte_sr, yte
    )

    print_result("BASE ONLY", logits_base, yte, params_base)
    print_result("BASE + SR CONTEXT", logits_fuse, yte, params_fuse)
    print_context_diagnostics(logits_base, logits_fuse, yte, xte_gate)

    if args.second_touch_gate:
        gated_base, meta_base = apply_second_touch_gate(
            logits_base,
            support_retest=xte_gate[:, 0],
            resistance_retest=xte_gate[:, 1],
            distance_to_support=xte_gate[:, 2],
            distance_to_resistance=xte_gate[:, 3],
            near_threshold=args.gate_near_threshold,
            retest_threshold=args.gate_retest_threshold,
            gate_strength=args.gate_strength,
        )
        gated_fuse, meta_fuse = apply_second_touch_gate(
            logits_fuse,
            support_retest=xte_gate[:, 0],
            resistance_retest=xte_gate[:, 1],
            distance_to_support=xte_gate[:, 2],
            distance_to_resistance=xte_gate[:, 3],
            near_threshold=args.gate_near_threshold,
            retest_threshold=args.gate_retest_threshold,
            gate_strength=args.gate_strength,
        )

        print("\nGate config:", meta_fuse)
        print_result("BASE ONLY + SECOND-TOUCH GATE", gated_base, yte, params_base)
        print_result("BASE+SR + SECOND-TOUCH GATE", gated_fuse, yte, params_fuse)


if __name__ == "__main__":
    main()
