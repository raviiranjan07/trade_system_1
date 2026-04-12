"""Standard evaluators: compute all required metrics per protocol.

One function per protocol. Each takes predictions + labels and returns a dict
with every metric the protocol requires, using the exact names it expects.

This file is a calculator — it never touches the filesystem.
"""

import numpy as np
from pathlib import Path

FEES_BPS = 8  # round-trip fees (locked in CLAUDE.md)


def evaluate_sr_bounce_break(
    preds: np.ndarray,
    labels: np.ndarray,
    mfe_bps: np.ndarray | None = None,
    mae_bps: np.ndarray | None = None,
    split_name: str = "test",
) -> dict:
    """Compute all metrics for the sr_bounce_break_v1 protocol.

    Args:
        preds:      model predictions, array of 0 (BREAK) or 1 (BOUNCE)
        labels:     actual outcomes, array of 0 (BREAK) or 1 (BOUNCE)
        mfe_bps:    favorable excursion in bps per sample (for trading metrics)
        mae_bps:    adverse excursion in bps per sample (for trading metrics)
        split_name: prefix for metric keys ("test", "val", "train")

    Returns:
        dict with all required + optional metrics for sr_bounce_break_v1.
        Keys are prefixed with split_name (e.g., "test_accuracy").
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    n = len(labels)

    if n == 0:
        return {f"{split_name}_accuracy": 0.0, f"n_{split_name}": 0}

    # --- ML metrics ---
    correct = (preds == labels)
    accuracy = correct.mean()

    # Per-class metrics
    bounce_mask = (labels == 1)
    break_mask = (labels == 0)
    pred_bounce = (preds == 1)
    pred_break = (preds == 0)

    tp_bounce = (pred_bounce & bounce_mask).sum()
    fp_bounce = (pred_bounce & break_mask).sum()
    fn_bounce = (~pred_bounce & bounce_mask).sum()

    tp_break = (pred_break & break_mask).sum()
    fp_break = (pred_break & bounce_mask).sum()
    fn_break = (~pred_break & break_mask).sum()

    precision_bounce = tp_bounce / max(tp_bounce + fp_bounce, 1)
    recall_bounce = tp_bounce / max(tp_bounce + fn_bounce, 1)
    precision_break = tp_break / max(tp_break + fp_break, 1)
    recall_break = tp_break / max(tp_break + fn_break, 1)

    f1_bounce = _f1(precision_bounce, recall_bounce)
    f1_break = _f1(precision_break, recall_break)
    f1_macro = (f1_bounce + f1_break) / 2

    # Confusion matrix: [[TN, FP], [FN, TP]] with bounce=positive
    confusion = [
        [int(tp_break), int(fp_bounce)],
        [int(fn_bounce), int(tp_bounce)],
    ]

    # Baseline: always predict majority class from this split
    majority_rate = max(bounce_mask.mean(), break_mask.mean())
    baseline_accuracy = float(majority_rate)
    delta_vs_baseline = float(accuracy - baseline_accuracy)

    # Class balance
    class_balance = float(bounce_mask.mean())

    metrics = {
        f"{split_name}_accuracy": round(float(accuracy), 4),
        f"{split_name}_precision_bounce": round(float(precision_bounce), 4),
        f"{split_name}_precision_break": round(float(precision_break), 4),
        f"{split_name}_recall_bounce": round(float(recall_bounce), 4),
        f"{split_name}_recall_break": round(float(recall_break), 4),
        f"{split_name}_f1": round(float(f1_macro), 4),
        f"{split_name}_f1_bounce": round(float(f1_bounce), 4),
        f"{split_name}_f1_break": round(float(f1_break), 4),
        f"{split_name}_confusion_matrix": confusion,
        f"n_{split_name}": int(n),
        f"class_balance_{split_name}": round(class_balance, 4),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "delta_vs_baseline": round(delta_vs_baseline, 4),
    }

    # --- Trading metrics (only if MFE/MAE provided) ---
    if mfe_bps is not None and mae_bps is not None:
        mfe_bps = np.asarray(mfe_bps)
        mae_bps = np.asarray(mae_bps)
        trading = _compute_trading_metrics(preds, labels, mfe_bps, mae_bps, split_name)
        metrics.update(trading)

    return metrics


def _f1(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_trading_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    mfe_bps: np.ndarray,
    mae_bps: np.ndarray,
    split_name: str,
) -> dict:
    """Compute trading-specific metrics for predicted bounces.

    When the model predicts BOUNCE, we assume we take the trade.
    MFE = how far it went in our favor, MAE = how far it went against us.
    """
    # Only trades where model predicts BOUNCE (we enter)
    traded = (preds == 1)
    if traded.sum() == 0:
        return {
            f"{split_name}_signal_win_rate": 0.0,
            f"{split_name}_expected_value_bps": 0.0,
            f"{split_name}_expected_value_bps_after_fees": 0.0,
            f"{split_name}_profit_factor": 0.0,
            f"{split_name}_avg_winner_bps": 0.0,
            f"{split_name}_avg_loser_bps": 0.0,
            f"{split_name}_n_trades": 0,
        }

    trade_labels = labels[traded]
    trade_mfe = mfe_bps[traded]
    trade_mae = mae_bps[traded]

    winners = (trade_labels == 1)  # model said bounce, it did bounce
    losers = ~winners

    win_rate = winners.mean()
    n_trades = int(traded.sum())

    avg_winner = float(trade_mfe[winners].mean()) if winners.any() else 0.0
    avg_loser = float(trade_mae[losers].mean()) if losers.any() else 0.0

    gross_profit = float(trade_mfe[winners].sum()) if winners.any() else 0.0
    gross_loss = float(trade_mae[losers].sum()) if losers.any() else 0.0
    profit_factor = gross_profit / max(gross_loss, 0.01)

    ev_bps = float(trade_mfe[winners].sum() - trade_mae[losers].sum()) / n_trades
    ev_bps_after_fees = ev_bps - FEES_BPS

    return {
        f"{split_name}_signal_win_rate": round(float(win_rate), 4),
        f"{split_name}_expected_value_bps": round(ev_bps, 2),
        f"{split_name}_expected_value_bps_after_fees": round(ev_bps_after_fees, 2),
        f"{split_name}_profit_factor": round(profit_factor, 2),
        f"{split_name}_avg_winner_bps": round(avg_winner, 2),
        f"{split_name}_avg_loser_bps": round(avg_loser, 2),
        f"{split_name}_n_trades": n_trades,
    }
