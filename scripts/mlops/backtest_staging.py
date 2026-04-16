"""Backtest the @staging model on 2025 OOS and write a metrics report.

Does not compare against any other model. Just a clean, honest backtest
of whatever ML_V1 @staging currently points to. Human reads the numbers
and decides whether to promote.

Output: data/reports/backtest_staging.json

Run:
    python scripts/mlops/backtest_staging.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mlflow
import yaml

from engine.backtest import run_backtest
from engine.config.loader import load_config

with open(REPO_ROOT / "configs/params.yaml") as _f:
    _p = yaml.safe_load(_f)
BACKTEST_START = _p["backtest"]["start"]
BACKTEST_END = _p["backtest"]["end"]
STAGING_DIR = REPO_ROOT / "models/ML_V1_staging"
REPORT_PATH = REPO_ROOT / "data/reports/backtest_staging.json"
TRADES_PATH = REPO_ROOT / "data/reports/backtest_staging_trades.parquet"


def compute_metrics(trades: list) -> dict:
    if not trades:
        return {"n_trades": 0, "total_bps": 0.0, "win_pct": 0.0, "pf": 0.0, "max_dd_bps": 0.0}

    tdf = pd.DataFrame([asdict(t) for t in trades])
    net = tdf["net_profit_bps"].astype(float)

    wins = net[net > 0]
    losses = net[net <= 0]
    gross_loss = abs(losses.sum()) or 1e-9
    pf = float(wins.sum() / gross_loss)

    equity = net.cumsum()
    max_dd = float((equity - equity.cummax()).min())

    # Breakdown by signal type
    by_signal = {}
    for st in sorted(tdf["signal_type"].unique()):
        s = tdf[tdf["signal_type"] == st]
        by_signal[str(st)] = {
            "n": int(len(s)),
            "win_pct": round(float((s["net_profit_bps"] > 0).mean()) * 100, 1),
            "total_bps": round(float(s["net_profit_bps"].sum()), 1),
        }

    # Breakdown by exit reason (PT_TARGET, STOP_LOSS, etc.) — tells us WHICH
    # exit rules make or lose money.
    by_exit = {}
    if "exit_reason" in tdf.columns:
        for er in sorted(tdf["exit_reason"].unique()):
            s = tdf[tdf["exit_reason"] == er]
            by_exit[str(er)] = {
                "n": int(len(s)),
                "win_pct": round(float((s["net_profit_bps"] > 0).mean()) * 100, 1),
                "total_bps": round(float(s["net_profit_bps"].sum()), 1),
                "avg_bps": round(float(s["net_profit_bps"].mean()), 2),
            }

    return {
        "n_trades": int(len(tdf)),
        "total_bps": round(float(net.sum()), 1),
        "win_pct": round(float((net > 0).mean()) * 100, 1),
        "pf": round(pf, 2),
        "max_dd_bps": round(max_dd, 1),
        "avg_trade_bps": round(float(net.mean()), 2),
        "by_signal_type": by_signal,
        "by_exit_reason": by_exit,
    }


def staging_version() -> int | None:
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT}/mlflow.db")
    try:
        mv = mlflow.MlflowClient().get_model_version_by_alias("ML_V1", "staging")
        return int(mv.version)
    except mlflow.exceptions.MlflowException:
        return None


def main() -> None:
    cfg = load_config()
    version = staging_version()

    print(f"Backtest window: {BACKTEST_START} .. {BACKTEST_END}")
    print(f"Model: ML_V1 v{version} @staging ({STAGING_DIR.relative_to(REPO_ROOT)})")

    if not (STAGING_DIR / "direction_model.onnx").exists():
        raise RuntimeError(f"Staging model not found at {STAGING_DIR}. Run train_mlp_v15 first.")

    trades = run_backtest(
        cfg,
        start=BACKTEST_START,
        end=BACKTEST_END,
        ml_model_dir=STAGING_DIR,
    )

    # Save per-trade records for the invariant verifier (verify_backtest.py).
    trades_df = pd.DataFrame([asdict(t) for t in trades])
    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_parquet(TRADES_PATH)

    metrics = compute_metrics(trades)

    print(
        f"\n{metrics['n_trades']} trades | {metrics['win_pct']}% win | "
        f"{metrics['total_bps']:+.0f} bps | PF {metrics['pf']} | DD {metrics['max_dd_bps']:+.0f}"
    )
    print(f"Avg trade: {metrics['avg_trade_bps']:+.1f} bps")
    print("\nBy signal type:")
    for st, s in metrics["by_signal_type"].items():
        print(f"  {st:<20} {s['n']:>4}t  {s['win_pct']:>5.1f}% win  {s['total_bps']:>+8.0f} bps")

    if metrics.get("by_exit_reason"):
        print("\nBy exit reason:")
        for er, s in metrics["by_exit_reason"].items():
            print(
                f"  {er:<20} {s['n']:>4}t  {s['win_pct']:>5.1f}% win  "
                f"{s['total_bps']:>+8.0f} bps  avg {s['avg_bps']:>+6.1f}"
            )

    report = {
        "window": {"start": BACKTEST_START, "end": BACKTEST_END},
        "model": {"name": "ML_V1", "version": version, "alias": "staging"},
        "metrics": metrics,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
