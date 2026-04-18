"""Backtest Report Builder — enforces schema v2.0.

Every backtest result MUST go through this builder. It:
1. Computes metrics with LONG/SHORT split
2. Computes exit distribution
3. Tags scope (model, mode, exit_version, signals)
4. Validates all required fields exist
5. Saves to standard JSON format

Usage:
    from mlops.backtest_report import build_report, validate_report, save_report
    report = build_report(trades_df, model="ml_v3", mode="independent", ...)
    save_report(report, path)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

SCHEMA_VERSION = "2.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "configs/schemas/backtest_report.yaml"


def _compute_metrics(df):
    """Compute trading metrics for a dataframe of trades."""
    if len(df) == 0:
        return {"n": 0, "bps": 0.0, "pf": 0.0, "win_pct": 0.0,
                "stop_pct": 0.0, "dd": 0.0, "avg": 0.0}
    wins = df[df["net_profit_bps"] > 0]
    losses = df[df["net_profit_bps"] <= 0]
    gw = float(wins["net_profit_bps"].sum()) if len(wins) else 0
    gl = abs(float(losses["net_profit_bps"].sum())) if len(losses) else 1
    stops = df[df["exit_reason"] == "STOP_LOSS"]
    eq = df["net_profit_bps"].cumsum()
    dd = float((eq - eq.cummax()).min()) if len(eq) else 0
    return {
        "n": int(len(df)),
        "bps": round(float(df["net_profit_bps"].sum()), 1),
        "pf": round(gw / gl, 3) if gl > 0 else 0.0,
        "win_pct": round(len(wins) / len(df) * 100, 1),
        "stop_pct": round(len(stops) / len(df) * 100, 1),
        "dd": round(dd, 1),
        "avg": round(float(df["net_profit_bps"].mean()), 1),
    }


def _compute_exit_distribution(df):
    """Per exit-reason breakdown."""
    dist = {}
    for reason, g in df.groupby("exit_reason"):
        dist[reason] = {
            "n": int(len(g)),
            "bps": round(float(g["net_profit_bps"].sum()), 1),
            "avg": round(float(g["net_profit_bps"].mean()), 1),
        }
    return dist


def build_report(
    trades_df: pd.DataFrame,
    model: str,
    mode: str,
    exit_version: str,
    start: str,
    end: str,
    period_type: str = "test",
    v14_included: Optional[bool] = None,
) -> dict:
    """Build a schema-compliant backtest report.

    Args:
        trades_df: DataFrame with columns: signal_type, direction, net_profit_bps, exit_reason
        model: model identifier (ml_v1, ml_v2_attention, ml_v3, v14)
        mode: "independent", "mixed", or "v14_only"
        exit_version: "v1" or "v2"
        start: backtest start date
        end: backtest end date
        period_type: "train", "val", "test", or "full"
        v14_included: explicit override; auto-detected from mode if None
    """
    if v14_included is None:
        v14_included = mode == "mixed" or mode == "v14_only"

    # Determine which signals belong to this model
    signal_prefix = {
        "ml_v3": "ML_V3_",
        "ml_v2_attention": "ML_ATTN_",
        "ml_v1": "ML_",
        "v14": "",  # all V1.4 signals
    }.get(model, "")

    if model == "v14":
        ml = trades_df[trades_df["signal_type"].isin(
            ["V12_LONG", "V12_SHORT", "BEAR_LONG", "BULL_SHORT"])]
    elif model == "ml_v1":
        ml = trades_df[trades_df["signal_type"].str.startswith("ML_") &
                        ~trades_df["signal_type"].str.startswith("ML_ATTN_") &
                        ~trades_df["signal_type"].str.startswith("ML_V3_")]
    else:
        ml = trades_df[trades_df["signal_type"].str.startswith(signal_prefix)]

    signals_included = sorted(ml["signal_type"].unique().tolist()) if len(ml) > 0 else []

    ml_long = ml[ml["direction"] == "LONG"]
    ml_short = ml[ml["direction"] == "SHORT"]

    report = {
        # Scope
        "model": model,
        "mode": mode,
        "signals_included": signals_included,
        "v14_included": v14_included,
        "exit_version": exit_version,
        # Period
        "start": start,
        "end": end,
        "period_type": period_type,
        # Metrics
        "metrics": {
            "all": _compute_metrics(ml),
            "long": _compute_metrics(ml_long),
            "short": _compute_metrics(ml_short),
        },
        # Exit breakdown
        "exit_distribution": _compute_exit_distribution(ml),
        # Meta
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return report


def validate_report(report: dict) -> list[str]:
    """Validate report against schema. Returns list of errors (empty = valid)."""
    errors = []

    required_top = ["model", "mode", "signals_included", "v14_included",
                     "exit_version", "start", "end", "period_type",
                     "metrics", "exit_distribution", "schema_version", "generated_at"]
    for field in required_top:
        if field not in report:
            errors.append(f"Missing required field: {field}")

    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"Schema version mismatch: expected {SCHEMA_VERSION}, got {report.get('schema_version')}")

    if report.get("mode") not in ("independent", "mixed", "v14_only"):
        errors.append(f"Invalid mode: {report.get('mode')}")

    if report.get("exit_version") not in ("v1", "v2"):
        errors.append(f"Invalid exit_version: {report.get('exit_version')}")

    # Check metrics structure
    metrics = report.get("metrics", {})
    for split in ["all", "long", "short"]:
        if split not in metrics:
            errors.append(f"Missing metrics.{split}")
        else:
            for key in ["n", "bps", "pf", "win_pct", "stop_pct", "dd", "avg"]:
                if key not in metrics[split]:
                    errors.append(f"Missing metrics.{split}.{key}")

    return errors


def validate_comparable(reports: list[dict]) -> list[str]:
    """Check that reports can be compared (same scope). Returns errors."""
    errors = []
    if len(reports) < 2:
        return errors

    ref = reports[0]
    must_match = ["exit_version", "start", "end"]

    # Mode: "independent" and "v14_only" are both single-model tests — compatible
    def normalize_mode(m):
        return "single" if m in ("independent", "v14_only") else m

    for i, r in enumerate(reports[1:], 1):
        for field in must_match:
            if ref.get(field) != r.get(field):
                errors.append(
                    f"Cannot compare: {ref['model']} has {field}={ref.get(field)} "
                    f"but {r['model']} has {field}={r.get(field)}")
        if normalize_mode(ref.get("mode")) != normalize_mode(r.get("mode")):
            errors.append(
                f"Cannot compare: {ref['model']} has mode={ref.get('mode')} "
                f"but {r['model']} has mode={r.get('mode')} (mixed vs single)")

    return errors


def save_report(report: dict, path: Path) -> None:
    """Validate and save report to JSON."""
    errors = validate_report(report)
    if errors:
        raise ValueError(f"Invalid report: {errors}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
