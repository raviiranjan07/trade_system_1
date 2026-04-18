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


def _load_params() -> dict:
    """Load params.yaml once."""
    params_path = REPO_ROOT / "configs/params.yaml"
    if not params_path.exists():
        return {}
    with open(params_path) as f:
        return yaml.safe_load(f) or {}


def _is_ml_model(report: dict) -> bool:
    """True if this report is for an ML model (not V1.4 rule-based)."""
    return report.get("model") != "v14" and report.get("mode") != "v14_only"


def _check_period_leakage(report: dict, params: dict) -> list[str]:
    """Gate 1: Backtest period must NOT overlap with train or val periods.

    ML models must only be final-backtested on the test period.
    Val was used for threshold sweep — including it inflates numbers.
    Train was used for weight fitting — including it is catastrophic.
    """
    errors = []
    if not _is_ml_model(report):
        return errors

    model = report.get("model", "")
    split = params.get(model, {}).get("split", {})
    if not split:
        return errors

    bt_start = report.get("start", "")
    bt_end = report.get("end", "")
    train_start = split.get("train_start", "")
    train_end = split.get("train_end", "")
    val_start = split.get("val_start", "")
    val_end = split.get("val_end", "")
    test_start = split.get("test_start", "")
    test_end = split.get("test_end", "")

    # Backtest must not overlap with training period
    if train_end and bt_start <= train_end:
        errors.append(
            f"LEAKAGE: {model} backtest starts {bt_start} but training ends {train_end}. "
            f"Backtest overlaps with training data!")

    # Backtest must not overlap with validation period
    if val_start and val_end and bt_start < val_end:
        errors.append(
            f"LEAKAGE: {model} backtest starts {bt_start} but val period is "
            f"{val_start} to {val_end}. Sweep optimized on val — "
            f"backtest must start at {test_start}!")

    # Backtest must not extend beyond test period
    if test_end and bt_end > test_end:
        errors.append(
            f"LEAKAGE: {model} backtest ends {bt_end} but test period ends {test_end}. "
            f"Data beyond test_end has no split assignment!")

    return errors


def _check_param_mismatch(report: dict, params: dict) -> list[str]:
    """Gate 2: Warn (not block) if sweep winner differs from inference params.

    This is a WARNING because:
    - In a DVC pipeline, sweep runs before backtest in the same run —
      no chance to update params.yaml in between
    - The sweep result might be from a contaminated (old) run
    - User must manually review sweep results and decide whether to adopt

    Returns warnings as errors only if sweep was validated on clean data.
    """
    warnings = []
    if not _is_ml_model(report):
        return warnings

    model = report.get("model", "")
    inference = params.get(model, {}).get("inference", {})
    if not inference:
        return warnings

    # Check if sweep report exists
    sweep_path = REPO_ROOT / "data/reports" / model / "threshold_sweep.json"
    if not sweep_path.exists():
        return warnings  # no sweep done yet — nothing to compare

    with open(sweep_path) as f:
        sweep = json.load(f)

    # Only trust sweep if it was run on val period (not contaminated)
    val_period = sweep.get("val_period", "")
    split = params.get(model, {}).get("split", {})
    expected_val = f"{split.get('val_start', '')} to {split.get('val_end', '')}"
    if val_period != expected_val:
        # Sweep was run on different dates — don't trust its winner
        return warnings

    winner = sweep.get("winner", {})
    if not winner:
        return warnings

    sweep_long = winner.get("conf_long")
    sweep_short = winner.get("conf_short")
    actual_long = inference.get("conf_long")
    actual_short = inference.get("conf_short")

    mismatches = []
    if sweep_long is not None and actual_long != sweep_long:
        mismatches.append(f"conf_long: sweep={sweep_long}, params={actual_long}")
    if sweep_short is not None and actual_short != sweep_short:
        mismatches.append(f"conf_short: sweep={sweep_short}, params={actual_short}")

    if mismatches:
        warnings.append(
            f"PARAM INFO: {model} sweep winner differs from params.yaml: "
            f"{'; '.join(mismatches)}. "
            f"Review sweep results and update params.yaml if appropriate.")

    return warnings


def _check_model_staleness(report: dict) -> list[str]:
    """Gate 3: Model must have been trained on current data.

    Reads training_manifest.json from the model directory and checks that
    the data hashes match current files. If data changed but model wasn't
    retrained, backtest results are stale.
    """
    errors = []
    if not _is_ml_model(report):
        return errors

    model = report.get("model", "")

    # Map model name to directory
    model_dirs = {
        "ml_v1": REPO_ROOT / "models/ML_V1_staging",
        "ml_v2_attention": REPO_ROOT / "models/ML_V2_ATTENTION",
        "ml_v3": REPO_ROOT / "models/ML_V3_staging",
    }
    model_dir = model_dirs.get(model)
    if not model_dir or not model_dir.exists():
        errors.append(f"STALENESS: {model} model directory not found: {model_dir}")
        return errors

    manifest_path = model_dir / "training_manifest.json"
    if not manifest_path.exists():
        errors.append(
            f"STALENESS: {model} has no training_manifest.json in {model_dir.name}. "
            f"Cannot verify if model is trained on current data!")
        return errors

    import hashlib

    def _md5_of(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    with open(manifest_path) as f:
        manifest = json.load(f)

    data = manifest.get("data", {})
    for key, expected_md5 in data.items():
        if not key.endswith("_md5"):
            continue
        base = key[:-4]  # strip "_md5"
        rel_path = data.get(f"{base}_path")
        if not rel_path:
            continue
        full = REPO_ROOT / rel_path
        if not full.exists():
            errors.append(f"STALENESS: {model} trained on {rel_path} but file is missing!")
            continue
        actual_md5 = _md5_of(full)
        if actual_md5 != expected_md5:
            errors.append(
                f"STALENESS: {model} was trained on {base} with hash {expected_md5[:8]}... "
                f"but current file hash is {actual_md5[:8]}... "
                f"Data changed — model needs retraining!")

    return errors


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

    # === ML-specific gates (skip for V1.4) ===
    params = _load_params()

    # Gate 1 (HARD): Period leakage — backtest must not overlap train/val
    errors.extend(_check_period_leakage(report, params))

    # Gate 2 (SOFT): Param mismatch — logged as warning, doesn't block
    # (sweep runs in same pipeline as backtest — no chance to update in between)
    warnings = _check_param_mismatch(report, params)
    if warnings:
        import logging
        _logger = logging.getLogger(__name__)
        for w in warnings:
            _logger.warning(w)

    # Gate 3 (HARD): Model staleness — data changed but model not retrained
    errors.extend(_check_model_staleness(report))

    return errors


def validate_comparable(reports: list[dict]) -> list[str]:
    """Check that reports can be compared (same scope). Returns errors.

    Rules:
    - exit_version must match across all reports
    - mode must be compatible (independent/v14_only are both single-model)
    - date ranges must match WITHIN the same category:
      * All ML models must have the same start/end (test period)
      * V1.4 may have a wider range (no leakage risk) — that's OK
    """
    errors = []
    if len(reports) < 2:
        return errors

    # exit_version must match
    ref = reports[0]
    for r in reports[1:]:
        if ref.get("exit_version") != r.get("exit_version"):
            errors.append(
                f"Cannot compare: {ref['model']} has exit_version={ref.get('exit_version')} "
                f"but {r['model']} has exit_version={r.get('exit_version')}")

    # Mode compatibility
    def normalize_mode(m):
        return "single" if m in ("independent", "v14_only") else m

    for r in reports[1:]:
        if normalize_mode(ref.get("mode")) != normalize_mode(r.get("mode")):
            errors.append(
                f"Cannot compare: {ref['model']} has mode={ref.get('mode')} "
                f"but {r['model']} has mode={r.get('mode')} (mixed vs single)")

    # Date ranges: ML models must match each other
    ml_reports = [r for r in reports if r.get("mode") != "v14_only"]
    if len(ml_reports) >= 2:
        ml_ref = ml_reports[0]
        for r in ml_reports[1:]:
            for field in ["start", "end"]:
                if ml_ref.get(field) != r.get(field):
                    errors.append(
                        f"Cannot compare ML models: {ml_ref['model']} has {field}={ml_ref.get(field)} "
                        f"but {r['model']} has {field}={r.get(field)}")

    return errors


def save_report(report: dict, path: Path) -> None:
    """Validate and save report to JSON."""
    errors = validate_report(report)
    if errors:
        raise ValueError(f"Invalid report: {errors}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
