"""Promote a model from @staging to @production.

Atomic two-step promotion: copy staging files to production path, then
swap MLflow aliases (retire current production, alias staging as production).

Usage:
    python scripts/mlops/promote.py ML_V1                  # interactive
    python scripts/mlops/promote.py ML_V1 --dry-run        # preview
    python scripts/mlops/promote.py ML_V1 --force          # skip prompt
    python scripts/mlops/promote.py all                    # all registered models
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mlflow

PROMOTION_LOG = REPO_ROOT / "docs/PROMOTION_LOG.md"

# Per-model metadata. Add a block here when a new model is introduced.
MODELS = {
    "ML_V1": {
        "staging_dir": REPO_ROOT / "models/ML_V1_staging",
        "production_dir": REPO_ROOT / "models/ML_V1",
    },
    "ML_V2_ATTENTION": {
        "staging_dir": REPO_ROOT / "models/ML_V2_ATTENTION_staging",
        "production_dir": REPO_ROOT / "models/ML_V2_ATTENTION",
    },
    "ML_V3": {
        "staging_dir": REPO_ROOT / "models/ML_V3_staging",
        "production_dir": REPO_ROOT / "models/ML_V3",
    },
}

# Minimum metrics a @staging model must beat to be promotable.
GATE_THRESHOLDS = {
    "min_pf": 1.3,
    "min_n_trades": 50,
}


def _client():
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT}/mlflow.db")
    return mlflow.MlflowClient()


def _run_verifier(staging_dir: Path) -> tuple[bool, str]:
    """Run src/mlops/verify.py against staging. Returns (passed, reason)."""
    result = subprocess.run(
        [sys.executable, "src/mlops/verify.py", str(staging_dir.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, "all verifier checks passed"
    return False, f"verifier failed:\n{result.stdout}\n{result.stderr}"


def _check_backtest_metrics(model_key: str) -> tuple[bool, str, dict | None]:
    """Read the latest backtest report for this model and check thresholds."""
    # Map model names to report directory keys
    dir_map = {
        "ml_v1": "ml_v1",
        "ml_v2_attention": "ml_v2_attention",
        "ml_v3": "ml_v3",
        "v14": "v14",
    }
    report_key = dir_map.get(model_key.lower(), model_key.lower())
    report_path = REPO_ROOT / "data/reports" / report_key / "backtest.json"

    if not report_path.exists():
        return False, f"no backtest report found at {report_path.relative_to(REPO_ROOT)}", None

    report = json.loads(report_path.read_text())

    # Schema v2.0 reports have metrics.all.{pf, n, bps, ...}
    m = report.get("metrics", {}).get("all", report.get("metrics", {}))
    pf = m.get("pf", 0)
    n = m.get("n", m.get("n_trades", 0))
    bps = m.get("bps", m.get("total_bps", 0))

    failures = []
    if pf < GATE_THRESHOLDS["min_pf"]:
        failures.append(f"PF {pf} < min {GATE_THRESHOLDS['min_pf']}")
    if n < GATE_THRESHOLDS["min_n_trades"]:
        failures.append(f"trades {n} < min {GATE_THRESHOLDS['min_n_trades']}")

    if failures:
        return False, "; ".join(failures), m
    return True, f"PF {pf}, {n} trades, {bps:+.0f} bps", m


def run_gates(model_name: str, mc: dict, model_key: str) -> tuple[bool, list[str]]:
    """Run all pre-promotion checks. Returns (all_passed, list_of_issues)."""
    issues = []

    if not mc["staging_dir"].exists():
        return False, [f"staging dir missing: {mc['staging_dir']}"]

    manifest = mc["staging_dir"] / "training_manifest.json"
    if not manifest.exists():
        issues.append(f"training_manifest.json missing in {mc['staging_dir']}")

    # Verifier
    passed, msg = _run_verifier(mc["staging_dir"])
    if not passed:
        issues.append(f"verifier: {msg}")

    # Backtest metrics
    bt_passed, bt_msg, _ = _check_backtest_metrics(model_key)
    if not bt_passed:
        issues.append(f"backtest gate: {bt_msg}")

    # MLflow has @staging
    client = _client()
    try:
        mv = client.get_model_version_by_alias(model_name, "staging")
        staging_version = mv.version
    except mlflow.exceptions.MlflowException as e:
        issues.append(f"no @staging alias in MLflow: {e}")
        staging_version = None

    return len(issues) == 0, issues


def promote_one(model_name: str, model_key: str, dry_run: bool = False) -> dict:
    """Promote a single model. Returns result dict for the audit log."""
    mc = MODELS[model_name]
    client = _client()

    print(f"\n{'=' * 60}")
    print(f"Promoting {model_name}")
    print('=' * 60)

    # Gate 1: all pre-checks
    passed, issues = run_gates(model_name, mc, model_key)
    if not passed:
        print(f"PRE-PROMOTION GATES FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        return {"model": model_name, "status": "aborted_gates", "issues": issues}
    print(f"Gates passed.")

    # Read versions
    staging_version = client.get_model_version_by_alias(model_name, "staging").version
    try:
        prod_version = client.get_model_version_by_alias(model_name, "production").version
    except mlflow.exceptions.MlflowException:
        prod_version = None

    print(f"  Current @production: v{prod_version}")
    print(f"  Promoting @staging:  v{staging_version}")

    _, _, metrics = _check_backtest_metrics(model_key)

    # Don't delete if staging == production (same version, nothing to retire)
    should_delete_old = prod_version is not None and str(prod_version) != str(staging_version)

    if dry_run:
        print(f"[DRY RUN] would:")
        print(f"  - Copy {mc['staging_dir'].name}/* -> {mc['production_dir'].name}/")
        print(f"  - MLflow alias: v{staging_version} @staging -> @production")
        if should_delete_old:
            print(f"  - DELETE MLflow version v{prod_version} (was @production)")
        elif prod_version is not None:
            print(f"  - @production already at v{prod_version} (same as staging, no deletion)")
        return {"model": model_name, "status": "dry_run", "would_promote_version": staging_version}

    # Step 1: copy files
    mc["production_dir"].mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_file in mc["staging_dir"].iterdir():
        if src_file.is_file() and src_file.name != "desktop.ini":
            dst = mc["production_dir"] / src_file.name
            try:
                shutil.copy2(src_file, dst)
                copied += 1
            except OSError:
                # Fallback: binary copy for files Windows can't handle via copy2
                with open(src_file, "rb") as fin, open(dst, "wb") as fout:
                    fout.write(fin.read())
                copied += 1
    print(f"Files copied ({copied}): {mc['staging_dir'].name}/ -> {mc['production_dir'].name}/")

    # Step 2: alias swap + delete old
    client.set_registered_model_alias(model_name, "production", staging_version)
    print(f"MLflow: v{staging_version} -> @production")
    if should_delete_old:
        client.delete_model_version(model_name, prod_version)
        print(f"MLflow: v{prod_version} DELETED (was @production)")

    # Step 3: tag the run
    new_run_id = client.get_model_version_by_alias(model_name, "production").run_id
    promoted_at = datetime.now().isoformat(timespec="seconds")
    client.set_model_version_tag(model_name, staging_version, "promoted_at", promoted_at)
    if prod_version is not None:
        client.set_model_version_tag(model_name, staging_version, "replaced_version", str(prod_version))
    print(f"Tagged v{staging_version} run with promoted_at={promoted_at}")

    return {
        "model": model_name,
        "status": "promoted",
        "promoted_version": staging_version,
        "retired_version": prod_version,
        "promoted_at": promoted_at,
        "metrics": metrics,
    }


def append_audit_log(results: list[dict]) -> None:
    PROMOTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not PROMOTION_LOG.exists():
        PROMOTION_LOG.write_text("# Model Promotion Log\n\n")

    entry = ["\n---\n", f"## {datetime.now().isoformat(timespec='seconds')}\n"]
    for r in results:
        if r["status"] == "promoted":
            entry.append(f"### {r['model']}: v{r['retired_version']} -> v{r['promoted_version']}\n")
            entry.append(f"- Retired: v{r['retired_version']}\n")
            entry.append(f"- New production: v{r['promoted_version']}\n")
            m = r.get("metrics") or {}
            n = m.get("n", m.get("n_trades", 0))
            bps = m.get("bps", m.get("total_bps", 0))
            dd = m.get("dd", m.get("max_dd_bps", 0))
            entry.append(f"- Metrics: PF {m.get('pf')}, "
                         f"{n} trades, "
                         f"{bps:+.0f} bps, "
                         f"DD {dd:+.0f}\n")
        else:
            entry.append(f"### {r['model']}: {r['status']}\n")
            if "issues" in r:
                for i in r["issues"]:
                    entry.append(f"  - {i}\n")

    with open(PROMOTION_LOG, "a") as f:
        f.writelines(entry)
    print(f"\nAudit log updated: {PROMOTION_LOG.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model name (e.g. ML_V1) or 'all'")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    # Resolve target list
    if args.model == "all":
        targets = list(MODELS.keys())
    elif args.model in MODELS:
        targets = [args.model]
    else:
        print(f"Unknown model: {args.model}. Available: {list(MODELS.keys())}", file=sys.stderr)
        return 2

    # Key used for backtest report filename lookup (lowercase)
    key_map = {"ML_V1": "ml_v1", "ML_V2_ATTENTION": "ml_v2_attention", "ML_V3": "ml_v3"}

    if not args.force and not args.dry_run:
        print(f"About to promote: {', '.join(targets)}")
        resp = input("Continue? [y/N] ")
        if resp.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1

    results = []
    for name in targets:
        r = promote_one(name, key_map.get(name, name.lower()), dry_run=args.dry_run)
        results.append(r)

    if not args.dry_run:
        append_audit_log(results)

    # Exit summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['model']:<22} {r['status']}")
    print()

    if not args.dry_run and any(r["status"] == "promoted" for r in results):
        print("NEXT STEPS:")
        print("  1. Restart bot to pick up new model files")
        print("  2. Monitor live trades for 3-5 days")
        print("  3. Compare live PnL to honest backtest expectations")
        print()
        print(f"Audit log: {PROMOTION_LOG.relative_to(REPO_ROOT)}")

    return 0 if all(r["status"] in ("promoted", "dry_run") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
