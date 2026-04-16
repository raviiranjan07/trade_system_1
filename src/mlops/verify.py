"""Generic model-training verifier.

Reads <model_dir>/training_manifest.json and runs checks against the
declared invariants. Works for any trained model (ML_V1, Attention,
XGBoost, etc.) — training scripts just need to write a valid manifest.

Run:
    python -m mlops.verify models/ML_V1_staging
    python -m mlops.verify models/direction_attention
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


# ---------- file helpers ----------

def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- checks ----------

def check_date_splits_disjoint(manifest: dict) -> list[CheckResult]:
    """Declared date ranges must not overlap."""
    results = []
    split = manifest["split"]
    if split.get("method") != "date_based":
        return results  # skip — different method declared

    t = pd.Timestamp
    train = (t(split["train"][0]), t(split["train"][1]))
    val = (t(split["val"][0]), t(split["val"][1]))
    test = (t(split["test"][0]), t(split["test"][1]))

    def overlaps(a, b):
        return not (a[1] < b[0] or b[1] < a[0])

    results.append(CheckResult(
        "train vs val disjoint",
        not overlaps(train, val),
        f"train={train}, val={val}",
    ))
    results.append(CheckResult(
        "train vs test disjoint",
        not overlaps(train, test),
        f"train={train}, test={test}",
    ))
    results.append(CheckResult(
        "val vs test disjoint",
        not overlaps(val, test),
        f"val={val}, test={test}",
    ))
    results.append(CheckResult(
        "train ends before test starts",
        train[1] < test[0],
        f"train_end={train[1]}, test_start={test[0]}",
    ))
    return results


def check_data_hashes_match_current(manifest: dict) -> list[CheckResult]:
    """Files the model was trained on must still hash the same.

    Manifest format: data.<name>_md5 + data.<name>_path for each tracked file.
    """
    results = []
    data = manifest.get("data", {})
    for key, expected_md5 in data.items():
        if not key.endswith("_md5"):
            continue
        base = key[:-4]  # strip "_md5"
        path = data.get(f"{base}_path")
        if not path:
            continue
        full = REPO_ROOT / path
        if not full.exists():
            results.append(CheckResult(f"{base} file exists", False, f"missing: {full}"))
            continue
        actual = _md5_file(full)
        results.append(CheckResult(
            f"{base} matches training-time hash",
            actual == expected_md5,
            f"declared={expected_md5[:8]}, actual={actual[:8]}",
        ))
    return results


def check_mlflow_registered(manifest: dict) -> list[CheckResult]:
    """Model must be registered in MLflow with the declared alias."""
    results = []
    ml = manifest.get("mlflow")
    if not ml:
        return results

    try:
        import mlflow
    except ImportError:
        return [CheckResult("mlflow import", False, "mlflow not installed")]

    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT}/mlflow.db")
    client = mlflow.MlflowClient()

    model_name = manifest["model_name"]
    try:
        rm = client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException as e:
        return [CheckResult(f"{model_name} registered", False, str(e))]

    results.append(CheckResult(f"{model_name} exists in registry", True))

    alias = ml.get("alias")
    version = str(ml.get("registered_version")) if ml.get("registered_version") else None
    if alias and version:
        aliased = rm.aliases.get(alias)
        results.append(CheckResult(
            f"{model_name}@{alias} points to v{version}",
            str(aliased) == version,
            f"actual @{alias} -> v{aliased}",
        ))
    return results


def check_scaler_fit_on_train_only(manifest: dict, model_dir: Path) -> list[CheckResult]:
    """If scaler declared fit_on=train_only, verify by recomputing on the declared train range."""
    results = []
    scaler_meta = manifest.get("scaler", {})
    if scaler_meta.get("fit_on") != "train_only":
        return results  # skip — different policy declared

    scaler_path = model_dir / "scaler.npz"
    if not scaler_path.exists():
        return [CheckResult("scaler.npz exists", False, str(scaler_path))]

    # Training script must tell us how to recompute. Look for feature_recipe.
    recipe = manifest.get("feature_recipe")
    if not recipe:
        # Can't verify scaler without knowing which features, which data file,
        # or how to compute them. Still a useful check to flag this.
        return [CheckResult(
            "scaler verification available",
            False,
            "manifest lacks feature_recipe — cannot recompute scaler for check",
        )]

    # Dynamic import — any model can plug in by declaring compute_fn in manifest.
    # Format: "package.module.function_name"
    compute_fn_path = recipe.get("compute_fn")
    if not compute_fn_path:
        return [CheckResult(
            "scaler verification supported",
            False,
            "manifest.feature_recipe.compute_fn is missing",
        )]

    try:
        import importlib
        module_path, fn_name = compute_fn_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        compute_fn = getattr(module, fn_name)
    except (ImportError, AttributeError, ValueError) as e:
        return [CheckResult(
            f"import compute_fn {compute_fn_path}",
            False,
            str(e),
        )]

    src_path = REPO_ROOT / recipe["source_parquet"]
    df = pd.read_parquet(src_path)
    result = compute_fn(df)
    feat_raw = result[0] if isinstance(result, tuple) else result
    feat_raw = np.nan_to_num(feat_raw, nan=0.0, posinf=0.0, neginf=0.0)

    split = manifest["split"]
    train_start = pd.Timestamp(split["train"][0])
    train_end = pd.Timestamp(split["train"][1]) + pd.Timedelta(days=1)
    mask = np.asarray((df.index >= train_start) & (df.index < train_end))
    train_feat = feat_raw[mask]

    actual = np.load(scaler_path)
    actual_mean = actual["mean"]

    # Weaker but robust check: scaler must differ from a full-data scaler.
    # If fit_on=train_only holds, train-subset mean != full-data mean.
    # (Exact match against training-subset mean would require replicating
    # training's label filter — the manifest doesn't declare that yet, so
    # we verify the weaker invariant that catches the main leakage pattern.)
    full_mean = feat_raw.mean(axis=0)
    diff_full = float(np.abs(actual_mean - full_mean).max())
    results.append(CheckResult(
        "scaler is not full-data scaler (train_only declared)",
        diff_full > 1e-4,
        f"|scaler_mean - full_data_mean|.max = {diff_full:.6f}; "
        f"if this is ~0, scaler was likely fit on all data (leakage).",
    ))

    # Sanity: scaler mean is reasonable (close to train-range mean in magnitude)
    train_mean = train_feat.mean(axis=0)
    diff_train = float(np.abs(actual_mean - train_mean).max())
    # Allow up to 3x std deviation — label filtering can shift means modestly
    train_std = train_feat.std(axis=0)
    tolerance = float(3 * np.max(train_std))
    results.append(CheckResult(
        "scaler mean is plausibly within train range",
        diff_train < tolerance,
        f"|scaler_mean - train_range_mean|.max = {diff_train:.4f}, tol = {tolerance:.4f}",
    ))
    return results


# ---------- driver ----------

ALL_CHECKS = [
    check_date_splits_disjoint,
    check_data_hashes_match_current,
    check_mlflow_registered,
    check_scaler_fit_on_train_only,
]


def verify_model(model_dir: Path) -> list[CheckResult]:
    manifest_path = model_dir / "training_manifest.json"
    if not manifest_path.exists():
        return [CheckResult(
            "training_manifest.json exists",
            False,
            f"not found: {manifest_path}",
        )]

    manifest = json.loads(manifest_path.read_text())

    results: list[CheckResult] = [CheckResult("manifest loaded", True)]
    for fn in ALL_CHECKS:
        try:
            # some checks need model_dir; try both signatures
            sig = fn.__code__.co_varnames[:fn.__code__.co_argcount]
            if "model_dir" in sig:
                results.extend(fn(manifest, model_dir))
            else:
                results.extend(fn(manifest))
        except Exception as exc:
            results.append(CheckResult(f"{fn.__name__} crashed", False, str(exc)))

    return results


def print_results(results: list[CheckResult]) -> int:
    fails = 0
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        suffix = f" — {r.detail}" if r.detail else ""
        print(f"  [{tag}] {r.name}{suffix}")
        if not r.passed:
            fails += 1
    print()
    if fails:
        print(f"*** {fails} checks FAILED ***")
    else:
        print(f"All {len(results)} checks PASS.")
    return fails


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python src/mlops/verify.py <model_dir>", file=sys.stderr)
        return 2

    model_dir = Path(sys.argv[1]).resolve()
    if not model_dir.is_dir():
        print(f"not a directory: {model_dir}", file=sys.stderr)
        return 2

    print(f"Verifying {model_dir.relative_to(REPO_ROOT)}:\n")
    results = verify_model(model_dir)
    fails = print_results(results)

    # Write a verification report for downstream stages to depend on.
    report = {
        "model_dir": str(model_dir.relative_to(REPO_ROOT)),
        "checks": [{"name": r.name, "passed": r.passed, "detail": r.detail} for r in results],
        "passed": fails == 0,
        "n_passed": sum(1 for r in results if r.passed),
        "n_failed": fails,
    }
    reports_dir = REPO_ROOT / "data/reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"verification_{model_dir.name}.json"
    out_path.write_text(json.dumps(report, indent=2))

    return 1 if fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
