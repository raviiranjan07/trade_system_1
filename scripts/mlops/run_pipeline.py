"""Run a model's end-to-end DVC pipeline by name.

Reads configs/pipelines.yaml and executes `dvc repro` with the declared
stages. Future models are added by editing the yaml only.

Usage:
    python scripts/mlops/run_pipeline.py ml_v1              # normal (only stale stages)
    python scripts/mlops/run_pipeline.py ml_v1 --force      # force re-run, keeps new version
    python scripts/mlops/run_pipeline.py ml_v1 --test       # force + auto-cleanup new versions
    python scripts/mlops/run_pipeline.py --list             # list available pipelines

`--test` mode runs the full pipeline end-to-end (implies --force) and then
deletes any MLflow model versions created during the run. Use this to validate
the pipeline without polluting the registry with duplicate versions.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "configs/pipelines.yaml"


def load_manifest() -> dict:
    with open(MANIFEST) as f:
        return yaml.safe_load(f) or {}


def list_pipelines(manifest: dict) -> int:
    if not manifest:
        print("No pipelines declared in configs/pipelines.yaml")
        return 0
    print("Available pipelines:\n")
    for name, meta in manifest.items():
        desc = meta.get("description", "")
        stages = meta.get("stages", [])
        print(f"  {name}")
        if desc:
            print(f"    {desc}")
        print(f"    stages ({len(stages)}): {' -> '.join(stages)}")
        print()
    return 0


def _model_registry_name(pipeline_name: str) -> str:
    """Map pipeline name -> MLflow registered model name.

    Convention: pipeline name uppercased. ml_v1 -> ML_V1.
    Adjust here when adding a new pipeline if its registry name differs.
    """
    return pipeline_name.upper()


def _get_client():
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT}/mlflow.db")
    return mlflow.MlflowClient()


def _snapshot_registry(model_name: str) -> tuple[set[str], dict[str, str]]:
    """Return (version_set, {alias: version}) for `model_name`."""
    try:
        client = _get_client()
        import mlflow
    except ImportError:
        return set(), {}
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        rm = client.get_registered_model(model_name)
        return {str(v.version) for v in versions}, dict(rm.aliases)
    except mlflow.exceptions.MlflowException:
        return set(), {}


def _restore_aliases_and_delete(model_name: str, pre_versions: set[str],
                                 pre_aliases: dict[str, str],
                                 post_versions: set[str]) -> None:
    """Test-mode cleanup:
      1. Re-point any alias the test moved back to its pre-run version.
      2. Delete new (non-pre-existing) versions.
    """
    client = _get_client()
    import mlflow

    try:
        rm_post = client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        return
    post_aliases = dict(rm_post.aliases)

    # Restore aliases that moved
    for alias, pre_v in pre_aliases.items():
        if post_aliases.get(alias) != pre_v:
            print(f"  Restoring alias @{alias}: v{post_aliases.get(alias)} -> v{pre_v}")
            client.set_registered_model_alias(model_name, alias, pre_v)

    # Delete new versions
    new_versions = post_versions - pre_versions
    if not new_versions:
        return
    for v in sorted(new_versions, key=int):
        try:
            client.delete_model_version(model_name, v)
            print(f"  Deleted {model_name} v{v}")
        except mlflow.exceptions.MlflowException as e:
            print(f"  Could not delete v{v}: {e}")


def run(model_name: str, force: bool, test: bool) -> int:
    manifest = load_manifest()
    if model_name not in manifest:
        print(f"Unknown pipeline: {model_name}", file=sys.stderr)
        print(f"Available: {', '.join(manifest)}", file=sys.stderr)
        return 2

    stages = manifest[model_name].get("stages", [])
    if not stages:
        print(f"Pipeline '{model_name}' has no stages declared.", file=sys.stderr)
        return 2

    registry_name = _model_registry_name(model_name)
    pre_versions: set[str] = set()
    pre_aliases: dict[str, str] = {}
    if test:
        force = True  # --test implies --force so stages actually re-run
        pre_versions, pre_aliases = _snapshot_registry(registry_name)
        print(f"[TEST MODE] Pre-run {registry_name}: versions={sorted(pre_versions, key=int) or '(none)'}, aliases={pre_aliases or '(none)'}")

    cmd = ["dvc", "repro"]
    if force:
        cmd.append("-f")
    cmd.extend(stages)

    print(f"Running pipeline '{model_name}' ({len(stages)} stages):")
    print(f"  $ {' '.join(cmd)}\n")
    # Force UTF-8 so child process output (e.g. torch.onnx emojis) doesn't
    # crash on Windows cp1252 when stdout is piped.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    rc = subprocess.call(cmd, cwd=REPO_ROOT, env=env)

    if test:
        post_versions, _ = _snapshot_registry(registry_name)
        new_versions = post_versions - pre_versions
        print()
        if new_versions or pre_aliases:
            print(f"[TEST MODE] Restoring {registry_name} to pre-test state:")
            _restore_aliases_and_delete(registry_name, pre_versions, pre_aliases, post_versions)
        else:
            print(f"[TEST MODE] No new versions were created (nothing to clean up).")
        print(f"[TEST MODE] Registry state restored.")

    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a model pipeline by name.")
    parser.add_argument("model_name", nargs="?", help="Pipeline name from configs/pipelines.yaml")
    parser.add_argument("-f", "--force", action="store_true", help="Force re-run (pass -f to dvc repro)")
    parser.add_argument("--test", action="store_true", help="End-to-end test mode: forces run, then deletes any MLflow versions it created")
    parser.add_argument("-l", "--list", action="store_true", help="List declared pipelines")
    args = parser.parse_args()

    if args.list or (args.model_name is None):
        return list_pipelines(load_manifest())

    return run(args.model_name, args.force, args.test)


if __name__ == "__main__":
    sys.exit(main())
