"""Experiment runner: the main entry point for tracked experiments.

Usage:
    from mlops.runner import run_experiment

    with run_experiment(
        experiment_name="stage9_sr_advisor",
        protocol_name="sr_bounce_break_v1",
        config_path="experiments/brain/SR/config.yaml",
        params={"lr": 0.001, "model": "conv1d"},
        primary_metric="test_accuracy",
        notes="First baseline run",
    ) as run:
        # ... training code ...
        run.log_metrics(metrics)
        run.log_artifact("model.pt")

Ties together: git.py, tracking.py, protocol.py, registry.py, evaluation.py.
"""

import io
import json
import shutil
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from mlops import git, tracking
from mlops.protocol import Protocol, load_protocol, validate_metrics, validate_artifacts
from mlops.registry import RegistryRow, append_run


class RunnerError(Exception):
    """Raised when the runner itself fails (not the experiment)."""


class _TeeWriter:
    """Write to both a file and the original stream."""

    def __init__(self, file_handle, original_stream):
        self._file = file_handle
        self._original = original_stream

    def write(self, text):
        self._original.write(text)
        self._file.write(text)
        self._file.flush()

    def flush(self):
        self._original.flush()
        self._file.flush()

    def fileno(self):
        return self._original.fileno()


class Run:
    """Active experiment run. Returned by run_experiment() context manager.

    Provides methods to log metrics, artifacts, and tags during the run.
    """

    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        protocol: Protocol,
        primary_metric: str,
        mlflow_run_id: str,
    ):
        self.run_id = run_id
        self.run_dir = run_dir
        self.protocol = protocol
        self.primary_metric = primary_metric
        self.mlflow_run_id = mlflow_run_id
        self._metrics: dict = {}
        self._notes: str = ""

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric."""
        self._metrics[key] = value
        tracking.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Log multiple metrics at once.

        Non-numeric values (e.g., confusion matrix) are stored in metrics.json
        but only numeric values are sent to MLflow.
        """
        self._metrics.update(metrics)
        tracking.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path, artifact_subdir: str | None = None) -> None:
        """Copy a file into the run's artifacts folder and log to MLflow."""
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Artifact not found: {src}")

        dest_dir = self.run_dir / "artifacts"
        if artifact_subdir:
            dest_dir = dest_dir / artifact_subdir
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        tracking.log_artifact(str(dest))

    def log_plot(self, local_path: str | Path) -> None:
        """Copy a plot file into the run's plots folder and log to MLflow."""
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Plot not found: {src}")

        plots_dir = self.run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        dest = plots_dir / src.name
        shutil.copy2(src, dest)
        tracking.log_artifact(str(dest), artifact_path="plots")

    def set_tag(self, key: str, value) -> None:
        """Set a metadata tag on the MLflow run."""
        tracking.set_tag(key, value)

    def set_note(self, text: str) -> None:
        """Set a human-readable note for this run."""
        self._notes = text


def _generate_run_id() -> str:
    """Generate a timestamped run ID: YYYY-MM-DD_HHMMSS_<6hex>."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    return f"{ts}_{short_id}"


@contextmanager
def run_experiment(
    experiment_name: str,
    protocol_name: str,
    config_path: str | Path | None = None,
    params: dict | None = None,
    primary_metric: str = "test_accuracy",
    notes: str = "",
    experiments_dir: Path | None = None,
):
    """Context manager that wraps an experiment run.

    On entry:
        - Loads protocol, gets git info
        - Creates run folder with config/protocol/git snapshots
        - Starts MLflow run, logs params and git tags
        - Tees stdout to stdout.log
        - Yields a Run object

    On normal exit:
        - Validates metrics and artifacts against protocol
        - Writes metrics.json
        - Ends MLflow run (FINISHED)
        - Appends registry row

    On exception:
        - Writes error.log
        - Ends MLflow run (FAILED)
        - Appends registry row with FAILED status
        - Re-raises the exception
    """
    start_time = time.time()
    start_dt = datetime.now()
    run_id = _generate_run_id()
    params = params or {}

    # Load protocol
    protocol = load_protocol(protocol_name)

    # Git info
    git_info = git.get_git_info()

    # Create run folder
    base_dir = experiments_dir or Path("experiments")
    run_dir = base_dir / experiment_name / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)

    # Snapshot config
    if config_path is not None:
        config_src = Path(config_path)
        if config_src.exists():
            shutil.copy2(config_src, run_dir / "config.yaml")

    # Snapshot protocol
    protocol_src = Path("configs/protocols") / f"{protocol_name}.yaml"
    if protocol_src.exists():
        shutil.copy2(protocol_src, run_dir / "protocol.yaml")

    # Write git info
    with open(run_dir / "git_info.txt", "w") as f:
        for k, v in git_info.items():
            f.write(f"{k}: {v}\n")

    # Init MLflow and start run
    tracking.init()
    mlflow_run_id = tracking.start_run(experiment_name, run_name=run_id)

    # Log params and git tags
    tracking.log_params(params)
    tracking.set_tags({
        "git.commit": git_info.get("commit", ""),
        "git.branch": git_info.get("branch", ""),
        "git.dirty": str(git_info.get("dirty", "")),
        "protocol": protocol_name,
        "run_id": run_id,
    })

    # Create Run object
    run = Run(
        run_id=run_id,
        run_dir=run_dir,
        protocol=protocol,
        primary_metric=primary_metric,
        mlflow_run_id=mlflow_run_id,
    )
    run._notes = notes

    # Tee stdout to log file
    log_file = open(run_dir / "stdout.log", "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeWriter(log_file, original_stdout)
    sys.stderr = _TeeWriter(log_file, original_stderr)

    status = "FAILED"
    user_exception = None
    try:
        yield run
        # Normal exit — validate metrics first (before writing metrics.json)
        validate_metrics(run._metrics, protocol)
        status = "FINISHED"

    except Exception as exc:
        user_exception = exc
        with open(run_dir / "error.log", "w") as ef:
            ef.write(traceback.format_exc())

    finally:
        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

        # Write metrics.json (whatever was logged, even on failure)
        with open(run_dir / "metrics.json", "w") as mf:
            json.dump(run._metrics, mf, indent=2, default=str)

        # Validate artifacts AFTER metrics.json is written (it's a required artifact)
        if status == "FINISHED":
            try:
                validate_artifacts(run_dir, protocol)
            except Exception as exc:
                status = "FAILED"
                user_exception = exc
                with open(run_dir / "error.log", "w") as ef:
                    ef.write(traceback.format_exc())

        # End MLflow run
        tracking.end_run(status=status)

        # Compute duration
        duration_s = round(time.time() - start_time, 1)

        # Get primary metric value
        primary_value = run._metrics.get(primary_metric, "")

        # Append to registry
        row = RegistryRow(
            run_id=run_id,
            experiment_name=experiment_name,
            protocol_name=protocol_name,
            status=status,
            start_time=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            duration_s=duration_s,
            git_commit=git_info.get("commit", ""),
            git_dirty=str(git_info.get("dirty", "")),
            git_branch=git_info.get("branch", ""),
            config_path=str(config_path or ""),
            artifacts_dir=str(run_dir),
            primary_metric_name=primary_metric,
            primary_metric_value=str(primary_value),
            notes=run._notes,
        )
        append_run(row)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Run {run_id} — {status}")
        print(f"Protocol: {protocol_name}")
        if primary_value != "":
            print(f"{primary_metric} = {primary_value}")
            baseline = run._metrics.get("baseline_accuracy")
            delta = run._metrics.get("delta_vs_baseline")
            if delta is not None:
                print(f"delta_vs_baseline = {delta}")
        print(f"Duration: {duration_s}s")
        print(f"Folder: {run_dir}")
        print(f"{'='*60}")

        # Re-raise if there was an exception
        if user_exception is not None:
            raise user_exception
