"""Run registry: appends one row per experiment run to a CSV file.

The registry is the human-readable index of all runs. Git-tracked,
grep-able, Excel-compatible. Complements MLflow's database.

Only writes. Never reads, queries, sorts, or deletes.
"""

import csv
from dataclasses import dataclass, fields, asdict
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None  # Windows: file locking not available via fcntl

DEFAULT_REGISTRY_PATH = Path("experiments/mlops_registry.csv")

COLUMNS = [
    "run_id",
    "experiment_name",
    "protocol_name",
    "status",
    "start_time",
    "duration_s",
    "git_commit",
    "git_dirty",
    "git_branch",
    "config_path",
    "artifacts_dir",
    "primary_metric_name",
    "primary_metric_value",
    "notes",
]


@dataclass
class RegistryRow:
    """One row in the registry CSV."""
    run_id: str = ""
    experiment_name: str = ""
    protocol_name: str = ""
    status: str = ""
    start_time: str = ""
    duration_s: float = 0.0
    git_commit: str = ""
    git_dirty: str = ""
    git_branch: str = ""
    config_path: str = ""
    artifacts_dir: str = ""
    primary_metric_name: str = ""
    primary_metric_value: str = ""
    notes: str = ""


def append_run(row: RegistryRow, registry_path: Path | None = None) -> None:
    """Append a single row to the registry CSV.

    Creates the file with a header if it doesn't exist.
    Uses file locking to handle concurrent writes safely.
    """
    path = registry_path or DEFAULT_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists() and path.stat().st_size > 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        if fcntl is not None:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except OSError:
                pass

        writer = csv.DictWriter(f, fieldnames=COLUMNS)

        if not file_exists:
            writer.writeheader()

        row_dict = asdict(row)
        writer.writerow({k: row_dict.get(k, "") for k in COLUMNS})
