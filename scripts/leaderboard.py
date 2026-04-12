"""Leaderboard: print a sorted comparison table of all runs for an experiment.

Usage:
  PYTHONPATH=src python scripts/leaderboard.py --experiment stage9_sr_advisor
  PYTHONPATH=src python scripts/leaderboard.py --experiment stage9_sr_advisor --sort test_f1
  PYTHONPATH=src python scripts/leaderboard.py --all
"""

import argparse
import csv
import json
from pathlib import Path


DEFAULT_EXPERIMENTS_DIR = Path("experiments")
DEFAULT_REGISTRY_PATH = DEFAULT_EXPERIMENTS_DIR / "mlops_registry.csv"


def load_registry(registry_path: Path) -> list[dict]:
    """Load all rows from the registry CSV."""
    if not registry_path.exists():
        return []
    with open(registry_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_run_metrics(run_dir: Path) -> dict | None:
    """Load metrics.json from a run folder."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def build_leaderboard(
    experiment_name: str | None = None,
    sort_by: str = "primary_metric_value",
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> list[dict]:
    """Build a sorted leaderboard from registry + metrics.json files."""
    rows = load_registry(registry_path)

    if experiment_name:
        rows = [r for r in rows if r.get("experiment_name") == experiment_name]

    entries = []
    for row in rows:
        run_dir = Path(row.get("artifacts_dir", ""))
        metrics = load_run_metrics(run_dir) if run_dir.exists() else None

        entry = {
            "run_id": row.get("run_id", ""),
            "experiment": row.get("experiment_name", ""),
            "protocol": row.get("protocol_name", ""),
            "status": row.get("status", ""),
            "duration_s": row.get("duration_s", ""),
            "git_commit": row.get("git_commit", ""),
            "primary_metric": row.get("primary_metric_name", ""),
            "primary_value": row.get("primary_metric_value", ""),
            "notes": row.get("notes", ""),
        }

        if metrics:
            entry["test_accuracy"] = metrics.get("test_accuracy", "")
            entry["test_f1"] = metrics.get("test_f1", "")
            entry["delta_vs_baseline"] = metrics.get("delta_vs_baseline", "")
            entry["n_test"] = metrics.get("n_test", "")
            entry["test_confident_accuracy"] = metrics.get("test_confident_accuracy", "")

        entries.append(entry)

    def sort_key(e):
        if sort_by == "primary_metric_value":
            val = e.get("primary_value", "")
        else:
            val = e.get(sort_by, "")
        try:
            return float(val)
        except (ValueError, TypeError):
            return -999

    entries.sort(key=sort_key, reverse=True)
    return entries


def print_leaderboard(entries: list[dict], experiment_name: str | None = None):
    """Print a formatted leaderboard table to stdout."""
    if not entries:
        print("No runs found.")
        return

    title = f"LEADERBOARD: {experiment_name}" if experiment_name else "LEADERBOARD: all experiments"
    protocol = entries[0].get("protocol", "")
    if protocol:
        title += f"  (protocol: {protocol})"

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)

    header = f"{'RUN_ID':<30} {'test_acc':>8} {'delta':>7} {'test_f1':>7} {'n_test':>7} {'dur(s)':>7} {'status':<10} {'notes'}"
    print(header)
    print("-" * 120)

    for e in entries:
        test_acc = e.get("test_accuracy", "")
        delta = e.get("delta_vs_baseline", "")
        test_f1 = e.get("test_f1", "")
        n_test = e.get("n_test", "")
        dur = e.get("duration_s", "")
        status = e.get("status", "")
        notes = e.get("notes", "")
        run_id = e.get("run_id", "")

        test_acc_str = f"{float(test_acc):.4f}" if test_acc != "" else "—"
        delta_str = f"+{float(delta):.3f}" if delta != "" else "—"
        f1_str = f"{float(test_f1):.4f}" if test_f1 != "" else "—"
        n_str = str(int(float(n_test))) if n_test != "" else "—"
        dur_str = str(dur) if dur else "—"

        if len(notes) > 40:
            notes = notes[:37] + "..."

        print(f"{run_id:<30} {test_acc_str:>8} {delta_str:>7} {f1_str:>7} {n_str:>7} {dur_str:>7} {status:<10} {notes}")

    print("=" * 120)
    print(f"Total runs: {len(entries)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Print experiment leaderboard")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Filter to this experiment name (default: all)")
    parser.add_argument("--sort", type=str, default="primary_metric_value",
                        help="Sort by this field (default: primary_metric_value)")
    parser.add_argument("--registry", type=str, default=str(DEFAULT_REGISTRY_PATH),
                        help="Path to registry CSV")
    args = parser.parse_args()

    entries = build_leaderboard(
        experiment_name=args.experiment,
        sort_by=args.sort,
        registry_path=Path(args.registry),
    )
    print_leaderboard(entries, args.experiment)


if __name__ == "__main__":
    main()
