"""Protocol layer: loads and validates experiment protocols.

A Protocol defines the locked rules for a problem (label, splits, required
metrics, required artifacts). Configs vary per run; the protocol stays fixed.

The runner uses validate_metrics() and validate_artifacts() to enforce that
every run conforms to its declared protocol.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


class ProtocolViolationError(Exception):
    """Raised when a run violates its declared protocol."""


@dataclass(frozen=True)
class Protocol:
    """Locked rules for a problem. Immutable contract between runs."""
    name: str
    version: int
    description: str
    label_spec: dict
    data_split: dict
    required_metrics: tuple[str, ...]
    optional_metrics: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    baseline: dict
    tags: dict


def load_protocol(name_or_path: str, protocols_dir: Path | None = None) -> Protocol:
    """Load a protocol from YAML.

    Args:
        name_or_path: either a bare name ("sr_bounce_break_v1") or a full path
                      to a YAML file. Bare names are resolved under
                      <protocols_dir>/<name>.yaml.
        protocols_dir: optional override for the protocol directory.
                       Defaults to configs/protocols/.

    Returns:
        An immutable Protocol instance.
    """
    if name_or_path.endswith(".yaml") or "/" in name_or_path or "\\" in name_or_path:
        path = Path(name_or_path)
    else:
        base = protocols_dir or Path("configs/protocols")
        path = base / f"{name_or_path}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Protocol not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if "name" not in data:
        raise ValueError(f"Protocol YAML {path} is missing required field: 'name'")

    return Protocol(
        name=data["name"],
        version=int(data.get("version", 1)),
        description=data.get("description", ""),
        label_spec=data.get("label_spec", {}),
        data_split=data.get("data_split", {}),
        required_metrics=tuple(data.get("required_metrics", [])),
        optional_metrics=tuple(data.get("optional_metrics", [])),
        required_artifacts=tuple(data.get("required_artifacts", [])),
        baseline=data.get("baseline", {}),
        tags=data.get("tags", {}),
    )


def validate_metrics(logged: dict, protocol: Protocol) -> None:
    """Raise ProtocolViolationError if any required metric is missing.

    Extra metrics (not in required or optional) are allowed.
    """
    missing = set(protocol.required_metrics) - set(logged.keys())
    if missing:
        raise ProtocolViolationError(
            f"Protocol '{protocol.name}' requires metrics that were not logged: "
            f"{sorted(missing)}"
        )


def validate_artifacts(run_dir: Path, protocol: Protocol) -> None:
    """Raise ProtocolViolationError if any required artifact file is missing.

    Paths in required_artifacts are relative to run_dir.
    """
    missing: list[str] = []
    for rel_path in protocol.required_artifacts:
        if not (run_dir / rel_path).exists():
            missing.append(rel_path)
    if missing:
        raise ProtocolViolationError(
            f"Protocol '{protocol.name}' requires artifacts that were not produced: "
            f"{missing} (in {run_dir})"
        )
