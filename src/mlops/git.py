"""Git integration for MLOps runner.

Provides get_git_info() which returns the current git state as a dict.
Used by runner.py to log code version with every run.
"""

import subprocess
from pathlib import Path


def _run_git(args: list[str], cwd: Path | None = None) -> str | None:
    """Run a git command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_git_info(repo_root: Path | None = None) -> dict:
    """Return the current git state as a dict.

    Returns:
        dict with keys:
            commit:      7-char short hash, or None
            commit_full: 40-char full hash, or None
            branch:      current branch name, or None
            dirty:       True if uncommitted changes exist (bool)
            last_msg:    last commit message one-liner, or None

    All fields are None if not in a git repo or git is not installed.
    dirty defaults to False in that case.
    """
    commit_full = _run_git(["rev-parse", "HEAD"], repo_root)
    if commit_full is None:
        return {
            "commit": None,
            "commit_full": None,
            "branch": None,
            "dirty": False,
            "last_msg": None,
        }

    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    status = _run_git(["status", "--porcelain"], repo_root)
    last_msg = _run_git(["log", "-1", "--format=%s"], repo_root)

    return {
        "commit": commit_full[:7],
        "commit_full": commit_full,
        "branch": branch,
        "dirty": bool(status),
        "last_msg": last_msg,
    }
