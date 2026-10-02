"""Git provenance for figure sidecars.

A trimmed, dependency-free port of the EPS ``orchestrate.provenance`` module:
the same ``GitProvenance`` record and the three helpers ``paper_plots`` calls,
with the EPS-specific argv[0] tracking probe dropped.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_git(args: list[str], cwd: Path | None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd or REPO_ROOT), *args],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return out.stdout


@dataclass(frozen=True)
class GitProvenance:
    """Commit identity plus working-tree cleanliness at render time."""

    commit_sha: str  # 8-hex short sha, or "unknown"
    dirty: bool | None  # None when the status probe failed
    dirty_paths: list[str] = field(default_factory=list)
    argv0_path: str | None = None
    argv0_state: str | None = None
    commit_sha_full: str | None = None


def _git_short_sha(cwd: Path | None = None) -> str:
    """Return the 8-hex short sha of HEAD, or ``"unknown"`` on any failure."""
    out = _run_git(["rev-parse", "HEAD"], cwd)
    return out.strip()[:8] if out and out.strip() else "unknown"


git_short_sha = _git_short_sha


def git_provenance(cwd: Path | None = None, argv0: str | None = None) -> GitProvenance:
    """Capture the current git provenance for reproducibility metadata."""
    full = _run_git(["rev-parse", "HEAD"], cwd)
    full = full.strip() if full and full.strip() else None
    status = _run_git(["status", "--porcelain", "-uno"], cwd)
    if status is None:
        dirty, paths = None, []
    else:
        paths = [line[3:] for line in status.splitlines() if line.strip()]
        dirty = bool(paths)
    return GitProvenance(
        commit_sha=full[:8] if full else "unknown",
        dirty=dirty,
        dirty_paths=paths,
        argv0_path=argv0,
        argv0_state=None,
        commit_sha_full=full,
    )


def commit_string(prov: GitProvenance) -> str:
    """Human-legible ``<sha>`` or ``<sha>+dirty`` for non-JSON channels."""
    return f"{prov.commit_sha}+dirty" if prov.dirty else prov.commit_sha


def as_metadata_dict(prov: GitProvenance, *, phase: str | None = None) -> dict[str, object]:
    """Render the provenance as reproducibility-metadata dict fields."""
    out: dict[str, object] = {
        "commit_sha": prov.commit_sha,
        "commit_sha_full": prov.commit_sha_full,
        "dirty": prov.dirty,
        "dirty_paths": list(prov.dirty_paths),
    }
    if phase is not None:
        out["phase"] = phase
    return out
