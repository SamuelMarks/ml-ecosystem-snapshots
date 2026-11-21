"""Total snapshot path resolution utilities for the CLI."""

from __future__ import annotations

import glob
import os
import typing

from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result

from ml_ecosystem_snapshots.exceptions import SnapshotNotFoundError
from ml_ecosystem_snapshots.introspection.runtime_boundary import get_member_dict


def _build_candidate_snapshot_dirs() -> tuple[str, ...]:
    """Build the ordered tuple of existing snapshot search directories.

    Returns:
        Deduplicated immutable tuple of normalized existing directory paths.
    """
    import ml_ecosystem_snapshots.cli as cli_pkg
    from ml_ecosystem_snapshots.index import get_cache_dir
    from ml_ecosystem_snapshots.network_env import get_custom_snapshots_paths

    raw_cli_file = get_member_dict(cli_pkg).get("__file__", __file__)
    cli_file = os.path.abspath(str(raw_cli_file))
    if os.path.basename(cli_file) == "__init__.py":
        pkg_dir = os.path.dirname(os.path.dirname(cli_file))
    else:
        pkg_dir = os.path.dirname(cli_file)
    repo_root = os.path.dirname(os.path.dirname(pkg_dir))

    search_dirs: tuple[str, ...] = (
        *get_custom_snapshots_paths(),
        os.path.join(pkg_dir, "snapshots"),
        os.path.join(pkg_dir, "frameworks"),
        os.path.join(repo_root, "snapshots"),
        os.path.join(get_cache_dir(), "snapshots"),
        os.path.join(os.getcwd(), "snapshots"),
    )

    seen_dirs: set[str] = set()
    candidate_dirs: list[str] = []
    for d in search_dirs:
        norm_d = os.path.normpath(d)
        if norm_d not in seen_dirs and os.path.isdir(norm_d):
            seen_dirs.add(norm_d)
            candidate_dirs.append(norm_d)
    return tuple(candidate_dirs)


def _search_candidate_dirs(
    basename: str,
    candidate_dirs: typing.Sequence[str],
) -> str | None:
    """Search `candidate_dirs` for an exact, `.json`, or glob match for `basename`.

    Args:
        basename: Base filename or framework prefix.
        candidate_dirs: Existing snapshot directories to search.

    Returns:
        Resolved file path if found, or `None`.
    """
    for d in candidate_dirs:
        exact_candidate = os.path.join(d, basename)
        if os.path.exists(exact_candidate):
            return exact_candidate
        json_candidate = os.path.join(d, basename + ".json")
        if os.path.exists(json_candidate):
            return json_candidate

    all_matches: list[str] = []
    for d in candidate_dirs:
        matches = sorted(glob.glob(os.path.join(d, f"{basename}*.json")))
        if matches:
            all_matches.extend(matches)

    if len(all_matches) == 1:
        return all_matches[0]
    if len(all_matches) > 1:
        return sorted(all_matches)[-1]
    return None


def resolve_snapshot_path(path: str) -> Result[SnapshotNotFoundError, str]:
    """Resolve a snapshot path totally into `Result[SnapshotNotFoundError, str]`.

    Args:
        path: File path, filename, or framework prefix to resolve.

    Returns:
        `Ok(resolved_path)` if found, else `Err(SnapshotNotFoundError)`.
    """
    if os.path.exists(path):
        return Ok(path)
    if os.path.exists(path + ".json"):
        return Ok(path + ".json")

    basename = os.path.basename(path)
    candidate_dirs = _build_candidate_snapshot_dirs()
    found = _search_candidate_dirs(basename, candidate_dirs)
    if found is not None:
        return Ok(found)

    return Err(SnapshotNotFoundError(f"Snapshot file not found: {path}"))


__all__ = ["resolve_snapshot_path"]
