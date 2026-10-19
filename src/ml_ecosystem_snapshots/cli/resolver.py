"""Snapshot path resolution utilities for the CLI."""

from __future__ import annotations

import os

from ml_ecosystem_snapshots.exceptions import SnapshotNotFoundError


def resolve_snapshot_path(path: str, must_exist: bool = False) -> str:
    """Resolve a snapshot path, looking in local and package snapshot directories if needed.

    Args:
        path: The file path, filename, or framework prefix to resolve.
        must_exist: Whether to raise `SnapshotNotFoundError` when the path cannot be resolved.

    Returns:
        The resolved full path if found, or the original path otherwise.

    Raises:
        SnapshotNotFoundError: If `must_exist` is True and no matching snapshot file exists.
    """
    import ml_ecosystem_snapshots.cli as cli_pkg
    from ml_ecosystem_snapshots.index import get_cache_dir
    from ml_ecosystem_snapshots.network_env import get_custom_snapshots_paths

    if os.path.exists(path):
        return path

    if os.path.exists(path + ".json"):
        return path + ".json"

    basename = os.path.basename(path)
    cli_file = os.path.abspath(cli_pkg.__file__)
    if os.path.basename(cli_file) == "__init__.py":
        pkg_dir = os.path.dirname(os.path.dirname(cli_file))
    else:
        pkg_dir = os.path.dirname(cli_file)
    repo_root = os.path.dirname(os.path.dirname(pkg_dir))

    search_dirs: list[str] = get_custom_snapshots_paths() + [
        os.path.join(pkg_dir, "snapshots"),
        os.path.join(pkg_dir, "frameworks"),
        os.path.join(repo_root, "snapshots"),
        os.path.join(get_cache_dir(), "snapshots"),
        os.path.join(os.getcwd(), "snapshots"),
    ]

    seen_dirs = set()
    candidate_dirs: list[str] = []
    for d in search_dirs:
        norm_d = os.path.normpath(d)
        if norm_d not in seen_dirs and os.path.isdir(norm_d):
            seen_dirs.add(norm_d)
            candidate_dirs.append(norm_d)

    for d in candidate_dirs:
        exact_candidate = os.path.join(d, basename)
        if os.path.exists(exact_candidate):
            return exact_candidate
        json_candidate = os.path.join(d, basename + ".json")
        if os.path.exists(json_candidate):
            return json_candidate

    import glob

    all_matches: list[str] = []
    for d in candidate_dirs:
        matches = glob.glob(os.path.join(d, f"{basename}*.json"))
        if matches:
            all_matches.extend(matches)

    if len(all_matches) == 1:
        return all_matches[0]
    elif len(all_matches) > 1:
        return sorted(all_matches)[-1]

    if must_exist:
        raise SnapshotNotFoundError(f"Snapshot file not found: {path}")
    return path


__all__ = ["resolve_snapshot_path"]
