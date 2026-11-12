"""MCP snapshot cache adapter delegating to `serialization.repository`."""

from __future__ import annotations

import logging

from ml_ecosystem_snapshots.api import (
    FRAMEWORK_COLLECTORS as FRAMEWORK_COLLECTORS,
)
from ml_ecosystem_snapshots.api import (
    CollectorFn,
)
from ml_ecosystem_snapshots.api import (
    extract_snapshot as _imported_extract_snapshot,
)
from ml_ecosystem_snapshots.models import JsonValue
from ml_ecosystem_snapshots.network_env import is_offline_mode
from ml_ecosystem_snapshots.serialization.repository import (
    DEFAULT_SNAPSHOT_REPOSITORY,
    ExtractSnapshotFn,
    SnapshotRepository,
    SnapshotRuntimeContext,
    clear_snapshot_cache,
    get_cached_snapshot,
    load_or_extract_snapshot,
)

_LOGGER = logging.getLogger(__name__)
extract_snapshot: ExtractSnapshotFn | None = _imported_extract_snapshot
_SNAPSHOT_CACHE: dict[str, dict[str, JsonValue]] = DEFAULT_SNAPSHOT_REPOSITORY.store


def get_framework_snapshot(
    framework: str,
    version: str | None = None,
    prefer_bundled: bool | None = None,
) -> dict[str, JsonValue]:
    """Retrieve or build a cached snapshot for a target framework.

    Args:
        framework: Name of the framework (e.g. 'torch', 'jax', 'nvidia_sass').
        version: Optional framework version (e.g. '2.4.0').
        prefer_bundled: Optional flag controlling whether bundled on-disk snapshots are checked first.

    Returns:
        The snapshot dictionary containing categorized GhostRefs.
    """
    active_collectors: dict[str, CollectorFn] = FRAMEWORK_COLLECTORS
    ctx = SnapshotRuntimeContext(
        active_extract=extract_snapshot,
        active_collectors=active_collectors,
        default_extract_ref=_imported_extract_snapshot,
        anchor_file=__file__,
        offline_fn=is_offline_mode,
        logger=_LOGGER,
    )
    return load_or_extract_snapshot(
        framework=framework,
        version=version,
        prefer_bundled=prefer_bundled,
        active_cache=_SNAPSHOT_CACHE,
        runtime_ctx=ctx,
    )


def resolve_snapshot(
    framework: str,
    version: str | None = None,
    prefer_bundled: bool | None = None,
    repository: SnapshotRepository | None = None,
) -> dict[str, JsonValue]:
    """Resolve a framework snapshot via an explicit `SnapshotRepository` or default loader.

    Args:
        framework: Target framework identifier.
        version: Optional framework version constraint.
        prefer_bundled: Optional flag to prefer bundled snapshots.
        repository: Optional injected `SnapshotRepository` instance.

    Returns:
        Categorized snapshot dictionary.
    """
    if repository is not None:
        return repository.get(framework=framework, version=version, prefer_bundled=prefer_bundled)
    if prefer_bundled is not None:
        return get_framework_snapshot(framework, version=version, prefer_bundled=prefer_bundled)
    if version is not None:
        return get_framework_snapshot(framework, version=version)
    return get_framework_snapshot(framework)


__all__ = [
    "DEFAULT_SNAPSHOT_REPOSITORY",
    "ExtractSnapshotFn",
    "FRAMEWORK_COLLECTORS",
    "SnapshotRepository",
    "clear_snapshot_cache",
    "extract_snapshot",
    "get_cached_snapshot",
    "get_framework_snapshot",
    "is_offline_mode",
    "resolve_snapshot",
]
