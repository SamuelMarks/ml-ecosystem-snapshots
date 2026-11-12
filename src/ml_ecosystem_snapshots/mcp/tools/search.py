"""Thin MCP adapter for prefix, substring, and FTS5 symbol search (`search_apis`)."""

from __future__ import annotations

from ml_ecosystem_snapshots.grounding.lookup import search_snapshot_apis
from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository, resolve_snapshot
from ml_ecosystem_snapshots.mcp.tools.translation import CONCEPT_ALIAS_MAP


def search_apis(
    framework: str,
    query: str,
    limit: int = 10,
    version: str | None = None,
    repository: SnapshotRepository | None = None,
) -> list[str]:
    """Search available APIs or instructions in a framework snapshot matching a query.

    Args:
        framework: Target framework name.
        query: Search term or prefix.
        limit: Maximum number of matches to return.
        version: Optional framework version string.
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        List of matching API paths or mnemonics.
    """
    return search_snapshot_apis(
        framework=framework,
        query=query,
        snap_loader=lambda: resolve_snapshot(framework, version=version, repository=repository),
        limit=limit,
        version=version,
        concept_map=CONCEPT_ALIAS_MAP,
    )


__all__ = ["search_apis"]
