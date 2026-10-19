"""MCP tool module for API signature retrieval and overload resolution."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository, resolve_snapshot


def get_api_signature(
    framework: str,
    api_path: str,
    version: Optional[str] = None,
    repository: Optional[SnapshotRepository] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve the exact ground-truth GhostRef signature for an API.

    Args:
        framework: The framework name.
        api_path: The canonical API path (e.g. 'torch.sum').
        version: Optional framework version string.
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        The serialized GhostRef dictionary or None if not found.
    """
    try:
        from ml_ecosystem_snapshots.index import lookup_symbol

        cached_sym = lookup_symbol(framework, api_path, version=version)
        if cached_sym:
            return cached_sym
    except Exception:
        pass

    snap = resolve_snapshot(framework, version=version, repository=repository)
    for _cat, items in snap.get("categories", {}).items():
        for item in items:
            if (
                item.get("api_path") == api_path
                or item.get("name") == api_path
                or item.get("mnemonic") == api_path
            ):
                return dict(item)
            if api_path in item.get("aliases", []):
                return dict(item)
    return None


__all__ = ["get_api_signature"]
