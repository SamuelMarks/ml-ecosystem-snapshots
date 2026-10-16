"""MCP tool module for API signature retrieval and overload resolution."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ml_ecosystem_snapshots.mcp.cache import get_framework_snapshot


def get_api_signature(
    framework: str, api_path: str, version: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Retrieve the exact ground-truth GhostRef signature for an API.

    Args:
        framework: The framework name.
        api_path: The canonical API path (e.g. 'torch.sum').
        version: Optional framework version string.

    Returns:
        The serialized GhostRef dictionary or None if not found.
    """
    import ml_ecosystem_snapshots.mcp_server as _mcp_server

    active_get_snap = getattr(
        _mcp_server, "get_framework_snapshot", get_framework_snapshot
    )

    try:
        from ml_ecosystem_snapshots.index import lookup_symbol

        cached_sym = lookup_symbol(framework, api_path, version=version)
        if cached_sym:
            return cached_sym
    except Exception:
        pass

    snap = active_get_snap(framework, version=version)
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
