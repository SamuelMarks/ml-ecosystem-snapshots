"""MCP tool module for API signature retrieval and overload resolution."""

from __future__ import annotations

from ml_ecosystem_snapshots.mcp.cache import SnapshotRepository, resolve_snapshot
from ml_ecosystem_snapshots.models import JsonValue


def get_api_signature(
    framework: str,
    api_path: str,
    version: str | None = None,
    repository: SnapshotRepository | None = None,
) -> dict[str, JsonValue] | None:
    """Look up the exact ground-truth API signature for a given framework and path.

    Args:
        framework: Target framework name (e.g. 'torch', 'jax').
        api_path: Canonical API path or mnemonic (e.g. 'torch.nn.Linear', 'FADD').
        version: Optional framework version string (e.g. '2.4.0').
        repository: Optional `SnapshotRepository` instance for snapshot lookup.

    Returns:
        Dictionary of the matched GhostRef, or None if not found.
    """
    try:
        from ml_ecosystem_snapshots.index import IndexedSymbolRecord, lookup_symbol

        cached_sym: IndexedSymbolRecord | dict[str, JsonValue] | None = lookup_symbol(
            framework, api_path, version=version
        )
        if cached_sym is not None:
            if isinstance(cached_sym, IndexedSymbolRecord):
                return dict(cached_sym.data)
            assert isinstance(cached_sym, dict)
            return dict(cached_sym)
    except Exception:
        pass

    snap = resolve_snapshot(framework, version=version, repository=repository)
    raw_cats = snap.get("categories") or {}
    assert isinstance(raw_cats, dict)
    for _cat, items in raw_cats.items():
        assert isinstance(items, list)
        for item in items:
            assert isinstance(item, dict)
            if (
                item.get("api_path") == api_path
                or item.get("name") == api_path
                or item.get("mnemonic") == api_path
            ):
                return dict(item)
            raw_aliases = item.get("aliases")
            if isinstance(raw_aliases, list) and api_path in raw_aliases:
                return dict(item)
    return None


__all__ = ["get_api_signature"]
