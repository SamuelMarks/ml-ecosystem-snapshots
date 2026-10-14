"""Thread-safe snapshot repository and caching layer for MCP tool queries."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional
import ml_ecosystem_snapshots.mcp_server as _mcp_server


class SnapshotRepository:
    """Thread-safe repository managing in-memory snapshot caching and lookup."""

    def __init__(
        self, backing_store: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """Initialize the snapshot repository.

        Args:
            backing_store: Optional dictionary backing store; defaults to `_SNAPSHOT_CACHE`.
        """
        self._lock = threading.RLock()
        self._store: Dict[str, Dict[str, Any]] = (
            backing_store if backing_store is not None else _mcp_server._SNAPSHOT_CACHE
        )

    def get(
        self,
        framework: str,
        version: Optional[str] = None,
        prefer_bundled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Retrieve or load a cached snapshot for `framework`.

        Args:
            framework: Target framework identifier (e.g. 'torch', 'jax', 'nvidia_sass').
            version: Optional version constraint string.
            prefer_bundled: Optional flag controlling bundled snapshot preference.

        Returns:
            Categorized snapshot dictionary.
        """
        with self._lock:
            return _mcp_server.get_framework_snapshot(
                framework, version=version, prefer_bundled=prefer_bundled
            )

    def clear(self) -> None:
        """Clear all cached snapshots from the backing store."""
        with self._lock:
            self._store.clear()

    def preload(
        self, framework: str, data: Dict[str, Any], version: Optional[str] = None
    ) -> None:
        """Preload a snapshot dictionary into the repository under `framework` and optional `version`.

        Args:
            framework: Framework name.
            data: Snapshot envelope dictionary.
            version: Optional version string.
        """
        clean_fw = framework.lower().strip()
        cache_key = f"{clean_fw}@{version}" if version else clean_fw
        with self._lock:
            self._store[cache_key] = data


__all__ = ["SnapshotRepository"]
