"""Thread-safe snapshot repository and caching layer for MCP tool queries."""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from ml_ecosystem_snapshots.network_env import (
    get_custom_snapshots_paths,
    is_offline_mode,
)

_SNAPSHOT_CACHE: Dict[str, Dict[str, Any]] = {}


def get_framework_snapshot(
    framework: str,
    version: Optional[str] = None,
    prefer_bundled: Optional[bool] = None,
) -> Dict[str, Any]:
    """Retrieve or build a cached snapshot for a target framework.

    Args:
        framework: Name of the framework (e.g. 'torch', 'jax', 'nvidia_sass').
        version: Optional framework version (e.g. '2.4.0').
        prefer_bundled: Optional flag controlling whether bundled on-disk snapshots
            are checked before calling `extract_snapshot`. Defaults to True when
            using the standard `extract_snapshot` implementation.

    Returns:
        The snapshot dictionary containing categorized GhostRefs.
    """
    import ml_ecosystem_snapshots.mcp_server as _mcp_server

    active_cache: Dict[str, Dict[str, Any]] = getattr(
        _mcp_server, "_SNAPSHOT_CACHE", _SNAPSHOT_CACHE
    )
    active_extract: Optional[Callable[..., Dict[str, Any]]] = getattr(
        _mcp_server, "extract_snapshot", None
    )
    active_extract_default: Optional[Callable[..., Dict[str, Any]]] = getattr(
        _mcp_server, "_extract_snapshot_fn", None
    )
    active_collectors: Dict[str, Any] = getattr(_mcp_server, "FRAMEWORK_COLLECTORS", {})
    active_custom_paths = getattr(
        _mcp_server, "get_custom_snapshots_paths", get_custom_snapshots_paths
    )
    active_offline = getattr(_mcp_server, "is_offline_mode", is_offline_mode)

    clean_fw = framework.lower().strip()
    cache_key = f"{clean_fw}@{version}" if version else clean_fw
    if cache_key not in active_cache:
        loaded_data = None
        source_kind = "not_found"
        use_bundled = (
            prefer_bundled
            if prefer_bundled is not None
            else (active_extract is None or active_extract is active_extract_default)
        )
        if use_bundled:
            from ml_ecosystem_snapshots.index import get_cache_dir

            mcp_file = getattr(_mcp_server, "__file__", None)
            base_dir = (
                os.path.dirname(mcp_file)
                if mcp_file
                else os.path.dirname(os.path.dirname(__file__))
            )
            clean_ver = version.lstrip("vV") if version else None

            custom_dirs: List[Tuple[str, str]] = [
                (cd, "custom_path") for cd in active_custom_paths()
            ]
            candidates: List[Tuple[str, str]] = custom_dirs + [
                (os.path.join(base_dir, "snapshots"), "bundled_package"),
                (os.path.join(base_dir, "frameworks"), "bundled_package"),
                (os.path.join(get_cache_dir(), "snapshots"), "local_cache"),
            ]
            for candidate_dir, source_label in candidates:
                if os.path.isdir(candidate_dir):
                    candidate_files = [
                        f for f in os.listdir(candidate_dir) if f.endswith(".json")
                    ]
                    candidate_files.sort(reverse=True)
                    for fname in candidate_files:
                        matches = False
                        if clean_ver:
                            if (
                                fname
                                in (
                                    f"{clean_fw}_v{clean_ver}.json",
                                    f"{clean_fw}_{clean_ver}.json",
                                    f"{clean_fw}_v{version}.json",
                                    f"{clean_fw}_{version}.json",
                                )
                                or fname.startswith(f"{clean_fw}_v{clean_ver}")
                                or fname.startswith(f"{clean_fw}_{clean_ver}")
                            ):
                                matches = True
                        else:
                            if fname.startswith(clean_fw) or fname.startswith(
                                f"{clean_fw}_"
                            ):
                                matches = True
                        if matches:
                            try:
                                with open(
                                    os.path.join(candidate_dir, fname),
                                    "r",
                                    encoding="utf-8",
                                ) as f:
                                    data = json.load(f)
                                    if "categories" in data:
                                        loaded_data = data
                                    elif isinstance(data, dict) and (
                                        "operations" in data or "instructions" in data
                                    ):
                                        items = (
                                            data.get("operations")
                                            or data.get("instructions")
                                            or []
                                        )
                                        loaded_data = {
                                            **data,
                                            "categories": {"UTIL": items},
                                        }
                                    elif isinstance(data, list):
                                        loaded_data = {"categories": {"UTIL": data}}
                                    source_kind = source_label
                                    break
                            except (OSError, ValueError):
                                pass
                    if loaded_data:
                        break

        if loaded_data:
            loaded_data["_snapshot_source"] = source_kind
            active_cache[cache_key] = loaded_data
        elif (
            active_extract is not None
            and clean_fw in active_collectors
            and (
                not active_offline()
                or clean_fw
                in (
                    "nvidia_sass",
                    "amd_rdna",
                    "mlir",
                    "stablehlo",
                    "nvidia_ptx",
                    "html_dsl",
                    "latex_dsl",
                    "tikz",
                )
            )
        ):
            try:
                data = active_extract(clean_fw)
                data["_snapshot_source"] = "runtime_introspection"
                active_cache[cache_key] = data
            except Exception:
                active_cache[cache_key] = {"categories": {}}
        else:
            active_cache[cache_key] = {"categories": {}}
    return active_cache[cache_key]


class SnapshotRepository:
    """Thread-safe repository managing in-memory framework snapshot caches."""

    def __init__(
        self, backing_store: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """Initialize the snapshot repository with a thread lock and backing dictionary.

        Args:
            backing_store: Optional dictionary to store cached snapshots. Defaults to
                the canonical `_SNAPSHOT_CACHE`.
        """
        self._lock = threading.RLock()
        self._store: Dict[str, Dict[str, Any]] = (
            backing_store if backing_store is not None else _SNAPSHOT_CACHE
        )

    def clear(self) -> None:
        """Clear all cached framework snapshots."""
        with self._lock:
            self._store.clear()

    def get(
        self,
        framework: str,
        version: Optional[str] = None,
        prefer_bundled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Retrieve or load a snapshot for a target framework.

        Args:
            framework: Framework name (e.g. 'torch', 'jax', 'nvidia_sass').
            version: Optional version constraint.
            prefer_bundled: Optional flag to prefer bundled snapshots over live introspection.

        Returns:
            Categorized snapshot dictionary.
        """
        with self._lock:
            return get_framework_snapshot(
                framework=framework, version=version, prefer_bundled=prefer_bundled
            )

    def put(
        self, framework: str, data: Dict[str, Any], version: Optional[str] = None
    ) -> None:
        """Explicitly insert a snapshot dictionary into the repository cache.

        Args:
            framework: Framework name.
            data: Categorized snapshot dictionary.
            version: Optional version string.
        """
        clean_fw = framework.lower().strip()
        cache_key = f"{clean_fw}@{version}" if version else clean_fw
        with self._lock:
            self._store[cache_key] = data

    def preload(
        self, framework: str, data: Dict[str, Any], version: Optional[str] = None
    ) -> None:
        """Preload a snapshot dictionary into the repository cache for testing or warming.

        Args:
            framework: Framework name.
            data: Categorized snapshot dictionary.
            version: Optional version string.
        """
        self.put(framework=framework, data=data, version=version)


__all__ = [
    "SnapshotRepository",
    "_SNAPSHOT_CACHE",
    "get_framework_snapshot",
]
