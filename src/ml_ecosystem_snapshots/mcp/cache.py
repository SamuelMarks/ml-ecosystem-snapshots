"""Thread-safe snapshot repository and caching layer for MCP tool queries."""

from __future__ import annotations

import json
import os
import threading
import typing

from ml_ecosystem_snapshots.api import (
    CollectorFn,
    FRAMEWORK_COLLECTORS as FRAMEWORK_COLLECTORS,
    extract_snapshot as _imported_extract_snapshot,
)
from ml_ecosystem_snapshots.models import JsonValue
from ml_ecosystem_snapshots.network_env import (
    get_custom_snapshots_paths,
    is_offline_mode,
)


class ExtractSnapshotFn(typing.Protocol):
    """Protocol for framework snapshot extraction callable."""

    def __call__(
        self,
        framework_name: str,
        include_nonpublic: bool = False,
    ) -> dict[str, JsonValue]:
        """Extract a framework snapshot dictionary.

        Args:
            framework_name: Target framework identifier.
            include_nonpublic: Whether to include non-public symbols.

        Returns:
            Extracted snapshot dictionary.
        """
        ...


extract_snapshot: ExtractSnapshotFn | None = _imported_extract_snapshot

_SNAPSHOT_CACHE: dict[str, dict[str, JsonValue]] = {}


def get_framework_snapshot(
    framework: str,
    version: str | None = None,
    prefer_bundled: bool | None = None,
) -> dict[str, JsonValue]:
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
    active_cache: dict[str, dict[str, JsonValue]] = _SNAPSHOT_CACHE
    active_extract: ExtractSnapshotFn | None = extract_snapshot
    active_collectors: dict[str, CollectorFn] = FRAMEWORK_COLLECTORS
    active_custom_paths = get_custom_snapshots_paths
    active_offline = is_offline_mode

    clean_fw = framework.lower().strip()
    cache_key = f"{clean_fw}@{version}" if version else clean_fw
    if cache_key not in active_cache:
        loaded_data: dict[str, JsonValue] | None = None
        source_kind = "not_found"
        use_bundled = (
            prefer_bundled
            if prefer_bundled is not None
            else (
                active_extract is None or active_extract == _imported_extract_snapshot
            )
        )
        if use_bundled:
            from ml_ecosystem_snapshots.index import get_cache_dir

            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            clean_ver = version.lstrip("vV") if version else None

            custom_dirs: list[tuple[str, str]] = [
                (cd, "custom_path") for cd in active_custom_paths()
            ]
            candidates: list[tuple[str, str]] = custom_dirs + [
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
                                    data: JsonValue = json.load(f)
                                    if isinstance(data, dict) and "categories" in data:
                                        loaded_data = dict(data)
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
                extracted = active_extract(clean_fw)
                extracted["_snapshot_source"] = "runtime_introspection"
                active_cache[cache_key] = extracted
            except Exception:
                active_cache[cache_key] = {"categories": {}}
        else:
            active_cache[cache_key] = {"categories": {}}
    return active_cache[cache_key]


class SnapshotRepository:
    """Thread-safe repository managing in-memory framework snapshot caches."""

    def __init__(
        self, backing_store: dict[str, dict[str, JsonValue]] | None = None
    ) -> None:
        """Initialize the snapshot repository with a thread lock and backing dictionary.

        Args:
            backing_store: Optional dictionary to store cached snapshots. Defaults to
                the canonical `_SNAPSHOT_CACHE`.
        """
        self._lock = threading.RLock()
        self._store: dict[str, dict[str, JsonValue]] = (
            backing_store if backing_store is not None else _SNAPSHOT_CACHE
        )

    def clear(self) -> None:
        """Clear all cached framework snapshots."""
        with self._lock:
            self._store.clear()

    def get(
        self,
        framework: str,
        version: str | None = None,
        prefer_bundled: bool | None = None,
    ) -> dict[str, JsonValue]:
        """Retrieve or load a snapshot for a target framework.

        Args:
            framework: Framework name (e.g. 'torch', 'jax', 'nvidia_sass').
            version: Optional version constraint.
            prefer_bundled: Optional flag to prefer bundled snapshots over live introspection.

        Returns:
            Categorized snapshot dictionary.
        """
        with self._lock:
            clean_fw = framework.lower().strip()
            cache_key = f"{clean_fw}@{version}" if version else clean_fw
            if cache_key in self._store:
                return self._store[cache_key]
            snap = get_framework_snapshot(
                framework=framework, version=version, prefer_bundled=prefer_bundled
            )
            self._store[cache_key] = snap
            return snap

    def put(
        self,
        framework: str,
        data: dict[str, JsonValue],
        version: str | None = None,
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
        self,
        framework: str,
        data: dict[str, JsonValue],
        version: str | None = None,
    ) -> None:
        """Preload a snapshot dictionary into the repository cache for testing or warming.

        Args:
            framework: Framework name.
            data: Categorized snapshot dictionary.
            version: Optional version string.
        """
        self.put(framework=framework, data=data, version=version)


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
        return repository.get(
            framework=framework, version=version, prefer_bundled=prefer_bundled
        )
    if prefer_bundled is not None:
        return get_framework_snapshot(
            framework, version=version, prefer_bundled=prefer_bundled
        )
    if version is not None:
        return get_framework_snapshot(framework, version=version)
    return get_framework_snapshot(framework)


__all__ = [
    "ExtractSnapshotFn",
    "FRAMEWORK_COLLECTORS",
    "SnapshotRepository",
    "_SNAPSHOT_CACHE",
    "extract_snapshot",
    "get_framework_snapshot",
    "resolve_snapshot",
]
