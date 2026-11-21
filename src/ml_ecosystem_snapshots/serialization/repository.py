"""Thread-safe snapshot repository, closed source ADTs, and caching layer."""

from __future__ import annotations

import enum
import json
import logging
import os
import threading
import typing

import pydantic
import typing_extensions
from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result

from ml_ecosystem_snapshots.exceptions import SnapshotError, SnapshotNotFoundError
from ml_ecosystem_snapshots.models import JsonValue, WireSerializableProtocol
from ml_ecosystem_snapshots.network_env import (
    get_custom_snapshots_paths,
    is_offline_mode,
)
from ml_ecosystem_snapshots.registry import (
    FRAMEWORK_COLLECTORS as _DEFAULT_COLLECTORS,
)
from ml_ecosystem_snapshots.registry import (
    CollectorFn,
)


def _default_extract_snapshot(
    framework_name: str,
    include_nonpublic: bool = False,
) -> ExtractedSnapshotPayload:
    """Delegate live snapshot extraction lazily to `ml_ecosystem_snapshots.api.extract_snapshot`.

    Args:
        framework_name: Target framework identifier.
        include_nonpublic: Whether to include non-public symbols.

    Returns:
        `Result[SnapshotError, ProvenSnapshot[EnvelopeValidPhase]]` from `extract_snapshot`.
    """
    from ml_ecosystem_snapshots.api import extract_snapshot

    return extract_snapshot(framework_name, include_nonpublic=include_nonpublic)


_LOGGER = logging.getLogger(__name__)


class SnapshotSource(str, enum.Enum):
    """Closed enumeration of snapshot resolution origins."""

    BUNDLED_PACKAGE = "bundled_package"
    CUSTOM_PATH = "custom_path"
    LOCAL_CACHE = "local_cache"
    RUNTIME_INTROSPECTION = "runtime_introspection"
    NOT_FOUND = "not_found"


class ResolvedSnapshot(pydantic.BaseModel):
    """Immutable resolved snapshot payload paired with its `SnapshotSource` and version.

    Attributes:
        payload: Categorized snapshot dictionary.
        source: Origin `SnapshotSource` variant.
        version: Resolved snapshot version string (or `'unknown'`).
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    payload: dict[str, JsonValue]
    source: SnapshotSource
    version: str


ExtractedSnapshotPayload: typing_extensions.TypeAlias = (
    "Result[SnapshotError, WireSerializableProtocol | dict[str, JsonValue]]"
    " | WireSerializableProtocol"
    " | dict[str, JsonValue]"
)


class ExtractSnapshotFn(typing.Protocol):
    """Protocol for framework snapshot extraction callables."""

    def __call__(
        self,
        framework_name: str,
        include_nonpublic: bool = False,
    ) -> ExtractedSnapshotPayload:
        """Extract a framework snapshot dictionary or `Result` of `ProvenSnapshot`.

        Args:
            framework_name: Target framework identifier.
            include_nonpublic: Whether to include non-public symbols.

        Returns:
            Extracted snapshot payload or `Result`.
        """
        ...


_CACHE_LOCK = threading.RLock()
_DEFAULT_SNAPSHOT_STORE: dict[str, dict[str, JsonValue]] = {}


def clear_snapshot_cache() -> None:
    """Thread-safely clear all entries in the default snapshot cache."""
    with _CACHE_LOCK:
        _DEFAULT_SNAPSHOT_STORE.clear()


def get_cached_snapshot(cache_key: str) -> dict[str, JsonValue] | None:
    """Thread-safely retrieve a cached snapshot dictionary by key.

    Args:
        cache_key: Cache key (e.g. `'torch'` or `'torch@2.4.0'`).

    Returns:
        Cached snapshot dictionary or `None` if not present.
    """
    with _CACHE_LOCK:
        return _DEFAULT_SNAPSHOT_STORE.get(cache_key)


def _matches_snapshot_filename(
    fname: str,
    clean_fw: str,
    version: str | None,
    clean_ver: str | None,
) -> bool:
    """Check whether a snapshot JSON filename matches the requested framework and version.

    Args:
        fname: Candidate `.json` filename.
        clean_fw: Normalized lowercase framework identifier.
        version: Optional requested version string.
        clean_ver: Optional version string stripped of leading `v`/`V`.

    Returns:
        True if the filename matches the target framework and version constraint.
    """
    if clean_ver:
        return (
            fname
            in (
                f"{clean_fw}_v{clean_ver}.json",
                f"{clean_fw}_{clean_ver}.json",
                f"{clean_fw}_v{version}.json",
                f"{clean_fw}_{version}.json",
            )
            or fname.startswith(f"{clean_fw}_v{clean_ver}")
            or fname.startswith(f"{clean_fw}_{clean_ver}")
        )
    return fname.startswith(clean_fw) or fname.startswith(f"{clean_fw}_")


def _coerce_snapshot_json_payload(data: JsonValue) -> dict[str, JsonValue] | None:
    """Normalize a raw JSON payload purely into a categorized snapshot dictionary.

    Args:
        data: Parsed JSON value from a snapshot file.

    Returns:
        Fresh dictionary containing a `'categories'` mapping, or `None` if unrecognized.
    """
    if isinstance(data, dict) and "categories" in data:
        return dict(data)
    if isinstance(data, dict) and ("operations" in data or "instructions" in data):
        items = data.get("operations") or data.get("instructions") or []
        return {
            **data,
            "categories": {"UTIL": items},
        }
    if isinstance(data, list):
        return {"categories": {"UTIL": data}}
    return None


def _load_bundled_snapshot(
    clean_fw: str,
    version: str | None,
    custom_paths: typing.Sequence[str],
    anchor_file: str | None = None,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, JsonValue] | None, SnapshotSource]:
    """Scan custom, bundled, and local cache directories for a matching snapshot JSON file.

    Args:
        clean_fw: Normalized lowercase framework identifier.
        version: Optional version constraint string.
        custom_paths: Sequence of custom snapshot search directory paths.
        anchor_file: Optional file path whose parent directory contains `snapshots/` and `frameworks/`.
        logger: Optional logger instance for diagnostic messages.

    Returns:
        Tuple of `(loaded_snapshot_dict_or_none, SnapshotSource)`.
    """
    from ml_ecosystem_snapshots.index import get_cache_dir

    active_logger = logger or _LOGGER
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(anchor_file or __file__)))
    clean_ver = version.lstrip("vV") if version else None

    custom_dirs: list[tuple[str, SnapshotSource]] = [(cd, SnapshotSource.CUSTOM_PATH) for cd in custom_paths]
    candidates: list[tuple[str, SnapshotSource]] = custom_dirs + [
        (os.path.join(base_dir, "snapshots"), SnapshotSource.BUNDLED_PACKAGE),
        (os.path.join(base_dir, "frameworks"), SnapshotSource.BUNDLED_PACKAGE),
        (os.path.join(get_cache_dir(), "snapshots"), SnapshotSource.LOCAL_CACHE),
    ]
    for candidate_dir, source_kind in candidates:
        if os.path.isdir(candidate_dir):
            candidate_files = sorted(
                (f for f in os.listdir(candidate_dir) if f.endswith(".json")),
                reverse=True,
            )
            for fname in candidate_files:
                if _matches_snapshot_filename(fname, clean_fw, version, clean_ver):
                    try:
                        with open(os.path.join(candidate_dir, fname), encoding="utf-8") as f:
                            loaded_data = _coerce_snapshot_json_payload(json.load(f))
                            if loaded_data:
                                return loaded_data, source_kind
                    except (OSError, ValueError) as exc:
                        active_logger.debug("Skipping unreadable snapshot file %s: %s", fname, exc)
    return None, SnapshotSource.NOT_FOUND


def _unwrap_extracted_snapshot_dict(
    raw_extracted: ExtractedSnapshotPayload,
) -> Result[SnapshotError, dict[str, JsonValue]]:
    """Normalize an `ExtractedSnapshotPayload` into `Result[SnapshotError, dict[str, JsonValue]]`.

    Args:
        raw_extracted: Result, `WireSerializableProtocol`, or dictionary from an extractor hook.

    Returns:
        `Ok(dict)` or `Err(SnapshotError)`.
    """
    if isinstance(raw_extracted, Err):
        return raw_extracted
    inner = raw_extracted.value if isinstance(raw_extracted, Ok) else raw_extracted
    if isinstance(inner, WireSerializableProtocol):
        return Ok(inner.to_wire_dict())
    return Ok(dict(inner))


class SnapshotRuntimeContext:
    """Runtime hooks passed into `load_or_extract_snapshot`."""

    def __init__(
        self,
        active_extract: ExtractSnapshotFn | None = _default_extract_snapshot,
        active_collectors: dict[str, CollectorFn] | None = None,
        default_extract_ref: ExtractSnapshotFn | None = _default_extract_snapshot,
        anchor_file: str | None = None,
        offline_fn: typing.Callable[[], bool] = is_offline_mode,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize runtime hooks for snapshot loading and extraction.

        Args:
            active_extract: Active framework snapshot extractor callable.
            active_collectors: Optional collector registry mapping.
            default_extract_ref: Reference to the default extractor callable.
            anchor_file: Optional anchor file path for bundled snapshot discovery.
            offline_fn: Zero-argument callable returning True in offline mode.
            logger: Optional logger instance for diagnostic logging.
        """
        self.active_extract = active_extract
        self.active_collectors = active_collectors
        self.default_extract_ref = default_extract_ref
        self.anchor_file = anchor_file
        self.offline_fn = offline_fn
        self.logger = logger


_STATIC_DOMAIN_TARGETS: typing.Final[frozenset[str]] = frozenset(
    {
        "nvidia_sass",
        "amd_rdna",
        "mlir",
        "stablehlo",
        "nvidia_ptx",
        "html_dsl",
        "latex_dsl",
        "tikz",
    }
)


def load_or_extract_snapshot(
    framework: str,
    version: str | None = None,
    prefer_bundled: bool | None = None,
    active_cache: dict[str, dict[str, JsonValue]] | None = None,
    runtime_ctx: SnapshotRuntimeContext | None = None,
) -> dict[str, JsonValue]:
    """Retrieve or build a cached snapshot for a target framework without mutating loaded dicts.

    Args:
        framework: Name of the framework (e.g. `'torch'`, `'jax'`, `'nvidia_sass'`).
        version: Optional framework version (e.g. `'2.4.0'`).
        prefer_bundled: Optional flag controlling whether bundled on-disk snapshots are checked first.
        active_cache: Optional backing store dictionary (defaults to `_DEFAULT_SNAPSHOT_STORE`).
        runtime_ctx: Optional `SnapshotRuntimeContext` specifying runtime hooks.

    Returns:
        Categorized snapshot dictionary.
    """
    ctx = runtime_ctx or SnapshotRuntimeContext()
    store = active_cache if active_cache is not None else _DEFAULT_SNAPSHOT_STORE
    collectors = ctx.active_collectors if ctx.active_collectors is not None else _DEFAULT_COLLECTORS
    active_logger = ctx.logger or _LOGGER

    clean_fw = framework.lower().strip()
    cache_key = f"{clean_fw}@{version}" if version else clean_fw
    if cache_key not in store:
        loaded_data: dict[str, JsonValue] | None = None
        source_kind = SnapshotSource.NOT_FOUND
        use_bundled = (
            prefer_bundled
            if prefer_bundled is not None
            else (ctx.active_extract is None or ctx.active_extract == ctx.default_extract_ref)
        )
        if use_bundled:
            loaded_data, source_kind = _load_bundled_snapshot(
                clean_fw=clean_fw,
                version=version,
                custom_paths=get_custom_snapshots_paths(),
                anchor_file=ctx.anchor_file,
                logger=active_logger,
            )

        if loaded_data:
            store[cache_key] = {**loaded_data, "_snapshot_source": source_kind.value}
        elif (
            ctx.active_extract is not None
            and clean_fw in collectors
            and (not ctx.offline_fn() or clean_fw in _STATIC_DOMAIN_TARGETS)
        ):
            try:
                ext_res = _unwrap_extracted_snapshot_dict(ctx.active_extract(clean_fw))
                if isinstance(ext_res, Ok):
                    store[cache_key] = {
                        **ext_res.value,
                        "_snapshot_source": SnapshotSource.RUNTIME_INTROSPECTION.value,
                    }
                else:
                    active_logger.debug("Runtime snapshot extraction failed for %s: %s", clean_fw, ext_res.error)
                    store[cache_key] = {"categories": {}}
            except (
                ImportError,
                AttributeError,
                TypeError,
                ValueError,
                OSError,
                RuntimeError,
                LookupError,
                SyntaxError,
                SnapshotError,
            ) as exc:
                active_logger.debug("Runtime snapshot extraction failed for %s: %s", clean_fw, exc)
                store[cache_key] = {"categories": {}}
        else:
            store[cache_key] = {"categories": {}}
    return store[cache_key]


class SnapshotLoaderFn(typing.Protocol):
    """Protocol for custom snapshot repository loader callables."""

    def __call__(
        self,
        framework: str,
        version: str | None = None,
        prefer_bundled: bool | None = None,
    ) -> dict[str, JsonValue]:
        """Load a categorized snapshot dictionary for `framework`.

        Args:
            framework: Framework name.
            version: Optional version constraint.
            prefer_bundled: Optional bundled-first preference.

        Returns:
            Categorized snapshot dictionary.
        """
        ...


class SnapshotRepository:
    """Thread-safe repository managing in-memory framework snapshot caches."""

    def __init__(
        self,
        backing_store: dict[str, dict[str, JsonValue]] | None = None,
        loader_fn: SnapshotLoaderFn | None = None,
    ) -> None:
        """Initialize the snapshot repository with a thread lock and backing dictionary.

        Args:
            backing_store: Optional dictionary to store cached snapshots.
            loader_fn: Optional snapshot loader callback `(framework, version, prefer_bundled) -> dict`.
        """
        self._lock = threading.RLock()
        self._store: dict[str, dict[str, JsonValue]] = (
            backing_store if backing_store is not None else _DEFAULT_SNAPSHOT_STORE
        )
        self._loader_fn = loader_fn

    @property
    def store(self) -> dict[str, dict[str, JsonValue]]:
        """Return the underlying snapshot dictionary store.

        Returns:
            Mapping from cache key to snapshot dictionary.
        """
        return self._store

    def clear(self) -> None:
        """Clear all cached framework snapshots."""
        with self._lock:
            self._store.clear()

    def pop(
        self,
        cache_key: str,
        default: dict[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue] | None:
        """Remove and return a cached snapshot entry by key if present.

        Args:
            cache_key: Cache key (e.g. `'torch'` or `'torch@2.4.0'`).
            default: Optional fallback value when `cache_key` is absent.

        Returns:
            Removed snapshot dictionary or `default`.
        """
        with self._lock:
            return self._store.pop(cache_key, default)

    def get(
        self,
        framework: str,
        version: str | None = None,
        prefer_bundled: bool | None = None,
    ) -> dict[str, JsonValue]:
        """Retrieve or load a snapshot for a target framework.

        Args:
            framework: Framework name (e.g. `'torch'`, `'jax'`, `'nvidia_sass'`).
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
            loader = self._loader_fn or load_or_extract_snapshot
            snap = loader(framework=framework, version=version, prefer_bundled=prefer_bundled)
            self._store[cache_key] = snap
            return snap

    def resolve_result(
        self,
        framework: str,
        version: str | None = None,
        prefer_bundled: bool | None = None,
    ) -> Result[SnapshotNotFoundError, ResolvedSnapshot]:
        """Resolve a framework snapshot totally into `Result[SnapshotNotFoundError, ResolvedSnapshot]`.

        Args:
            framework: Framework name.
            version: Optional version constraint.
            prefer_bundled: Optional flag to prefer bundled snapshots.

        Returns:
            `Ok(ResolvedSnapshot)` when categories are non-empty, else `Err(SnapshotNotFoundError)`.
        """
        snap = self.get(framework=framework, version=version, prefer_bundled=prefer_bundled)
        raw_cats = snap.get("categories")
        if not isinstance(raw_cats, dict) or not raw_cats:
            return Err(SnapshotNotFoundError(f"Snapshot not found or empty for framework {framework!r}."))
        raw_src = str(snap.get("_snapshot_source") or SnapshotSource.BUNDLED_PACKAGE.value)
        src = {m.value: m for m in SnapshotSource}.get(raw_src, SnapshotSource.BUNDLED_PACKAGE)
        ver = str(snap.get("version") or version or "unknown")
        return Ok(ResolvedSnapshot(payload=dict(snap), source=src, version=ver))

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
            self._store[cache_key] = dict(data)

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


DEFAULT_SNAPSHOT_REPOSITORY = SnapshotRepository()


def resolve_repository_snapshot(
    framework: str,
    version: str | None = None,
    prefer_bundled: bool | None = None,
    repository: SnapshotRepository | None = None,
) -> dict[str, JsonValue]:
    """Resolve a framework snapshot via an explicit `SnapshotRepository` or `DEFAULT_SNAPSHOT_REPOSITORY`.

    Args:
        framework: Target framework identifier.
        version: Optional framework version constraint.
        prefer_bundled: Optional flag to prefer bundled snapshots.
        repository: Optional injected `SnapshotRepository` instance.

    Returns:
        Categorized snapshot dictionary.
    """
    active_repo = repository if repository is not None else DEFAULT_SNAPSHOT_REPOSITORY
    return active_repo.get(framework=framework, version=version, prefer_bundled=prefer_bundled)


__all__ = [
    "DEFAULT_SNAPSHOT_REPOSITORY",
    "ExtractSnapshotFn",
    "ExtractedSnapshotPayload",
    "ResolvedSnapshot",
    "SnapshotLoaderFn",
    "SnapshotRepository",
    "SnapshotRuntimeContext",
    "SnapshotSource",
    "clear_snapshot_cache",
    "get_cached_snapshot",
    "load_or_extract_snapshot",
    "resolve_repository_snapshot",
]
