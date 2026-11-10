"""Static AST and docstring parser adapters (Griffe and CDD) for Ghost introspection."""

from __future__ import annotations

import logging
import threading
import typing

import typing_extensions

from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    GriffeNodeProtocol,
)
from ml_ecosystem_snapshots.network_env import is_offline_mode
from ml_ecosystem_snapshots.utils import (
    get_framework_docstring_parser,
    resolve_griffe_parser,
)

if typing.TYPE_CHECKING:
    import griffe

logger = logging.getLogger("ml_ecosystem_snapshots.introspection.static_parsers")


class GriffeLoaderProto(typing.Protocol):
    """Protocol for a callable Griffe loader stored in `_GRIFFE_CACHE`."""

    def __call__(
        self,
        objspec: str | None = None,
        /,
        *,
        docstring_parser: griffe.Parser | None = None,
    ) -> griffe.Module | griffe.Object | griffe.Alias | GriffeNodeProtocol:
        """Load a Griffe module or object.

        Args:
            objspec: Optional module or package name.
            docstring_parser: Optional Griffe docstring parser.

        Returns:
            Loaded Griffe module, object, or alias.
        """
        ...


GriffeCacheEntry: typing_extensions.TypeAlias = "griffe.Module | griffe.Object | griffe.Alias | GriffeNodeProtocol | GriffeLoaderProto | None"


class GriffeModuleCache:
    """Thread-safe cache for statically loaded Griffe AST modules and objects."""

    def __init__(self, backing_store: dict[str, GriffeCacheEntry] | None = None) -> None:
        """Initialize the thread-safe Griffe module cache.

        Args:
            backing_store: Optional dictionary backing store.
        """
        self._lock = threading.RLock()
        self._store: dict[str, GriffeCacheEntry] = backing_store if backing_store is not None else {}

    @property
    def store(self) -> dict[str, GriffeCacheEntry]:
        """Return the underlying cache dictionary.

        Returns:
            Dictionary mapping module keys to `GriffeCacheEntry` values.
        """
        return self._store

    def get(self, key: str) -> GriffeCacheEntry:
        """Retrieve a cached Griffe entry by key.

        Args:
            key: Module path or cache key.

        Returns:
            Cached `GriffeCacheEntry` or `None` if absent.
        """
        with self._lock:
            return self._store.get(key)

    def contains(self, key: str) -> bool:
        """Return whether `key` is present in the cache.

        Args:
            key: Module path or cache key.

        Returns:
            True if `key` is stored in the cache.
        """
        with self._lock:
            return key in self._store

    def put(self, key: str, value: GriffeCacheEntry) -> None:
        """Store a `GriffeCacheEntry` under `key`.

        Args:
            key: Module path or cache key.
            value: Griffe AST node or loader callable to cache.
        """
        with self._lock:
            self._store[key] = value

    def clear(self) -> None:
        """Clear all entries from the Griffe module cache."""
        with self._lock:
            self._store.clear()


_DEFAULT_GRIFFE_CACHE = GriffeModuleCache()
_GRIFFE_CACHE: dict[str, GriffeCacheEntry] = _DEFAULT_GRIFFE_CACHE.store

logging.getLogger("griffe").setLevel(logging.CRITICAL)


def load_griffe_module_safely(module_path: str, framework_name: str | None = None) -> GriffeCacheEntry:
    """Statically load a Python module or package via Griffe with caching and parser resolution.

    Args:
        module_path: Dotted Python module path (e.g. 'chex', 'grain', 'orbax.checkpoint', 'optax').
        framework_name: Optional framework identifier used to select the Griffe docstring parser.

    Returns:
        Loaded Griffe AST node (`Module`, `Object`, `Alias`, or `GriffeNodeProtocol`), or `None`
        if offline mode is active or static loading fails.
    """
    if is_offline_mode():
        return None
    if module_path in _GRIFFE_CACHE:
        return _GRIFFE_CACHE[module_path]
    try:
        import griffe as _griffe_mod

        fw_key = framework_name or module_path.split(".")[0]
        parser = resolve_griffe_parser(get_framework_docstring_parser(fw_key))
        loaded = _griffe_mod.load(module_path, docstring_parser=parser)
        assert isinstance(loaded, (_griffe_mod.Module, _griffe_mod.Object, _griffe_mod.Alias))
        _GRIFFE_CACHE[module_path] = loaded
        return loaded
    except (ImportError, OSError, SyntaxError, ValueError, KeyError, RuntimeError, TypeError, AssertionError) as exc:
        logger.debug("Failed to statically load Griffe module '%s': %s", module_path, exc)
        return None


def preload_griffe_cache(frameworks: list[str] | None = None) -> None:
    """Preload Griffe AST trees for target ML frameworks into the module cache.

    Args:
        frameworks: Optional list of package names to preload. If None,
            preloads common ML packages if installed.
    """
    targets = (
        frameworks
        if frameworks is not None
        else [
            "torch",
            "jax",
            "tensorflow",
            "keras",
            "mlx",
            "scipy",
            "chex",
            "grain",
            "orbax",
            "optax",
        ]
    )
    for target in targets:
        if target not in _GRIFFE_CACHE:
            try:
                import griffe as _griffe_mod

                parser = resolve_griffe_parser(get_framework_docstring_parser(target))
                loaded = _griffe_mod.load(target, docstring_parser=parser)
                assert isinstance(loaded, (_griffe_mod.Module, _griffe_mod.Object, _griffe_mod.Alias))
                _GRIFFE_CACHE[target] = loaded
            except (ImportError, OSError, SyntaxError, ValueError, KeyError, RuntimeError, TypeError, AssertionError) as exc:
                logger.debug("Failed to preload Griffe cache for '%s': %s", target, exc)


__all__ = [
    "GriffeCacheEntry",
    "GriffeLoaderProto",
    "GriffeModuleCache",
    "GriffeNodeProtocol",
    "load_griffe_module_safely",
    "preload_griffe_cache",
]
