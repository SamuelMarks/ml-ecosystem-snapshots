"""Static AST and docstring parser adapters (Griffe and CDD) for Ghost introspection."""

from __future__ import annotations

import logging
import typing
import typing_extensions

from ml_ecosystem_snapshots.network_env import is_offline_mode
from ml_ecosystem_snapshots.utils import (
    get_framework_docstring_parser,
    resolve_griffe_parser,
)

if typing.TYPE_CHECKING:
    import griffe


@typing.runtime_checkable
class GriffeNodeProtocol(typing.Protocol):
    """Structural protocol for Griffe AST nodes (Module, Object, Function, Class, Alias)."""

    is_public: bool


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

_GRIFFE_CACHE: dict[str, GriffeCacheEntry] = {}

logging.getLogger("griffe").setLevel(logging.CRITICAL)


def load_griffe_module_safely(
    module_path: str, framework_name: str | None = None
) -> GriffeCacheEntry:
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
        assert isinstance(
            loaded, (_griffe_mod.Module, _griffe_mod.Object, _griffe_mod.Alias)
        )
        _GRIFFE_CACHE[module_path] = loaded
        return loaded
    except Exception:
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
                assert isinstance(
                    loaded, (_griffe_mod.Module, _griffe_mod.Object, _griffe_mod.Alias)
                )
                _GRIFFE_CACHE[target] = loaded
            except Exception:
                pass


__all__ = [
    "GriffeCacheEntry",
    "_GRIFFE_CACHE",
    "load_griffe_module_safely",
    "preload_griffe_cache",
]
