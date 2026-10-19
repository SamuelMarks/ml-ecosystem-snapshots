"""Static AST and docstring parser adapters (Griffe and CDD) for Ghost introspection."""

from __future__ import annotations

import logging
import typing
from typing_extensions import TypeAlias

import griffe

from ml_ecosystem_snapshots.utils import (
    get_framework_docstring_parser,
    resolve_griffe_parser,
)

GriffeCacheEntry: TypeAlias = typing.Union[
    griffe.Module, griffe.Object, griffe.Alias, object, None
]

_GRIFFE_CACHE: dict[str, GriffeCacheEntry] = {}

logging.getLogger("griffe").setLevel(logging.CRITICAL)


def preload_griffe_cache(frameworks: list[str] | None = None) -> None:
    """Preload Griffe AST trees for target ML frameworks into the module cache.

    Args:
        frameworks: Optional list of package names to preload. If None,
            preloads common ML packages if installed.
    """
    targets = (
        frameworks
        if frameworks is not None
        else ["torch", "jax", "tensorflow", "keras", "mlx", "scipy"]
    )
    for target in targets:
        if target not in _GRIFFE_CACHE:
            try:
                parser = resolve_griffe_parser(get_framework_docstring_parser(target))
                loaded = griffe.load(target, docstring_parser=parser)
                assert isinstance(loaded, (griffe.Module, griffe.Object, griffe.Alias))
                _GRIFFE_CACHE[target] = loaded
            except Exception:
                pass


__all__ = ["GriffeCacheEntry", "_GRIFFE_CACHE", "preload_griffe_cache"]
