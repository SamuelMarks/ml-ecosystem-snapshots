"""Static AST and docstring parser adapters (Griffe and CDD) for Ghost introspection."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ml_ecosystem_snapshots.utils import (
    get_framework_docstring_parser,
    resolve_griffe_parser,
)

_GRIFFE_CACHE: Dict[str, Any] = {}

logging.getLogger("griffe").setLevel(logging.CRITICAL)


def preload_griffe_cache(frameworks: Optional[List[str]] = None) -> None:
    """Preload Griffe AST trees for target ML frameworks into the module cache.

    Args:
        frameworks: Optional list of package names to preload. If None,
            preloads common ML packages if installed.
    """
    import griffe

    targets = (
        frameworks
        if frameworks is not None
        else ["torch", "jax", "tensorflow", "keras", "mlx", "scipy"]
    )
    for target in targets:
        if target not in _GRIFFE_CACHE:
            try:
                parser = resolve_griffe_parser(get_framework_docstring_parser(target))
                _GRIFFE_CACHE[target] = griffe.load(target, docstring_parser=parser)
            except Exception:
                pass


__all__ = ["_GRIFFE_CACHE", "preload_griffe_cache"]
