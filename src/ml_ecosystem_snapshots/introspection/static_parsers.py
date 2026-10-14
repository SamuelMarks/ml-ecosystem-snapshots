"""Static AST and docstring parser adapters (Griffe and CDD) for Ghost introspection."""

from __future__ import annotations

from typing import List, Optional
from ml_ecosystem_snapshots.models import preload_griffe_cache as _preload_griffe_cache


def preload_griffe_cache(frameworks: Optional[List[str]] = None) -> None:
    """Preload Griffe AST trees for target ML frameworks into the module cache.

    Args:
        frameworks: Optional list of package names to preload.
    """
    _preload_griffe_cache(frameworks=frameworks)


__all__ = ["preload_griffe_cache"]
