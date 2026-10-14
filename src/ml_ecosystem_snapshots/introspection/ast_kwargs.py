"""AST analysis utilities for discovering accepted `**kwargs` keys in Python functions."""

from __future__ import annotations

from typing import Any, List, Optional
from ml_ecosystem_snapshots.models import (
    extract_accepted_kwargs_from_ast as _extract_accepted_kwargs_from_ast,
)


def extract_accepted_kwargs_from_ast(func: Any) -> Optional[List[str]]:
    """Statically analyze a function AST to extract keyword arguments accessed via `**kwargs`.

    Args:
        func: Live function or method object to inspect.

    Returns:
        Sorted list of discovered keyword argument names, or None when unavailable.
    """
    return _extract_accepted_kwargs_from_ast(func)


__all__ = ["extract_accepted_kwargs_from_ast"]
