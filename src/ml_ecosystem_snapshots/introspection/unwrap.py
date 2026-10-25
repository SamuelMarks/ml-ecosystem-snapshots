"""Decorator and compiler wrapper unwrapping utilities for live Python symbols."""

from __future__ import annotations

from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    InspectableTarget,
    has_runtime_attr,
    read_runtime_attr,
)


def unwrap_decorated_target(
    obj: InspectableTarget, max_depth: int = 10
) -> InspectableTarget:
    """Recursively unwrap nested decorators and framework compiler wrappers (`__wrapped__`, `_orig_mod`, etc.).

    Args:
        obj: Target Python callable or class symbol.
        max_depth: Maximum recursion depth to guard against cyclic wrapper attributes.

    Returns:
        The innermost unwrapped Python callable or class symbol.
    """
    unwrapped: InspectableTarget = obj
    for _ in range(max_depth):
        if has_runtime_attr(unwrapped, "__wrapped__"):
            unwrapped = read_runtime_attr(unwrapped, "__wrapped__", unwrapped)
        elif has_runtime_attr(unwrapped, "_python_function"):
            unwrapped = read_runtime_attr(unwrapped, "_python_function", unwrapped)
        elif has_runtime_attr(unwrapped, "__original_fn"):
            unwrapped = read_runtime_attr(unwrapped, "__original_fn", unwrapped)
        elif has_runtime_attr(unwrapped, "_original_fn"):
            unwrapped = read_runtime_attr(unwrapped, "_original_fn", unwrapped)
        elif has_runtime_attr(unwrapped, "_orig_mod"):
            unwrapped = read_runtime_attr(unwrapped, "_orig_mod", unwrapped)
        else:
            break
    return unwrapped


__all__ = ["unwrap_decorated_target"]
