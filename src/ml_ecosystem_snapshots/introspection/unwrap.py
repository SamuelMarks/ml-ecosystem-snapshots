"""Decorator and compiler wrapper unwrapping utilities for live Python symbols."""

from __future__ import annotations

from typing import Any


def unwrap_decorated_target(obj: Any, max_depth: int = 10) -> Any:
    """Recursively unwrap nested decorators and framework compiler wrappers (`__wrapped__`, `_orig_mod`, etc.).

    Args:
        obj: Target Python callable or class object.
        max_depth: Maximum recursion depth to guard against cyclic wrapper attributes.

    Returns:
        The innermost unwrapped Python callable or class object.
    """
    unwrapped = obj
    for _ in range(max_depth):
        if hasattr(unwrapped, "__wrapped__"):
            unwrapped = unwrapped.__wrapped__
        elif hasattr(unwrapped, "_python_function"):
            unwrapped = unwrapped._python_function
        elif hasattr(unwrapped, "__original_fn"):
            unwrapped = getattr(unwrapped, "__original_fn")
        elif hasattr(unwrapped, "_original_fn"):
            unwrapped = unwrapped._original_fn
        elif hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod
        else:
            break
    return unwrapped


__all__ = ["unwrap_decorated_target"]
