"""Decorator and compiler wrapper unwrapping utilities for live Python symbols."""

from __future__ import annotations

from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    InspectableTarget,
    OriginalFnWrapperProtocol,
    OrigModWrapperProtocol,
    PythonFunctionWrapperProtocol,
    get_member_dict,
)


def unwrap_decorated_target(obj: InspectableTarget, max_depth: int = 10) -> InspectableTarget:
    """Recursively unwrap nested decorators and framework compiler wrappers (`__wrapped__`, `_orig_mod`, etc.).

    Args:
        obj: Target Python callable or class symbol.
        max_depth: Maximum recursion depth to guard against cyclic wrapper attributes.

    Returns:
        The innermost unwrapped Python callable or class symbol.
    """
    unwrapped: InspectableTarget = obj
    for _ in range(max_depth):
        members = get_member_dict(unwrapped)
        if "__wrapped__" in members and members["__wrapped__"] is not None:
            unwrapped = members["__wrapped__"]
        elif isinstance(unwrapped, PythonFunctionWrapperProtocol):
            unwrapped = unwrapped._python_function
        elif "__original_fn" in members and members["__original_fn"] is not None:
            unwrapped = members["__original_fn"]
        elif isinstance(unwrapped, OriginalFnWrapperProtocol):
            unwrapped = unwrapped._original_fn
        elif isinstance(unwrapped, OrigModWrapperProtocol):
            unwrapped = unwrapped._orig_mod
        else:
            break
    return unwrapped


__all__ = ["unwrap_decorated_target"]
