"""Runtime reflection and introspection boundary protocols for live Python targets.

Isolates low-level dynamic Python object attribute access, wrapper unwrapping,
and structural callable/descriptor protocols into a single strongly typed
boundary module, cleanly separating `LivePythonTarget` from `StaticGriffeNode`
and `JsonValue`.
"""

from __future__ import annotations

import enum
import inspect
import logging
import types
import typing

import griffe
import typing_extensions

from ml_ecosystem_snapshots.models import JsonValue, ScalarValue

_LOGGER = logging.getLogger(__name__)

STATIC_GRIFFE_TYPES = (
    griffe.Module,
    griffe.Class,
    griffe.Function,
    griffe.Attribute,
    griffe.Object,
    griffe.Alias,
)
GRIFFE_FUNCTION_TYPES = (griffe.Function, griffe.Alias)
GRIFFE_CLASS_TYPES = (griffe.Class, griffe.Alias)
GRIFFE_DOCSTRING_TYPE = griffe.Docstring
GRIFFE_ERROR_TYPE = griffe.GriffeError

StaticGriffeNode: typing_extensions.TypeAlias = (
    "griffe.Module | griffe.Class | griffe.Function | griffe.Attribute | griffe.Object | griffe.Alias"
)


@typing.runtime_checkable
class PythonFunctionWrapperProtocol(typing.Protocol):
    """Structural protocol for TensorFlow `@tf.function` wrappers exposing `_python_function`."""

    _python_function: LivePythonTarget


@typing.runtime_checkable
class OriginalFnWrapperProtocol(typing.Protocol):
    """Structural protocol for decorator wrappers exposing `_original_fn`."""

    _original_fn: LivePythonTarget


@typing.runtime_checkable
class OrigModWrapperProtocol(typing.Protocol):
    """Structural protocol for `torch.compile` wrappers exposing `_orig_mod`."""

    _orig_mod: LivePythonTarget


@typing.runtime_checkable
class InspectableCallableProtocol(typing.Protocol):
    """Structural protocol for callable Python runtime targets."""

    def __call__(self, *args: JsonValue, **kwargs: JsonValue) -> JsonValue:
        """Invoke the callable target.

        Args:
            args: Variadic positional arguments.
            kwargs: Variadic keyword arguments.

        Returns:
            Result of invocation.
        """


@typing.runtime_checkable
class DescriptorProtocol(typing.Protocol):
    """Structural protocol for Python descriptors (`property`, `classmethod`, `staticmethod`, C-descriptors)."""

    def __get__(
        self, instance: LivePythonTarget, owner: type | None = None
    ) -> InspectableCallableProtocol | ScalarValue:
        """Retrieve the descriptor value from `instance` or `owner`.

        Args:
            instance: Instance object or None.
            owner: Owner class type or None.

        Returns:
            Bound or unbound descriptor value.
        """


@typing.runtime_checkable
class WrappedTargetProtocol(typing.Protocol):
    """Structural protocol for Python runtime instances with custom string/repr formatting."""

    def __repr__(self) -> str:
        """Return string representation of the runtime target.

        Returns:
            String representation.
        """


LivePythonTarget: typing_extensions.TypeAlias = (
    "types.BuiltinFunctionType"
    " | types.MethodType"
    " | types.ModuleType"
    " | types.EllipsisType"
    " | type"
    " | property"
    " | enum.Enum"
    " | InspectableCallableProtocol"
    " | DescriptorProtocol"
    " | WrappedTargetProtocol"
    " | PythonFunctionWrapperProtocol"
    " | OriginalFnWrapperProtocol"
    " | OrigModWrapperProtocol"
    " | tuple[LivePythonTarget, ...]"
    " | list[LivePythonTarget]"
    " | set[ScalarValue]"
    " | frozenset[ScalarValue]"
    " | dict[str, LivePythonTarget]"
    " | ScalarValue"
)

DefaultValueInput: typing_extensions.TypeAlias = LivePythonTarget

_SourceInspectableType: typing_extensions.TypeAlias = (
    "types.ModuleType | type | types.MethodType | types.TracebackType | types.FrameType | types.CodeType"
)


def _is_source_inspectable(
    target: LivePythonTarget | StaticGriffeNode,
) -> typing_extensions.TypeGuard[_SourceInspectableType]:
    """Narrow `target` for `inspect.getsource`.

    Args:
        target: Runtime symbol to inspect.

    Returns:
        True if `target` is a module, class, method, traceback, frame, code object, or callable.
    """
    return callable(target) or isinstance(
        target,
        (
            types.ModuleType,
            type,
            types.MethodType,
            types.TracebackType,
            types.FrameType,
            types.CodeType,
        ),
    )


def _is_signature_inspectable(
    target: LivePythonTarget | StaticGriffeNode,
) -> typing_extensions.TypeGuard[typing.Callable[..., LivePythonTarget]]:
    """Narrow `target` for `inspect.signature`.

    Args:
        target: Runtime symbol to inspect.

    Returns:
        True if `target` is callable.
    """
    return callable(target)


def get_member_dict(target: LivePythonTarget | StaticGriffeNode) -> dict[str, LivePythonTarget]:
    """Return a dictionary of accessible members on `target` via `inspect.getmembers`.

    Args:
        target: Runtime symbol, module, or Griffe AST node to inspect.

    Returns:
        Dictionary mapping member names to their values.
    """
    if target is None:
        return {}
    try:
        raw: list[tuple[str, LivePythonTarget]] = inspect.getmembers(target)
        return dict(raw)
    except (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        LookupError,
        ImportError,
        OSError,
        griffe.GriffeError,
    ):
        _LOGGER.debug("Failed to enumerate runtime members on target", exc_info=True)
        return {}


def get_runtime_source(target: LivePythonTarget | StaticGriffeNode) -> str:
    """Retrieve the Python source code string for a live callable or class target.

    Args:
        target: Runtime callable, class, or module to inspect.

    Returns:
        Python source string from `inspect.getsource`.

    Raises:
        TypeError: If `target` is None or not a source-inspectable object.
    """
    if not _is_source_inspectable(target):
        raise TypeError("Cannot retrieve source for None")
    return inspect.getsource(target)


def get_runtime_signature(target: LivePythonTarget | StaticGriffeNode) -> inspect.Signature:
    """Retrieve the `inspect.Signature` for a live callable or class target.

    Args:
        target: Runtime callable or class to inspect.

    Returns:
        Resolved `inspect.Signature` instance.

    Raises:
        TypeError: If `target` is None or not callable.
    """
    if not _is_signature_inspectable(target):
        raise TypeError("Cannot retrieve signature for None")
    return inspect.signature(target)


__all__ = [
    "DefaultValueInput",
    "DescriptorProtocol",
    "GRIFFE_CLASS_TYPES",
    "GRIFFE_DOCSTRING_TYPE",
    "GRIFFE_ERROR_TYPE",
    "GRIFFE_FUNCTION_TYPES",
    "InspectableCallableProtocol",
    "LivePythonTarget",
    "OrigModWrapperProtocol",
    "OriginalFnWrapperProtocol",
    "PythonFunctionWrapperProtocol",
    "STATIC_GRIFFE_TYPES",
    "StaticGriffeNode",
    "WrappedTargetProtocol",
    "get_member_dict",
    "get_runtime_signature",
    "get_runtime_source",
]
