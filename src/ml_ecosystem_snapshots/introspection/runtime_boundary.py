"""Runtime reflection and introspection boundary protocols for live Python targets.

Isolates low-level dynamic Python object attribute access, wrapper unwrapping,
and structural callable/descriptor protocols into a single strongly typed
boundary module.
"""

from __future__ import annotations

import enum
import inspect
import types
import typing

import typing_extensions

from ml_ecosystem_snapshots.models import JsonValue, ScalarValue

try:
    import griffe
except ImportError:  # pragma: no cover
    pass


@typing.runtime_checkable
class NamedKindProtocol(typing.Protocol):
    """Structural protocol for parameter kind enums/objects exposing a `.name` attribute."""

    @property
    def name(self) -> str:
        """Return parameter kind name.

        Returns:
            Parameter kind name string.
        """
        ...


@typing.runtime_checkable
class GriffeDocstringProtocol(typing.Protocol):
    """Structural protocol for a Griffe docstring object exposing a `.value` string."""

    @property
    def value(self) -> str:
        """Return docstring text value.

        Returns:
            Raw docstring text.
        """
        ...


@typing.runtime_checkable
class GriffeParameterProtocol(typing.Protocol):
    """Structural protocol for a Griffe AST parameter object."""

    @property
    def name(self) -> str:
        """Return parameter name.

        Returns:
            Parameter name string.
        """
        ...

    @property
    def default(self) -> ScalarValue:
        """Return parameter default value.

        Returns:
            Default scalar representation.
        """
        ...

    @property
    def annotation(self) -> ScalarValue:
        """Return parameter type annotation.

        Returns:
            Annotation scalar representation.
        """
        ...


@typing.runtime_checkable
class GriffeParameterWithKindProtocol(typing.Protocol):
    """Structural protocol for a Griffe parameter exposing a `.kind` attribute."""

    @property
    def kind(self) -> NamedKindProtocol | str | None:
        """Return parameter kind.

        Returns:
            Parameter kind object or string.
        """
        ...


@typing.runtime_checkable
class GriffeObjectProtocol(typing.Protocol):
    """Structural protocol for a Griffe AST object with `name` and `is_public`."""

    name: str
    is_public: bool


@typing.runtime_checkable
class GriffeMembersProtocol(typing.Protocol):
    """Structural protocol for a Griffe node exposing a `members` mapping."""

    @property
    def members(self) -> typing.Mapping[str, InspectableTarget]:
        """Return members mapping.

        Returns:
            Mapping of member names to inspectable targets.
        """
        ...


@typing.runtime_checkable
class GriffeParametersProtocol(typing.Protocol):
    """Structural protocol for a Griffe node exposing a `parameters` iterable."""

    @property
    def parameters(self) -> typing.Iterable[GriffeParameterProtocol]:
        """Return function parameters.

        Returns:
            Iterable of Griffe parameter objects.
        """
        ...


@typing.runtime_checkable
class GriffeOverloadsProtocol(typing.Protocol):
    """Structural protocol for a Griffe node exposing an `overloads` sequence."""

    @property
    def overloads(self) -> list[InspectableTarget]:
        """Return function overloads list.

        Returns:
            List of overloaded Griffe function definitions.
        """
        ...


@typing.runtime_checkable
class GriffeReturnsProtocol(typing.Protocol):
    """Structural protocol for a Griffe node exposing a `returns` annotation."""

    @property
    def returns(self) -> InspectableTarget:
        """Return function return annotation.

        Returns:
            Return annotation expression or None.
        """
        ...


@typing.runtime_checkable
class GriffeDocstringNodeProtocol(typing.Protocol):
    """Structural protocol for a Griffe node exposing a `docstring` object."""

    @property
    def docstring(self) -> GriffeDocstringProtocol | str | None:
        """Return attached docstring object.

        Returns:
            Attached Griffe docstring object, string, or None.
        """
        ...


@typing.runtime_checkable
class GriffeClassProtocol(typing.Protocol):
    """Structural protocol for a Griffe AST class node."""

    @property
    def is_class(self) -> bool:
        """Return whether the node represents a class.

        Returns:
            True if the node is a class.
        """
        ...

    @property
    def members(self) -> typing.Mapping[str, GriffeNodeProtocol]:
        """Return class members mapping.

        Returns:
            Mapping of member names to Griffe AST nodes.
        """
        ...


@typing.runtime_checkable
class GriffeFunctionProtocol(typing.Protocol):
    """Structural protocol for a Griffe AST function node."""

    @property
    def is_function(self) -> bool:
        """Return whether the node represents a function.

        Returns:
            True if the node is a function.
        """
        ...

    @property
    def parameters(self) -> typing.Iterable[GriffeParameterProtocol]:
        """Return function parameters.

        Returns:
            Iterable of Griffe parameter objects.
        """
        ...


@typing.runtime_checkable
class GriffeNodeProtocol(typing.Protocol):
    """Structural protocol for Griffe AST nodes (Module, Object, Function, Class, Alias)."""

    is_public: bool


@typing.runtime_checkable
class GriffeInspectableNodeProtocol(typing.Protocol):
    """Structural protocol for Griffe AST nodes exposing `is_class` and `is_function`."""

    @property
    def is_class(self) -> bool:
        """Return whether the node represents a class.

        Returns:
            True if the node is a class.
        """
        ...

    @property
    def is_function(self) -> bool:
        """Return whether the node represents a function.

        Returns:
            True if the node is a function.
        """
        ...


@typing.runtime_checkable
class PythonFunctionWrapperProtocol(typing.Protocol):
    """Structural protocol for TensorFlow `@tf.function` wrappers exposing `_python_function`."""

    _python_function: InspectableTarget


@typing.runtime_checkable
class OriginalFnWrapperProtocol(typing.Protocol):
    """Structural protocol for decorator wrappers exposing `_original_fn`."""

    _original_fn: InspectableTarget


@typing.runtime_checkable
class OrigModWrapperProtocol(typing.Protocol):
    """Structural protocol for `torch.compile` wrappers exposing `_orig_mod`."""

    _orig_mod: InspectableTarget


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

    def __get__(self, instance: InspectableTarget, owner: type | None = None) -> InspectableCallableProtocol | ScalarValue:
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


InspectableTarget: typing_extensions.TypeAlias = (
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
    " | NamedKindProtocol"
    " | GriffeDocstringProtocol"
    " | GriffeNodeProtocol"
    " | GriffeInspectableNodeProtocol"
    " | GriffeParameterProtocol"
    " | GriffeParameterWithKindProtocol"
    " | GriffeMembersProtocol"
    " | GriffeParametersProtocol"
    " | GriffeOverloadsProtocol"
    " | GriffeReturnsProtocol"
    " | GriffeDocstringNodeProtocol"
    " | typing.Iterable[GriffeParameterProtocol]"
    " | GriffeObjectProtocol"
    " | GriffeClassProtocol"
    " | GriffeFunctionProtocol"
    " | griffe.Module"
    " | griffe.Object"
    " | griffe.Alias"
    " | tuple[InspectableTarget, ...]"
    " | list[InspectableTarget]"
    " | set[ScalarValue]"
    " | frozenset[ScalarValue]"
    " | dict[str, InspectableTarget]"
    " | ScalarValue"
)

DefaultValueInput: typing_extensions.TypeAlias = InspectableTarget

_SourceInspectableType: typing_extensions.TypeAlias = "types.ModuleType | type | types.MethodType | types.TracebackType | types.FrameType | types.CodeType"


def _is_source_inspectable(
    target: InspectableTarget,
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
    target: InspectableTarget,
) -> typing_extensions.TypeGuard[typing.Callable[..., InspectableTarget]]:
    """Narrow `target` for `inspect.signature`.

    Args:
        target: Runtime symbol to inspect.

    Returns:
        True if `target` is callable.
    """
    return callable(target)


def get_member_dict(target: InspectableTarget) -> dict[str, InspectableTarget]:
    """Return a dictionary of accessible members on `target` via `inspect.getmembers`.

    Args:
        target: Runtime symbol, module, or AST node to inspect.

    Returns:
        Dictionary mapping member names to their values.
    """
    if target is None:
        return {}
    try:
        raw: list[tuple[str, InspectableTarget]] = inspect.getmembers(target)
        return dict(raw)
    except Exception:
        return {}


def get_runtime_source(target: InspectableTarget) -> str:
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


def get_runtime_signature(target: InspectableTarget) -> inspect.Signature:
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
    "GriffeClassProtocol",
    "GriffeDocstringNodeProtocol",
    "GriffeDocstringProtocol",
    "GriffeFunctionProtocol",
    "GriffeInspectableNodeProtocol",
    "GriffeMembersProtocol",
    "GriffeNodeProtocol",
    "GriffeObjectProtocol",
    "GriffeOverloadsProtocol",
    "GriffeParameterProtocol",
    "GriffeParameterWithKindProtocol",
    "GriffeParametersProtocol",
    "GriffeReturnsProtocol",
    "InspectableCallableProtocol",
    "InspectableTarget",
    "NamedKindProtocol",
    "OrigModWrapperProtocol",
    "OriginalFnWrapperProtocol",
    "PythonFunctionWrapperProtocol",
    "WrappedTargetProtocol",
    "get_member_dict",
    "get_runtime_signature",
    "get_runtime_source",
]
