"""Runtime reflection and introspection boundary protocols for live Python targets.

Isolates low-level dynamic Python object attribute access, wrapper unwrapping,
and structural callable/descriptor protocols into a single strongly typed
boundary module.
"""

from __future__ import annotations

import enum
import inspect
import operator
import types
import typing
import typing_extensions

from ml_ecosystem_snapshots.models import JsonValue, ScalarValue

try:
    import griffe
except ImportError:  # pragma: no cover
    pass


@typing.runtime_checkable
class GriffeParameterProtocol(typing.Protocol):
    """Structural protocol for a Griffe AST parameter object."""

    @property
    def name(self) -> str:
        """Return parameter name."""
        ...

    @property
    def default(self) -> ScalarValue:
        """Return parameter default value."""
        ...

    @property
    def annotation(self) -> ScalarValue:
        """Return parameter type annotation."""
        ...


@typing.runtime_checkable
class GriffeObjectProtocol(typing.Protocol):
    """Structural protocol for a Griffe AST object with `name` and `is_public`."""

    name: str
    is_public: bool


@typing.runtime_checkable
class GriffeClassProtocol(typing.Protocol):
    """Structural protocol for a Griffe AST class node."""

    @property
    def is_class(self) -> bool:
        """Return whether the node represents a class."""
        ...

    @property
    def members(self) -> typing.Mapping[str, GriffeNodeProtocol]:
        """Return class members mapping."""
        ...


@typing.runtime_checkable
class GriffeFunctionProtocol(typing.Protocol):
    """Structural protocol for a Griffe AST function node."""

    @property
    def is_function(self) -> bool:
        """Return whether the node represents a function."""
        ...

    @property
    def parameters(self) -> typing.Iterable[GriffeParameterProtocol]:
        """Return function parameters."""
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
        """Return whether the node represents a class."""
        ...

    @property
    def is_function(self) -> bool:
        """Return whether the node represents a function."""
        ...


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
        self, instance: ScalarValue, owner: type | None = None
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


InspectableTarget: typing_extensions.TypeAlias = (
    "types.FunctionType"
    " | types.BuiltinFunctionType"
    " | types.MethodType"
    " | types.ModuleType"
    " | types.EllipsisType"
    " | type"
    " | property"
    " | enum.Enum"
    " | InspectableCallableProtocol"
    " | DescriptorProtocol"
    " | WrappedTargetProtocol"
    " | GriffeNodeProtocol"
    " | GriffeInspectableNodeProtocol"
    " | GriffeParameterProtocol"
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


def read_runtime_attr(
    target: InspectableTarget,
    attr_name: str,
    default: InspectableTarget = None,
) -> InspectableTarget:
    """Safely read a named attribute from a live runtime target.

    Args:
        target: Runtime symbol or module to inspect.
        attr_name: Attribute name to look up.
        default: Fallback value returned if the attribute is absent or raises.

    Returns:
        Attribute value or `default`.
    """
    try:
        val: InspectableTarget = operator.attrgetter(attr_name)(target)
        return val
    except Exception:
        return default


def has_runtime_attr(target: InspectableTarget, attr_name: str) -> bool:
    """Safely check whether a live runtime target exposes `attr_name`.

    Args:
        target: Runtime symbol or module to inspect.
        attr_name: Attribute name to check.

    Returns:
        True if the attribute exists on `target`, False otherwise.
    """
    try:
        operator.attrgetter(attr_name)(target)
        return True
    except Exception:
        return False


def get_runtime_source(target: InspectableTarget) -> str:
    """Retrieve the Python source code string for a live callable or class target.

    Args:
        target: Runtime callable, class, or module to inspect.

    Returns:
        Python source string from `inspect.getsource`.
    """
    getsource_fn: typing.Callable[[InspectableTarget], str] = operator.attrgetter(
        "getsource"
    )(inspect)
    return getsource_fn(target)


def get_runtime_signature(target: InspectableTarget) -> inspect.Signature:
    """Retrieve the `inspect.Signature` for a live callable or class target.

    Args:
        target: Runtime callable or class to inspect.

    Returns:
        Resolved `inspect.Signature` instance.
    """
    sig_fn: typing.Callable[[InspectableTarget], inspect.Signature] = (
        operator.attrgetter("signature")(inspect)
    )
    return sig_fn(target)


__all__ = [
    "DefaultValueInput",
    "DescriptorProtocol",
    "GriffeClassProtocol",
    "GriffeFunctionProtocol",
    "GriffeInspectableNodeProtocol",
    "GriffeNodeProtocol",
    "GriffeObjectProtocol",
    "GriffeParameterProtocol",
    "InspectableCallableProtocol",
    "InspectableTarget",
    "WrappedTargetProtocol",
    "get_runtime_signature",
    "get_runtime_source",
    "has_runtime_attr",
    "read_runtime_attr",
]
