"""Models for Ghost API representations.

Ghost Core: Introspection Abstraction Layer.

This module provides deeply immutable data structures and total `Result`-returning
JSON guards required to decouple framework analysis from the live environment.
"""

from __future__ import annotations

import enum
import typing

import pydantic
import typing_extensions
from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam as ExtendedGhostParam,
)
from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostRef as ExtendedGhostRef,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostInstructionRef as GhostInstructionRef,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostIsaRef as GhostIsaRef,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostMlirRef as GhostMlirRef,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostOperationRef as GhostOperationRef,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostOverload as GhostOverload,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostParam as GhostParam,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostPythonRef as GhostPythonRef,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostRef as GhostRef,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostResult as GhostResult,
)
from ml_switcheroo_ir.schema.ghost import (
    IRParameterRole as IRParameterRole,
)
from ml_switcheroo_ir.schema.ghost import (
    OperandDirection as OperandDirection,
)
from ml_switcheroo_ir.schema.ghost import (
    ParameterKind as ParameterKind,
)
from ml_switcheroo_ir.schema.ghost import (
    SnapshotEnvelope as SnapshotEnvelope,
)
from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result, value_or
from ml_switcheroo_ir.types import JsonScalar as ScalarValue

from ml_ecosystem_snapshots.exceptions import SnapshotSchemaError

JsonValue: typing_extensions.TypeAlias = pydantic.JsonValue

SCHEMA_VERSION: typing.Final[str] = "2.0.0"


@typing.runtime_checkable
class WireSerializableProtocol(typing.Protocol):
    """Protocol for verified snapshot wrappers exposing a deterministic wire dictionary."""

    def to_wire_dict(self) -> dict[str, JsonValue]:
        """Project this verified snapshot wrapper into a JSON-serializable wire dictionary.

        Returns:
            JSON-serializable snapshot dictionary.
        """
        ...


class SymbolKind(str, enum.Enum):
    """Canonical symbol kind enumeration for `GhostRef.kind` across Python, IR, ISA, runtime, and DSL domains."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    OPERATION = "operation"
    INSTRUCTION = "instruction"
    COLLECTIVE = "collective"
    ELEMENT = "element"


class RawParamSpec(pydantic.BaseModel):
    """Typed specification for a raw canonical parameter entry."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: str = "POSITIONAL_OR_KEYWORD"
    default: str | None = None
    annotation: str | None = None
    type: str | None = None
    description: str | None = None
    direction: str | None = None
    role: str | None = None


class CanonicalAttrSpec(pydantic.BaseModel):
    """Typed specification for an attribute inside a canonical operation."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str
    type: str = "any"
    required: bool = False
    default: ScalarValue | tuple[ScalarValue, ...] = None
    options: tuple[ScalarValue, ...] | None = None


class CanonicalOpSpec(pydantic.BaseModel):
    """Typed specification for a canonical fallback operation or instruction."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str = ""
    mnemonic: str = ""
    api_path: str = ""
    kind: str = "function"
    params: tuple[RawParamSpec, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    attributes: tuple[CanonicalAttrSpec, ...] = ()
    docstring: str | None = None
    description: str | None = None
    category: str | None = None
    submodule: str | None = None
    address_space: str | None = None


class SnapshotSymbolModel(ExtendedGhostRef):
    """Immutable runtime-validated snapshot symbol model extending `ExtendedGhostRef`."""

    model_config = pydantic.ConfigDict(extra="allow", frozen=True)

    kwargs: tuple[str, ...] = ()


class IsaInstructionRecordModel(pydantic.BaseModel):
    """Immutable runtime-validated hardware ISA instruction record (NVIDIA SASS, PTX, AMD RDNA)."""

    model_config = pydantic.ConfigDict(extra="allow", frozen=True)

    mnemonic: str = ""
    category: str = ""
    description: str = ""
    encoding: str | None = None
    architecture: str | tuple[str, ...] | None = None
    min_sm: str | None = None
    modifiers: tuple[str, ...] = ()
    operands: tuple[JsonValue, ...] = ()
    supported_types: tuple[str, ...] = ()
    vector_widths: tuple[str, ...] = ()
    state_spaces: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()


class TableGenRecordModel(pydantic.BaseModel):
    """Immutable runtime-validated MLIR / StableHLO TableGen operation record."""

    model_config = pydantic.ConfigDict(extra="allow", frozen=True)

    api_path: str = ""
    name: str = ""
    class_name: str = ""
    dialect: str = ""
    kind: str = "function"
    is_public: bool = True
    has_varargs: bool = False
    environment_tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    overloads: tuple[dict[str, JsonValue], ...] = ()
    params: tuple[dict[str, JsonValue], ...] = ()
    operands: tuple[dict[str, JsonValue], ...] = ()
    attributes: tuple[dict[str, JsonValue], ...] = ()
    results: tuple[dict[str, JsonValue], ...] = ()
    returns_type: str | None = None
    returns_description: str | None = None
    raises: tuple[str, ...] = ()
    docstring: str | None = None
    description: str | None = None
    summary: str | None = None
    traits: tuple[str, ...] = ()
    regions: tuple[dict[str, JsonValue] | str, ...] = ()
    verification_rules: dict[str, JsonValue] = pydantic.Field(default_factory=dict)
    region_signatures: dict[str, JsonValue] = pydantic.Field(default_factory=dict)
    assembly_format: str | None = None


def ensure_json_dict(
    value: JsonValue,
    context: str = "JSON object",
) -> Result[SnapshotSchemaError, dict[str, JsonValue]]:
    """Validate totally at runtime that `value` is a JSON dictionary.

    Args:
        value: Parsed JSON value to validate.
        context: Human-readable context included in error diagnostics.

    Returns:
        `Ok(dict)` if `value` is a `dict`, else `Err(SnapshotSchemaError)`.
    """
    if not isinstance(value, dict):
        return Err(SnapshotSchemaError(f"Expected {context} to be a dict, got {value!r}."))
    return Ok(value)


def ensure_json_list(
    value: JsonValue,
    context: str = "JSON array",
) -> Result[SnapshotSchemaError, tuple[JsonValue, ...]]:
    """Validate totally at runtime that `value` is a JSON array, returning an immutable `tuple`.

    Args:
        value: Parsed JSON value to validate.
        context: Human-readable context included in error diagnostics.

    Returns:
        `Ok(tuple)` if `value` is a `list`, else `Err(SnapshotSchemaError)`.
    """
    if not isinstance(value, list):
        return Err(SnapshotSchemaError(f"Expected {context} to be a list, got {value!r}."))
    return Ok(tuple(value))


_PresentT = typing.TypeVar("_PresentT")


def ensure_present(
    value: _PresentT | None,
    context: str = "required value",
) -> Result[SnapshotSchemaError, _PresentT]:
    """Validate totally at runtime that `value` is not `None`.

    Args:
        value: Optional value expected to be populated.
        context: Human-readable context included in error diagnostics.

    Returns:
        `Ok(value)` if non-`None`, else `Err(SnapshotSchemaError)`.
    """
    if value is None:
        return Err(SnapshotSchemaError(f"Expected {context} to be non-None."))
    return Ok(value)


def json_dict_or_empty(value: JsonValue) -> dict[str, JsonValue]:
    """Fold `ensure_json_dict(value)` to a dictionary or `{}` when `value` is non-dict.

    Args:
        value: Candidate JSON value.

    Returns:
        Shallow copy of `value` when a `dict`, otherwise an empty dictionary `{}`.
    """
    res = ensure_json_dict(value)
    return dict(value_or(res, {}))


def json_tuple_or_empty(value: JsonValue) -> tuple[JsonValue, ...]:
    """Fold `ensure_json_list(value)` to an immutable `tuple` or `()` when `value` is non-list.

    Args:
        value: Candidate JSON value.

    Returns:
        Immutable tuple of `JsonValue` items when `value` is a `list`, otherwise `()`.
    """
    return value_or(ensure_json_list(value), ())


__all__ = [
    "CanonicalAttrSpec",
    "CanonicalOpSpec",
    "ExtendedGhostParam",
    "ExtendedGhostRef",
    "GhostInstructionRef",
    "GhostIsaRef",
    "GhostMlirRef",
    "GhostOperationRef",
    "GhostOverload",
    "GhostParam",
    "GhostPythonRef",
    "GhostRef",
    "GhostResult",
    "IRParameterRole",
    "IsaInstructionRecordModel",
    "JsonValue",
    "OperandDirection",
    "ParameterKind",
    "RawParamSpec",
    "SCHEMA_VERSION",
    "ScalarValue",
    "SnapshotEnvelope",
    "SnapshotSymbolModel",
    "SymbolKind",
    "TableGenRecordModel",
    "WireSerializableProtocol",
    "ensure_json_dict",
    "ensure_json_list",
    "ensure_present",
    "json_dict_or_empty",
    "json_tuple_or_empty",
]
