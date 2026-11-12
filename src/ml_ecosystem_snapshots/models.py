"""Models for Ghost API representations.

Ghost Core: Introspection Abstraction Layer.

This module provides the data structures required to decouple framework
analysis from the live environment. It enables the system to operate in
"Ghost Mode" (WASM/CI) by working against cached snapshots instead of
requiring heavy libraries (Torch/TensorFlow) to be installed.
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
from ml_switcheroo_ir.types import JsonScalar as ScalarValue

JsonValue: typing_extensions.TypeAlias = pydantic.JsonValue

SCHEMA_VERSION: typing.Final[str] = "2.0.0"


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
    default: ScalarValue | list[ScalarValue] = None
    options: list[ScalarValue] | None = None


class CanonicalOpSpec(pydantic.BaseModel):
    """Typed specification for a canonical fallback operation or instruction."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    name: str = ""
    mnemonic: str = ""
    api_path: str = ""
    kind: str = "function"
    params: list[RawParamSpec] = pydantic.Field(default_factory=list)
    inputs: list[str] = pydantic.Field(default_factory=list)
    outputs: list[str] = pydantic.Field(default_factory=list)
    attributes: list[CanonicalAttrSpec] = pydantic.Field(default_factory=list)
    docstring: str | None = None
    description: str | None = None
    category: str | None = None
    submodule: str | None = None
    address_space: str | None = None


class SnapshotSymbolModel(ExtendedGhostRef):
    """Runtime-validated snapshot symbol model extending `ExtendedGhostRef`."""

    model_config = pydantic.ConfigDict(extra="allow")

    kwargs: list[str] = pydantic.Field(default_factory=list)


class IsaInstructionRecordModel(pydantic.BaseModel):
    """Runtime-validated hardware ISA instruction record (NVIDIA SASS, PTX, AMD RDNA)."""

    model_config = pydantic.ConfigDict(extra="allow")

    mnemonic: str = ""
    category: str = ""
    description: str = ""
    encoding: str | None = None
    architecture: str | list[str] | None = None
    min_sm: str | None = None
    modifiers: list[str] = pydantic.Field(default_factory=list)
    operands: list[JsonValue] = pydantic.Field(default_factory=list)
    supported_types: list[str] = pydantic.Field(default_factory=list)
    vector_widths: list[str] = pydantic.Field(default_factory=list)
    state_spaces: list[str] = pydantic.Field(default_factory=list)
    scopes: list[str] = pydantic.Field(default_factory=list)


class TableGenRecordModel(pydantic.BaseModel):
    """Runtime-validated MLIR / StableHLO TableGen operation record."""

    model_config = pydantic.ConfigDict(extra="allow")

    api_path: str = ""
    name: str = ""
    class_name: str = ""
    dialect: str = ""
    kind: str = "function"
    is_public: bool = True
    has_varargs: bool = False
    environment_tags: list[str] = pydantic.Field(default_factory=list)
    aliases: list[str] = pydantic.Field(default_factory=list)
    overloads: list[dict[str, JsonValue]] = pydantic.Field(default_factory=list)
    params: list[dict[str, JsonValue]] = pydantic.Field(default_factory=list)
    operands: list[dict[str, JsonValue]] = pydantic.Field(default_factory=list)
    attributes: list[dict[str, JsonValue]] = pydantic.Field(default_factory=list)
    results: list[dict[str, JsonValue]] = pydantic.Field(default_factory=list)
    returns_type: str | None = None
    returns_description: str | None = None
    raises: list[str] = pydantic.Field(default_factory=list)
    docstring: str | None = None
    description: str | None = None
    summary: str | None = None
    traits: list[str] = pydantic.Field(default_factory=list)
    regions: list[dict[str, JsonValue] | str] = pydantic.Field(default_factory=list)
    verification_rules: dict[str, JsonValue] = pydantic.Field(default_factory=dict)
    region_signatures: dict[str, JsonValue] = pydantic.Field(default_factory=dict)
    assembly_format: str | None = None


def ensure_json_dict(value: JsonValue, context: str = "JSON object") -> dict[str, JsonValue]:
    """Validate at runtime that `value` is a JSON dictionary, surviving `python -O`.

    Args:
        value: Parsed JSON value to validate.
        context: Human-readable context included in error diagnostics.

    Returns:
        Validated shallow dictionary mapping strings to `JsonValue`.

    Raises:
        SnapshotSchemaError: If `value` is not a `dict`.
    """
    from ml_ecosystem_snapshots.exceptions import SnapshotSchemaError

    if not isinstance(value, dict):
        raise SnapshotSchemaError(f"Expected {context} to be a dict, got {value!r}.")
    return value


def ensure_json_list(value: JsonValue, context: str = "JSON array") -> list[JsonValue]:
    """Validate at runtime that `value` is a JSON list, surviving `python -O`.

    Args:
        value: Parsed JSON value to validate.
        context: Human-readable context included in error diagnostics.

    Returns:
        Validated list of `JsonValue` items.

    Raises:
        SnapshotSchemaError: If `value` is not a `list`.
    """
    from ml_ecosystem_snapshots.exceptions import SnapshotSchemaError

    if not isinstance(value, list):
        raise SnapshotSchemaError(f"Expected {context} to be a list, got {value!r}.")
    return value


_PresentT = typing.TypeVar("_PresentT")


def ensure_present(value: _PresentT | None, context: str = "required value") -> _PresentT:
    """Validate at runtime that `value` is not `None`, surviving `python -O`.

    Args:
        value: Optional value expected to be populated.
        context: Human-readable context included in error diagnostics.

    Returns:
        Non-`None` value of type `_PresentT`.

    Raises:
        SnapshotSchemaError: If `value` is `None`.
    """
    from ml_ecosystem_snapshots.exceptions import SnapshotSchemaError

    if value is None:
        raise SnapshotSchemaError(f"Expected {context} to be non-None.")
    return value


__all__ = [
    "CanonicalAttrSpec",
    "CanonicalOpSpec",
    "ExtendedGhostParam",
    "ExtendedGhostRef",
    "GhostInstructionRef",
    "GhostIsaRef",
    "GhostMlirRef",
    "GhostOperationRef",
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
    "ensure_json_dict",
    "ensure_json_list",
    "ensure_present",
]
