"""Models for Ghost API representations.

Ghost Core: Introspection Abstraction Layer.

This module provides the data structures required to decouple framework
analysis from the live environment. It enables the system to operate in
"Ghost Mode" (WASM/CI) by working against cached snapshots instead of
requiring heavy libraries (Torch/TensorFlow) to be installed.
"""

from __future__ import annotations

import typing
from typing_extensions import TypeAlias

from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam as ExtendedGhostParam,
    ExtendedGhostRef as ExtendedGhostRef,
    GhostInstructionRef as GhostInstructionRef,
    GhostIsaRef as GhostIsaRef,
    GhostMlirRef as GhostMlirRef,
    GhostOperationRef as GhostOperationRef,
    GhostParam as GhostParam,
    GhostPythonRef as GhostPythonRef,
    GhostRef as GhostRef,
    GhostResult as GhostResult,
    IRParameterRole as IRParameterRole,
    OperandDirection as OperandDirection,
    ParameterKind as ParameterKind,
    SnapshotEnvelope as SnapshotEnvelope,
)
import pydantic

ScalarValue: TypeAlias = typing.Union[str, int, float, bool, None]
JsonValue: TypeAlias = pydantic.JsonValue


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


class SnapshotSymbolDict(dict[str, JsonValue]):
    """Typed dictionary representation of a serialized GhostRef or TableGen record."""

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal[
            "name",
            "api_path",
            "kind",
            "class_name",
            "dialect",
            "mnemonic",
            "encoding",
            "category",
        ],
    ) -> str: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal[
            "returns_type",
            "returns_description",
            "docstring",
            "description",
            "signature_completeness",
        ],
    ) -> str | None: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal["is_public", "has_varargs", "is_c_extension"],
    ) -> bool: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal["params", "operands", "attributes", "results"],
    ) -> list[dict[str, str]]: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal[
            "traits",
            "regions",
            "modifiers",
            "aliases",
            "kwargs",
            "environment_tags",
            "raises",
            "accepted_kwargs",
            "supported_types",
            "vector_widths",
            "state_spaces",
            "scopes",
        ],
    ) -> list[str]: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal[
            "verification_rules", "region_signatures", "domain_metadata"
        ],
    ) -> dict[str, JsonValue]: ...

    @typing.overload
    def __getitem__(self, key: str) -> JsonValue: ...

    def __getitem__(
        self, key: str
    ) -> (
        str | bool | list[dict[str, str]] | list[str] | dict[str, JsonValue] | JsonValue
    ):
        """Retrieve a symbol record field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        return super().__getitem__(key)


class IsaInstructionRecordModel(dict[str, JsonValue]):
    """Typed dictionary representation of an ISA instruction record."""

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal["mnemonic", "encoding", "category", "min_sm"],
    ) -> str: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal["description"],
    ) -> str | None: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal[
            "modifiers",
            "supported_types",
            "vector_widths",
            "state_spaces",
            "scopes",
        ],
    ) -> list[str]: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal["operands"],
    ) -> list[list[str]]: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal["architecture"],
    ) -> str | list[str]: ...

    @typing.overload
    def __getitem__(self, key: str) -> JsonValue: ...

    def __getitem__(self, key: str) -> str | list[str] | list[list[str]] | JsonValue:
        """Retrieve an ISA instruction record field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        return super().__getitem__(key)


TableGenRecordModel: TypeAlias = SnapshotSymbolDict
SnapshotCategoriesDict: TypeAlias = dict[str, list[SnapshotSymbolDict]]


class SnapshotEnvelopeDict(dict[str, JsonValue]):
    """Typed dictionary representation of a serialized framework snapshot envelope."""

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["categories"]
    ) -> SnapshotCategoriesDict: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal[
            "schema_version",
            "target",
            "version",
            "generated_at",
            "source_type",
            "upstream_commit",
            "upstream_version",
        ],
    ) -> str: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["supported_microarchitectures"]
    ) -> list[str]: ...

    @typing.overload
    def __getitem__(self, key: str) -> JsonValue: ...

    def __getitem__(
        self, key: str
    ) -> SnapshotCategoriesDict | str | list[str] | JsonValue:
        """Retrieve a snapshot envelope field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        return super().__getitem__(key)


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
    "ScalarValue",
    "SnapshotCategoriesDict",
    "SnapshotEnvelope",
    "SnapshotEnvelopeDict",
    "SnapshotSymbolDict",
    "TableGenRecordModel",
]
