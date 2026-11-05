"""Models for Ghost API representations.

Ghost Core: Introspection Abstraction Layer.

This module provides the data structures required to decouple framework
analysis from the live environment. It enables the system to operate in
"Ghost Mode" (WASM/CI) by working against cached snapshots instead of
requiring heavy libraries (Torch/TensorFlow) to be installed.
"""

from __future__ import annotations

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


class IntSlice(typing.Protocol):
    """Structural protocol representing a `slice` object with integer bounds."""

    if typing.TYPE_CHECKING:

        @property
        def start(self) -> int | None:
            """Return slice start index.

            Returns:
                Start index or None.
            """
            ...

        @property
        def stop(self) -> int | None:
            """Return slice stop index.

            Returns:
                Stop index or None.
            """
            ...

        @property
        def step(self) -> int | None:
            """Return slice step stride.

            Returns:
                Step stride or None.
            """
            ...


class JsonStringDict(dict[str, JsonValue]):
    """Dictionary subtype of `dict[str, JsonValue]` whose values are strings."""

    if typing.TYPE_CHECKING:

        def __getitem__(self, key: str) -> str:
            """Return string value for `key`.

            Args:
                key: Dictionary key.

            Returns:
                String value.
            """
            ...


class JsonStringDictList(list[JsonValue]):
    """List subtype of `list[JsonValue]` whose elements are `JsonStringDict`."""

    if typing.TYPE_CHECKING:

        @typing.overload
        def __getitem__(self, index: typing.SupportsIndex) -> JsonStringDict: ...

        @typing.overload
        def __getitem__(self, index: IntSlice) -> list[JsonValue]: ...

        def __getitem__(self, index: typing.SupportsIndex | IntSlice) -> JsonStringDict | list[JsonValue]:
            """Return element or slice at `index`.

            Args:
                index: Integer index or slice.

            Returns:
                `JsonStringDict` element or list slice.
            """
            ...

        def __iter__(self) -> typing.Iterator[JsonStringDict]:
            """Iterate over `JsonStringDict` elements.

            Yields:
                `JsonStringDict` items.
            """
            ...


class JsonStringList(list[JsonValue]):
    """List subtype of `list[JsonValue]` whose elements are strings."""

    if typing.TYPE_CHECKING:

        @typing.overload
        def __getitem__(self, index: typing.SupportsIndex) -> str: ...

        @typing.overload
        def __getitem__(self, index: IntSlice) -> list[JsonValue]: ...

        def __getitem__(self, index: typing.SupportsIndex | IntSlice) -> str | list[JsonValue]:
            """Return string element or slice at `index`.

            Args:
                index: Integer index or slice.

            Returns:
                String element or list slice.
            """
            ...

        def __iter__(self) -> typing.Iterator[str]:
            """Iterate over string elements.

            Yields:
                String items.
            """
            ...


class JsonNestedStringList(list[JsonValue]):
    """List subtype of `list[JsonValue]` whose elements are `JsonStringList`."""

    if typing.TYPE_CHECKING:

        @typing.overload
        def __getitem__(self, index: typing.SupportsIndex) -> JsonStringList: ...

        @typing.overload
        def __getitem__(self, index: IntSlice) -> list[JsonValue]: ...

        def __getitem__(self, index: typing.SupportsIndex | IntSlice) -> JsonStringList | list[JsonValue]:
            """Return `JsonStringList` element or slice at `index`.

            Args:
                index: Integer index or slice.

            Returns:
                `JsonStringList` element or list slice.
            """
            ...

        def __iter__(self) -> typing.Iterator[JsonStringList]:
            """Iterate over `JsonStringList` elements.

            Yields:
                `JsonStringList` items.
            """
            ...


class SnapshotSymbolDict(dict[str, JsonValue]):
    """Typed dictionary representation of a serialized GhostRef or TableGen record."""

    if typing.TYPE_CHECKING:

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
        ) -> JsonStringDictList: ...

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
        ) -> JsonStringList: ...

        @typing.overload
        def __getitem__(
            self,
            key: typing.Literal["verification_rules", "region_signatures", "domain_metadata"],
        ) -> dict[str, JsonValue]: ...

        @typing.overload
        def __getitem__(self, key: str) -> JsonValue: ...

        def __getitem__(self, key: str) -> JsonValue:
            """Retrieve a symbol record field by dictionary key.

            Args:
                key: Field name.

            Returns:
                Field value.
            """
            ...


class IsaInstructionRecordModel(dict[str, JsonValue]):
    """Typed dictionary representation of an ISA instruction record."""

    if typing.TYPE_CHECKING:

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
        ) -> JsonStringList: ...

        @typing.overload
        def __getitem__(
            self,
            key: typing.Literal["operands"],
        ) -> JsonNestedStringList: ...

        @typing.overload
        def __getitem__(
            self,
            key: typing.Literal["architecture"],
        ) -> str | JsonStringList: ...

        @typing.overload
        def __getitem__(self, key: str) -> JsonValue: ...

        def __getitem__(self, key: str) -> JsonValue:
            """Retrieve an ISA instruction record field by dictionary key.

            Args:
                key: Field name.

            Returns:
                Field value.
            """
            ...


TableGenRecordModel: typing_extensions.TypeAlias = SnapshotSymbolDict


class JsonSymbolList(list[JsonValue]):
    """List subtype of `list[JsonValue]` whose elements are `SnapshotSymbolDict`."""

    if typing.TYPE_CHECKING:

        @typing.overload
        def __getitem__(self, index: typing.SupportsIndex) -> SnapshotSymbolDict: ...

        @typing.overload
        def __getitem__(self, index: IntSlice) -> list[JsonValue]: ...

        def __getitem__(self, index: typing.SupportsIndex | IntSlice) -> SnapshotSymbolDict | list[JsonValue]:
            """Return `SnapshotSymbolDict` element or slice at `index`.

            Args:
                index: Integer index or slice.

            Returns:
                `SnapshotSymbolDict` element or list slice.
            """
            ...

        def __iter__(self) -> typing.Iterator[SnapshotSymbolDict]:
            """Iterate over `SnapshotSymbolDict` elements.

            Yields:
                `SnapshotSymbolDict` items.
            """
            ...


class JsonCategoriesDict(dict[str, JsonValue]):
    """Dictionary subtype of `dict[str, JsonValue]` mapping category names to `JsonSymbolList`."""

    if typing.TYPE_CHECKING:

        def __getitem__(self, key: str) -> JsonSymbolList:
            """Return `JsonSymbolList` for `key`.

            Args:
                key: Category name.

            Returns:
                `JsonSymbolList` for the category.
            """
            ...


SnapshotCategoriesDict: typing_extensions.TypeAlias = dict[str, JsonSymbolList]


class SnapshotEnvelopeDict(dict[str, JsonValue]):
    """Typed dictionary representation of a serialized framework snapshot envelope."""

    if typing.TYPE_CHECKING:

        @typing.overload
        def __getitem__(self, key: typing.Literal["categories"]) -> JsonCategoriesDict: ...

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
        def __getitem__(self, key: typing.Literal["supported_microarchitectures"]) -> JsonStringList: ...

        @typing.overload
        def __getitem__(self, key: str) -> JsonValue: ...

        def __getitem__(self, key: str) -> JsonValue:
            """Retrieve a snapshot envelope field by dictionary key.

            Args:
                key: Field name.

            Returns:
                Field value.
            """
            ...


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
