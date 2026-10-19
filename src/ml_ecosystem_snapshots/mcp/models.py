"""Pydantic response schemas for MCP tool payloads."""

from __future__ import annotations

import typing

import pydantic

from ml_ecosystem_snapshots.models import JsonValue


class _McpResultModel(pydantic.BaseModel):
    """Base Pydantic model for MCP tool results with dictionary access helpers."""

    model_config = pydantic.ConfigDict(extra="forbid")


class HallucinationCheckResult(_McpResultModel):
    """Typed response payload for `check_hallucination`."""

    api_exists: bool = False
    is_hallucinated: bool = False
    invalid_kwargs: list[str] = pydantic.Field(default_factory=list)
    canonical_params: list[str] = pydantic.Field(default_factory=list)
    has_unconstrained_kwargs: bool = False
    reason: str = ""
    snapshot_source: str = ""
    signature_completeness: str = ""
    is_c_extension: bool = False
    dtype_errors: list[str] | None = None
    rank_errors: list[str] | None = None
    shape_errors: list[str] | None = None
    warning: str | None = None

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal[
            "api_exists",
            "is_hallucinated",
            "has_unconstrained_kwargs",
            "is_c_extension",
        ],
    ) -> bool: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal[
            "invalid_kwargs",
            "canonical_params",
            "dtype_errors",
            "rank_errors",
            "shape_errors",
        ],
    ) -> list[str]: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal[
            "reason", "snapshot_source", "signature_completeness", "warning"
        ],
    ) -> str: ...

    @typing.overload
    def __getitem__(self, key: str) -> list[str] | JsonValue: ...

    def __getitem__(self, key: str) -> bool | str | list[str] | JsonValue:
        """Retrieve a result field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        val: bool | str | list[str] | JsonValue = getattr(self, key)
        return val

    @typing.overload
    def get(
        self,
        key: typing.Literal[
            "api_exists",
            "is_hallucinated",
            "has_unconstrained_kwargs",
            "is_c_extension",
        ],
        default: bool = False,
    ) -> bool: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal[
            "invalid_kwargs",
            "canonical_params",
            "dtype_errors",
            "rank_errors",
            "shape_errors",
        ],
        default: list[str] | None = None,
    ) -> list[str]: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal[
            "reason", "snapshot_source", "signature_completeness", "warning"
        ],
        default: str = "",
    ) -> str: ...

    @typing.overload
    def get(self, key: str, default: JsonValue = None) -> list[str] | JsonValue: ...

    def get(
        self,
        key: str,
        default: bool | str | list[str] | JsonValue = None,
    ) -> bool | str | list[str] | JsonValue:
        """Retrieve a result field by key with an optional default.

        Args:
            key: Field name.
            default: Default value when absent or None.

        Returns:
            Field value or `default`.
        """
        val: bool | str | list[str] | JsonValue = getattr(self, key, None)
        return val if val is not None else default


class AntiPatternExplanationResult(_McpResultModel):
    """Typed response payload for `explain_anti_pattern`."""

    framework: str = ""
    api_path: str = ""
    hallucinated_argument: str = ""
    is_known_anti_pattern: bool = False
    canonical_argument: str | None = None
    explanation: str = ""
    canonical_params: list[str] | None = None
    canonical_example: str = ""

    @typing.overload
    def __getitem__(self, key: typing.Literal["is_known_anti_pattern"]) -> bool: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["canonical_params"]) -> list[str]: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["canonical_argument"]) -> str | None: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal[
            "framework",
            "api_path",
            "hallucinated_argument",
            "explanation",
            "canonical_example",
        ],
    ) -> str: ...

    @typing.overload
    def __getitem__(self, key: str) -> bool | str | list[str] | None: ...

    def __getitem__(self, key: str) -> bool | str | list[str] | None:
        """Retrieve a result field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        val: bool | str | list[str] | None = getattr(self, key)
        return val


class CodeBlockFinding(_McpResultModel):
    """Typed finding item within `CodeBlockCheckResult`."""

    line: int = 0
    type: str = ""
    framework: str | None = None
    target: str = ""
    reason: str = ""
    invalid_kwargs: list[str] | None = None

    @typing.overload
    def __getitem__(self, key: typing.Literal["line"]) -> int: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["invalid_kwargs"]) -> list[str]: ...

    @typing.overload
    def __getitem__(self, key: str) -> int | str | list[str]: ...

    def __getitem__(self, key: str) -> int | str | list[str]:
        """Retrieve a finding field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        val: int | str | list[str] = getattr(self, key)
        return val


class CodeBlockCheckResult(_McpResultModel):
    """Typed response payload for `check_code_block`."""

    total_analyzed: int
    hallucinations_detected: int
    is_valid: bool
    findings: list[CodeBlockFinding] = pydantic.Field(default_factory=list)

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["total_analyzed", "hallucinations_detected"]
    ) -> int: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["is_valid"]) -> bool: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["findings"]
    ) -> list[CodeBlockFinding]: ...

    @typing.overload
    def __getitem__(
        self, key: str
    ) -> int | bool | list[CodeBlockFinding] | JsonValue: ...

    def __getitem__(self, key: str) -> int | bool | list[CodeBlockFinding] | JsonValue:
        """Retrieve a result field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        val: int | bool | list[CodeBlockFinding] | JsonValue = getattr(self, key)
        return val


class IsaCheckResult(_McpResultModel):
    """Typed response payload for `check_sass_instruction`, `check_rdna_instruction`, and `check_ptx_instruction`."""

    is_valid: bool = False
    mnemonic_exists: bool = False
    encoding: str | None = None
    category: str | None = None
    min_sm: str | None = None
    supported_types: list[str] | None = None
    supported_architectures: list[str] | None = None
    valid_modifiers: list[str] | None = None
    errors: list[str] = pydantic.Field(default_factory=list)

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["is_valid", "mnemonic_exists"]
    ) -> bool: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal[
            "supported_types",
            "supported_architectures",
            "valid_modifiers",
            "errors",
        ],
    ) -> list[str]: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["encoding", "category", "min_sm"]
    ) -> str | None: ...

    @typing.overload
    def __getitem__(self, key: str) -> list[str] | JsonValue: ...

    def __getitem__(self, key: str) -> bool | str | list[str] | JsonValue:
        """Retrieve an ISA check field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        val: bool | str | list[str] | JsonValue = getattr(self, key)
        return val

    @typing.overload
    def get(
        self,
        key: typing.Literal["is_valid", "mnemonic_exists"],
        default: bool = False,
    ) -> bool: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal[
            "supported_types",
            "supported_architectures",
            "valid_modifiers",
            "errors",
        ],
        default: list[str] | None = None,
    ) -> list[str]: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["encoding", "category", "min_sm"],
        default: str | None = None,
    ) -> str | None: ...

    @typing.overload
    def get(self, key: str, default: JsonValue = None) -> list[str] | JsonValue: ...

    def get(
        self,
        key: str,
        default: bool | str | list[str] | JsonValue = None,
    ) -> bool | str | list[str] | JsonValue:
        """Retrieve an ISA check field by key with default.

        Args:
            key: Field name.
            default: Default value when absent or None.

        Returns:
            Field value or `default`.
        """
        val: bool | str | list[str] | JsonValue = getattr(self, key, None)
        return val if val is not None else default


class IrCheckResult(_McpResultModel):
    """Typed response payload for `check_mlir_op` and `check_stablehlo_op`."""

    is_valid: bool = False
    op_exists: bool = False
    expected_operands: list[str | dict[str, JsonValue]] | None = None
    expected_attributes: list[str] | None = None
    errors: list[str] = pydantic.Field(default_factory=list)

    @typing.overload
    def __getitem__(self, key: typing.Literal["is_valid", "op_exists"]) -> bool: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["expected_operands"]
    ) -> list[str | dict[str, JsonValue]]: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["expected_attributes", "errors"]
    ) -> list[str]: ...

    @typing.overload
    def __getitem__(
        self, key: str
    ) -> list[str] | list[str | dict[str, JsonValue]] | JsonValue: ...

    def __getitem__(
        self, key: str
    ) -> bool | list[str] | list[str | dict[str, JsonValue]] | JsonValue:
        """Retrieve an IR check field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        val: bool | list[str] | list[str | dict[str, JsonValue]] | JsonValue = getattr(
            self, key
        )
        return val

    @typing.overload
    def get(
        self,
        key: typing.Literal["is_valid", "op_exists"],
        default: bool = False,
    ) -> bool: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["expected_operands"],
        default: list[str | dict[str, JsonValue]] | None = None,
    ) -> list[str | dict[str, JsonValue]]: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["expected_attributes", "errors"],
        default: list[str] | None = None,
    ) -> list[str]: ...

    @typing.overload
    def get(
        self, key: str, default: JsonValue = None
    ) -> list[str] | list[str | dict[str, JsonValue]] | JsonValue: ...

    def get(
        self,
        key: str,
        default: (
            bool | list[str] | list[str | dict[str, JsonValue]] | JsonValue
        ) = None,
    ) -> bool | list[str] | list[str | dict[str, JsonValue]] | JsonValue:
        """Retrieve an IR check field by key with default.

        Args:
            key: Field name.
            default: Default value when absent or None.

        Returns:
            Field value or `default`.
        """
        val: bool | list[str] | list[str | dict[str, JsonValue]] | JsonValue = getattr(
            self, key, None
        )
        return val if val is not None else default


class DomainOpCheckResult(_McpResultModel):
    """Typed response payload for `check_domain_op_generic` and domain verifiers."""

    is_valid: bool
    op_exists: bool
    expected_inputs: list[str] = pydantic.Field(default_factory=list)
    expected_attributes: list[str] = pydantic.Field(default_factory=list)
    errors: list[str] = pydantic.Field(default_factory=list)

    @typing.overload
    def __getitem__(self, key: typing.Literal["is_valid", "op_exists"]) -> bool: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal["expected_inputs", "expected_attributes", "errors"],
    ) -> list[str]: ...

    @typing.overload
    def __getitem__(self, key: str) -> list[str] | JsonValue: ...

    def __getitem__(self, key: str) -> bool | list[str] | JsonValue:
        """Retrieve a result field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        val: bool | list[str] | JsonValue = getattr(self, key)
        return val

    @typing.overload
    def get(
        self,
        key: typing.Literal["is_valid", "op_exists"],
        default: bool = False,
    ) -> bool: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["expected_inputs", "expected_attributes", "errors"],
        default: list[str] | None = None,
    ) -> list[str]: ...

    @typing.overload
    def get(self, key: str, default: JsonValue = None) -> list[str] | JsonValue: ...

    def get(
        self,
        key: str,
        default: bool | list[str] | JsonValue = None,
    ) -> bool | list[str] | JsonValue:
        """Retrieve a result field by key with an optional default.

        Args:
            key: Field name.
            default: Default value when absent.

        Returns:
            Field value or `default`.
        """
        val: bool | list[str] | JsonValue = getattr(self, key, None)
        return val if val is not None else default


class ConceptTranslationResult(_McpResultModel):
    """Typed response payload for `translate_concept_arguments`."""

    concept: str = ""
    description: str | None = None
    source_framework: str = ""
    target_framework: str = ""
    translated_kwargs: dict[str, JsonValue] = pydantic.Field(default_factory=dict)
    unmapped_kwargs: dict[str, JsonValue] = pydantic.Field(default_factory=dict)
    error: str | None = None
    available_concepts: list[str] | None = None

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal["concept", "source_framework", "target_framework"],
    ) -> str: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["translated_kwargs", "unmapped_kwargs"]
    ) -> dict[str, JsonValue]: ...

    @typing.overload
    def __getitem__(self, key: str) -> JsonValue: ...

    def __getitem__(self, key: str) -> str | dict[str, JsonValue] | JsonValue:
        """Retrieve a translation result field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        val: str | dict[str, JsonValue] | JsonValue = getattr(self, key)
        return val


__all__ = [
    "AntiPatternExplanationResult",
    "CodeBlockCheckResult",
    "CodeBlockFinding",
    "ConceptTranslationResult",
    "DomainOpCheckResult",
    "HallucinationCheckResult",
    "IrCheckResult",
    "IsaCheckResult",
]
