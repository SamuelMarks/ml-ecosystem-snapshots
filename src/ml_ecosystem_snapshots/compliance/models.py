"""Typed Pydantic report schemas for compliance scoring and snippet verification."""

from __future__ import annotations

import typing
import typing_extensions

import pydantic

from ml_ecosystem_snapshots.models import JsonValue

ParamSignatureTuple: typing_extensions.TypeAlias = tuple[
    str, str, "str | None", "str | None"
]


class ComplianceMismatchEntry(pydantic.BaseModel):
    """Typed mismatch entry within `ComplianceScoreReport`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    api_path: str
    expected: list[tuple[str, str, str | None, str | None]] = pydantic.Field(
        default_factory=list
    )
    actual: list[tuple[str, str, str | None, str | None]] = pydantic.Field(
        default_factory=list
    )
    reason: str | None = None

    @typing.overload
    def __getitem__(self, key: typing.Literal["api_path"]) -> str: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["expected", "actual"]
    ) -> list[ParamSignatureTuple]: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["reason"]) -> str | None: ...

    @typing.overload
    def __getitem__(self, key: str) -> str | list[ParamSignatureTuple] | JsonValue: ...

    def __getitem__(self, key: str) -> str | list[ParamSignatureTuple] | JsonValue:
        """Retrieve a mismatch entry field by dictionary key.

        Args:
            key: Field name ('api_path', 'expected', 'actual', or 'reason').

        Returns:
            Field value.
        """
        if key == "api_path":
            return self.api_path
        if key == "expected":
            return self.expected
        if key == "actual":
            return self.actual
        return self.reason


class ComplianceScoreReport(pydantic.BaseModel):
    """Typed compliance report returned by `score_compliance`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    score_percentage: float = 0.0
    total_endpoints: int = 0
    matched: list[str] = pydantic.Field(default_factory=list)
    missing: list[str] = pydantic.Field(default_factory=list)
    mismatched: list[ComplianceMismatchEntry] = pydantic.Field(default_factory=list)
    opaque_signatures: list[str] = pydantic.Field(default_factory=list)
    warnings: list[str] = pydantic.Field(default_factory=list)

    @typing.overload
    def __getitem__(self, key: typing.Literal["score_percentage"]) -> float: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["total_endpoints"]) -> int: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal["matched", "missing", "opaque_signatures", "warnings"],
    ) -> list[str]: ...

    @typing.overload
    def __getitem__(
        self, key: typing.Literal["mismatched"]
    ) -> list[ComplianceMismatchEntry]: ...

    @typing.overload
    def __getitem__(
        self, key: str
    ) -> float | int | list[str] | list[ComplianceMismatchEntry] | JsonValue: ...

    def __getitem__(
        self, key: str
    ) -> float | int | list[str] | list[ComplianceMismatchEntry] | JsonValue:
        """Retrieve a compliance report field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        if key == "score_percentage":
            return self.score_percentage
        if key == "total_endpoints":
            return self.total_endpoints
        if key == "matched":
            return self.matched
        if key == "missing":
            return self.missing
        if key == "mismatched":
            return self.mismatched
        if key == "opaque_signatures":
            return self.opaque_signatures
        return self.warnings if key == "warnings" else None

    @typing.overload
    def get(
        self,
        key: typing.Literal["score_percentage"],
        default: float = 0.0,
    ) -> float: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["total_endpoints"],
        default: int = 0,
    ) -> int: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["matched", "missing", "opaque_signatures", "warnings"],
        default: list[str] | None = None,
    ) -> list[str]: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["mismatched"],
        default: list[ComplianceMismatchEntry] | None = None,
    ) -> list[ComplianceMismatchEntry]: ...

    @typing.overload
    def get(
        self,
        key: str,
        default: (
            float | int | list[str] | list[ComplianceMismatchEntry] | JsonValue
        ) = None,
    ) -> float | int | list[str] | list[ComplianceMismatchEntry] | JsonValue: ...

    def get(
        self,
        key: str,
        default: (
            float | int | list[str] | list[ComplianceMismatchEntry] | JsonValue
        ) = None,
    ) -> float | int | list[str] | list[ComplianceMismatchEntry] | JsonValue:
        """Retrieve a compliance report field by key with an optional default.

        Args:
            key: Field name.
            default: Default value when absent.

        Returns:
            Field value or `default`.
        """
        res = self[key]
        return default if res is None else res


class SnippetComplianceReport(pydantic.BaseModel):
    """Typed compliance report returned by MLIR, WGSL, SASS, and RDNA snippet verifiers."""

    model_config = pydantic.ConfigDict(extra="forbid")

    is_compliant: bool = False
    total_ops: int = 0
    verified_ops: int = 0
    total_instructions: int = 0
    verified_instructions: int = 0
    errors: list[str] = pydantic.Field(default_factory=list)

    @typing.overload
    def __getitem__(self, key: typing.Literal["is_compliant"]) -> bool: ...

    @typing.overload
    def __getitem__(
        self,
        key: typing.Literal[
            "total_ops",
            "verified_ops",
            "total_instructions",
            "verified_instructions",
        ],
    ) -> int: ...

    @typing.overload
    def __getitem__(self, key: typing.Literal["errors"]) -> list[str]: ...

    @typing.overload
    def __getitem__(self, key: str) -> list[str] | JsonValue: ...

    def __getitem__(self, key: str) -> bool | int | list[str] | JsonValue:
        """Retrieve a snippet compliance field by dictionary key.

        Args:
            key: Field name.

        Returns:
            Field value.
        """
        if key == "is_compliant":
            return self.is_compliant
        if key == "total_ops":
            return self.total_ops
        if key == "verified_ops":
            return self.verified_ops
        if key == "total_instructions":
            return self.total_instructions
        if key == "verified_instructions":
            return self.verified_instructions
        return self.errors if key == "errors" else None

    @typing.overload
    def get(
        self,
        key: typing.Literal["is_compliant"],
        default: bool = False,
    ) -> bool: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal[
            "total_ops",
            "verified_ops",
            "total_instructions",
            "verified_instructions",
        ],
        default: int = 0,
    ) -> int: ...

    @typing.overload
    def get(
        self,
        key: typing.Literal["errors"],
        default: list[str] | None = None,
    ) -> list[str]: ...

    @typing.overload
    def get(
        self,
        key: str,
        default: bool | int | list[str] | JsonValue = None,
    ) -> bool | int | list[str] | JsonValue: ...

    def get(
        self,
        key: str,
        default: bool | int | list[str] | JsonValue = None,
    ) -> bool | int | list[str] | JsonValue:
        """Retrieve a snippet compliance field by key with an optional default.

        Args:
            key: Field name.
            default: Default value when absent.

        Returns:
            Field value or `default`.
        """
        res = self[key]
        return default if res is None else res


__all__ = [
    "ComplianceMismatchEntry",
    "ComplianceScoreReport",
    "ParamSignatureTuple",
    "SnippetComplianceReport",
]
