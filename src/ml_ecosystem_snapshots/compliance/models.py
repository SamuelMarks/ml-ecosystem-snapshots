"""Typed Pydantic report schemas for compliance scoring and snippet verification."""

from __future__ import annotations

import pydantic
import typing_extensions

ParamSignatureTuple: typing_extensions.TypeAlias = tuple[str, str, "str | None", "str | None"]


class NamespaceAlignmentRule(pydantic.BaseModel):
    """Configurable rule set for normalizing and aligning target symbol paths to reference namespaces."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    prefix_map: dict[str, str] = pydantic.Field(default_factory=dict)
    subpackage_roots: tuple[str, ...] = ()
    node_prefix_rewrites: dict[str, str] = pydantic.Field(default_factory=dict)
    path_rewrites: tuple[tuple[str, str], ...] = ()


class ComplianceMismatchEntry(pydantic.BaseModel):
    """Typed mismatch entry within `ComplianceScoreReport`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    api_path: str
    expected: list[tuple[str, str, str | None, str | None]] = pydantic.Field(default_factory=list)
    actual: list[tuple[str, str, str | None, str | None]] = pydantic.Field(default_factory=list)
    reason: str | None = None


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


class SnippetComplianceReport(pydantic.BaseModel):
    """Typed compliance report returned by MLIR, WGSL, SASS, and RDNA snippet verifiers."""

    model_config = pydantic.ConfigDict(extra="forbid")

    is_compliant: bool = False
    total_ops: int = 0
    verified_ops: int = 0
    total_instructions: int = 0
    verified_instructions: int = 0
    errors: list[str] = pydantic.Field(default_factory=list)


__all__ = [
    "ComplianceMismatchEntry",
    "ComplianceScoreReport",
    "NamespaceAlignmentRule",
    "ParamSignatureTuple",
    "SnippetComplianceReport",
]
