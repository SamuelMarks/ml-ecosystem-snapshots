"""Data models, typed check results, and diagnostic report structures for the Grounding SDK."""

from __future__ import annotations

import pydantic
from ml_switcheroo_ir.validation.grounding import (
    DiagnosticSeverity,
    GroundingDiagnostic,
    GroundingReport,
)

from ml_ecosystem_snapshots.models import JsonValue


class _VerificationResultModel(pydantic.BaseModel):
    """Base Pydantic model for structured verification results."""

    model_config = pydantic.ConfigDict(extra="forbid")


class HallucinationCheckResult(_VerificationResultModel):
    """Typed verification result for Python API signature and call checks."""

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


class AntiPatternExplanationResult(_VerificationResultModel):
    """Typed verification result explaining hallucinated arguments and anti-patterns."""

    framework: str = ""
    api_path: str = ""
    hallucinated_argument: str = ""
    is_known_anti_pattern: bool = False
    canonical_argument: str | None = None
    explanation: str = ""
    canonical_params: list[str] | None = None
    canonical_example: str = ""


class CodeBlockFinding(_VerificationResultModel):
    """Typed finding item within `CodeBlockCheckResult`."""

    line: int = 0
    type: str = ""
    framework: str | None = None
    target: str = ""
    reason: str = ""
    invalid_kwargs: list[str] | None = None


class CodeBlockCheckResult(_VerificationResultModel):
    """Typed verification result for Python/MLIR/SASS/RDNA code blocks."""

    total_analyzed: int
    hallucinations_detected: int
    is_valid: bool
    findings: list[CodeBlockFinding] = pydantic.Field(default_factory=list)


class IsaCheckResult(_VerificationResultModel):
    """Typed verification result for SASS, RDNA, and PTX hardware ISA checks."""

    is_valid: bool = False
    mnemonic_exists: bool = False
    encoding: str | None = None
    category: str | None = None
    min_sm: str | None = None
    supported_types: list[str] | None = None
    supported_architectures: list[str] | None = None
    valid_modifiers: list[str] | None = None
    errors: list[str] = pydantic.Field(default_factory=list)


class IrCheckResult(_VerificationResultModel):
    """Typed verification result for MLIR and StableHLO operation checks."""

    is_valid: bool = False
    op_exists: bool = False
    expected_operands: list[str | dict[str, JsonValue]] | None = None
    expected_attributes: list[str] | None = None
    errors: list[str] = pydantic.Field(default_factory=list)


class DomainOpCheckResult(_VerificationResultModel):
    """Typed verification result for domain operation checks."""

    is_valid: bool
    op_exists: bool
    expected_inputs: list[str] = pydantic.Field(default_factory=list)
    expected_attributes: list[str] = pydantic.Field(default_factory=list)
    errors: list[str] = pydantic.Field(default_factory=list)


class ConceptTranslationResult(_VerificationResultModel):
    """Typed verification result for cross-framework concept argument translation."""

    concept: str = ""
    description: str | None = None
    source_framework: str = ""
    target_framework: str = ""
    translated_kwargs: dict[str, JsonValue] = pydantic.Field(default_factory=dict)
    unmapped_kwargs: dict[str, JsonValue] = pydantic.Field(default_factory=dict)
    error: str | None = None
    available_concepts: list[str] | None = None


__all__ = [
    "AntiPatternExplanationResult",
    "CodeBlockCheckResult",
    "CodeBlockFinding",
    "ConceptTranslationResult",
    "DiagnosticSeverity",
    "DomainOpCheckResult",
    "GroundingDiagnostic",
    "GroundingReport",
    "HallucinationCheckResult",
    "IrCheckResult",
    "IsaCheckResult",
]
