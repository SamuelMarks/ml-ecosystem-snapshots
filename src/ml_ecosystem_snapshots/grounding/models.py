"""Immutable data models, typed check results, and diagnostic report structures for the Grounding SDK."""

from __future__ import annotations

import pydantic
from ml_switcheroo_ir.validation.grounding import (
    DiagnosticSeverity,
    GroundingDiagnostic,
    GroundingReport,
)

from ml_ecosystem_snapshots.models import JsonValue


class _VerificationResultModel(pydantic.BaseModel):
    """Frozen base Pydantic model for structured verification results."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class HallucinationCheckResult(_VerificationResultModel):
    """Immutable verification result for Python API signature and call checks."""

    api_exists: bool = False
    is_hallucinated: bool = False
    invalid_kwargs: tuple[str, ...] = ()
    canonical_params: tuple[str, ...] = ()
    has_unconstrained_kwargs: bool = False
    reason: str = ""
    snapshot_source: str = ""
    signature_completeness: str = ""
    is_c_extension: bool = False
    dtype_errors: tuple[str, ...] | None = None
    rank_errors: tuple[str, ...] | None = None
    shape_errors: tuple[str, ...] | None = None
    warning: str | None = None


class AntiPatternExplanationResult(_VerificationResultModel):
    """Immutable verification result explaining hallucinated arguments and anti-patterns."""

    framework: str = ""
    api_path: str = ""
    hallucinated_argument: str = ""
    is_known_anti_pattern: bool = False
    canonical_argument: str | None = None
    explanation: str = ""
    canonical_params: tuple[str, ...] | None = None
    canonical_example: str = ""


class CodeBlockFinding(_VerificationResultModel):
    """Immutable finding item within `CodeBlockCheckResult`."""

    line: int = 0
    type: str = ""
    framework: str | None = None
    target: str = ""
    reason: str = ""
    invalid_kwargs: tuple[str, ...] | None = None


class CodeBlockCheckResult(_VerificationResultModel):
    """Immutable verification result for Python/MLIR/SASS/RDNA code blocks."""

    total_analyzed: int
    hallucinations_detected: int
    is_valid: bool
    findings: tuple[CodeBlockFinding, ...] = ()


class IsaCheckResult(_VerificationResultModel):
    """Immutable verification result for SASS, RDNA, and PTX hardware ISA checks."""

    is_valid: bool = False
    mnemonic_exists: bool = False
    encoding: str | None = None
    category: str | None = None
    min_sm: str | None = None
    supported_types: tuple[str, ...] | None = None
    supported_architectures: tuple[str, ...] | None = None
    valid_modifiers: tuple[str, ...] | None = None
    errors: tuple[str, ...] = ()


class IrCheckResult(_VerificationResultModel):
    """Immutable verification result for MLIR and StableHLO operation checks."""

    is_valid: bool = False
    op_exists: bool = False
    expected_operands: tuple[str | dict[str, JsonValue], ...] | None = None
    expected_attributes: tuple[str, ...] | None = None
    errors: tuple[str, ...] = ()


class DomainOpCheckResult(_VerificationResultModel):
    """Immutable verification result for domain operation checks."""

    is_valid: bool
    op_exists: bool
    expected_inputs: tuple[str, ...] = ()
    expected_attributes: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class ConceptTranslationResult(_VerificationResultModel):
    """Immutable verification result for cross-framework concept argument translation."""

    concept: str = ""
    description: str | None = None
    source_framework: str = ""
    target_framework: str = ""
    translated_kwargs: dict[str, JsonValue] = pydantic.Field(default_factory=dict)
    unmapped_kwargs: dict[str, JsonValue] = pydantic.Field(default_factory=dict)
    error: str | None = None
    available_concepts: tuple[str, ...] | None = None


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
