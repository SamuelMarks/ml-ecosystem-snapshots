"""TypedDict response schemas for MCP tool payloads."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class HallucinationCheckResult(TypedDict, total=False):
    """Typed response payload for `check_hallucination`."""

    api_exists: bool
    is_hallucinated: bool
    invalid_kwargs: List[str]
    canonical_params: List[str]
    has_unconstrained_kwargs: bool
    reason: str
    snapshot_source: str
    signature_completeness: str
    is_c_extension: bool
    dtype_errors: List[str]
    rank_errors: List[str]
    shape_errors: List[str]
    warning: str


class AntiPatternExplanationResult(TypedDict, total=False):
    """Typed response payload for `explain_anti_pattern`."""

    framework: str
    api_path: str
    hallucinated_argument: str
    is_known_anti_pattern: bool
    canonical_argument: Optional[str]
    explanation: str
    canonical_params: List[str]
    canonical_example: str


class CodeBlockFinding(TypedDict, total=False):
    """Typed finding item within `CodeBlockCheckResult`."""

    line: int
    type: str
    framework: str
    target: str
    reason: str
    invalid_kwargs: List[str]


class CodeBlockCheckResult(TypedDict):
    """Typed response payload for `check_code_block`."""

    total_analyzed: int
    hallucinations_detected: int
    is_valid: bool
    findings: List[CodeBlockFinding]


class IsaCheckResult(TypedDict, total=False):
    """Typed response payload for `check_sass_instruction`, `check_rdna_instruction`, and `check_ptx_instruction`."""

    is_valid: bool
    mnemonic_exists: bool
    encoding: Optional[str]
    category: str
    min_sm: str
    supported_types: List[str]
    supported_architectures: List[str]
    valid_modifiers: List[str]
    errors: List[str]


class IrCheckResult(TypedDict, total=False):
    """Typed response payload for `check_mlir_op` and `check_stablehlo_op`."""

    is_valid: bool
    op_exists: bool
    expected_operands: List[Any]
    expected_attributes: List[str]
    errors: List[str]


class DomainOpCheckResult(TypedDict):
    """Typed response payload for `check_domain_op_generic` and domain verifiers."""

    is_valid: bool
    op_exists: bool
    expected_inputs: List[str]
    expected_attributes: List[str]
    errors: List[str]


class ConceptTranslationResult(TypedDict):
    """Typed response payload for `translate_concept_arguments`."""

    concept: str
    source_framework: str
    target_framework: str
    translated_kwargs: Dict[str, Any]
    unmapped_kwargs: Dict[str, Any]


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
