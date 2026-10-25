"""Compliance checking subpackage.

Provides static Python AST extraction, snapshot vs. target compliance scoring,
assembly/IR snippet compliance checkers, and broadcasting/dtype promotion rules.
"""

from __future__ import annotations

import griffe

from ml_ecosystem_snapshots.compliance.assembly_verifier import (
    check_mlir_text_compliance,
    check_rdna_assembly_compliance,
    check_sass_assembly_compliance,
    check_wgsl_shader_compliance,
)
from ml_ecosystem_snapshots.compliance.ast_extractor import (
    DEFAULT_NAMESPACE_PREFIX_MAP,
    _temporary_sys_path,
    align_namespace,
    extract_target_ast,
    extract_target_refs,
    extract_target_refs_single,
    get_module_info_from_path,
)
from ml_ecosystem_snapshots.compliance.dtype_rules import (
    DTYPE_BITWIDTHS,
    estimate_tensor_memory_bytes,
    get_dtype_bitwidth,
    validate_broadcast_shapes,
    validate_dtype_for_op,
    validate_matmul_shapes,
)
from ml_ecosystem_snapshots.compliance.models import (
    ComplianceMismatchEntry,
    ComplianceScoreReport,
    ParamSignatureTuple,
    SnippetComplianceReport,
)
from ml_ecosystem_snapshots.compliance.scorer import score_compliance

__all__ = [
    "ComplianceMismatchEntry",
    "ComplianceScoreReport",
    "DEFAULT_NAMESPACE_PREFIX_MAP",
    "DTYPE_BITWIDTHS",
    "ParamSignatureTuple",
    "SnippetComplianceReport",
    "_temporary_sys_path",
    "align_namespace",
    "check_mlir_text_compliance",
    "check_rdna_assembly_compliance",
    "check_sass_assembly_compliance",
    "check_wgsl_shader_compliance",
    "estimate_tensor_memory_bytes",
    "extract_target_ast",
    "extract_target_refs",
    "extract_target_refs_single",
    "get_dtype_bitwidth",
    "get_module_info_from_path",
    "griffe",
    "score_compliance",
    "validate_broadcast_shapes",
    "validate_dtype_for_op",
    "validate_matmul_shapes",
]
