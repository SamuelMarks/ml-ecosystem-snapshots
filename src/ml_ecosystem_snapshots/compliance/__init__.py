"""Compliance checking subpackage.

Provides static Python AST extraction, pure in-memory module source auditing,
snapshot vs. target compliance scoring, assembly/IR snippet compliance checkers,
and broadcasting/dtype promotion rules.
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
    align_namespace,
    analyze_source_ir_fw_compliance,
    extract_refs_from_source,
    extract_target_ast,
    extract_target_refs,
    extract_target_refs_single,
    get_module_info_from_path,
    normalize_source_module_name,
)
from ml_ecosystem_snapshots.compliance.dtype_rules import (
    DTYPE_BITWIDTHS,
    canonicalize_ir_dtype_name,
    estimate_tensor_memory_bytes,
    get_dtype_bitwidth,
    validate_broadcast_shapes,
    validate_dtype_for_op,
    validate_matmul_shapes,
)
from ml_ecosystem_snapshots.compliance.models import (
    ComplianceMismatchEntry,
    ComplianceReport,
    ComplianceScoreReport,
    NamespaceAlignmentRule,
    ParamSignatureTuple,
    SnippetComplianceReport,
)
from ml_ecosystem_snapshots.compliance.scorer import (
    audit_module_sources,
    score_compliance,
)

__all__ = [
    "ComplianceMismatchEntry",
    "ComplianceReport",
    "ComplianceScoreReport",
    "DTYPE_BITWIDTHS",
    "NamespaceAlignmentRule",
    "ParamSignatureTuple",
    "SnippetComplianceReport",
    "align_namespace",
    "analyze_source_ir_fw_compliance",
    "audit_module_sources",
    "canonicalize_ir_dtype_name",
    "check_mlir_text_compliance",
    "check_rdna_assembly_compliance",
    "check_sass_assembly_compliance",
    "check_wgsl_shader_compliance",
    "estimate_tensor_memory_bytes",
    "extract_refs_from_source",
    "extract_target_ast",
    "extract_target_refs",
    "extract_target_refs_single",
    "get_dtype_bitwidth",
    "get_module_info_from_path",
    "griffe",
    "normalize_source_module_name",
    "score_compliance",
    "validate_broadcast_shapes",
    "validate_dtype_for_op",
    "validate_matmul_shapes",
]
