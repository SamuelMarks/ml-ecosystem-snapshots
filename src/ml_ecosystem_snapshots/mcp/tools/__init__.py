"""Domain-specific MCP tool implementations subpackage."""

from __future__ import annotations

from ml_ecosystem_snapshots.mcp.tools.domain_ops import (
    DOMAIN_OP_CONFIGS,
    DomainOpCheckConfig,
    build_domain_op_tool_schemas,
    check_array_api_op,
    check_aten_op,
    check_chex_op,
    check_cpp_op,
    check_domain_op_generic,
    check_flash_attention_op,
    check_grain_op,
    check_metal_op,
    check_nccl_op,
    check_onnx_op,
    check_optax_op,
    check_orbax_op,
    check_safetensors_op,
    check_scipy_op,
    check_torchaudio_op,
    check_torchvision_op,
    check_wasm_instruction,
    check_webgl_op,
    check_wgsl_op,
)
from ml_ecosystem_snapshots.mcp.tools.hallucination import (
    DTYPE_CANONICAL_ALIASES,
    _matches_rank_constraint,
    check_code_block,
    check_hallucination,
    explain_anti_pattern,
    normalize_dtype_name,
)
from ml_ecosystem_snapshots.mcp.tools.ir import check_mlir_op, check_stablehlo_op
from ml_ecosystem_snapshots.mcp.tools.isa import (
    check_ptx_instruction,
    check_rdna_instruction,
    check_sass_instruction,
)
from ml_ecosystem_snapshots.mcp.tools.search import search_apis
from ml_ecosystem_snapshots.mcp.tools.signature import get_api_signature
from ml_ecosystem_snapshots.mcp.tools.translation import (
    CONCEPT_ALIAS_MAP,
    DEFAULT_CONCEPT_MAP,
    load_concept_map,
    translate_concept_arguments,
)

__all__ = [
    "CONCEPT_ALIAS_MAP",
    "DEFAULT_CONCEPT_MAP",
    "DOMAIN_OP_CONFIGS",
    "DTYPE_CANONICAL_ALIASES",
    "DomainOpCheckConfig",
    "_matches_rank_constraint",
    "build_domain_op_tool_schemas",
    "check_array_api_op",
    "check_aten_op",
    "check_chex_op",
    "check_code_block",
    "check_cpp_op",
    "check_domain_op_generic",
    "check_flash_attention_op",
    "check_grain_op",
    "check_hallucination",
    "check_metal_op",
    "check_mlir_op",
    "check_nccl_op",
    "check_onnx_op",
    "check_optax_op",
    "check_orbax_op",
    "check_ptx_instruction",
    "check_rdna_instruction",
    "check_safetensors_op",
    "check_sass_instruction",
    "check_scipy_op",
    "check_stablehlo_op",
    "check_torchaudio_op",
    "check_torchvision_op",
    "check_wasm_instruction",
    "check_webgl_op",
    "check_wgsl_op",
    "explain_anti_pattern",
    "get_api_signature",
    "load_concept_map",
    "normalize_dtype_name",
    "search_apis",
    "translate_concept_arguments",
]
