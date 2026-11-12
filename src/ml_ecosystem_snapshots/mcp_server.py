"""Model Context Protocol (MCP) Server for ML Framework Snapshots.

Exposes ground-truth framework signatures, assembly instruction verifiers,
and MLIR/StableHLO dialect verification tools over a standard JSON-RPC stdio
interface for code-generation agents.
"""

from __future__ import annotations

import typing

from ml_ecosystem_snapshots.mcp.cache import (
    DEFAULT_SNAPSHOT_REPOSITORY,
    ExtractSnapshotFn,
    SnapshotRepository,
    clear_snapshot_cache,
    get_cached_snapshot,
    get_framework_snapshot,
    resolve_snapshot,
)

try:
    from ml_ecosystem_snapshots.api import (
        FRAMEWORK_COLLECTORS as _imported_collectors,
    )
    from ml_ecosystem_snapshots.api import (
        CollectorFn,
    )
    from ml_ecosystem_snapshots.api import (
        extract_snapshot as _imported_extract_snapshot,
    )

    FRAMEWORK_COLLECTORS: dict[str, CollectorFn] = _imported_collectors
    extract_snapshot: ExtractSnapshotFn | None = _imported_extract_snapshot
except ImportError:
    FRAMEWORK_COLLECTORS = {}
    extract_snapshot = None

from ml_ecosystem_snapshots.mcp.models import (
    AntiPatternExplanationResult,
    CodeBlockCheckResult,
    CodeBlockFinding,
    ConceptTranslationResult,
    DomainOpCheckResult,
    HallucinationCheckResult,
    IrCheckResult,
    IsaCheckResult,
)
from ml_ecosystem_snapshots.mcp.protocol import (
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    run_stdio_loop,
)
from ml_ecosystem_snapshots.mcp.registry import (
    McpToolSpec,
    dispatch_tool_call,
    get_mcp_tools_list,
    handle_mcp_message,
)
from ml_ecosystem_snapshots.mcp.tools import (
    CONCEPT_ALIAS_MAP,
    DEFAULT_CONCEPT_MAP,
    DOMAIN_OP_CONFIGS,
    DTYPE_CANONICAL_ALIASES,
    DomainOpCheckConfig,
    build_domain_op_tool_schemas,
    check_array_api_op,
    check_aten_op,
    check_chex_op,
    check_code_block,
    check_cpp_op,
    check_domain_op_generic,
    check_flash_attention_op,
    check_grain_op,
    check_hallucination,
    check_metal_op,
    check_mlir_op,
    check_nccl_op,
    check_onnx_op,
    check_optax_op,
    check_orbax_op,
    check_ptx_instruction,
    check_rdna_instruction,
    check_safetensors_op,
    check_sass_instruction,
    check_scipy_op,
    check_stablehlo_op,
    check_torchaudio_op,
    check_torchvision_op,
    check_wasm_instruction,
    check_webgl_op,
    check_wgsl_op,
    explain_anti_pattern,
    get_api_signature,
    load_concept_map,
    matches_rank_constraint,
    normalize_dtype_name,
    search_apis,
    translate_concept_arguments,
)
from ml_ecosystem_snapshots.network_env import (
    get_custom_snapshots_paths,
    is_offline_mode,
)


def run_mcp_server(
    input_stream: typing.TextIO | None = None,
    output_stream: typing.TextIO | None = None,
    repository: SnapshotRepository | None = None,
) -> None:
    """Run the stdio MCP JSON-RPC loop for code-generation agent queries.

    Args:
        input_stream: Input stream to read from (defaults to sys.stdin).
        output_stream: Output stream to write to (defaults to sys.stdout).
        repository: Optional injected `SnapshotRepository` instance.
    """
    run_stdio_loop(
        lambda msg: handle_mcp_message(msg, repository=repository),
        input_stream=input_stream,
        output_stream=output_stream,
    )


__all__ = [
    "AntiPatternExplanationResult",
    "CONCEPT_ALIAS_MAP",
    "CodeBlockCheckResult",
    "CodeBlockFinding",
    "ConceptTranslationResult",
    "DEFAULT_CONCEPT_MAP",
    "DEFAULT_SNAPSHOT_REPOSITORY",
    "DOMAIN_OP_CONFIGS",
    "DTYPE_CANONICAL_ALIASES",
    "DomainOpCheckConfig",
    "DomainOpCheckResult",
    "FRAMEWORK_COLLECTORS",
    "HallucinationCheckResult",
    "IrCheckResult",
    "IsaCheckResult",
    "JsonRpcError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "McpToolSpec",
    "SnapshotRepository",
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
    "clear_snapshot_cache",
    "dispatch_tool_call",
    "explain_anti_pattern",
    "extract_snapshot",
    "get_api_signature",
    "get_cached_snapshot",
    "get_custom_snapshots_paths",
    "get_framework_snapshot",
    "get_mcp_tools_list",
    "handle_mcp_message",
    "is_offline_mode",
    "load_concept_map",
    "matches_rank_constraint",
    "normalize_dtype_name",
    "resolve_snapshot",
    "run_mcp_server",
    "run_stdio_loop",
    "search_apis",
    "translate_concept_arguments",
]
