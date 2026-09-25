"""FlashAttention & Fused Attention Kernels Snapshot Extractor.

Extracts FlashAttention-2/3, PagedAttention, and RaggedAttention kernel parameter signatures,
cache indexing layouts, and attention attributes matching `custom_ops.json`.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostPythonRef,
    GhostRef,
    ParameterKind,
    SemanticTier,
)
from ..models import GhostInspector


CANONICAL_FLASH_ATTN_OPS: List[Dict[str, Any]] = [
    # FlashAttention-2 Core
    {
        "name": "flash_attn_func",
        "api_path": "flash_attn.flash_attn_func",
        "params": [
            {"name": "q", "kind": "POSITIONAL_ONLY"},
            {"name": "k", "kind": "POSITIONAL_ONLY"},
            {"name": "v", "kind": "POSITIONAL_ONLY"},
            {"name": "dropout_p", "kind": "KEYWORD_ONLY", "default": "0.0"},
            {"name": "softmax_scale", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "causal", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "window_size", "kind": "KEYWORD_ONLY", "default": "(-1, -1)"},
            {"name": "alibi_slopes", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "deterministic", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Compute FlashAttention forward pass on dense tensors.",
    },
    # FlashAttention Varlen
    {
        "name": "flash_attn_varlen_func",
        "api_path": "flash_attn.flash_attn_varlen_func",
        "params": [
            {"name": "q", "kind": "POSITIONAL_ONLY"},
            {"name": "k", "kind": "POSITIONAL_ONLY"},
            {"name": "v", "kind": "POSITIONAL_ONLY"},
            {"name": "cu_seqlens_q", "kind": "POSITIONAL_ONLY"},
            {"name": "cu_seqlens_k", "kind": "POSITIONAL_ONLY"},
            {"name": "max_seqlen_q", "kind": "POSITIONAL_ONLY"},
            {"name": "max_seqlen_k", "kind": "POSITIONAL_ONLY"},
            {"name": "dropout_p", "kind": "KEYWORD_ONLY", "default": "0.0"},
            {"name": "softmax_scale", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "causal", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Compute FlashAttention forward pass on variable length sequence batches.",
    },
    # PagedAttention (vLLM style KV cache)
    {
        "name": "paged_attention_v1",
        "api_path": "vllm.paged_attention_v1",
        "params": [
            {"name": "out", "kind": "POSITIONAL_ONLY"},
            {"name": "query", "kind": "POSITIONAL_ONLY"},
            {"name": "key_cache", "kind": "POSITIONAL_ONLY"},
            {"name": "value_cache", "kind": "POSITIONAL_ONLY"},
            {"name": "num_kv_heads", "kind": "POSITIONAL_ONLY"},
            {"name": "scale", "kind": "POSITIONAL_ONLY"},
            {"name": "block_tables", "kind": "POSITIONAL_ONLY"},
            {"name": "context_lens", "kind": "POSITIONAL_ONLY"},
            {"name": "block_size", "kind": "POSITIONAL_ONLY"},
            {"name": "max_context_len", "kind": "POSITIONAL_ONLY"},
            {"name": "alibi_slopes", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "kv_cache_dtype", "kind": "KEYWORD_ONLY", "default": "'auto'"},
        ],
        "docstring": "PagedAttention forward pass using block-allocated KV cache.",
    },
    # Ragged PagedAttention
    {
        "name": "ragged_paged_attention",
        "api_path": "ml_switcheroo_compiler.ops.ragged_paged_attention",
        "params": [
            {"name": "query", "kind": "POSITIONAL_ONLY"},
            {"name": "key_cache", "kind": "POSITIONAL_ONLY"},
            {"name": "value_cache", "kind": "POSITIONAL_ONLY"},
            {"name": "block_tables", "kind": "POSITIONAL_ONLY"},
            {"name": "context_lens", "kind": "POSITIONAL_ONLY"},
            {"name": "block_size", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Ragged PagedAttention kernel execution on non-rectangular tokens.",
    },
    # Custom Dialect Ops matching custom_ops.json
    {
        "name": "FlashAttention",
        "api_path": "ml.switcheroo.custom.FlashAttention",
        "params": [
            {"name": "Q", "kind": "POSITIONAL_ONLY"},
            {"name": "K", "kind": "POSITIONAL_ONLY"},
            {"name": "V", "kind": "POSITIONAL_ONLY"},
            {"name": "causal", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "scale", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Custom FlashAttention specification.",
    },
    {
        "name": "ScaledDotProductAttention",
        "api_path": "ml.switcheroo.custom.ScaledDotProductAttention",
        "params": [
            {"name": "query", "kind": "POSITIONAL_ONLY"},
            {"name": "key", "kind": "POSITIONAL_ONLY"},
            {"name": "value", "kind": "POSITIONAL_ONLY"},
            {"name": "attn_mask", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "scale", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "dropout_p", "kind": "KEYWORD_ONLY", "default": "0.0"},
            {"name": "is_causal", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Scaled Dot-Product Attention operator.",
    },
    {
        "name": "RoPE",
        "api_path": "ml.switcheroo.custom.RoPE",
        "params": [
            {"name": "X", "kind": "POSITIONAL_ONLY"},
            {"name": "cos", "kind": "POSITIONAL_ONLY"},
            {"name": "sin", "kind": "POSITIONAL_ONLY"},
            {"name": "dim", "kind": "KEYWORD_ONLY", "default": "-1"},
        ],
        "docstring": "Rotary Positional Embedding (RoPE) operator.",
    },
]


def _discover_flash_attn_module() -> Any:
    """Safely import flash_attn or vllm if available.

    Returns:
        The flash_attn or vllm module object or None.
    """
    for mod_name in ("flash_attn", "vllm"):
        try:
            return importlib.import_module(mod_name)
        except (ImportError, Exception):
            pass
    return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect FlashAttention, PagedAttention, and custom attention operator signatures.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRef objects representing attention operations.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.NEURAL_OPS,
        SemanticTier.LAYER,
        SemanticTier.UTIL,
    ):
        return []

    mod = _discover_flash_attn_module()
    if mod is None:
        refs: List[GhostRef] = []
        for op in CANONICAL_FLASH_ATTN_OPS:
            params = [
                GhostParam(
                    name=p["name"],
                    kind=getattr(ParameterKind, p.get("kind", "POSITIONAL_OR_KEYWORD")),
                    default=p.get("default"),
                )
                for p in op.get("params", [])
            ]
            refs.append(
                GhostPythonRef(
                    name=op["name"],
                    api_path=op["api_path"],
                    kind="function",
                    params=params,
                    docstring=op.get("docstring", f"Attention kernel {op['name']}."),
                    environment_tags=["attention", "fused_kernel", "custom_ops"],
                    domain_metadata={"domain": "attention"},
                )
            )
        return refs

    results: List[GhostRef] = []
    seen: set[str] = set()

    for op in CANONICAL_FLASH_ATTN_OPS:
        name = op["name"]
        if hasattr(mod, name):
            obj = getattr(mod, name)
            try:
                ref = GhostInspector.inspect(obj, op["api_path"], is_public=True)
                ref.environment_tags = list(ref.environment_tags or ()) + [
                    "attention",
                    "fused_kernel",
                ]
                results.append(ref)
                seen.add(op["api_path"])
            except Exception:
                pass

    # Fill any missed ops from canonical definitions
    for op in CANONICAL_FLASH_ATTN_OPS:
        if op["api_path"] not in seen:
            params = [
                GhostParam(
                    name=p["name"],
                    kind=getattr(ParameterKind, p.get("kind", "POSITIONAL_OR_KEYWORD")),
                    default=p.get("default"),
                )
                for p in op.get("params", [])
            ]
            results.append(
                GhostPythonRef(
                    name=op["name"],
                    api_path=op["api_path"],
                    kind="function",
                    params=params,
                    docstring=op.get("docstring", f"Attention kernel {op['name']}."),
                    environment_tags=["attention", "fused_kernel", "custom_ops"],
                    domain_metadata={"domain": "attention"},
                )
            )

    return results
