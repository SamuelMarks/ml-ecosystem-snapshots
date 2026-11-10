"""FlashAttention & Fused Attention Kernels Snapshot Extractor.

Extracts FlashAttention-2/3, PagedAttention, and RaggedAttention kernel parameter signatures,
cache indexing layouts, and attention attributes matching `custom_ops.json`.
"""

from __future__ import annotations

import importlib
import inspect
import types

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalPythonOpSpec,
    CanonicalPythonOpTuple,
    build_canonical_python_ops,
    build_canonical_python_ref,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import InspectableTarget

_FLASH_ATTN_OP_DEFS: list[CanonicalPythonOpTuple] = [
    (
        "flash_attn_func",
        "flash_attn.flash_attn_func",
        "function",
        [
            ("q", "POSITIONAL_ONLY", None),
            ("k", "POSITIONAL_ONLY", None),
            ("v", "POSITIONAL_ONLY", None),
            ("dropout_p", "KEYWORD_ONLY", "0.0"),
            ("softmax_scale", "KEYWORD_ONLY", "None"),
            ("causal", "KEYWORD_ONLY", "False"),
            ("window_size", "KEYWORD_ONLY", "(-1, -1)"),
            ("alibi_slopes", "KEYWORD_ONLY", "None"),
            ("deterministic", "KEYWORD_ONLY", "False"),
        ],
        "Compute FlashAttention forward pass on dense tensors.",
    ),
    (
        "flash_attn_varlen_func",
        "flash_attn.flash_attn_varlen_func",
        "function",
        [
            ("q", "POSITIONAL_ONLY", None),
            ("k", "POSITIONAL_ONLY", None),
            ("v", "POSITIONAL_ONLY", None),
            ("cu_seqlens_q", "POSITIONAL_ONLY", None),
            ("cu_seqlens_k", "POSITIONAL_ONLY", None),
            ("max_seqlen_q", "POSITIONAL_ONLY", None),
            ("max_seqlen_k", "POSITIONAL_ONLY", None),
            ("dropout_p", "KEYWORD_ONLY", "0.0"),
            ("softmax_scale", "KEYWORD_ONLY", "None"),
            ("causal", "KEYWORD_ONLY", "False"),
        ],
        "Compute FlashAttention forward pass on variable length sequence batches.",
    ),
    (
        "paged_attention_v1",
        "vllm.paged_attention_v1",
        "function",
        [
            ("out", "POSITIONAL_ONLY", None),
            ("query", "POSITIONAL_ONLY", None),
            ("key_cache", "POSITIONAL_ONLY", None),
            ("value_cache", "POSITIONAL_ONLY", None),
            ("num_kv_heads", "POSITIONAL_ONLY", None),
            ("scale", "POSITIONAL_ONLY", None),
            ("block_tables", "POSITIONAL_ONLY", None),
            ("context_lens", "POSITIONAL_ONLY", None),
            ("block_size", "POSITIONAL_ONLY", None),
            ("max_context_len", "POSITIONAL_ONLY", None),
            ("alibi_slopes", "KEYWORD_ONLY", "None"),
            ("kv_cache_dtype", "KEYWORD_ONLY", "'auto'"),
        ],
        "PagedAttention forward pass using block-allocated KV cache.",
    ),
    (
        "ragged_paged_attention",
        "ml_switcheroo_compiler.ops.ragged_paged_attention",
        "function",
        [("query", "POSITIONAL_ONLY", None), ("key_cache", "POSITIONAL_ONLY", None), ("value_cache", "POSITIONAL_ONLY", None), ("block_tables", "POSITIONAL_ONLY", None), ("context_lens", "POSITIONAL_ONLY", None), ("block_size", "POSITIONAL_ONLY", None)],
        "Ragged PagedAttention kernel execution on non-rectangular tokens.",
    ),
    ("FlashAttention", "ml.switcheroo.custom.FlashAttention", "function", [("Q", "POSITIONAL_ONLY", None), ("K", "POSITIONAL_ONLY", None), ("V", "POSITIONAL_ONLY", None), ("causal", "KEYWORD_ONLY", "False"), ("scale", "KEYWORD_ONLY", "None")], "Custom FlashAttention specification."),
    (
        "ScaledDotProductAttention",
        "ml.switcheroo.custom.ScaledDotProductAttention",
        "function",
        [("query", "POSITIONAL_ONLY", None), ("key", "POSITIONAL_ONLY", None), ("value", "POSITIONAL_ONLY", None), ("attn_mask", "KEYWORD_ONLY", "None"), ("scale", "KEYWORD_ONLY", "None"), ("dropout_p", "KEYWORD_ONLY", "0.0"), ("is_causal", "KEYWORD_ONLY", "False")],
        "Scaled Dot-Product Attention operator.",
    ),
    ("RoPE", "ml.switcheroo.custom.RoPE", "function", [("X", "POSITIONAL_ONLY", None), ("cos", "POSITIONAL_ONLY", None), ("sin", "POSITIONAL_ONLY", None), ("dim", "KEYWORD_ONLY", "-1")], "Rotary Positional Embedding (RoPE) operator."),
]

CANONICAL_FLASH_ATTN_OPS: list[CanonicalPythonOpSpec] = build_canonical_python_ops(_FLASH_ATTN_OP_DEFS)


def _discover_flash_attn_module() -> types.ModuleType | None:
    """Safely import flash_attn or vllm if available.

    Returns:
        The flash_attn or vllm module object or None.
    """
    for mod_name in ("flash_attn", "vllm"):
        try:
            return importlib.import_module(mod_name)
        except Exception as exc:
            del exc
    return None


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
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
        return [
            build_canonical_python_ref(
                op,
                default_doc_prefix="Attention kernel",
                environment_tags=("attention", "fused_kernel", "custom_ops"),
                domain_metadata={"domain": "attention"},
            )
            for op in CANONICAL_FLASH_ATTN_OPS
        ]

    results: list[GhostRef] = []
    seen: set[str] = set()
    mod_members: dict[str, InspectableTarget] = dict(inspect.getmembers(mod))

    for op in CANONICAL_FLASH_ATTN_OPS:
        name = op.name
        if name in mod_members:
            obj: InspectableTarget = mod_members[name]
            try:
                ref = GhostInspector.inspect(obj, op.api_path, is_public=True)
                ref.environment_tags = list(ref.environment_tags or ()) + [
                    "attention",
                    "fused_kernel",
                ]
                results.append(ref)
                seen.add(op.api_path)
            except Exception as exc:
                del exc

    for op in CANONICAL_FLASH_ATTN_OPS:
        if op.api_path not in seen:
            results.append(
                build_canonical_python_ref(
                    op,
                    default_doc_prefix="Attention kernel",
                    environment_tags=("attention", "fused_kernel", "custom_ops"),
                    domain_metadata={"domain": "attention"},
                )
            )

    return results
