"""PyTorch ATen C++ Dispatch Table Snapshot Extractor.

Extracts low-level ATen operator dispatch schemas, overloads, and argument
types under the canonical `aten.*` domain matching `ATEN_REGISTRY`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostPythonRef,
    GhostRef,
    ParameterKind,
    SemanticTier,
)


CANONICAL_ATEN_OPS: List[Dict[str, Any]] = [
    # Math & Binary Arithmetic
    {
        "name": "add",
        "api_path": "aten.add",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "other", "kind": "POSITIONAL_ONLY"},
            {"name": "alpha", "kind": "KEYWORD_ONLY", "default": "1.0"},
        ],
        "docstring": "ATen addition dispatch kernel.",
    },
    {
        "name": "sub",
        "api_path": "aten.sub",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "other", "kind": "POSITIONAL_ONLY"},
            {"name": "alpha", "kind": "KEYWORD_ONLY", "default": "1.0"},
        ],
        "docstring": "ATen subtraction dispatch kernel.",
    },
    {
        "name": "mul",
        "api_path": "aten.mul",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "other", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "ATen elementwise multiplication dispatch kernel.",
    },
    {
        "name": "div",
        "api_path": "aten.div",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "other", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "ATen division dispatch kernel.",
    },
    {
        "name": "matmul",
        "api_path": "aten.matmul",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "other", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "ATen matrix multiplication dispatch kernel.",
    },
    {
        "name": "mm",
        "api_path": "aten.mm",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "mat2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "ATen 2D matrix multiplication dispatch kernel.",
    },
    {
        "name": "bmm",
        "api_path": "aten.bmm",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "mat2", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "ATen batch matrix multiplication dispatch kernel.",
    },
    # Neural network activations & ops
    {
        "name": "relu",
        "api_path": "aten.relu",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "ATen rectified linear unit dispatch kernel.",
    },
    {
        "name": "sigmoid",
        "api_path": "aten.sigmoid",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "ATen sigmoid activation dispatch kernel.",
    },
    {
        "name": "tanh",
        "api_path": "aten.tanh",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "ATen hyperbolic tangent activation dispatch kernel.",
    },
    {
        "name": "gelu",
        "api_path": "aten.gelu",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "approximate", "kind": "KEYWORD_ONLY", "default": "'none'"},
        ],
        "docstring": "ATen Gaussian Error Linear Unit activation dispatch kernel.",
    },
    {
        "name": "softmax",
        "api_path": "aten.softmax",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "dim", "kind": "POSITIONAL_ONLY"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "ATen softmax dispatch kernel.",
    },
    {
        "name": "log_softmax",
        "api_path": "aten.log_softmax",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "dim", "kind": "POSITIONAL_ONLY"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "ATen log-softmax dispatch kernel.",
    },
    # Reductions
    {
        "name": "sum",
        "api_path": "aten.sum",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "dim", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdim", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "ATen sum reduction dispatch kernel.",
    },
    {
        "name": "mean",
        "api_path": "aten.mean",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "dim", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdim", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "ATen mean reduction dispatch kernel.",
    },
    {
        "name": "max",
        "api_path": "aten.max",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "dim", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdim", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "ATen maximum reduction dispatch kernel.",
    },
    {
        "name": "min",
        "api_path": "aten.min",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "dim", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "keepdim", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "ATen minimum reduction dispatch kernel.",
    },
    # Tensor layout & manipulation
    {
        "name": "view",
        "api_path": "aten.view",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "size", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "ATen tensor view reshape kernel.",
    },
    {
        "name": "reshape",
        "api_path": "aten.reshape",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "shape", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "ATen tensor reshape kernel.",
    },
    {
        "name": "transpose",
        "api_path": "aten.transpose",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "dim0", "kind": "POSITIONAL_ONLY"},
            {"name": "dim1", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "ATen dimension transpose kernel.",
    },
    {
        "name": "permute",
        "api_path": "aten.permute",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "dims", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "ATen tensor dimension permutation kernel.",
    },
    {
        "name": "squeeze",
        "api_path": "aten.squeeze",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "dim", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "ATen singleton dimension squeeze kernel.",
    },
    {
        "name": "unsqueeze",
        "api_path": "aten.unsqueeze",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "dim", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "ATen dimension unsqueeze kernel.",
    },
    {
        "name": "clone",
        "api_path": "aten.clone",
        "params": [
            {"name": "self", "kind": "POSITIONAL_ONLY"},
            {"name": "memory_format", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "ATen tensor memory clone kernel.",
    },
]


def _resolve_torch() -> Any:
    """Resolve torch module from import or sys.modules.

    Returns:
        The torch module, or None if unavailable.
    """
    try:
        import torch

        return torch
    except (ImportError, Exception):
        return None


def _get_torch_ops_aten() -> Any:
    """Safely import PyTorch torch.ops.aten namespace.

    Returns:
        The torch.ops.aten namespace object or None if not available.
    """
    try:
        mod = _resolve_torch()
        if mod is not None:
            ops = getattr(mod, "ops", None)
            if ops is not None:
                return getattr(ops, "aten", None)
    except Exception:
        pass
    return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect PyTorch ATen C++ dispatch operator signatures.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRef objects representing ATen C++ operations.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.NEURAL_OPS,
        SemanticTier.UTIL,
    ):
        return []

    aten_ops = _get_torch_ops_aten()
    if aten_ops is None:
        refs: List[GhostRef] = []
        for op in CANONICAL_ATEN_OPS:
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
                    docstring=op.get(
                        "docstring", f"ATen {op['name']} dispatch kernel."
                    ),
                    environment_tags=["aten", "dispatch", "c_extension"],
                    domain_metadata={"domain": "aten"},
                )
            )
        return refs

    from .torch import get_aten_op_schema

    results: List[GhostRef] = []
    seen: set[str] = set()

    for op in CANONICAL_ATEN_OPS:
        name = op["name"]
        if hasattr(aten_ops, name):
            schemas = get_aten_op_schema(name)
            if schemas:
                primary = schemas[0]
                params = [
                    GhostParam(
                        name=p["name"],
                        kind=getattr(
                            ParameterKind, p.get("kind", "POSITIONAL_OR_KEYWORD")
                        ),
                        default=p.get("default"),
                    )
                    for p in primary.get("params", [])
                ]
                ref = GhostPythonRef(
                    name=name,
                    api_path=f"aten.{name}",
                    kind="function",
                    params=params,
                    docstring=f"ATen {name} C++ dispatch kernel.",
                    environment_tags=["aten", "dispatch", "c_extension"],
                    domain_metadata={"domain": "aten"},
                )
                results.append(ref)
                seen.add(name)

    # Fill any missed ops from canonical definitions
    for op in CANONICAL_ATEN_OPS:
        if op["name"] not in seen:
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
                    docstring=op.get(
                        "docstring", f"ATen {op['name']} dispatch kernel."
                    ),
                    environment_tags=["aten", "dispatch", "c_extension"],
                    domain_metadata={"domain": "aten"},
                )
            )

    return results
