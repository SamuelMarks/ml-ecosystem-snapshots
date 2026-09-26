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
    {
        "name": "pow",
        "api_path": "aten.pow",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "exponent", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen pow C++ dispatch kernel.",
    },
    {
        "name": "neg",
        "api_path": "aten.neg",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen neg C++ dispatch kernel.",
    },
    {
        "name": "abs",
        "api_path": "aten.abs",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen abs C++ dispatch kernel.",
    },
    {
        "name": "exp",
        "api_path": "aten.exp",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen exp C++ dispatch kernel.",
    },
    {
        "name": "log",
        "api_path": "aten.log",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen log C++ dispatch kernel.",
    },
    {
        "name": "sqrt",
        "api_path": "aten.sqrt",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen sqrt C++ dispatch kernel.",
    },
    {
        "name": "sin",
        "api_path": "aten.sin",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen sin C++ dispatch kernel.",
    },
    {
        "name": "cos",
        "api_path": "aten.cos",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen cos C++ dispatch kernel.",
    },
    {
        "name": "tan",
        "api_path": "aten.tan",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen tan C++ dispatch kernel.",
    },
    {
        "name": "silu",
        "api_path": "aten.silu",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen silu C++ dispatch kernel.",
    },
    {
        "name": "addmm",
        "api_path": "aten.addmm",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "mat1", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "mat2", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "beta", "kind": "KEYWORD_ONLY", "default": "1"},
            {"name": "alpha", "kind": "KEYWORD_ONLY", "default": "1"},
        ],
        "docstring": "ATen addmm C++ dispatch kernel.",
    },
    {
        "name": "linear",
        "api_path": "aten.linear",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "weight", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "bias", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
        ],
        "docstring": "ATen linear C++ dispatch kernel.",
    },
    {
        "name": "convolution",
        "api_path": "aten.convolution",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "weight", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "bias", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "stride", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "padding", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dilation", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "transposed", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "output_padding", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "groups", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen convolution C++ dispatch kernel.",
    },
    {
        "name": "conv2d",
        "api_path": "aten.conv2d",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "weight", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "bias", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "stride", "kind": "POSITIONAL_OR_KEYWORD", "default": "[1, 1]"},
            {"name": "padding", "kind": "POSITIONAL_OR_KEYWORD", "default": "[0, 0]"},
            {"name": "dilation", "kind": "POSITIONAL_OR_KEYWORD", "default": "[1, 1]"},
            {"name": "groups", "kind": "POSITIONAL_OR_KEYWORD", "default": "1"},
        ],
        "docstring": "ATen conv2d C++ dispatch kernel.",
    },
    {
        "name": "layer_norm",
        "api_path": "aten.layer_norm",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "normalized_shape", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "weight", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "bias", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "eps", "kind": "POSITIONAL_OR_KEYWORD", "default": "1e-05"},
            {
                "name": "cudnn_enable",
                "kind": "POSITIONAL_OR_KEYWORD",
                "default": "True",
            },
        ],
        "docstring": "ATen layer_norm C++ dispatch kernel.",
    },
    {
        "name": "rms_norm",
        "api_path": "aten.rms_norm",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "normalized_shape", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "weight", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "eps", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
        ],
        "docstring": "ATen rms_norm C++ dispatch kernel.",
    },
    {
        "name": "group_norm",
        "api_path": "aten.group_norm",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "num_groups", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "weight", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "bias", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "eps", "kind": "POSITIONAL_OR_KEYWORD", "default": "1e-05"},
            {
                "name": "cudnn_enabled",
                "kind": "POSITIONAL_OR_KEYWORD",
                "default": "True",
            },
        ],
        "docstring": "ATen group_norm C++ dispatch kernel.",
    },
    {
        "name": "batch_norm",
        "api_path": "aten.batch_norm",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "weight", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "bias", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "running_mean", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "running_var", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "training", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "momentum", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "eps", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "cudnn_enabled", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen batch_norm C++ dispatch kernel.",
    },
    {
        "name": "embedding",
        "api_path": "aten.embedding",
        "params": [
            {"name": "weight", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "indices", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "padding_idx", "kind": "POSITIONAL_OR_KEYWORD", "default": "-1"},
            {
                "name": "scale_grad_by_freq",
                "kind": "POSITIONAL_OR_KEYWORD",
                "default": "False",
            },
            {"name": "sparse", "kind": "POSITIONAL_OR_KEYWORD", "default": "False"},
        ],
        "docstring": "ATen embedding C++ dispatch kernel.",
    },
    {
        "name": "scaled_dot_product_attention",
        "api_path": "aten.scaled_dot_product_attention",
        "params": [
            {"name": "query", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "key", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "value", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "attn_mask", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "dropout_p", "kind": "POSITIONAL_OR_KEYWORD", "default": "0.0"},
            {"name": "is_causal", "kind": "POSITIONAL_OR_KEYWORD", "default": "False"},
            {"name": "scale", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "enable_gqa", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "ATen scaled_dot_product_attention C++ dispatch kernel.",
    },
    {
        "name": "t",
        "api_path": "aten.t",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen t C++ dispatch kernel.",
    },
    {
        "name": "flatten",
        "api_path": "aten.flatten",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "start_dim", "kind": "POSITIONAL_OR_KEYWORD", "default": "0"},
            {"name": "end_dim", "kind": "POSITIONAL_OR_KEYWORD", "default": "-1"},
        ],
        "docstring": "ATen flatten C++ dispatch kernel.",
    },
    {
        "name": "cat",
        "api_path": "aten.cat",
        "params": [
            {"name": "tensors", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dim", "kind": "POSITIONAL_OR_KEYWORD", "default": "0"},
        ],
        "docstring": "ATen cat C++ dispatch kernel.",
    },
    {
        "name": "split",
        "api_path": "aten.split",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "split_size", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dim", "kind": "POSITIONAL_OR_KEYWORD", "default": "0"},
        ],
        "docstring": "ATen split C++ dispatch kernel.",
    },
    {
        "name": "slice",
        "api_path": "aten.slice",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dim", "kind": "POSITIONAL_OR_KEYWORD", "default": "0"},
            {"name": "start", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "end", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "step", "kind": "POSITIONAL_OR_KEYWORD", "default": "1"},
        ],
        "docstring": "ATen slice C++ dispatch kernel.",
    },
    {
        "name": "select",
        "api_path": "aten.select",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dim", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "index", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen select C++ dispatch kernel.",
    },
    {
        "name": "index_select",
        "api_path": "aten.index_select",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dim", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "index", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen index_select C++ dispatch kernel.",
    },
    {
        "name": "prod",
        "api_path": "aten.prod",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "ATen prod C++ dispatch kernel.",
    },
    {
        "name": "argmax",
        "api_path": "aten.argmax",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dim", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "keepdim", "kind": "POSITIONAL_OR_KEYWORD", "default": "False"},
        ],
        "docstring": "ATen argmax C++ dispatch kernel.",
    },
    {
        "name": "argmin",
        "api_path": "aten.argmin",
        "params": [
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dim", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "keepdim", "kind": "POSITIONAL_OR_KEYWORD", "default": "False"},
        ],
        "docstring": "ATen argmin C++ dispatch kernel.",
    },
    {
        "name": "zeros",
        "api_path": "aten.zeros",
        "params": [
            {"name": "size", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "names", "kind": "KEYWORD_ONLY"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "layout", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "pin_memory", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "ATen zeros C++ dispatch kernel.",
    },
    {
        "name": "ones",
        "api_path": "aten.ones",
        "params": [
            {"name": "size", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "names", "kind": "KEYWORD_ONLY"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "layout", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "pin_memory", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "ATen ones C++ dispatch kernel.",
    },
    {
        "name": "empty",
        "api_path": "aten.empty",
        "params": [
            {"name": "size", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "layout", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "pin_memory", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "memory_format", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "ATen empty C++ dispatch kernel.",
    },
    {
        "name": "full",
        "api_path": "aten.full",
        "params": [
            {"name": "size", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "fill_value", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "names", "kind": "KEYWORD_ONLY"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "layout", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "pin_memory", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "ATen full C++ dispatch kernel.",
    },
    {
        "name": "arange",
        "api_path": "aten.arange",
        "params": [
            {"name": "end", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dtype", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "layout", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "pin_memory", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "ATen arange C++ dispatch kernel.",
    },
    {
        "name": "where",
        "api_path": "aten.where",
        "params": [
            {"name": "condition", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "input", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "other", "kind": "POSITIONAL_OR_KEYWORD"},
        ],
        "docstring": "ATen where C++ dispatch kernel.",
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
