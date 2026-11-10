"""PyTorch ATen C++ Dispatch Table Snapshot Extractor.

Extracts low-level ATen operator dispatch schemas, overloads, and argument
types under the canonical `aten.*` domain matching `ATEN_REGISTRY`.
"""

from __future__ import annotations

import inspect
import types
import typing

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostPythonRef,
    GhostRef,
    ParameterKind,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalPythonOpSpec,
    CanonicalPythonOpTuple,
    build_canonical_python_ops,
    build_canonical_python_ref,
)
from ml_ecosystem_snapshots.frameworks.torch import (
    _coerce_aten_schema,
    get_aten_op_schema,
)


@typing.runtime_checkable
class _AtenOpsContainerProtocol(typing.Protocol):
    """Protocol for `torch.ops` exposing the `.aten` namespace."""

    aten: types.ModuleType


@typing.runtime_checkable
class _TorchWithOpsProtocol(typing.Protocol):
    """Protocol for the `torch` module exposing `.ops`."""

    ops: _AtenOpsContainerProtocol


_ATEN_OP_DEFS: list[CanonicalPythonOpTuple] = [
    ("add", "aten.add", "function", [("self", "POSITIONAL_ONLY", None), ("other", "POSITIONAL_ONLY", None), ("alpha", "KEYWORD_ONLY", "1.0")], "ATen addition dispatch kernel."),
    ("sub", "aten.sub", "function", [("self", "POSITIONAL_ONLY", None), ("other", "POSITIONAL_ONLY", None), ("alpha", "KEYWORD_ONLY", "1.0")], "ATen subtraction dispatch kernel."),
    ("mul", "aten.mul", "function", [("self", "POSITIONAL_ONLY", None), ("other", "POSITIONAL_ONLY", None)], "ATen elementwise multiplication dispatch kernel."),
    ("div", "aten.div", "function", [("self", "POSITIONAL_ONLY", None), ("other", "POSITIONAL_ONLY", None)], "ATen division dispatch kernel."),
    ("matmul", "aten.matmul", "function", [("self", "POSITIONAL_ONLY", None), ("other", "POSITIONAL_ONLY", None)], "ATen matrix multiplication dispatch kernel."),
    ("mm", "aten.mm", "function", [("self", "POSITIONAL_ONLY", None), ("mat2", "POSITIONAL_ONLY", None)], "ATen 2D matrix multiplication dispatch kernel."),
    ("bmm", "aten.bmm", "function", [("self", "POSITIONAL_ONLY", None), ("mat2", "POSITIONAL_ONLY", None)], "ATen batch matrix multiplication dispatch kernel."),
    ("relu", "aten.relu", "function", [("self", "POSITIONAL_ONLY", None)], "ATen rectified linear unit dispatch kernel."),
    ("sigmoid", "aten.sigmoid", "function", [("self", "POSITIONAL_ONLY", None)], "ATen sigmoid activation dispatch kernel."),
    ("tanh", "aten.tanh", "function", [("self", "POSITIONAL_ONLY", None)], "ATen hyperbolic tangent activation dispatch kernel."),
    ("gelu", "aten.gelu", "function", [("self", "POSITIONAL_ONLY", None), ("approximate", "KEYWORD_ONLY", "'none'")], "ATen Gaussian Error Linear Unit activation dispatch kernel."),
    ("softmax", "aten.softmax", "function", [("self", "POSITIONAL_ONLY", None), ("dim", "POSITIONAL_ONLY", None), ("dtype", "KEYWORD_ONLY", "None")], "ATen softmax dispatch kernel."),
    ("log_softmax", "aten.log_softmax", "function", [("self", "POSITIONAL_ONLY", None), ("dim", "POSITIONAL_ONLY", None), ("dtype", "KEYWORD_ONLY", "None")], "ATen log-softmax dispatch kernel."),
    ("sum", "aten.sum", "function", [("self", "POSITIONAL_ONLY", None), ("dim", "KEYWORD_ONLY", "None"), ("keepdim", "KEYWORD_ONLY", "False"), ("dtype", "KEYWORD_ONLY", "None")], "ATen sum reduction dispatch kernel."),
    ("mean", "aten.mean", "function", [("self", "POSITIONAL_ONLY", None), ("dim", "KEYWORD_ONLY", "None"), ("keepdim", "KEYWORD_ONLY", "False"), ("dtype", "KEYWORD_ONLY", "None")], "ATen mean reduction dispatch kernel."),
    ("max", "aten.max", "function", [("self", "POSITIONAL_ONLY", None), ("dim", "KEYWORD_ONLY", "None"), ("keepdim", "KEYWORD_ONLY", "False")], "ATen maximum reduction dispatch kernel."),
    ("min", "aten.min", "function", [("self", "POSITIONAL_ONLY", None), ("dim", "KEYWORD_ONLY", "None"), ("keepdim", "KEYWORD_ONLY", "False")], "ATen minimum reduction dispatch kernel."),
    ("view", "aten.view", "function", [("self", "POSITIONAL_ONLY", None), ("size", "POSITIONAL_ONLY", None)], "ATen tensor view reshape kernel."),
    ("reshape", "aten.reshape", "function", [("self", "POSITIONAL_ONLY", None), ("shape", "POSITIONAL_ONLY", None)], "ATen tensor reshape kernel."),
    ("transpose", "aten.transpose", "function", [("self", "POSITIONAL_ONLY", None), ("dim0", "POSITIONAL_ONLY", None), ("dim1", "POSITIONAL_ONLY", None)], "ATen dimension transpose kernel."),
    ("permute", "aten.permute", "function", [("self", "POSITIONAL_ONLY", None), ("dims", "POSITIONAL_ONLY", None)], "ATen tensor dimension permutation kernel."),
    ("squeeze", "aten.squeeze", "function", [("self", "POSITIONAL_ONLY", None), ("dim", "KEYWORD_ONLY", "None")], "ATen singleton dimension squeeze kernel."),
    ("unsqueeze", "aten.unsqueeze", "function", [("self", "POSITIONAL_ONLY", None), ("dim", "POSITIONAL_ONLY", None)], "ATen dimension unsqueeze kernel."),
    ("clone", "aten.clone", "function", [("self", "POSITIONAL_ONLY", None), ("memory_format", "KEYWORD_ONLY", "None")], "ATen tensor memory clone kernel."),
    ("pow", "aten.pow", "function", [("input", "POSITIONAL_OR_KEYWORD", None), ("exponent", "POSITIONAL_OR_KEYWORD", None)], "ATen pow C++ dispatch kernel."),
    ("neg", "aten.neg", "function", [("input", "POSITIONAL_OR_KEYWORD", None)], "ATen neg C++ dispatch kernel."),
    ("abs", "aten.abs", "function", [("input", "POSITIONAL_OR_KEYWORD", None)], "ATen abs C++ dispatch kernel."),
    ("exp", "aten.exp", "function", [("input", "POSITIONAL_OR_KEYWORD", None)], "ATen exp C++ dispatch kernel."),
    ("log", "aten.log", "function", [("input", "POSITIONAL_OR_KEYWORD", None)], "ATen log C++ dispatch kernel."),
    ("sqrt", "aten.sqrt", "function", [("input", "POSITIONAL_OR_KEYWORD", None)], "ATen sqrt C++ dispatch kernel."),
    ("sin", "aten.sin", "function", [("input", "POSITIONAL_OR_KEYWORD", None)], "ATen sin C++ dispatch kernel."),
    ("cos", "aten.cos", "function", [("input", "POSITIONAL_OR_KEYWORD", None)], "ATen cos C++ dispatch kernel."),
    ("tan", "aten.tan", "function", [("input", "POSITIONAL_OR_KEYWORD", None)], "ATen tan C++ dispatch kernel."),
    ("silu", "aten.silu", "function", [("input", "POSITIONAL_OR_KEYWORD", None)], "ATen silu C++ dispatch kernel."),
    ("addmm", "aten.addmm", "function", [("input", "POSITIONAL_OR_KEYWORD", None), ("mat1", "POSITIONAL_OR_KEYWORD", None), ("mat2", "POSITIONAL_OR_KEYWORD", None), ("beta", "KEYWORD_ONLY", "1"), ("alpha", "KEYWORD_ONLY", "1")], "ATen addmm C++ dispatch kernel."),
    ("linear", "aten.linear", "function", [("input", "POSITIONAL_OR_KEYWORD", None), ("weight", "POSITIONAL_OR_KEYWORD", None), ("bias", "POSITIONAL_OR_KEYWORD", "None")], "ATen linear C++ dispatch kernel."),
    (
        "convolution",
        "aten.convolution",
        "function",
        [
            ("input", "POSITIONAL_OR_KEYWORD", None),
            ("weight", "POSITIONAL_OR_KEYWORD", None),
            ("bias", "POSITIONAL_OR_KEYWORD", None),
            ("stride", "POSITIONAL_OR_KEYWORD", None),
            ("padding", "POSITIONAL_OR_KEYWORD", None),
            ("dilation", "POSITIONAL_OR_KEYWORD", None),
            ("transposed", "POSITIONAL_OR_KEYWORD", None),
            ("output_padding", "POSITIONAL_OR_KEYWORD", None),
            ("groups", "POSITIONAL_OR_KEYWORD", None),
        ],
        "ATen convolution C++ dispatch kernel.",
    ),
    (
        "conv2d",
        "aten.conv2d",
        "function",
        [
            ("input", "POSITIONAL_OR_KEYWORD", None),
            ("weight", "POSITIONAL_OR_KEYWORD", None),
            ("bias", "POSITIONAL_OR_KEYWORD", "None"),
            ("stride", "POSITIONAL_OR_KEYWORD", "[1, 1]"),
            ("padding", "POSITIONAL_OR_KEYWORD", "[0, 0]"),
            ("dilation", "POSITIONAL_OR_KEYWORD", "[1, 1]"),
            ("groups", "POSITIONAL_OR_KEYWORD", "1"),
        ],
        "ATen conv2d C++ dispatch kernel.",
    ),
    (
        "layer_norm",
        "aten.layer_norm",
        "function",
        [("input", "POSITIONAL_OR_KEYWORD", None), ("normalized_shape", "POSITIONAL_OR_KEYWORD", None), ("weight", "POSITIONAL_OR_KEYWORD", "None"), ("bias", "POSITIONAL_OR_KEYWORD", "None"), ("eps", "POSITIONAL_OR_KEYWORD", "1e-05"), ("cudnn_enable", "POSITIONAL_OR_KEYWORD", "True")],
        "ATen layer_norm C++ dispatch kernel.",
    ),
    ("rms_norm", "aten.rms_norm", "function", [("input", "POSITIONAL_OR_KEYWORD", None), ("normalized_shape", "POSITIONAL_OR_KEYWORD", None), ("weight", "POSITIONAL_OR_KEYWORD", "None"), ("eps", "POSITIONAL_OR_KEYWORD", "None")], "ATen rms_norm C++ dispatch kernel."),
    (
        "group_norm",
        "aten.group_norm",
        "function",
        [("input", "POSITIONAL_OR_KEYWORD", None), ("num_groups", "POSITIONAL_OR_KEYWORD", None), ("weight", "POSITIONAL_OR_KEYWORD", "None"), ("bias", "POSITIONAL_OR_KEYWORD", "None"), ("eps", "POSITIONAL_OR_KEYWORD", "1e-05"), ("cudnn_enabled", "POSITIONAL_OR_KEYWORD", "True")],
        "ATen group_norm C++ dispatch kernel.",
    ),
    (
        "batch_norm",
        "aten.batch_norm",
        "function",
        [
            ("input", "POSITIONAL_OR_KEYWORD", None),
            ("weight", "POSITIONAL_OR_KEYWORD", None),
            ("bias", "POSITIONAL_OR_KEYWORD", None),
            ("running_mean", "POSITIONAL_OR_KEYWORD", None),
            ("running_var", "POSITIONAL_OR_KEYWORD", None),
            ("training", "POSITIONAL_OR_KEYWORD", None),
            ("momentum", "POSITIONAL_OR_KEYWORD", None),
            ("eps", "POSITIONAL_OR_KEYWORD", None),
            ("cudnn_enabled", "POSITIONAL_OR_KEYWORD", None),
        ],
        "ATen batch_norm C++ dispatch kernel.",
    ),
    (
        "embedding",
        "aten.embedding",
        "function",
        [("weight", "POSITIONAL_OR_KEYWORD", None), ("indices", "POSITIONAL_OR_KEYWORD", None), ("padding_idx", "POSITIONAL_OR_KEYWORD", "-1"), ("scale_grad_by_freq", "POSITIONAL_OR_KEYWORD", "False"), ("sparse", "POSITIONAL_OR_KEYWORD", "False")],
        "ATen embedding C++ dispatch kernel.",
    ),
    (
        "scaled_dot_product_attention",
        "aten.scaled_dot_product_attention",
        "function",
        [
            ("query", "POSITIONAL_OR_KEYWORD", None),
            ("key", "POSITIONAL_OR_KEYWORD", None),
            ("value", "POSITIONAL_OR_KEYWORD", None),
            ("attn_mask", "POSITIONAL_OR_KEYWORD", "None"),
            ("dropout_p", "POSITIONAL_OR_KEYWORD", "0.0"),
            ("is_causal", "POSITIONAL_OR_KEYWORD", "False"),
            ("scale", "KEYWORD_ONLY", "None"),
            ("enable_gqa", "KEYWORD_ONLY", "False"),
        ],
        "ATen scaled_dot_product_attention C++ dispatch kernel.",
    ),
    ("t", "aten.t", "function", [("input", "POSITIONAL_OR_KEYWORD", None)], "ATen t C++ dispatch kernel."),
    ("flatten", "aten.flatten", "function", [("input", "POSITIONAL_OR_KEYWORD", None), ("start_dim", "POSITIONAL_OR_KEYWORD", "0"), ("end_dim", "POSITIONAL_OR_KEYWORD", "-1")], "ATen flatten C++ dispatch kernel."),
    ("cat", "aten.cat", "function", [("tensors", "POSITIONAL_OR_KEYWORD", None), ("dim", "POSITIONAL_OR_KEYWORD", "0")], "ATen cat C++ dispatch kernel."),
    ("split", "aten.split", "function", [("input", "POSITIONAL_OR_KEYWORD", None), ("split_size", "POSITIONAL_OR_KEYWORD", None), ("dim", "POSITIONAL_OR_KEYWORD", "0")], "ATen split C++ dispatch kernel."),
    ("slice", "aten.slice", "function", [("input", "POSITIONAL_OR_KEYWORD", None), ("dim", "POSITIONAL_OR_KEYWORD", "0"), ("start", "POSITIONAL_OR_KEYWORD", "None"), ("end", "POSITIONAL_OR_KEYWORD", "None"), ("step", "POSITIONAL_OR_KEYWORD", "1")], "ATen slice C++ dispatch kernel."),
    ("select", "aten.select", "function", [("input", "POSITIONAL_OR_KEYWORD", None), ("dim", "POSITIONAL_OR_KEYWORD", None), ("index", "POSITIONAL_OR_KEYWORD", None)], "ATen select C++ dispatch kernel."),
    ("index_select", "aten.index_select", "function", [("input", "POSITIONAL_OR_KEYWORD", None), ("dim", "POSITIONAL_OR_KEYWORD", None), ("index", "POSITIONAL_OR_KEYWORD", None)], "ATen index_select C++ dispatch kernel."),
    ("prod", "aten.prod", "function", [("input", "POSITIONAL_OR_KEYWORD", None), ("dtype", "KEYWORD_ONLY", "None")], "ATen prod C++ dispatch kernel."),
    ("argmax", "aten.argmax", "function", [("input", "POSITIONAL_OR_KEYWORD", None), ("dim", "POSITIONAL_OR_KEYWORD", "None"), ("keepdim", "POSITIONAL_OR_KEYWORD", "False")], "ATen argmax C++ dispatch kernel."),
    ("argmin", "aten.argmin", "function", [("input", "POSITIONAL_OR_KEYWORD", None), ("dim", "POSITIONAL_OR_KEYWORD", "None"), ("keepdim", "POSITIONAL_OR_KEYWORD", "False")], "ATen argmin C++ dispatch kernel."),
    ("zeros", "aten.zeros", "function", [("size", "POSITIONAL_OR_KEYWORD", None), ("names", "KEYWORD_ONLY", None), ("dtype", "KEYWORD_ONLY", "None"), ("layout", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None"), ("pin_memory", "KEYWORD_ONLY", "None")], "ATen zeros C++ dispatch kernel."),
    ("ones", "aten.ones", "function", [("size", "POSITIONAL_OR_KEYWORD", None), ("names", "KEYWORD_ONLY", None), ("dtype", "KEYWORD_ONLY", "None"), ("layout", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None"), ("pin_memory", "KEYWORD_ONLY", "None")], "ATen ones C++ dispatch kernel."),
    (
        "empty",
        "aten.empty",
        "function",
        [("size", "POSITIONAL_OR_KEYWORD", None), ("dtype", "KEYWORD_ONLY", "None"), ("layout", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None"), ("pin_memory", "KEYWORD_ONLY", "None"), ("memory_format", "KEYWORD_ONLY", "None")],
        "ATen empty C++ dispatch kernel.",
    ),
    (
        "full",
        "aten.full",
        "function",
        [("size", "POSITIONAL_OR_KEYWORD", None), ("fill_value", "POSITIONAL_OR_KEYWORD", None), ("names", "KEYWORD_ONLY", None), ("dtype", "KEYWORD_ONLY", "None"), ("layout", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None"), ("pin_memory", "KEYWORD_ONLY", "None")],
        "ATen full C++ dispatch kernel.",
    ),
    ("arange", "aten.arange", "function", [("end", "POSITIONAL_OR_KEYWORD", None), ("dtype", "KEYWORD_ONLY", "None"), ("layout", "KEYWORD_ONLY", "None"), ("device", "KEYWORD_ONLY", "None"), ("pin_memory", "KEYWORD_ONLY", "None")], "ATen arange C++ dispatch kernel."),
    ("where", "aten.where", "function", [("condition", "POSITIONAL_OR_KEYWORD", None), ("input", "POSITIONAL_OR_KEYWORD", None), ("other", "POSITIONAL_OR_KEYWORD", None)], "ATen where C++ dispatch kernel."),
]

CANONICAL_ATEN_OPS: list[CanonicalPythonOpSpec] = build_canonical_python_ops(_ATEN_OP_DEFS)


def _resolve_torch() -> _TorchWithOpsProtocol | types.ModuleType | None:
    """Resolve torch module from import or sys.modules.

    Returns:
        The torch module, or None if unavailable.
    """
    try:
        import torch

        torch_mod: types.ModuleType = torch
        return torch_mod
    except Exception:
        return None


def _get_torch_ops_aten() -> types.ModuleType | None:
    """Safely import PyTorch torch.ops.aten namespace.

    Returns:
        The torch.ops.aten namespace object or None if not available.
    """
    try:
        mod = _resolve_torch()
        if isinstance(mod, _TorchWithOpsProtocol):
            return mod.ops.aten
    except Exception as exc:
        del exc
    return None


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
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
        return [
            build_canonical_python_ref(
                op,
                default_doc_prefix="ATen",
                environment_tags=("aten", "dispatch", "c_extension"),
                domain_metadata={"domain": "aten"},
            )
            for op in CANONICAL_ATEN_OPS
        ]

    results: list[GhostRef] = []
    seen: set[str] = set()
    aten_names = {name for name, _ in inspect.getmembers(aten_ops)}

    for op in CANONICAL_ATEN_OPS:
        name = op.name
        if name in aten_names:
            schemas = get_aten_op_schema(name)
            if schemas:
                primary = _coerce_aten_schema(schemas[0])
                params = [
                    GhostParam(
                        name=p.name,
                        kind=ParameterKind[p.kind],
                        default=p.default,
                    )
                    for p in primary.params
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

    for op in CANONICAL_ATEN_OPS:
        if op.name not in seen:
            results.append(
                build_canonical_python_ref(
                    op,
                    default_doc_prefix="ATen",
                    environment_tags=("aten", "dispatch", "c_extension"),
                    domain_metadata={"domain": "aten"},
                )
            )

    return results
