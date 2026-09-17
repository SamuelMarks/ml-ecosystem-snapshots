"""WebGPU WGSL Dialect Snapshot Extractor.

Extracts built-in functions, compute shader intrinsics, and operations from the WebGPU WGSL specification.
"""

import json
from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostOperationRef,
    GhostParam,
    GhostRef,
    GhostResult,
    IRParameterRole,
    OperandDirection,
    ParameterKind,
    SemanticTier,
)
from ..models import ExtendedGhostParam


CANONICAL_WGSL_OPS: List[Dict[str, Any]] = [
    {
        "name": "storageStore",
        "domain": "wgsl",
        "inputs": ["buffer", "index", "value"],
        "outputs": [],
        "attributes": [
            {
                "name": "address_space",
                "type": "str",
                "required": False,
                "default": "storage, read_write",
            }
        ],
        "description": "Store a value into a storage buffer at the given index.",
    },
    {
        "name": "storageLoad",
        "domain": "wgsl",
        "inputs": ["buffer", "index"],
        "outputs": ["value"],
        "attributes": [
            {
                "name": "address_space",
                "type": "str",
                "required": False,
                "default": "storage, read",
            }
        ],
        "description": "Load a value from a storage buffer at the given index.",
    },
    {
        "name": "workgroupBarrier",
        "domain": "wgsl",
        "inputs": [],
        "outputs": [],
        "attributes": [],
        "description": "Synchronize memory operations among invocations in a workgroup.",
    },
    {
        "name": "storageBarrier",
        "domain": "wgsl",
        "inputs": [],
        "outputs": [],
        "attributes": [],
        "description": "Synchronize storage buffer operations across invocations.",
    },
    {
        "name": "atomicAdd",
        "domain": "wgsl",
        "inputs": ["atomic_ptr", "value"],
        "outputs": ["old_value"],
        "attributes": [],
        "description": "Atomically add a value to an atomic variable and return previous value.",
    },
    {
        "name": "atomicSub",
        "domain": "wgsl",
        "inputs": ["atomic_ptr", "value"],
        "outputs": ["old_value"],
        "attributes": [],
        "description": "Atomically subtract a value from an atomic variable and return previous value.",
    },
    {
        "name": "atomicExchange",
        "domain": "wgsl",
        "inputs": ["atomic_ptr", "value"],
        "outputs": ["old_value"],
        "attributes": [],
        "description": "Atomically store a value into an atomic variable and return previous value.",
    },
]


def _load_wgsl_ops() -> List[Dict[str, Any]]:
    """Load WGSL operation definitions from ml_switcheroo_ir schema or fallback registry.

    Returns:
        List of WGSL operator schema dictionaries.
    """
    try:
        import importlib.resources as pkg_resources
        import ml_switcheroo_ir.schema as ir_schema

        fpath = pkg_resources.files(ir_schema).joinpath("wgsl_ops.json")
        with fpath.open("r", encoding="utf-8") as f:
            data = json.load(f)
            ops: List[Dict[str, Any]] = list(data.get("ops", []))
            if ops:
                return ops
    except Exception:
        pass
    return CANONICAL_WGSL_OPS


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect WGSL operations for the specified semantic tier.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include non-public operations.

    Returns:
        List of GhostOperationRef objects representing WGSL shader operations.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.NEURAL,
        SemanticTier.UTIL,
    ):
        return []

    ops = _load_wgsl_ops()
    refs: List[GhostRef] = []

    for op in ops:
        op_name = op["name"]
        api_path = f"wgsl.{op_name}"
        params: List[GhostParam] = []

        # SSA inputs
        for inp in op.get("inputs", []):
            direction = (
                OperandDirection.WRITE
                if "store" in op_name.lower() and inp == "value"
                else OperandDirection.READ
            )
            params.append(
                ExtendedGhostParam(
                    name=inp,
                    kind=ParameterKind.POSITIONAL_ONLY,
                    annotation="Value",
                    standardized_name=inp,
                    description=f"WGSL operand {inp}",
                    direction=direction,
                    role=IRParameterRole.OPERAND,
                )
            )

        # Attributes
        attrs_meta: Dict[str, Any] = {}
        for attr in op.get("attributes", []):
            attr_name = attr.get("name", "attr")
            attr_type = attr.get("type", "str")
            attr_def = attr.get("default")
            params.append(
                ExtendedGhostParam(
                    name=attr_name,
                    kind=ParameterKind.KEYWORD_ONLY,
                    annotation=attr_type,
                    default=str(attr_def) if attr_def is not None else None,
                    standardized_name=attr_name,
                    description=f"WGSL attribute {attr_name}",
                    role=IRParameterRole.ATTRIBUTE,
                )
            )
            attrs_meta[attr_name] = attr

        returns: List[GhostResult] = []
        for out in op.get("outputs", []):
            returns.append(GhostResult(name=out, type="Value"))

        returns_type = "Value" if returns else "None"

        provenance = {
            "source_type": "wgsl_spec",
            "upstream_version": "draft-2024",
            "domain": op.get("domain", "wgsl"),
        }

        refs.append(
            GhostOperationRef(
                name=op_name,
                api_path=api_path,
                kind="function",
                params=params,
                operands=[
                    p
                    for p in params
                    if getattr(p, "role", None) == IRParameterRole.OPERAND
                ],
                returns=returns,
                returns_type=returns_type,
                docstring=op.get("description", f"WebGPU WGSL {op_name} operation."),
                environment_tags=["webgpu"],
                domain_metadata=provenance,
                attributes=attrs_meta if attrs_meta else None,
            )
        )

    return refs
