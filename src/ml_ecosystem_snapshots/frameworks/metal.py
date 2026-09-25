"""Apple Metal Shading Language (MSL) Snapshot Extractor.

Extracts built-in compute functions, SIMD-group matrix operations, atomic intrinsics,
and threadgroup synchronization operations from the Apple Metal Shading Language specification.
"""

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


CANONICAL_METAL_OPS: List[Dict[str, Any]] = [
    # SIMD-group Matrix Operations (simdgroup_matrix)
    {
        "name": "simdgroup_multiply_accumulate",
        "category": "matrix",
        "address_space": "thread",
        "inputs": ["dest", "a", "b", "c"],
        "outputs": ["result"],
        "attributes": [
            {"name": "m", "type": "int", "default": "8"},
            {"name": "n", "type": "int", "default": "8"},
            {"name": "k", "type": "int", "default": "8"},
        ],
        "description": "Multiply matrix A and B, accumulate with C, and write to destination.",
    },
    {
        "name": "simdgroup_load",
        "category": "matrix",
        "address_space": "device",
        "inputs": ["dest", "ptr", "stride"],
        "outputs": ["matrix"],
        "attributes": [
            {"name": "transpose", "type": "bool", "default": "false"},
        ],
        "description": "Load matrix tiles from device or threadgroup memory into SIMD registers.",
    },
    {
        "name": "simdgroup_store",
        "category": "matrix",
        "address_space": "device",
        "inputs": ["src", "ptr", "stride"],
        "outputs": [],
        "attributes": [
            {"name": "transpose", "type": "bool", "default": "false"},
        ],
        "description": "Store matrix tiles from SIMD registers to device or threadgroup memory.",
    },
    # Synchronization and Barriers
    {
        "name": "threadgroup_barrier",
        "category": "synchronization",
        "address_space": "threadgroup",
        "inputs": ["mem_flags"],
        "outputs": [],
        "attributes": [
            {"name": "scope", "type": "str", "default": "mem_flags::mem_threadgroup"}
        ],
        "description": "Ensure all threads in a threadgroup reach this barrier and synchronize memory writes.",
    },
    {
        "name": "simdgroup_barrier",
        "category": "synchronization",
        "address_space": "threadgroup",
        "inputs": ["mem_flags"],
        "outputs": [],
        "attributes": [
            {"name": "scope", "type": "str", "default": "mem_flags::mem_none"}
        ],
        "description": "Synchronize execution and memory fences across threads within a SIMDgroup.",
    },
    # SIMD-group Collective Communications
    {
        "name": "simd_shuffle",
        "category": "collective",
        "address_space": "thread",
        "inputs": ["data", "lane"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Exchange data between SIMD lanes across the SIMD group.",
    },
    {
        "name": "simd_sum",
        "category": "collective",
        "address_space": "thread",
        "inputs": ["data"],
        "outputs": ["sum"],
        "attributes": [],
        "description": "Calculate prefix or total sum of scalar data across all active threads in the SIMD group.",
    },
    # Atomics
    {
        "name": "atomic_fetch_add_explicit",
        "category": "atomic",
        "address_space": "device",
        "inputs": ["object", "operand", "memory_order"],
        "outputs": ["original"],
        "attributes": [
            {"name": "order", "type": "str", "default": "memory_order_relaxed"}
        ],
        "description": "Atomically add operand to object and return original value.",
    },
    {
        "name": "atomic_fetch_max_explicit",
        "category": "atomic",
        "address_space": "device",
        "inputs": ["object", "operand", "memory_order"],
        "outputs": ["original"],
        "attributes": [
            {"name": "order", "type": "str", "default": "memory_order_relaxed"}
        ],
        "description": "Atomically compute maximum of operand and object and store.",
    },
    # Core Math Intrinsics
    {
        "name": "fma",
        "category": "math",
        "address_space": "thread",
        "inputs": ["a", "b", "c"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Fused multiply-add: (a * b) + c with a single rounding step.",
    },
    {
        "name": "rsqrt",
        "category": "math",
        "address_space": "thread",
        "inputs": ["x"],
        "outputs": ["result"],
        "attributes": [{"name": "fast_math", "type": "bool", "default": "true"}],
        "description": "Reciprocal square root: 1 / sqrt(x).",
    },
    {
        "name": "dot",
        "category": "math",
        "address_space": "thread",
        "inputs": ["x", "y"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Compute dot product of two vector types.",
    },
]


def _load_metal_ops() -> List[Dict[str, Any]]:
    """Load Metal Shading Language operations catalog.

    Returns:
        List of MSL operation specification dictionaries.
    """
    return CANONICAL_METAL_OPS


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect Apple Metal Shading Language (MSL) operation definitions.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostOperationRef objects representing MSL compute operations.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.NEURAL,
        SemanticTier.UTIL,
    ):
        return []

    ops = _load_metal_ops()
    refs: List[GhostRef] = []

    for op in ops:
        op_name = op["name"]
        addr_space = op.get("address_space", "thread")
        api_path = f"metal.{op_name}"
        params: List[GhostParam] = []

        # SSA operands
        for inp in op.get("inputs", []):
            direction = (
                OperandDirection.WRITE
                if "store" in op_name and inp in ("ptr", "dest")
                else OperandDirection.READ
            )
            params.append(
                ExtendedGhostParam(
                    name=inp,
                    kind=ParameterKind.POSITIONAL_ONLY,
                    annotation=f"{addr_space} T*",
                    standardized_name=inp,
                    description=f"MSL operand {inp} in {addr_space} address space",
                    direction=direction,
                    role=IRParameterRole.OPERAND,
                )
            )

        # Attributes
        attrs_meta: Dict[str, Any] = {}
        for attr in op.get("attributes", []):
            attr_name = attr.get("name", "attr")
            attr_type = attr.get("type", "Any")
            attr_def = attr.get("default")
            params.append(
                ExtendedGhostParam(
                    name=attr_name,
                    kind=ParameterKind.KEYWORD_ONLY,
                    annotation=attr_type,
                    default=str(attr_def) if attr_def is not None else None,
                    standardized_name=attr_name,
                    description=f"MSL attribute {attr_name}",
                    role=IRParameterRole.ATTRIBUTE,
                )
            )
            attrs_meta[attr_name] = attr

        returns: List[GhostResult] = []
        for out in op.get("outputs", []):
            returns.append(GhostResult(name=out, type="T"))

        returns_type = "T" if returns else "None"

        provenance = {
            "source_type": "metal_spec",
            "upstream_version": "3.2",
            "address_space": addr_space,
            "category": op.get("category", "compute"),
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
                docstring=op.get("description", f"Apple Metal {op_name} operation."),
                environment_tags=["metal", "msl"],
                domain_metadata=provenance,
                attributes=attrs_meta if attrs_meta else None,
            )
        )

    return refs
