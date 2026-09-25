"""C++17 and PyBind11 Native Tensor Runtime Snapshot Extractor.

Formalizes standard C++17 STL algorithms, mathematical intrinsics (<cmath>),
and PyBind11/LibTorch tensor buffer access functions.
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


CANONICAL_CPP_RUNTIME_OPS: List[Dict[str, Any]] = [
    # C++ STL <cmath> / <algorithm>
    {
        "name": "clamp",
        "category": "stl_algorithm",
        "inputs": ["v", "lo", "hi"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Clamps value v between lower limit lo and upper limit hi.",
    },
    {
        "name": "fma",
        "category": "stl_cmath",
        "inputs": ["x", "y", "z"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Computes (x * y) + z as a single ternary operation.",
    },
    {
        "name": "sqrt",
        "category": "stl_cmath",
        "inputs": ["arg"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Computes square root of arg.",
    },
    {
        "name": "exp",
        "category": "stl_cmath",
        "inputs": ["arg"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Computes e raised to the given power arg.",
    },
    {
        "name": "log",
        "category": "stl_cmath",
        "inputs": ["arg"],
        "outputs": ["result"],
        "attributes": [],
        "description": "Computes natural logarithm of arg.",
    },
    # PyBind11 & LibTorch Interop
    {
        "name": "data_ptr",
        "category": "interop",
        "inputs": ["tensor"],
        "outputs": ["ptr"],
        "attributes": [
            {"name": "dtype", "type": "str", "default": "'float'"},
        ],
        "description": "Retrieve raw contiguous memory pointer from PyTorch / LibTorch tensor.",
    },
    {
        "name": "from_blob",
        "category": "interop",
        "inputs": ["data", "sizes", "options"],
        "outputs": ["tensor"],
        "attributes": [],
        "description": "Construct an unmanaged PyTorch tensor view wrapping a raw memory pointer.",
    },
    {
        "name": "make_tuple",
        "category": "stl_utility",
        "inputs": ["args"],
        "outputs": ["tuple"],
        "attributes": [],
        "description": "Creates a std::tuple object deducing target element types from arguments.",
    },
]


def _load_cpp_runtime_ops() -> List[Dict[str, Any]]:
    """Load canonical C++17 and PyBind11 runtime operation specifications.

    Returns:
        List of C++ runtime operation dictionaries.
    """
    return CANONICAL_CPP_RUNTIME_OPS


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect C++17 and PyBind11 native runtime operations.

    Args:
        category: Target SemanticTier enum value.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostOperationRef objects representing C++ runtime functions.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.NEURAL,
        SemanticTier.UTIL,
    ):
        return []

    ops = _load_cpp_runtime_ops()
    refs: List[GhostRef] = []

    for op in ops:
        op_name = op["name"]
        api_path = f"cpp.{op_name}"
        params: List[GhostParam] = []

        # SSA operands
        for inp in op.get("inputs", []):
            direction = (
                OperandDirection.WRITE
                if op_name in ("from_blob", "data_ptr") and inp == "data"
                else OperandDirection.READ
            )
            params.append(
                ExtendedGhostParam(
                    name=inp,
                    kind=ParameterKind.POSITIONAL_ONLY,
                    annotation="float"
                    if inp in ("v", "lo", "hi", "x", "y", "z", "arg")
                    else "auto",
                    standardized_name=inp,
                    description=f"C++ operand {inp}",
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
                    description=f"C++ attribute {attr_name}",
                    role=IRParameterRole.ATTRIBUTE,
                )
            )
            attrs_meta[attr_name] = attr

        returns: List[GhostResult] = []
        for out in op.get("outputs", []):
            returns.append(GhostResult(name=out, type="auto"))

        returns_type = "auto" if returns else "void"

        provenance = {
            "source_type": "cpp_spec",
            "upstream_version": "17",
            "category": op.get("category", "runtime"),
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
                docstring=op.get("description", f"C++ {op_name} operation."),
                environment_tags=["cpp", "cpp17", "pybind11"],
                domain_metadata=provenance,
                attributes=attrs_meta if attrs_meta else None,
            )
        )

    return refs
