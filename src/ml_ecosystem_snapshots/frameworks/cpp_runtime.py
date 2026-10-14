"""C++17 and PyBind11 Native Tensor Runtime Snapshot Extractor.

Formalizes standard C++17 STL algorithms, mathematical intrinsics (<cmath>),
and PyBind11/LibTorch tensor buffer access functions.
"""

from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    OperandDirection,
    SemanticTier,
)


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

    from ml_ecosystem_snapshots.frameworks._base import build_ssa_operation_refs

    ops = _load_cpp_runtime_ops()
    return build_ssa_operation_refs(
        ops=ops,
        api_prefix="cpp",
        name_key="name",
        operand_annotation_fn=lambda op, inp: (
            "float" if inp in ("v", "lo", "hi", "x", "y", "z", "arg") else "auto"
        ),
        operand_description_fn=lambda op, inp: f"C++ operand {inp}",
        operand_direction_fn=lambda op, inp: (
            OperandDirection.WRITE
            if op["name"] in ("from_blob", "data_ptr") and inp == "data"
            else OperandDirection.READ
        ),
        attr_description_prefix="C++ attribute",
        result_type="auto",
        empty_return_type="void",
        provenance_fn=lambda op: {
            "source_type": "cpp_spec",
            "upstream_version": "17",
            "category": op.get("category", "runtime"),
        },
        default_doc_fn=lambda op_name: f"C++ {op_name} operation.",
        environment_tags=("cpp", "cpp17", "pybind11"),
    )
