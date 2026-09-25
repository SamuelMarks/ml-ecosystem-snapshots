"""Apache Arrow Compute Kernel and Transformation Snapshot Extractor.

Extracts vectorized compute functions, scalar and aggregate kernels, and option models
from pyarrow.compute.
"""

from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)
from ..models import GhostInspector, GhostPythonRef


CANONICAL_PYARROW_COMPUTE_OPS: List[Dict[str, Any]] = [
    {
        "name": "add",
        "api_path": "pyarrow.compute.add",
        "kind": "function",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "y", "kind": "POSITIONAL_ONLY"},
            {"name": "memory_pool", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Add the arguments element-wise.",
    },
    {
        "name": "subtract",
        "api_path": "pyarrow.compute.subtract",
        "kind": "function",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "y", "kind": "POSITIONAL_ONLY"},
            {"name": "memory_pool", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Subtract the arguments element-wise.",
    },
    {
        "name": "multiply",
        "api_path": "pyarrow.compute.multiply",
        "kind": "function",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "y", "kind": "POSITIONAL_ONLY"},
            {"name": "memory_pool", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Multiply the arguments element-wise.",
    },
    {
        "name": "divide",
        "api_path": "pyarrow.compute.divide",
        "kind": "function",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "y", "kind": "POSITIONAL_ONLY"},
            {"name": "memory_pool", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Divide the arguments element-wise.",
    },
    {
        "name": "sum",
        "api_path": "pyarrow.compute.sum",
        "kind": "function",
        "params": [
            {"name": "array", "kind": "POSITIONAL_ONLY"},
            {"name": "options", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "memory_pool", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Sum the values in a numerical array or chunked array.",
    },
    {
        "name": "mean",
        "api_path": "pyarrow.compute.mean",
        "kind": "function",
        "params": [
            {"name": "array", "kind": "POSITIONAL_ONLY"},
            {"name": "options", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "memory_pool", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Compute the mean of a numerical array or chunked array.",
    },
    {
        "name": "filter",
        "api_path": "pyarrow.compute.filter",
        "kind": "function",
        "params": [
            {"name": "data", "kind": "POSITIONAL_ONLY"},
            {"name": "mask", "kind": "POSITIONAL_ONLY"},
            {
                "name": "null_selection_behavior",
                "kind": "KEYWORD_ONLY",
                "default": "'drop'",
            },
            {"name": "memory_pool", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Filter with a boolean selection mask.",
    },
    {
        "name": "take",
        "api_path": "pyarrow.compute.take",
        "kind": "function",
        "params": [
            {"name": "data", "kind": "POSITIONAL_ONLY"},
            {"name": "indices", "kind": "POSITIONAL_ONLY"},
            {"name": "boundscheck", "kind": "KEYWORD_ONLY", "default": "True"},
            {"name": "memory_pool", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Select elements from an array by index array.",
    },
    {
        "name": "cast",
        "api_path": "pyarrow.compute.cast",
        "kind": "function",
        "params": [
            {"name": "target", "kind": "POSITIONAL_ONLY"},
            {"name": "target_type", "kind": "POSITIONAL_ONLY", "default": "None"},
            {"name": "safe", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "options", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "memory_pool", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Cast array or chunked array values to another data type.",
    },
]


def _get_pyarrow_compute() -> Any:
    """Lazily load pyarrow.compute module to avoid import overhead.

    Returns:
        The pyarrow.compute module or None if not installed.
    """
    try:
        import pyarrow.compute as pc

        return pc
    except (ImportError, Exception):
        return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect Arrow Compute vectorized functions and execution options.

    Args:
        category: Target SemanticTier enum value.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRef objects representing Arrow Compute functions.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.UTIL,
    ):
        return []

    pc = _get_pyarrow_compute()
    if pc is None:
        refs: List[GhostRef] = []
        for op in CANONICAL_PYARROW_COMPUTE_OPS:
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
                    kind=op.get("kind", "function"),
                    params=params,
                    docstring=op.get("docstring", f"PyArrow compute {op['name']} API."),
                    environment_tags=["pyarrow", "arrow_compute"],
                    domain_metadata={},
                )
            )
        return refs

    results: List[GhostRef] = []
    target_names = [
        "add",
        "subtract",
        "multiply",
        "divide",
        "abs",
        "sqrt",
        "sin",
        "cos",
        "exp",
        "ln",
        "log10",
        "log2",
        "equal",
        "not_equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "sum",
        "mean",
        "min",
        "max",
        "stddev",
        "variance",
        "filter",
        "take",
        "cast",
        "sort_indices",
    ]

    for name in target_names:
        if hasattr(pc, name):
            obj = getattr(pc, name)
            try:
                ref = GhostInspector.inspect(
                    obj, f"pyarrow.compute.{name}", is_public=True
                )
                ref.environment_tags = list(ref.environment_tags or ()) + [
                    "pyarrow",
                    "arrow_compute",
                ]
                results.append(ref)
            except Exception:
                pass

    return results
