"""Intel Data Parallel Extension for NumPy (DPNP) Snapshot Extractor.

Extracts array primitives, SYCL Unified Shared Memory (USM) allocators, queue bindings,
and accelerated math kernels from Intel DPNP.
"""

from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)
from ..models import GhostInspector, GhostPythonRef


CANONICAL_DPNP_OPS: List[Dict[str, Any]] = [
    {
        "name": "array",
        "api_path": "dpnp.array",
        "kind": "function",
        "params": [
            {"name": "obj", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dtype", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "copy", "kind": "POSITIONAL_OR_KEYWORD", "default": "True"},
            {"name": "order", "kind": "POSITIONAL_OR_KEYWORD", "default": "'K'"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "usm_type", "kind": "KEYWORD_ONLY", "default": "'device'"},
            {"name": "sycl_queue", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Create a DPNP ndarray on target SYCL device using specified USM allocation type.",
    },
    {
        "name": "empty",
        "api_path": "dpnp.empty",
        "kind": "function",
        "params": [
            {"name": "shape", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dtype", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "order", "kind": "POSITIONAL_OR_KEYWORD", "default": "'C'"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "usm_type", "kind": "KEYWORD_ONLY", "default": "'device'"},
            {"name": "sycl_queue", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Return a new array of given shape and type, without initializing entries.",
    },
    {
        "name": "zeros",
        "api_path": "dpnp.zeros",
        "kind": "function",
        "params": [
            {"name": "shape", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dtype", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "order", "kind": "POSITIONAL_OR_KEYWORD", "default": "'C'"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "usm_type", "kind": "KEYWORD_ONLY", "default": "'device'"},
            {"name": "sycl_queue", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Return a new array of given shape and type, filled with zeros on the SYCL device.",
    },
    {
        "name": "ones",
        "api_path": "dpnp.ones",
        "kind": "function",
        "params": [
            {"name": "shape", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "dtype", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "order", "kind": "POSITIONAL_OR_KEYWORD", "default": "'C'"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "usm_type", "kind": "KEYWORD_ONLY", "default": "'device'"},
            {"name": "sycl_queue", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Return a new array of given shape and type, filled with ones on the SYCL device.",
    },
    {
        "name": "matmul",
        "api_path": "dpnp.matmul",
        "kind": "function",
        "params": [
            {"name": "x1", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "x2", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "out", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "casting", "kind": "KEYWORD_ONLY", "default": "'same_kind'"},
            {"name": "order", "kind": "KEYWORD_ONLY", "default": "'K'"},
        ],
        "docstring": "Matrix product of two arrays offloaded via Intel oneMKL / SYCL.",
    },
    {
        "name": "dot",
        "api_path": "dpnp.dot",
        "kind": "function",
        "params": [
            {"name": "a", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "b", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "out", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
        ],
        "docstring": "Dot product of two arrays using Intel oneMKL gemm/gemv kernels.",
    },
    {
        "name": "asnumpy",
        "api_path": "dpnp.asnumpy",
        "kind": "function",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Copy DPNP device array back into host NumPy ndarray.",
    },
]


def _get_dpnp() -> Any:
    """Lazily import DPNP module if installed.

    Returns:
        The dpnp module or None if not installed.
    """
    try:
        import dpnp

        return dpnp
    except (ImportError, Exception):
        return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect DPNP array and math operation definitions.

    Args:
        category: Target SemanticTier enum value.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRef objects representing DPNP functions.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.UTIL,
    ):
        return []

    dp = _get_dpnp()
    if dp is None:
        refs: List[GhostRef] = []
        for op in CANONICAL_DPNP_OPS:
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
                    docstring=op.get("docstring", f"DPNP {op['name']} API."),
                    environment_tags=["dpnp", "sycl", "oneapi"],
                    domain_metadata={},
                )
            )
        return refs

    results: List[GhostRef] = []
    target_names = [
        "array",
        "empty",
        "zeros",
        "ones",
        "matmul",
        "dot",
        "asnumpy",
        "sum",
        "mean",
        "std",
        "reshape",
    ]

    for name in target_names:
        if hasattr(dp, name):
            obj = getattr(dp, name)
            try:
                ref = GhostInspector.inspect(obj, f"dpnp.{name}", is_public=True)
                ref.environment_tags = list(ref.environment_tags or ()) + [
                    "dpnp",
                    "sycl",
                    "oneapi",
                ]
                results.append(ref)
            except Exception:
                pass

    return results
