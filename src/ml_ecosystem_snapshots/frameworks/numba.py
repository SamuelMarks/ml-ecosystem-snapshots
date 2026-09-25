"""Numba JIT Kernel and Array Extractor.

Extracts function signatures, decorators (@njit, @vectorize, @guvectorize),
parallel primitives (prange), and typing primitives from Numba.
"""

from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)
from ..models import GhostInspector, GhostPythonRef


CANONICAL_NUMBA_OPS: List[Dict[str, Any]] = [
    {
        "name": "njit",
        "api_path": "numba.njit",
        "kind": "function",
        "params": [
            {"name": "signature_or_function", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "parallel", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "fastmath", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "boundscheck", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "cache", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "nogil", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "inline", "kind": "KEYWORD_ONLY", "default": "'never'"},
        ],
        "docstring": "Compile the decorated function in nopython mode without Python runtime overhead.",
    },
    {
        "name": "jit",
        "api_path": "numba.jit",
        "kind": "function",
        "params": [
            {"name": "signature_or_function", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "nopython", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "parallel", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "fastmath", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "forceobj", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Compile Python code into native machine instructions using LLVM.",
    },
    {
        "name": "vectorize",
        "api_path": "numba.vectorize",
        "kind": "function",
        "params": [
            {"name": "signatures", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "target", "kind": "KEYWORD_ONLY", "default": "'cpu'"},
            {"name": "nopython", "kind": "KEYWORD_ONLY", "default": "True"},
            {"name": "fastmath", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Build NumPy universal functions (ufunc) from scalar Python functions.",
    },
    {
        "name": "guvectorize",
        "api_path": "numba.guvectorize",
        "kind": "function",
        "params": [
            {"name": "signatures", "kind": "POSITIONAL_ONLY"},
            {"name": "layout", "kind": "POSITIONAL_ONLY"},
            {"name": "target", "kind": "KEYWORD_ONLY", "default": "'cpu'"},
            {"name": "nopython", "kind": "KEYWORD_ONLY", "default": "True"},
        ],
        "docstring": "Build generalized universal functions (gufunc) operating on multi-dimensional sub-arrays.",
    },
    {
        "name": "prange",
        "api_path": "numba.prange",
        "kind": "function",
        "params": [
            {"name": "start_or_stop", "kind": "POSITIONAL_ONLY"},
            {"name": "stop", "kind": "POSITIONAL_ONLY", "default": "None"},
            {"name": "step", "kind": "POSITIONAL_ONLY", "default": "None"},
        ],
        "docstring": "Explicit parallel loop constructor for use inside @njit(parallel=True) kernels.",
    },
    {
        "name": "stencil",
        "api_path": "numba.stencil",
        "kind": "function",
        "params": [
            {"name": "neighborhood", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "cval", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Decorator generating stencil update kernels over multi-dimensional arrays.",
    },
    {
        "name": "typeof",
        "api_path": "numba.typeof",
        "kind": "function",
        "params": [{"name": "val", "kind": "POSITIONAL_ONLY"}],
        "docstring": "Determine the Numba type representation of the given Python runtime object.",
    },
]


def _get_numba() -> Any:
    """Lazily load Numba module to prevent unnecessary import cost.

    Returns:
        The numba module or None if not installed.
    """
    try:
        import numba

        return numba
    except (ImportError, Exception):
        return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect Numba JIT and array primitives for a given semantic tier.

    Args:
        category: The target SemanticTier enum value.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of gathered GhostRef objects.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.UTIL,
    ):
        return []

    nb = _get_numba()
    if nb is None:
        refs: List[GhostRef] = []
        for op in CANONICAL_NUMBA_OPS:
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
                    docstring=op.get("docstring", f"Numba {op['name']} API."),
                    environment_tags=["numba", "jit"],
                    domain_metadata={},
                )
            )
        return refs

    results: List[GhostRef] = []
    target_names = [
        "jit",
        "njit",
        "vectorize",
        "guvectorize",
        "prange",
        "stencil",
        "cfunc",
        "typeof",
        "from_dtype",
    ]

    for name in target_names:
        if hasattr(nb, name):
            obj = getattr(nb, name)
            try:
                ref = GhostInspector.inspect(obj, f"numba.{name}", is_public=True)
                ref.environment_tags = list(ref.environment_tags or ()) + [
                    "numba",
                    "jit",
                ]
                results.append(ref)
            except Exception:
                pass

    return results
