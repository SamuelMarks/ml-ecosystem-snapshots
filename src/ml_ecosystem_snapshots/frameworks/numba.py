"""Numba JIT Kernel and Array Extractor.

Extracts function signatures, decorators (@njit, @vectorize, @guvectorize),
parallel primitives (prange), and typing primitives from Numba.
"""

from __future__ import annotations

import types

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    SemanticTier,
)
import pydantic

from ml_ecosystem_snapshots.frameworks._base import CanonicalPythonOpSpec
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector


CANONICAL_NUMBA_OPS: list[CanonicalPythonOpSpec] = pydantic.TypeAdapter(
    list[CanonicalPythonOpSpec]
).validate_python(
    [
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
)


def _get_numba() -> types.ModuleType | None:
    """Lazily load Numba module to prevent unnecessary import cost.

    Returns:
        The numba module or None if not installed.
    """
    try:
        import numba

        mod: types.ModuleType = numba
        return mod
    except Exception:
        return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
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

    from ml_ecosystem_snapshots.frameworks._base import collect_from_canonical_or_live

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
    return collect_from_canonical_or_live(
        live_module=_get_numba(),
        module_prefix="numba",
        target_names=target_names,
        canonical_ops=CANONICAL_NUMBA_OPS,
        default_doc_prefix="Numba",
        environment_tags=("numba", "jit"),
        inspector_fn=GhostInspector.inspect,
    )
