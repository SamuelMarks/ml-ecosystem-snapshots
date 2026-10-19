"""Intel Data Parallel Extension for NumPy (DPNP) Snapshot Extractor.

Extracts array primitives, SYCL Unified Shared Memory (USM) allocators, queue bindings,
and accelerated math kernels from Intel DPNP.
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


CANONICAL_DPNP_OPS: list[CanonicalPythonOpSpec] = pydantic.TypeAdapter(
    list[CanonicalPythonOpSpec]
).validate_python(
    [
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
)


def _get_dpnp() -> types.ModuleType | None:
    """Lazily import DPNP module if installed.

    Returns:
        The dpnp module or None if not installed.
    """
    try:
        import dpnp

        mod: types.ModuleType = dpnp
        return mod
    except Exception:
        return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
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

    from ml_ecosystem_snapshots.frameworks._base import collect_from_canonical_or_live

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
    return collect_from_canonical_or_live(
        live_module=_get_dpnp(),
        module_prefix="dpnp",
        target_names=target_names,
        canonical_ops=CANONICAL_DPNP_OPS,
        default_doc_prefix="DPNP",
        environment_tags=("dpnp", "sycl", "oneapi"),
        inspector_fn=GhostInspector.inspect,
    )
