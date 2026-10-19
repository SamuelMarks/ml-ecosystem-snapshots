"""Bohrium Heterogeneous Runtime and Array Snapshot Extractor.

Extracts array constructors, JIT kernel fusion decorators, synchronization primitives (flush),
and accelerated linear algebra routines from the Bohrium runtime engine.
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


CANONICAL_BOHRIUM_OPS: list[CanonicalPythonOpSpec] = pydantic.TypeAdapter(
    list[CanonicalPythonOpSpec]
).validate_python(
    [
        {
            "name": "array",
            "api_path": "bohrium.array",
            "kind": "function",
            "params": [
                {"name": "object", "kind": "POSITIONAL_OR_KEYWORD"},
                {"name": "dtype", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
                {"name": "copy", "kind": "POSITIONAL_OR_KEYWORD", "default": "True"},
                {"name": "order", "kind": "POSITIONAL_OR_KEYWORD", "default": "'K'"},
                {"name": "subok", "kind": "POSITIONAL_OR_KEYWORD", "default": "False"},
                {"name": "ndmin", "kind": "POSITIONAL_OR_KEYWORD", "default": "0"},
                {"name": "bh_force", "kind": "KEYWORD_ONLY", "default": "True"},
            ],
            "docstring": "Create a Bohrium-managed array allocated across target compute runtime.",
        },
        {
            "name": "empty",
            "api_path": "bohrium.empty",
            "kind": "function",
            "params": [
                {"name": "shape", "kind": "POSITIONAL_OR_KEYWORD"},
                {"name": "dtype", "kind": "POSITIONAL_OR_KEYWORD", "default": "float"},
                {"name": "order", "kind": "POSITIONAL_OR_KEYWORD", "default": "'C'"},
                {"name": "bh_force", "kind": "KEYWORD_ONLY", "default": "True"},
            ],
            "docstring": "Return a new Bohrium array of given shape and type, without initializing entries.",
        },
        {
            "name": "zeros",
            "api_path": "bohrium.zeros",
            "kind": "function",
            "params": [
                {"name": "shape", "kind": "POSITIONAL_OR_KEYWORD"},
                {"name": "dtype", "kind": "POSITIONAL_OR_KEYWORD", "default": "float"},
                {"name": "order", "kind": "POSITIONAL_OR_KEYWORD", "default": "'C'"},
                {"name": "bh_force", "kind": "KEYWORD_ONLY", "default": "True"},
            ],
            "docstring": "Return a new Bohrium array of given shape and type, filled with zeros.",
        },
        {
            "name": "ones",
            "api_path": "bohrium.ones",
            "kind": "function",
            "params": [
                {"name": "shape", "kind": "POSITIONAL_OR_KEYWORD"},
                {"name": "dtype", "kind": "POSITIONAL_OR_KEYWORD", "default": "float"},
                {"name": "order", "kind": "POSITIONAL_OR_KEYWORD", "default": "'C'"},
                {"name": "bh_force", "kind": "KEYWORD_ONLY", "default": "True"},
            ],
            "docstring": "Return a new Bohrium array of given shape and type, filled with ones.",
        },
        {
            "name": "flush",
            "api_path": "bohrium.flush",
            "kind": "function",
            "params": [],
            "docstring": "Force synchronization and execution of all lazily accumulated Bohrium array operations.",
        },
        {
            "name": "interwork",
            "api_path": "bohrium.interwork",
            "kind": "function",
            "params": [
                {"name": "array", "kind": "POSITIONAL_ONLY"},
            ],
            "docstring": "Convert a Bohrium array into a standard NumPy ndarray or bridge runtime buffers.",
        },
        {
            "name": "matmul",
            "api_path": "bohrium.matmul",
            "kind": "function",
            "params": [
                {"name": "x1", "kind": "POSITIONAL_ONLY"},
                {"name": "x2", "kind": "POSITIONAL_ONLY"},
            ],
            "docstring": "Matrix product of two Bohrium arrays.",
        },
        {
            "name": "dot",
            "api_path": "bohrium.dot",
            "kind": "function",
            "params": [
                {"name": "a", "kind": "POSITIONAL_ONLY"},
                {"name": "b", "kind": "POSITIONAL_ONLY"},
            ],
            "docstring": "Dot product of two Bohrium vectors or matrices.",
        },
    ]
)


def _get_bohrium() -> types.ModuleType | None:
    """Lazily load Bohrium module to avoid runtime overhead.

    Returns:
        The bohrium module or None if not installed.
    """
    try:
        import bohrium

        mod: types.ModuleType = bohrium
        return mod
    except Exception:
        return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Collect Bohrium array and runtime operation definitions.

    Args:
        category: Target SemanticTier enum value.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRef objects representing Bohrium functions.
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
        "flush",
        "interwork",
        "matmul",
        "dot",
        "sum",
        "mean",
        "reshape",
    ]
    return collect_from_canonical_or_live(
        live_module=_get_bohrium(),
        module_prefix="bohrium",
        target_names=target_names,
        canonical_ops=CANONICAL_BOHRIUM_OPS,
        default_doc_prefix="Bohrium",
        environment_tags=("bohrium", "runtime"),
        inspector_fn=GhostInspector.inspect,
    )
