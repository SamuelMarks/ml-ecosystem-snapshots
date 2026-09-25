"""Bohrium Heterogeneous Runtime and Array Snapshot Extractor.

Extracts array constructors, JIT kernel fusion decorators, synchronization primitives (flush),
and accelerated linear algebra routines from the Bohrium runtime engine.
"""

from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)
from ..models import GhostInspector, GhostPythonRef


CANONICAL_BOHRIUM_OPS: List[Dict[str, Any]] = [
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


def _get_bohrium() -> Any:
    """Lazily load Bohrium module to avoid runtime overhead.

    Returns:
        The bohrium module or None if not installed.
    """
    try:
        import bohrium

        return bohrium
    except (ImportError, Exception):
        return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
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

    bh = _get_bohrium()
    if bh is None:
        refs: List[GhostRef] = []
        for op in CANONICAL_BOHRIUM_OPS:
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
                    docstring=op.get("docstring", f"Bohrium {op['name']} API."),
                    environment_tags=["bohrium", "runtime"],
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
        "flush",
        "interwork",
        "matmul",
        "dot",
        "sum",
        "mean",
        "reshape",
    ]

    for name in target_names:
        if hasattr(bh, name):
            obj = getattr(bh, name)
            try:
                ref = GhostInspector.inspect(obj, f"bohrium.{name}", is_public=True)
                ref.environment_tags = list(ref.environment_tags or ()) + [
                    "bohrium",
                    "runtime",
                ]
                results.append(ref)
            except Exception:
                pass

    return results
