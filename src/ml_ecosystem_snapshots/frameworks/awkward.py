"""Awkward Array Ragged and Nested Layout Snapshot Extractor.

Extracts nested jagged array constructors, record builders, Arrow conversion utilities,
and ragged array manipulation routines from Awkward Array.
"""

from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    SemanticTier,
)
from ..models import GhostInspector


CANONICAL_AWKWARD_OPS: List[Dict[str, Any]] = [
    {
        "name": "Array",
        "api_path": "awkward.Array",
        "kind": "class",
        "params": [
            {"name": "data", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "behavior", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "with_name", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "check_valid", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "High-level jagged and nested array container with arbitrary depth.",
    },
    {
        "name": "Record",
        "api_path": "awkward.Record",
        "kind": "class",
        "params": [
            {"name": "data", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "behavior", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "with_name", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Scalar record representing a single structured element of an Awkward record array.",
    },
    {
        "name": "ArrayBuilder",
        "api_path": "awkward.ArrayBuilder",
        "kind": "class",
        "params": [
            {"name": "initial", "kind": "KEYWORD_ONLY", "default": "1024"},
            {"name": "resize", "kind": "KEYWORD_ONLY", "default": "2.0"},
            {"name": "behavior", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Growable nested structure builder for dynamically assembling Awkward arrays.",
    },
    {
        "name": "flatten",
        "api_path": "awkward.flatten",
        "kind": "function",
        "params": [
            {"name": "array", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "1"},
        ],
        "docstring": "Flatten one level of list depth in the nested array.",
    },
    {
        "name": "unflatten",
        "api_path": "awkward.unflatten",
        "kind": "function",
        "params": [
            {"name": "array", "kind": "POSITIONAL_ONLY"},
            {"name": "counts", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "0"},
        ],
        "docstring": "Group elements of an array into sublists according to specified counts.",
    },
    {
        "name": "pad_none",
        "api_path": "awkward.pad_none",
        "kind": "function",
        "params": [
            {"name": "array", "kind": "POSITIONAL_ONLY"},
            {"name": "target", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "1"},
            {"name": "clip", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Pad lists with None values until they reach the target length.",
    },
    {
        "name": "fill_none",
        "api_path": "awkward.fill_none",
        "kind": "function",
        "params": [
            {"name": "array", "kind": "POSITIONAL_ONLY"},
            {"name": "value", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "-1"},
        ],
        "docstring": "Replace None values in option-type lists with a concrete replacement scalar.",
    },
    {
        "name": "to_numpy",
        "api_path": "awkward.to_numpy",
        "kind": "function",
        "params": [
            {"name": "array", "kind": "POSITIONAL_ONLY"},
            {"name": "allow_missing", "kind": "KEYWORD_ONLY", "default": "True"},
        ],
        "docstring": "Convert a regular rectangular Awkward array into a standard NumPy ndarray.",
    },
    {
        "name": "from_numpy",
        "api_path": "awkward.from_numpy",
        "kind": "function",
        "params": [
            {"name": "array", "kind": "POSITIONAL_ONLY"},
            {"name": "regulararray", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "recordarray", "kind": "KEYWORD_ONLY", "default": "True"},
        ],
        "docstring": "Convert a NumPy ndarray or structured array into an Awkward array.",
    },
    {
        "name": "to_arrow",
        "api_path": "awkward.to_arrow",
        "kind": "function",
        "params": [
            {"name": "array", "kind": "POSITIONAL_ONLY"},
            {"name": "extensionarray", "kind": "KEYWORD_ONLY", "default": "True"},
        ],
        "docstring": "Convert an Awkward array into an Apache Arrow Array or ChunkedArray.",
    },
    {
        "name": "from_arrow",
        "api_path": "awkward.from_arrow",
        "kind": "function",
        "params": [
            {"name": "array", "kind": "POSITIONAL_ONLY"},
            {"name": "generate_bitmasks", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Convert an Apache Arrow Array, ChunkedArray, RecordBatch, or Table into Awkward.",
    },
]


def _get_awkward() -> Any:
    """Lazily load Awkward Array module to prevent import overhead.

    Returns:
        The awkward module or None if not installed.
    """
    try:
        import awkward as ak

        return ak
    except Exception:
        return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect Awkward Array constructors and transformation utilities.

    Args:
        category: Target SemanticTier enum value.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRef objects representing Awkward Array functions and classes.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.UTIL,
    ):
        return []

    from ml_ecosystem_snapshots.frameworks._base import collect_from_canonical_or_live

    target_names = [
        "Array",
        "Record",
        "ArrayBuilder",
        "flatten",
        "unflatten",
        "pad_none",
        "fill_none",
        "is_none",
        "where",
        "cartesian",
        "combinations",
        "zip",
        "unzip",
        "concatenate",
        "broadcast_arrays",
        "to_numpy",
        "from_numpy",
        "to_arrow",
        "from_arrow",
        "sum",
        "mean",
        "std",
    ]
    return collect_from_canonical_or_live(
        live_module=_get_awkward(),
        module_prefix="awkward",
        target_names=target_names,
        canonical_ops=CANONICAL_AWKWARD_OPS,
        default_doc_prefix="Awkward",
        environment_tags=("awkward", "ragged"),
        inspector_fn=GhostInspector.inspect,
    )
