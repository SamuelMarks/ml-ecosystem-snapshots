"""PyData Sparse (COO/GCXS/DOK) Format Snapshot Extractor.

Extracts multidimensional sparse array formats, constructors, arithmetic operations,
and linear algebra routines from the PyData Sparse library.
"""

from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    SemanticTier,
)
from ..models import GhostInspector


CANONICAL_SPARSE_OPS: List[Dict[str, Any]] = [
    {
        "name": "COO",
        "api_path": "sparse.COO",
        "kind": "class",
        "params": [
            {"name": "coords", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "data", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "shape", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "has_duplicates", "kind": "KEYWORD_ONLY", "default": "True"},
            {"name": "sorted", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "prune", "kind": "KEYWORD_ONLY", "default": "False"},
            {"name": "fill_value", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Coordinate list (COO) format multidimensional sparse array.",
    },
    {
        "name": "GCXS",
        "api_path": "sparse.GCXS",
        "kind": "class",
        "params": [
            {"name": "arg", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "shape", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "compressed_axes", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "prune", "kind": "KEYWORD_ONLY", "default": "False"},
        ],
        "docstring": "Generalized Compressed Sparse format supporting arbitrary compressed axes.",
    },
    {
        "name": "DOK",
        "api_path": "sparse.DOK",
        "kind": "class",
        "params": [
            {"name": "shape", "kind": "POSITIONAL_OR_KEYWORD"},
            {"name": "data", "kind": "POSITIONAL_OR_KEYWORD", "default": "None"},
            {"name": "fill_value", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Dictionary Of Keys (DOK) sparse array for incremental construction.",
    },
    {
        "name": "as_coo",
        "api_path": "sparse.as_coo",
        "kind": "function",
        "params": [
            {"name": "x", "kind": "POSITIONAL_ONLY"},
            {"name": "shape", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "fill_value", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Convert an arbitrary array-like object to COO sparse format.",
    },
    {
        "name": "dot",
        "api_path": "sparse.dot",
        "kind": "function",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
            {"name": "b", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Dot product of two sparse or dense arrays.",
    },
    {
        "name": "matmul",
        "api_path": "sparse.matmul",
        "kind": "function",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
            {"name": "b", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Matrix product of two sparse arrays following NumPy broadcasting rules.",
    },
    {
        "name": "tensordot",
        "api_path": "sparse.tensordot",
        "kind": "function",
        "params": [
            {"name": "a", "kind": "POSITIONAL_ONLY"},
            {"name": "b", "kind": "POSITIONAL_ONLY"},
            {"name": "axes", "kind": "KEYWORD_ONLY", "default": "2"},
        ],
        "docstring": "Compute tensor contraction along specified axes.",
    },
    {
        "name": "elemwise",
        "api_path": "sparse.elemwise",
        "kind": "function",
        "params": [
            {"name": "func", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Apply element-wise function across sparse arrays.",
    },
    {
        "name": "concatenate",
        "api_path": "sparse.concatenate",
        "kind": "function",
        "params": [
            {"name": "arrays", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "0"},
        ],
        "docstring": "Join a sequence of sparse arrays along an existing axis.",
    },
    {
        "name": "stack",
        "api_path": "sparse.stack",
        "kind": "function",
        "params": [
            {"name": "arrays", "kind": "POSITIONAL_ONLY"},
            {"name": "axis", "kind": "KEYWORD_ONLY", "default": "0"},
        ],
        "docstring": "Join a sequence of sparse arrays along a new axis.",
    },
]


def _get_sparse() -> Any:
    """Lazily load the PyData Sparse module to avoid BLAS/Accelerate initialization.

    Returns:
        The sparse module or None if not installed.
    """
    try:
        import sparse

        return sparse
    except Exception:
        return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect PyData Sparse array and operator definitions.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRef objects representing sparse constructors and operations.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.UTIL,
    ):
        return []

    from ml_ecosystem_snapshots.frameworks._base import collect_from_canonical_or_live

    target_names = [
        "COO",
        "GCXS",
        "DOK",
        "as_coo",
        "asnumpy",
        "dot",
        "matmul",
        "tensordot",
        "einsum",
        "elemwise",
        "concatenate",
        "stack",
        "diagonal",
        "eye",
        "zeros",
        "ones",
        "full",
    ]
    return collect_from_canonical_or_live(
        live_module=_get_sparse(),
        module_prefix="sparse",
        target_names=target_names,
        canonical_ops=CANONICAL_SPARSE_OPS,
        default_doc_prefix="Sparse",
        environment_tags=("sparse", "coo"),
        inspector_fn=GhostInspector.inspect,
    )
