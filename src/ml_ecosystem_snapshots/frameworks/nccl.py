"""NVIDIA NCCL & AMD RCCL Collective Communications Snapshot Extractor.

Extracts multi-GPU collective communication primitives (AllReduce, AllGather,
ReduceScatter, AllToAll, Broadcast, Point-to-Point) matching `COLLECTIVE_OPS_REGISTRY`.
"""

from __future__ import annotations

import ctypes

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    SemanticTier,
)

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalPythonOpSpec,
    CanonicalPythonOpTuple,
    build_canonical_python_ops,
    build_canonical_python_ref,
)

_NCCL_OP_DEFS: list[CanonicalPythonOpTuple] = [
    (
        "all_reduce",
        "nccl.all_reduce",
        "function",
        [("sendbuff", "POSITIONAL_ONLY", None), ("recvbuff", "POSITIONAL_ONLY", None), ("count", "POSITIONAL_ONLY", None), ("datatype", "KEYWORD_ONLY", "7"), ("op", "KEYWORD_ONLY", "0"), ("comm", "KEYWORD_ONLY", "None"), ("stream", "KEYWORD_ONLY", "None")],
        "Reduces data across all GPUs and delivers the result to every GPU.",
    ),
    (
        "all_gather",
        "nccl.all_gather",
        "function",
        [("sendbuff", "POSITIONAL_ONLY", None), ("recvbuff", "POSITIONAL_ONLY", None), ("sendcount", "POSITIONAL_ONLY", None), ("datatype", "KEYWORD_ONLY", "7"), ("comm", "KEYWORD_ONLY", "None"), ("stream", "KEYWORD_ONLY", "None")],
        "Gathers data from each GPU and distributes the combined result to all GPUs.",
    ),
    (
        "reduce_scatter",
        "nccl.reduce_scatter",
        "function",
        [("sendbuff", "POSITIONAL_ONLY", None), ("recvbuff", "POSITIONAL_ONLY", None), ("recvcount", "POSITIONAL_ONLY", None), ("datatype", "KEYWORD_ONLY", "7"), ("op", "KEYWORD_ONLY", "0"), ("comm", "KEYWORD_ONLY", "None"), ("stream", "KEYWORD_ONLY", "None")],
        "Reduces data across all GPUs and scatters equal parts of the result to each GPU.",
    ),
    (
        "all_to_all",
        "nccl.all_to_all",
        "function",
        [("sendbuff", "POSITIONAL_ONLY", None), ("recvbuff", "POSITIONAL_ONLY", None), ("count", "POSITIONAL_ONLY", None), ("datatype", "KEYWORD_ONLY", "7"), ("comm", "KEYWORD_ONLY", "None"), ("stream", "KEYWORD_ONLY", "None")],
        "Scatters data from each rank to all other ranks.",
    ),
    (
        "broadcast",
        "nccl.broadcast",
        "function",
        [("sendbuff", "POSITIONAL_ONLY", None), ("recvbuff", "POSITIONAL_ONLY", None), ("count", "POSITIONAL_ONLY", None), ("datatype", "KEYWORD_ONLY", "7"), ("root", "KEYWORD_ONLY", "0"), ("comm", "KEYWORD_ONLY", "None"), ("stream", "KEYWORD_ONLY", "None")],
        "Broadcasts data from root rank to all other ranks.",
    ),
    (
        "reduce",
        "nccl.reduce",
        "function",
        [("sendbuff", "POSITIONAL_ONLY", None), ("recvbuff", "POSITIONAL_ONLY", None), ("count", "POSITIONAL_ONLY", None), ("datatype", "KEYWORD_ONLY", "7"), ("op", "KEYWORD_ONLY", "0"), ("root", "KEYWORD_ONLY", "0"), ("comm", "KEYWORD_ONLY", "None"), ("stream", "KEYWORD_ONLY", "None")],
        "Reduces data across all GPUs and places the result on root rank.",
    ),
    (
        "send",
        "nccl.send",
        "function",
        [("sendbuff", "POSITIONAL_ONLY", None), ("count", "POSITIONAL_ONLY", None), ("datatype", "KEYWORD_ONLY", "7"), ("peer", "POSITIONAL_ONLY", None), ("comm", "KEYWORD_ONLY", "None"), ("stream", "KEYWORD_ONLY", "None")],
        "Point-to-point non-blocking send to peer rank.",
    ),
    (
        "recv",
        "nccl.recv",
        "function",
        [("recvbuff", "POSITIONAL_ONLY", None), ("count", "POSITIONAL_ONLY", None), ("datatype", "KEYWORD_ONLY", "7"), ("peer", "POSITIONAL_ONLY", None), ("comm", "KEYWORD_ONLY", "None"), ("stream", "KEYWORD_ONLY", "None")],
        "Point-to-point non-blocking receive from peer rank.",
    ),
    ("comm_init_rank", "nccl.comm_init_rank", "function", [("comm", "POSITIONAL_ONLY", None), ("nranks", "POSITIONAL_ONLY", None), ("commId", "POSITIONAL_ONLY", None), ("rank", "POSITIONAL_ONLY", None)], "Initialize an NCCL communicator for a single rank."),
    ("comm_destroy", "nccl.comm_destroy", "function", [("comm", "POSITIONAL_ONLY", None)], "Destroy an initialized NCCL communicator."),
]

CANONICAL_NCCL_OPS: list[CanonicalPythonOpSpec] = build_canonical_python_ops(_NCCL_OP_DEFS)


def _discover_nccl_library() -> ctypes.CDLL | None:
    """Attempt to discover and load libnccl.so or librccl.so.

    Returns:
        Loaded CDLL object or None if not installed.
    """
    for lib_name in (
        "libnccl.so.2",
        "libnccl.so",
        "librccl.so.1",
        "librccl.so",
        "libnccl.dylib",
    ):
        try:
            return ctypes.CDLL(lib_name)
        except Exception as exc:
            del exc
    return None


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect NVIDIA NCCL and AMD RCCL collective communication APIs.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRef objects representing collective communication primitives.
    """
    if category not in (
        SemanticTier.ARRAY_API,
        SemanticTier.UTIL,
    ):
        return []

    lib = _discover_nccl_library()
    hardware_tag = "rccl" if (lib and "rccl" in str(lib)) else "nccl"

    return [
        build_canonical_python_ref(
            op,
            default_doc_prefix="NCCL",
            environment_tags=("collective", hardware_tag, "gpu_communication"),
            domain_metadata={"domain": "collective", "hardware": hardware_tag},
        )
        for op in CANONICAL_NCCL_OPS
    ]
