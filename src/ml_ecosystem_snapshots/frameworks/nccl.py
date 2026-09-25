"""NVIDIA NCCL & AMD RCCL Collective Communications Snapshot Extractor.

Extracts multi-GPU collective communication primitives (AllReduce, AllGather,
ReduceScatter, AllToAll, Broadcast, Point-to-Point) matching `COLLECTIVE_OPS_REGISTRY`.
"""

from __future__ import annotations

import ctypes
from typing import Any, Dict, List, Optional

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostPythonRef,
    GhostRef,
    ParameterKind,
    SemanticTier,
)


CANONICAL_NCCL_OPS: List[Dict[str, Any]] = [
    # Collective Reductions
    {
        "name": "all_reduce",
        "api_path": "nccl.all_reduce",
        "params": [
            {"name": "sendbuff", "kind": "POSITIONAL_ONLY"},
            {"name": "recvbuff", "kind": "POSITIONAL_ONLY"},
            {"name": "count", "kind": "POSITIONAL_ONLY"},
            {"name": "datatype", "kind": "KEYWORD_ONLY", "default": "7"},
            {"name": "op", "kind": "KEYWORD_ONLY", "default": "0"},
            {"name": "comm", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "stream", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Reduces data across all GPUs and delivers the result to every GPU.",
    },
    {
        "name": "all_gather",
        "api_path": "nccl.all_gather",
        "params": [
            {"name": "sendbuff", "kind": "POSITIONAL_ONLY"},
            {"name": "recvbuff", "kind": "POSITIONAL_ONLY"},
            {"name": "sendcount", "kind": "POSITIONAL_ONLY"},
            {"name": "datatype", "kind": "KEYWORD_ONLY", "default": "7"},
            {"name": "comm", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "stream", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Gathers data from each GPU and distributes the combined result to all GPUs.",
    },
    {
        "name": "reduce_scatter",
        "api_path": "nccl.reduce_scatter",
        "params": [
            {"name": "sendbuff", "kind": "POSITIONAL_ONLY"},
            {"name": "recvbuff", "kind": "POSITIONAL_ONLY"},
            {"name": "recvcount", "kind": "POSITIONAL_ONLY"},
            {"name": "datatype", "kind": "KEYWORD_ONLY", "default": "7"},
            {"name": "op", "kind": "KEYWORD_ONLY", "default": "0"},
            {"name": "comm", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "stream", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Reduces data across all GPUs and scatters equal parts of the result to each GPU.",
    },
    {
        "name": "all_to_all",
        "api_path": "nccl.all_to_all",
        "params": [
            {"name": "sendbuff", "kind": "POSITIONAL_ONLY"},
            {"name": "recvbuff", "kind": "POSITIONAL_ONLY"},
            {"name": "count", "kind": "POSITIONAL_ONLY"},
            {"name": "datatype", "kind": "KEYWORD_ONLY", "default": "7"},
            {"name": "comm", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "stream", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Scatters data from each rank to all other ranks.",
    },
    {
        "name": "broadcast",
        "api_path": "nccl.broadcast",
        "params": [
            {"name": "sendbuff", "kind": "POSITIONAL_ONLY"},
            {"name": "recvbuff", "kind": "POSITIONAL_ONLY"},
            {"name": "count", "kind": "POSITIONAL_ONLY"},
            {"name": "datatype", "kind": "KEYWORD_ONLY", "default": "7"},
            {"name": "root", "kind": "KEYWORD_ONLY", "default": "0"},
            {"name": "comm", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "stream", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Broadcasts data from root rank to all other ranks.",
    },
    {
        "name": "reduce",
        "api_path": "nccl.reduce",
        "params": [
            {"name": "sendbuff", "kind": "POSITIONAL_ONLY"},
            {"name": "recvbuff", "kind": "POSITIONAL_ONLY"},
            {"name": "count", "kind": "POSITIONAL_ONLY"},
            {"name": "datatype", "kind": "KEYWORD_ONLY", "default": "7"},
            {"name": "op", "kind": "KEYWORD_ONLY", "default": "0"},
            {"name": "root", "kind": "KEYWORD_ONLY", "default": "0"},
            {"name": "comm", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "stream", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Reduces data across all GPUs and places the result on root rank.",
    },
    # Point to Point
    {
        "name": "send",
        "api_path": "nccl.send",
        "params": [
            {"name": "sendbuff", "kind": "POSITIONAL_ONLY"},
            {"name": "count", "kind": "POSITIONAL_ONLY"},
            {"name": "datatype", "kind": "KEYWORD_ONLY", "default": "7"},
            {"name": "peer", "kind": "POSITIONAL_ONLY"},
            {"name": "comm", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "stream", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Point-to-point non-blocking send to peer rank.",
    },
    {
        "name": "recv",
        "api_path": "nccl.recv",
        "params": [
            {"name": "recvbuff", "kind": "POSITIONAL_ONLY"},
            {"name": "count", "kind": "POSITIONAL_ONLY"},
            {"name": "datatype", "kind": "KEYWORD_ONLY", "default": "7"},
            {"name": "peer", "kind": "POSITIONAL_ONLY"},
            {"name": "comm", "kind": "KEYWORD_ONLY", "default": "None"},
            {"name": "stream", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Point-to-point non-blocking receive from peer rank.",
    },
    # Communicator Lifecycle
    {
        "name": "comm_init_rank",
        "api_path": "nccl.comm_init_rank",
        "params": [
            {"name": "comm", "kind": "POSITIONAL_ONLY"},
            {"name": "nranks", "kind": "POSITIONAL_ONLY"},
            {"name": "commId", "kind": "POSITIONAL_ONLY"},
            {"name": "rank", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Initialize an NCCL communicator for a single rank.",
    },
    {
        "name": "comm_destroy",
        "api_path": "nccl.comm_destroy",
        "params": [
            {"name": "comm", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Destroy an initialized NCCL communicator.",
    },
]


def _discover_nccl_library() -> Optional[ctypes.CDLL]:
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
        except (OSError, Exception):
            pass
    return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
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

    refs: List[GhostRef] = []
    lib = _discover_nccl_library()
    hardware_tag = "rccl" if (lib and "rccl" in str(lib)) else "nccl"

    for op in CANONICAL_NCCL_OPS:
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
                kind="function",
                params=params,
                docstring=op.get(
                    "docstring", f"NCCL {op['name']} collective operation."
                ),
                environment_tags=["collective", hardware_tag, "gpu_communication"],
                domain_metadata={"domain": "collective", "hardware": hardware_tag},
            )
        )

    return refs
