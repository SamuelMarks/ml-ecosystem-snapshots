"""SafeTensors API Snapshot Extractor.

Extracts serialization, deserialization, and memory-mapped model tensor loading
APIs for PyTorch, Flax, NumPy, and top-level SafeTensors operations.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostPythonRef,
    GhostRef,
    ParameterKind,
    SemanticTier,
)
from ..models import GhostInspector


CANONICAL_SAFETENSORS_OPS: List[Dict[str, Any]] = [
    # Top-level functions
    {
        "name": "safe_open",
        "api_path": "safetensors.safe_open",
        "submodule": "core",
        "params": [
            {"name": "filename", "kind": "POSITIONAL_ONLY"},
            {"name": "framework", "kind": "KEYWORD_ONLY", "default": "'pt'"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "'cpu'"},
        ],
        "docstring": "Open a safetensors file with memory mapping.",
    },
    # Torch bindings
    {
        "name": "save_file",
        "api_path": "safetensors.torch.save_file",
        "submodule": "torch",
        "params": [
            {"name": "tensors", "kind": "POSITIONAL_ONLY"},
            {"name": "filename", "kind": "POSITIONAL_ONLY"},
            {"name": "metadata", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Save a dictionary of PyTorch tensors into a safetensors file.",
    },
    {
        "name": "load_file",
        "api_path": "safetensors.torch.load_file",
        "submodule": "torch",
        "params": [
            {"name": "filename", "kind": "POSITIONAL_ONLY"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "'cpu'"},
        ],
        "docstring": "Load a safetensors file into a dictionary of PyTorch tensors.",
    },
    {
        "name": "save_model",
        "api_path": "safetensors.torch.save_model",
        "submodule": "torch",
        "params": [
            {"name": "model", "kind": "POSITIONAL_ONLY"},
            {"name": "filename", "kind": "POSITIONAL_ONLY"},
            {"name": "metadata", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Save a PyTorch model's state dict into a safetensors file.",
    },
    {
        "name": "load_model",
        "api_path": "safetensors.torch.load_model",
        "submodule": "torch",
        "params": [
            {"name": "model", "kind": "POSITIONAL_ONLY"},
            {"name": "filename", "kind": "POSITIONAL_ONLY"},
            {"name": "strict", "kind": "KEYWORD_ONLY", "default": "True"},
            {"name": "device", "kind": "KEYWORD_ONLY", "default": "'cpu'"},
        ],
        "docstring": "Load weights from a safetensors file directly into a PyTorch model.",
    },
    {
        "name": "save",
        "api_path": "safetensors.torch.save",
        "submodule": "torch",
        "params": [
            {"name": "tensors", "kind": "POSITIONAL_ONLY"},
            {"name": "metadata", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Serialize a dictionary of PyTorch tensors into safetensors bytes.",
    },
    {
        "name": "load",
        "api_path": "safetensors.torch.load",
        "submodule": "torch",
        "params": [
            {"name": "data", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Deserialize safetensors bytes into a dictionary of PyTorch tensors.",
    },
    # Flax bindings
    {
        "name": "save_file",
        "api_path": "safetensors.flax.save_file",
        "submodule": "flax",
        "params": [
            {"name": "tensors", "kind": "POSITIONAL_ONLY"},
            {"name": "filename", "kind": "POSITIONAL_ONLY"},
            {"name": "metadata", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Save a dictionary of Flax/JAX arrays into a safetensors file.",
    },
    {
        "name": "load_file",
        "api_path": "safetensors.flax.load_file",
        "submodule": "flax",
        "params": [
            {"name": "filename", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Load a safetensors file into a dictionary of Flax/JAX arrays.",
    },
    {
        "name": "save",
        "api_path": "safetensors.flax.save",
        "submodule": "flax",
        "params": [
            {"name": "tensors", "kind": "POSITIONAL_ONLY"},
            {"name": "metadata", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Serialize a dictionary of Flax/JAX arrays into safetensors bytes.",
    },
    {
        "name": "load",
        "api_path": "safetensors.flax.load",
        "submodule": "flax",
        "params": [
            {"name": "data", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Deserialize safetensors bytes into a dictionary of Flax/JAX arrays.",
    },
    # NumPy bindings
    {
        "name": "save_file",
        "api_path": "safetensors.numpy.save_file",
        "submodule": "numpy",
        "params": [
            {"name": "tensors", "kind": "POSITIONAL_ONLY"},
            {"name": "filename", "kind": "POSITIONAL_ONLY"},
            {"name": "metadata", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Save a dictionary of NumPy ndarrays into a safetensors file.",
    },
    {
        "name": "load_file",
        "api_path": "safetensors.numpy.load_file",
        "submodule": "numpy",
        "params": [
            {"name": "filename", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Load a safetensors file into a dictionary of NumPy ndarrays.",
    },
    {
        "name": "save",
        "api_path": "safetensors.numpy.save",
        "submodule": "numpy",
        "params": [
            {"name": "tensors", "kind": "POSITIONAL_ONLY"},
            {"name": "metadata", "kind": "KEYWORD_ONLY", "default": "None"},
        ],
        "docstring": "Serialize a dictionary of NumPy ndarrays into safetensors bytes.",
    },
    {
        "name": "load",
        "api_path": "safetensors.numpy.load",
        "submodule": "numpy",
        "params": [
            {"name": "data", "kind": "POSITIONAL_ONLY"},
        ],
        "docstring": "Deserialize safetensors bytes into a dictionary of NumPy ndarrays.",
    },
]


def _get_safetensors_submodule(submod: str) -> Any:
    """Safely import a specific SafeTensors submodule.

    Args:
        submod: Name of the SafeTensors submodule (e.g. 'torch', 'flax', 'numpy', or 'core').

    Returns:
        The submodule object or None if not available.
    """
    try:
        if submod == "core":
            return importlib.import_module("safetensors")
        return importlib.import_module(f"safetensors.{submod}")
    except (ImportError, Exception):
        return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect SafeTensors weight serialization and loading APIs.

    Args:
        category: The SemanticTier category.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRef objects representing SafeTensors operations.
    """
    if category not in (
        SemanticTier.DATALOADER,
        SemanticTier.ARRAY_API,
        SemanticTier.UTIL,
    ):
        return []

    target_submodules = ["core", "torch", "flax", "numpy"]
    loaded_submods: Dict[str, Any] = {}
    for sub in target_submodules:
        mod = _get_safetensors_submodule(sub)
        if mod is not None:
            loaded_submods[sub] = mod

    if not loaded_submods:
        refs: List[GhostRef] = []
        for op in CANONICAL_SAFETENSORS_OPS:
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
                    docstring=op.get("docstring", f"SafeTensors {op['name']} API."),
                    environment_tags=["safetensors", op.get("submodule", "core")],
                    domain_metadata={"submodule": op.get("submodule", "core")},
                )
            )
        return refs

    results: List[GhostRef] = []
    seen: set[str] = set()

    for op in CANONICAL_SAFETENSORS_OPS:
        sub = op.get("submodule", "core")
        if sub in loaded_submods:
            mod = loaded_submods[sub]
            name = op["name"]
            if hasattr(mod, name):
                obj = getattr(mod, name)
                try:
                    ref = GhostInspector.inspect(obj, op["api_path"], is_public=True)
                    ref.environment_tags = list(ref.environment_tags or ()) + [
                        "safetensors",
                        sub,
                    ]
                    results.append(ref)
                    seen.add(op["api_path"])
                except Exception:
                    pass

    # Fill any missed ops from canonical definitions
    for op in CANONICAL_SAFETENSORS_OPS:
        if op["api_path"] not in seen:
            params = [
                GhostParam(
                    name=p["name"],
                    kind=getattr(ParameterKind, p.get("kind", "POSITIONAL_OR_KEYWORD")),
                    default=p.get("default"),
                )
                for p in op.get("params", [])
            ]
            results.append(
                GhostPythonRef(
                    name=op["name"],
                    api_path=op["api_path"],
                    kind="function",
                    params=params,
                    docstring=op.get("docstring", f"SafeTensors {op['name']} API."),
                    environment_tags=["safetensors", op.get("submodule", "core")],
                    domain_metadata={"submodule": op.get("submodule", "core")},
                )
            )

    return results
