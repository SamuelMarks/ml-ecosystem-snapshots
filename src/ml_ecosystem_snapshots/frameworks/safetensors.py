"""SafeTensors API Snapshot Extractor.

Extracts serialization, deserialization, and memory-mapped model tensor loading
APIs for PyTorch, Flax, NumPy, and top-level SafeTensors operations.
"""

from __future__ import annotations

import importlib
import types

from ml_switcheroo_ir.schema.ghost import (
    GhostRef,
    SemanticTier,
)
import pydantic

from ml_ecosystem_snapshots.frameworks._base import CanonicalSubmoduleOpSpec
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector


CANONICAL_SAFETENSORS_OPS: list[CanonicalSubmoduleOpSpec] = pydantic.TypeAdapter(
    list[CanonicalSubmoduleOpSpec]
).validate_python(
    [
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
            "docstring": "Save a dictionary of NumPy arrays into a byte buffer.",
        },
    ]
)


def _get_safetensors_submodule(submod: str) -> types.ModuleType | None:
    """Safely import a specific SafeTensors submodule.

    Args:
        submod: Name of the SafeTensors submodule (e.g. 'torch', 'flax', 'numpy', 'core').

    Returns:
        The submodule object or None if not available.
    """
    try:
        if submod == "core":
            return importlib.import_module("safetensors")
        return importlib.import_module(f"safetensors.{submod}")
    except Exception:
        return None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
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
    loaded_submods: dict[str, types.ModuleType | object] = {}
    for sub in target_submodules:
        mod = _get_safetensors_submodule(sub)
        if mod is not None:
            loaded_submods[sub] = mod

    from ml_ecosystem_snapshots.frameworks._base import (
        collect_from_submodules_or_canonical,
    )

    return collect_from_submodules_or_canonical(
        loaded_submods=loaded_submods,
        canonical_ops=CANONICAL_SAFETENSORS_OPS,
        framework_tag="safetensors",
        default_submodule="core",
        default_doc_prefix="SafeTensors",
        inspector_fn=GhostInspector.inspect,
    )
