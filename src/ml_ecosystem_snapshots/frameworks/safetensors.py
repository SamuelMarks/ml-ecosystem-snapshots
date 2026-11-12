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

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalSubmoduleOpSpec,
    CanonicalSubmoduleOpTuple,
    build_canonical_submodule_ops,
    collect_from_submodules_or_canonical,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

_SAFETENSORS_OP_DEFS: list[CanonicalSubmoduleOpTuple] = [
    (
        "safe_open",
        "safetensors.safe_open",
        "function",
        "core",
        [
            ("filename", "POSITIONAL_ONLY", None),
            ("framework", "KEYWORD_ONLY", "'pt'"),
            ("device", "KEYWORD_ONLY", "'cpu'"),
        ],
        "Open a safetensors file with memory mapping.",
    ),
    (
        "save_file",
        "safetensors.torch.save_file",
        "function",
        "torch",
        [
            ("tensors", "POSITIONAL_ONLY", None),
            ("filename", "POSITIONAL_ONLY", None),
            ("metadata", "KEYWORD_ONLY", "None"),
        ],
        "Save a dictionary of PyTorch tensors into a safetensors file.",
    ),
    (
        "load_file",
        "safetensors.torch.load_file",
        "function",
        "torch",
        [("filename", "POSITIONAL_ONLY", None), ("device", "KEYWORD_ONLY", "'cpu'")],
        "Load a safetensors file into a dictionary of PyTorch tensors.",
    ),
    (
        "save_model",
        "safetensors.torch.save_model",
        "function",
        "torch",
        [
            ("model", "POSITIONAL_ONLY", None),
            ("filename", "POSITIONAL_ONLY", None),
            ("metadata", "KEYWORD_ONLY", "None"),
        ],
        "Save a PyTorch model's state dict into a safetensors file.",
    ),
    (
        "load_model",
        "safetensors.torch.load_model",
        "function",
        "torch",
        [
            ("model", "POSITIONAL_ONLY", None),
            ("filename", "POSITIONAL_ONLY", None),
            ("strict", "KEYWORD_ONLY", "True"),
            ("device", "KEYWORD_ONLY", "'cpu'"),
        ],
        "Load weights from a safetensors file directly into a PyTorch model.",
    ),
    (
        "save",
        "safetensors.torch.save",
        "function",
        "torch",
        [("tensors", "POSITIONAL_ONLY", None), ("metadata", "KEYWORD_ONLY", "None")],
        "Serialize a dictionary of PyTorch tensors into safetensors bytes.",
    ),
    (
        "load",
        "safetensors.torch.load",
        "function",
        "torch",
        [("data", "POSITIONAL_ONLY", None)],
        "Deserialize safetensors bytes into a dictionary of PyTorch tensors.",
    ),
    (
        "save_file",
        "safetensors.flax.save_file",
        "function",
        "flax",
        [
            ("tensors", "POSITIONAL_ONLY", None),
            ("filename", "POSITIONAL_ONLY", None),
            ("metadata", "KEYWORD_ONLY", "None"),
        ],
        "Save a dictionary of Flax/JAX arrays into a safetensors file.",
    ),
    (
        "load_file",
        "safetensors.flax.load_file",
        "function",
        "flax",
        [("filename", "POSITIONAL_ONLY", None)],
        "Load a safetensors file into a dictionary of Flax/JAX arrays.",
    ),
    (
        "save",
        "safetensors.flax.save",
        "function",
        "flax",
        [("tensors", "POSITIONAL_ONLY", None), ("metadata", "KEYWORD_ONLY", "None")],
        "Serialize a dictionary of Flax/JAX arrays into safetensors bytes.",
    ),
    (
        "load",
        "safetensors.flax.load",
        "function",
        "flax",
        [("data", "POSITIONAL_ONLY", None)],
        "Deserialize safetensors bytes into a dictionary of Flax/JAX arrays.",
    ),
    (
        "save_file",
        "safetensors.numpy.save_file",
        "function",
        "numpy",
        [
            ("tensors", "POSITIONAL_ONLY", None),
            ("filename", "POSITIONAL_ONLY", None),
            ("metadata", "KEYWORD_ONLY", "None"),
        ],
        "Save a dictionary of NumPy ndarrays into a safetensors file.",
    ),
    (
        "load_file",
        "safetensors.numpy.load_file",
        "function",
        "numpy",
        [("filename", "POSITIONAL_ONLY", None)],
        "Load a safetensors file into a dictionary of NumPy ndarrays.",
    ),
    (
        "save",
        "safetensors.numpy.save",
        "function",
        "numpy",
        [("tensors", "POSITIONAL_ONLY", None), ("metadata", "KEYWORD_ONLY", "None")],
        "Serialize a dictionary of NumPy ndarrays into safetensors bytes.",
    ),
    (
        "load",
        "safetensors.numpy.load",
        "function",
        "numpy",
        [("data", "POSITIONAL_ONLY", None)],
        "Save a dictionary of NumPy arrays into a byte buffer.",
    ),
]

CANONICAL_SAFETENSORS_OPS: list[CanonicalSubmoduleOpSpec] = build_canonical_submodule_ops(_SAFETENSORS_OP_DEFS)


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
    except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError, TypeError, ValueError, OSError):
        return None


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
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
    loaded_submods: dict[str, types.ModuleType] = {}
    for sub in target_submodules:
        mod = _get_safetensors_submodule(sub)
        if mod is not None:
            loaded_submods[sub] = mod

    return collect_from_submodules_or_canonical(
        loaded_submods=loaded_submods,
        canonical_ops=CANONICAL_SAFETENSORS_OPS,
        framework_tag="safetensors",
        default_submodule="core",
        default_doc_prefix="SafeTensors",
        inspector_fn=GhostInspector.inspect,
    )
