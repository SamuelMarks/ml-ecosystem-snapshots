"""Keras API Snapshot Extractor.

Provides functions to statically introspect the Keras library using Griffe and generate
GhostRefs for layers, losses, optimizers, and activations.
"""

from __future__ import annotations

import types

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import LivePythonTarget

griffe: types.ModuleType | None = None
try:
    import griffe as _griffe_mod

    griffe = _griffe_mod
    HAS_GRIFFE = True
except ImportError:
    griffe = None
    HAS_GRIFFE = False


def _scan_griffe_module(
    module_path: str,
    prefix: str,
    kind: str = "class",
    block_list: set[str] | None = None,
    include_nonpublic: bool = False,
) -> list[GhostRef]:
    """Statically scans a Keras module for members of a specific kind.

    Args:
        module_path: The griffe module path to inspect (e.g. 'keras.losses').
        prefix: The import prefix for the generated API path.
        kind: Expected kind ("class" or "function").
        block_list: A set of names to exclude from the scan.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing the discovered API members.
    """
    if not griffe:
        return []

    block_list = block_list or set()
    found: list[GhostRef] = []

    try:
        mod = griffe.load(module_path)
    except (
        SnapshotError,
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        LookupError,
        OSError,
        RuntimeError,
        SyntaxError,
        griffe.GriffeError,
    ):
        return []

    for name, member in mod.members.items():
        if (not include_nonpublic and name.startswith("_")) or name in block_list:
            continue

        is_class = member.is_class
        is_function = member.is_function

        if (kind == "class" and is_class) or (kind == "function" and is_function):
            member_target: LivePythonTarget = member
            found.append(GhostInspector.inspect(member_target, f"{prefix}.{name}"))

    return found


_KERAS_TIER_SPECS: dict[SemanticTier, tuple[str, str, set[str] | None]] = {
    SemanticTier.LOSS: ("keras.losses", "class", {"Loss", "Container"}),
    SemanticTier.OPTIMIZER: ("keras.optimizers", "class", {"Optimizer", "TFOptimizer"}),
    SemanticTier.ACTIVATION: ("keras.activations", "function", None),
    SemanticTier.LAYER: ("keras.layers", "class", {"Layer"}),
    SemanticTier.SCHEDULER: ("keras.optimizers.schedules", "class", {"LearningRateSchedule"}),
    SemanticTier.INITIALIZER: ("keras.initializers", "class", {"Initializer"}),
    SemanticTier.METRIC: ("keras.metrics", "class", {"Metric"}),
}


def _collect_static(category: SemanticTier, include_nonpublic: bool) -> list[GhostRef]:
    """Scan the Keras library for a specific API category via Griffe.

    Args:
        category: The SemanticTier enum value specifying what to scan.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of populated GhostRef objects.
    """
    if not griffe:
        return []

    tier_spec = _KERAS_TIER_SPECS.get(category)
    if tier_spec is not None:
        mod_path, kind, blocked = tier_spec
        return _scan_griffe_module(
            mod_path,
            mod_path,
            kind=kind,
            block_list=blocked,
            include_nonpublic=include_nonpublic,
        )

    if category == SemanticTier.ARRAY_API:
        ops_refs = _scan_griffe_module(
            "keras.ops",
            "keras.ops",
            kind="function",
            include_nonpublic=include_nonpublic,
        )
        image_refs = _scan_griffe_module(
            "keras.ops.image",
            "keras.ops.image",
            kind="function",
            include_nonpublic=include_nonpublic,
        )
        ops_refs.extend(image_refs)
        for ref in ops_refs:
            if ref.environment_tags is None:
                ref.environment_tags = []
            ref.environment_tags.extend(["backend:jax", "backend:torch", "backend:tensorflow"])
        return ops_refs

    return []


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Entrypoint to collect the Keras API signature for a given category.

    Args:
        category: The category of API to collect (e.g., LOSS).
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.
    """
    return _collect_static(category, include_nonpublic)
