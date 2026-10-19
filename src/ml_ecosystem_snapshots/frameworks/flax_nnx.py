"""Flax (NNX) API Snapshot Extractor.

Provides functions to dynamically introspect the Flax NNX library and
generate GhostRefs for layers. Defers to core JAX extractors for
losses, optimizers, and activations.
"""

from __future__ import annotations

import inspect
import types

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.frameworks.jax import collect_api as jax_collect_api
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.utils import get_all_members

nnx: types.ModuleType | None = None


def _get_nnx() -> types.ModuleType | None:
    """Lazily load flax.nnx to avoid circular import issues with jax.

    Returns:
        flax.nnx module or None if not installed.
    """
    global nnx
    if nnx is not None:
        return nnx
    try:
        import flax.nnx as _nnx

        nnx = _nnx
        return nnx
    except Exception:
        return None


def _scan_nnx_layers(include_nonpublic: bool) -> list[GhostRef]:
    """Scan `flax.nnx` module for classes inheriting from `nnx.Module`.

    Excludes the base `Module` class itself.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found NNX layers.

    """
    found: list[GhostRef] = []
    nnx_mod = _get_nnx()
    if not nnx_mod:
        return found
    try:
        for name, obj in get_all_members(nnx_mod):
            if not include_nonpublic and name.startswith("_"):
                continue
            if inspect.isclass(obj) and name != "Module":
                try:
                    if issubclass(obj, getattr(nnx_mod, "Module")):
                        found.append(GhostInspector.inspect(obj, f"flax.nnx.{name}"))
                except TypeError:
                    pass
    except Exception:
        pass

    return found


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Entrypoint to collect the Flax API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.

    """
    # Use core JAX scanning for losses, optimizers, activations, and array ops
    if category in [
        SemanticTier.LOSS,
        SemanticTier.OPTIMIZER,
        SemanticTier.ACTIVATION,
        SemanticTier.SCHEDULER,
        SemanticTier.INITIALIZER,
        SemanticTier.METRIC,
        SemanticTier.DATALOADER,
        SemanticTier.ARRAY_API,
    ]:
        return jax_collect_api(category, include_nonpublic)

    # Add Flax-specific neural layers
    if category == SemanticTier.LAYER:
        return _scan_nnx_layers(include_nonpublic)

    return []
