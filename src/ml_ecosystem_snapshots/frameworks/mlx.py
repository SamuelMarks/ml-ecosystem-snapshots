"""MLX API Snapshot Extractor.

Provides functions to dynamically introspect the MLX library and generate
GhostRefs for layers, losses, optimizers, and activations.
"""

from __future__ import annotations

import inspect
import types
import typing

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.utils import get_all_members


@typing.runtime_checkable
class _MlxNnWithLossesProtocol(typing.Protocol):
    """Protocol for `mlx.nn` module exposing a `.losses` submodule."""

    losses: types.ModuleType


@typing.runtime_checkable
class _MlxWithCoreProtocol(typing.Protocol):
    """Protocol for `mlx` module exposing a `.core` submodule."""

    core: types.ModuleType


mlx: types.ModuleType | None = None
try:
    import mlx as _mlx
    import mlx.core as _core  # noqa: F401
    import mlx.nn as _nn  # noqa: F401
    import mlx.optimizers as _optimizers  # noqa: F401

    mlx = _mlx
except ImportError:
    mlx = None

_MLX_ACTIVATION_NAMES: frozenset[str] = frozenset(
    {
        "relu",
        "gelu",
        "silu",
        "sigmoid",
        "tanh",
        "softmax",
        "elu",
        "mish",
        "hardswish",
        "leaky_relu",
        "prelu",
        "selu",
        "step",
        "log_softmax",
        "softplus",
        "hard_sigmoid",
        "hardsigmoid",
    }
)


def _collect_mlx_layers(nn_mod: types.ModuleType, include_nonpublic: bool) -> list[GhostRef]:
    """Collect MLX layer classes from `mlx.nn`.

    Args:
        nn_mod: Loaded `mlx.nn` module.
        include_nonpublic: Whether to include private symbols.

    Returns:
        List of `GhostRef` objects.
    """
    return [GhostInspector.inspect(obj, f"mlx.nn.{name}") for name, obj in get_all_members(nn_mod) if (include_nonpublic or not name.startswith("_")) and inspect.isclass(obj) and name[0].isupper()]


def _collect_mlx_activations(nn_mod: types.ModuleType, include_nonpublic: bool) -> list[GhostRef]:
    """Collect MLX activation functions/classes from `mlx.nn`.

    Args:
        nn_mod: Loaded `mlx.nn` module.
        include_nonpublic: Whether to include private symbols.

    Returns:
        List of `GhostRef` objects.
    """
    return [GhostInspector.inspect(obj, f"mlx.nn.{name}") for name, obj in get_all_members(nn_mod) if (include_nonpublic or not name.startswith("_")) and name.lower() in _MLX_ACTIVATION_NAMES]


def _collect_mlx_losses(nn_mod: types.ModuleType, include_nonpublic: bool) -> list[GhostRef]:
    """Collect MLX loss functions/classes from `mlx.nn.losses`.

    Args:
        nn_mod: Loaded `mlx.nn` module.
        include_nonpublic: Whether to include private symbols.

    Returns:
        List of `GhostRef` objects.
    """
    nn_members: dict[str, types.ModuleType] = dict(inspect.getmembers(nn_mod))
    losses_mod = nn_members.get("losses")
    if losses_mod is None:
        return []
    return [GhostInspector.inspect(obj, f"mlx.nn.losses.{name}") for name, obj in get_all_members(losses_mod) if (include_nonpublic or not name.startswith("_")) and (inspect.isfunction(obj) or inspect.isclass(obj)) and "loss" in name.lower()]


def _collect_mlx_optimizers(opt_mod: types.ModuleType, include_nonpublic: bool) -> list[GhostRef]:
    """Collect MLX optimizer classes from `mlx.optimizers`.

    Args:
        opt_mod: Loaded `mlx.optimizers` module.
        include_nonpublic: Whether to include private symbols.

    Returns:
        List of `GhostRef` objects.
    """
    return [GhostInspector.inspect(obj, f"mlx.optimizers.{name}") for name, obj in get_all_members(opt_mod) if inspect.isclass(obj) and (include_nonpublic or not name.startswith("_")) and name[0].isupper()]


def _collect_mlx_array_api(mlx_mod: types.ModuleType, include_nonpublic: bool) -> list[GhostRef]:
    """Collect MLX array API functions from `mlx.core` and its submodules.

    Args:
        mlx_mod: Loaded `mlx` root module.
        include_nonpublic: Whether to include private symbols.

    Returns:
        List of `GhostRef` objects.
    """
    mlx_members: dict[str, types.ModuleType] = dict(inspect.getmembers(mlx_mod))
    core_mod = mlx_members.get("core")
    if core_mod is None:
        return []
    results: list[GhostRef] = []
    core_members: dict[str, types.ModuleType] = dict(inspect.getmembers(core_mod))
    submods: list[tuple[types.ModuleType | None, str]] = [
        (core_mod, "mlx.core"),
        (core_members.get("random"), "mlx.core.random"),
        (core_members.get("linalg"), "mlx.core.linalg"),
        (core_members.get("fft"), "mlx.core.fft"),
    ]
    for smod, prefix in submods:
        if not smod:
            continue
        for name, obj in get_all_members(smod):
            if (include_nonpublic or not name.startswith("_")) and callable(obj) and not inspect.isclass(obj):
                try:
                    results.append(GhostInspector.inspect(obj, f"{prefix}.{name}"))
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    IndexError,
                    SyntaxError,
                    OSError,
                    ImportError,
                    RuntimeError,
                ):
                    pass
    return results


def _collect_live(category: SemanticTier, include_nonpublic: bool) -> list[GhostRef]:
    """Scan the live MLX library for a specific API category.

    Args:
        category: The SemanticTier enum value specifying what to scan.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of populated GhostRef objects.
    """
    if not mlx:
        return []

    results: list[GhostRef] = []
    try:
        if category == SemanticTier.LAYER:
            results = _collect_mlx_layers(mlx.nn, include_nonpublic)
        elif category == SemanticTier.ACTIVATION:
            results = _collect_mlx_activations(mlx.nn, include_nonpublic)
        elif category == SemanticTier.LOSS:
            results = _collect_mlx_losses(mlx.nn, include_nonpublic)
        elif category == SemanticTier.OPTIMIZER:
            results = _collect_mlx_optimizers(mlx.optimizers, include_nonpublic)
        elif category == SemanticTier.ARRAY_API:
            results = _collect_mlx_array_api(mlx, include_nonpublic)
    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        SyntaxError,
        OSError,
        ImportError,
        RuntimeError,
    ):
        pass

    return results


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Entrypoint to collect the MLX API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.
    """
    return _collect_live(category, include_nonpublic)
