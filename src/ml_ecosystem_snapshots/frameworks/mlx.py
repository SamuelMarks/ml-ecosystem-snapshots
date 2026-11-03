"""MLX API Snapshot Extractor.

Provides functions to dynamically introspect the MLX library and generate
GhostRefs for layers, losses, optimizers, and activations.
"""

from __future__ import annotations

import inspect
import types

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.utils import get_all_members

mlx: types.ModuleType | None = None
try:
    import mlx.core as _core  # noqa: F401
    import mlx.nn as _nn  # noqa: F401
    import mlx.optimizers as _optimizers  # noqa: F401
    import mlx as _mlx

    mlx = _mlx
except ImportError:
    mlx = None


def _collect_live(category: SemanticTier, include_nonpublic: bool) -> list[GhostRef]:
    """Scan the live MLX library for a specific API category.

    Args:
        category: The SemanticTier enum value specifying what to scan.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of populated GhostRef objects.

    """
    results: list[GhostRef] = []
    if not mlx:
        return results

    try:
        if category == SemanticTier.LAYER:
            for name, obj in get_all_members(mlx.nn):
                if (
                    (include_nonpublic or not name.startswith("_"))
                    and inspect.isclass(obj)
                    and name[0].isupper()
                ):
                    results.append(GhostInspector.inspect(obj, f"mlx.nn.{name}"))

        elif category == SemanticTier.ACTIVATION:
            target_names = {
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
            for name, obj in get_all_members(mlx.nn):
                if (
                    include_nonpublic or not name.startswith("_")
                ) and name.lower() in target_names:
                    results.append(GhostInspector.inspect(obj, f"mlx.nn.{name}"))

        elif category == SemanticTier.LOSS:
            if "losses" in dir(mlx.nn):
                for name, obj in get_all_members(mlx.nn.losses):
                    if not include_nonpublic and name.startswith("_"):
                        continue
                    if inspect.isfunction(obj) or inspect.isclass(obj):
                        if "loss" in name.lower():
                            results.append(
                                GhostInspector.inspect(obj, f"mlx.nn.losses.{name}")
                            )

        elif category == SemanticTier.OPTIMIZER:
            for name, obj in get_all_members(mlx.optimizers):
                if (
                    inspect.isclass(obj)
                    and (include_nonpublic or not name.startswith("_"))
                    and name[0].isupper()
                ):
                    results.append(
                        GhostInspector.inspect(obj, f"mlx.optimizers.{name}")
                    )
        elif category == SemanticTier.ARRAY_API:
            if "core" in dir(mlx):
                core_mod: types.ModuleType = mlx.core
                core_members: dict[str, types.ModuleType] = dict(
                    inspect.getmembers(core_mod)
                )
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
                        if (
                            (include_nonpublic or not name.startswith("_"))
                            and callable(obj)
                            and not inspect.isclass(obj)
                        ):
                            try:
                                results.append(
                                    GhostInspector.inspect(obj, f"{prefix}.{name}")
                                )
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


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Entrypoint to collect the MLX API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.

    """
    return _collect_live(category, include_nonpublic)
