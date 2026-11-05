"""Numpy API Snapshot Extractor."""

from __future__ import annotations

import inspect
import types
import typing

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import InspectableTarget


@typing.runtime_checkable
class _NumpyWithLinalgProtocol(typing.Protocol):
    """Protocol for `numpy` exposing `.linalg`."""

    linalg: types.ModuleType


@typing.runtime_checkable
class _NumpyWithFftProtocol(typing.Protocol):
    """Protocol for `numpy` exposing `.fft`."""

    fft: types.ModuleType


np: types.ModuleType | None = None
try:
    import numpy as _np

    np = _np
except ImportError:
    np = None


def _collect_numpy_callables(mod: types.ModuleType, prefix: str, target_names: list[str] | None = None) -> list[GhostRef]:
    """Collect public callable functions from a NumPy module or submodule.

    Args:
        mod: NumPy module or submodule (`numpy`, `numpy.linalg`, `numpy.fft`).
        prefix: Qualified API path prefix.
        target_names: Optional explicit list of function names to filter by.

    Returns:
        List of inspected `GhostRef` items.
    """
    results: list[GhostRef] = []
    members: dict[str, InspectableTarget] = dict(inspect.getmembers(mod))
    names = target_names if target_names is not None else sorted(members.keys())
    for name in names:
        if name.startswith("_") or name not in members:
            continue
        obj = members[name]
        if callable(obj) and (target_names is not None or not inspect.isclass(obj)):
            try:
                results.append(GhostInspector.inspect(obj, f"{prefix}.{name}", is_public=True))
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
    """Collect numpy API.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRefs.
    """
    if not np:
        return []

    if category == SemanticTier.ACTIVATION:
        return _collect_numpy_callables(np, "numpy", target_names=["tanh", "exp", "maximum", "minimum"])

    if category == SemanticTier.ARRAY_API:
        results = _collect_numpy_callables(np, "numpy")
        if isinstance(np, _NumpyWithLinalgProtocol):
            results.extend(_collect_numpy_callables(np.linalg, "numpy.linalg"))
        if isinstance(np, _NumpyWithFftProtocol):
            results.extend(_collect_numpy_callables(np.fft, "numpy.fft"))
        return results

    return []
