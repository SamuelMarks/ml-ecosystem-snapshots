"""Numpy API Snapshot Extractor."""

from __future__ import annotations

import inspect
import operator
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


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Collect numpy API.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRefs.
    """
    results: list[GhostRef] = []
    if not np:
        return results

    if category == SemanticTier.ACTIVATION:
        # Numpy doesn't have an "activation" module per se, but we can capture basic math ops
        np_names = set(dir(np))
        for name in ["tanh", "exp", "maximum", "minimum"]:
            if name in np_names:
                obj: InspectableTarget = operator.attrgetter(name)(np)
                if callable(obj):
                    try:
                        res = GhostInspector.inspect(
                            obj, f"numpy.{name}", is_public=True
                        )
                        results.append(res)
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
    elif category == SemanticTier.ARRAY_API:
        for name in dir(np):
            if not name.startswith("_"):
                obj_arr: InspectableTarget = operator.attrgetter(name)(np)
                if callable(obj_arr) and not inspect.isclass(obj_arr):
                    try:
                        res = GhostInspector.inspect(
                            obj_arr, f"numpy.{name}", is_public=True
                        )
                        results.append(res)
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

        # Introspect numpy.linalg.*
        if isinstance(np, _NumpyWithLinalgProtocol):
            for name in dir(np.linalg):
                if not name.startswith("_"):
                    obj_la: InspectableTarget = operator.attrgetter(name)(np.linalg)
                    if callable(obj_la) and not inspect.isclass(obj_la):
                        try:
                            results.append(
                                GhostInspector.inspect(
                                    obj_la, f"numpy.linalg.{name}", is_public=True
                                )
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

        # Introspect numpy.fft.*
        if isinstance(np, _NumpyWithFftProtocol):
            for name in dir(np.fft):
                if not name.startswith("_"):
                    obj_fft: InspectableTarget = operator.attrgetter(name)(np.fft)
                    if callable(obj_fft) and not inspect.isclass(obj_fft):
                        try:
                            results.append(
                                GhostInspector.inspect(
                                    obj_fft, f"numpy.fft.{name}", is_public=True
                                )
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
    return results
