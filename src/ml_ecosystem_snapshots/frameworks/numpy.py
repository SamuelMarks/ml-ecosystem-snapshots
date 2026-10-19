"""Numpy API Snapshot Extractor."""

from __future__ import annotations

import inspect
import types

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

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
        for name in ["tanh", "exp", "maximum", "minimum"]:
            if hasattr(np, name):
                obj = getattr(np, name)
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
                        Exception,
                    ):
                        pass
    elif category == SemanticTier.ARRAY_API:
        for name in dir(np):
            if not name.startswith("_"):
                obj = getattr(np, name, None)
                if callable(obj) and not inspect.isclass(obj):
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
                        Exception,
                    ):
                        pass

        # Introspect numpy.linalg.*
        if hasattr(np, "linalg"):
            for name in dir(np.linalg):
                if not name.startswith("_"):
                    obj = getattr(np.linalg, name)
                    if callable(obj) and not inspect.isclass(obj):
                        try:
                            results.append(
                                GhostInspector.inspect(
                                    obj, f"numpy.linalg.{name}", is_public=True
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
                            Exception,
                        ):
                            pass

        # Introspect numpy.fft.*
        if hasattr(np, "fft"):
            for name in dir(np.fft):
                if not name.startswith("_"):
                    obj = getattr(np.fft, name)
                    if callable(obj) and not inspect.isclass(obj):
                        try:
                            results.append(
                                GhostInspector.inspect(
                                    obj, f"numpy.fft.{name}", is_public=True
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
                            Exception,
                        ):
                            pass
    return results
