"""Dask API Snapshot Extractor."""

from __future__ import annotations

import types

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.frameworks._base import collect_dir_callables
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

da: types.ModuleType | None = None
try:
    import dask.array as _da

    da = _da
except Exception:
    da = None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Collect dask API.

    Args:
        category: The semantic tier to target.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of gathered API references.
    """
    results: list[GhostRef] = []

    if category == SemanticTier.ARRAY_API:
        if not da:
            return results
        submodules = [
            (da, "dask.array"),
            (getattr(da, "linalg", None), "dask.array.linalg"),
            (getattr(da, "fft", None), "dask.array.fft"),
        ]
        for submod_obj, prefix in submodules:
            if submod_obj is None:
                continue
            results.extend(
                collect_dir_callables(
                    mod_obj=submod_obj,
                    api_prefix=prefix,
                    include_nonpublic=include_nonpublic,
                    inspector_fn=GhostInspector.inspect,
                )
            )

    elif category == SemanticTier.UTIL:
        try:
            import dask as _dask_top

            results.extend(
                collect_dir_callables(
                    mod_obj=_dask_top,
                    api_prefix="dask",
                    include_nonpublic=include_nonpublic,
                    inspector_fn=GhostInspector.inspect,
                )
            )
        except Exception:
            pass

    return results
