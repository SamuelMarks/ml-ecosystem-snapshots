"""Dask API Snapshot Extractor."""

from __future__ import annotations

import logging
import types

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.frameworks._base import collect_dir_callables
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import get_member_dict

_LOGGER = logging.getLogger(__name__)

da: types.ModuleType | None = None
try:
    import dask.array as _da

    da = _da
except (
    ImportError,
    AttributeError,
    TypeError,
    ValueError,
    OSError,
    RuntimeError,
    LookupError,
    SyntaxError,
    AssertionError,
):
    _LOGGER.debug("Optional dask.array module unavailable", exc_info=True)
    da = None


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
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
        da_members = get_member_dict(da)
        linalg_mod = da_members.get("linalg")
        fft_mod = da_members.get("fft")
        submodules: list[tuple[types.ModuleType | None, str]] = [
            (da, "dask.array"),
            (linalg_mod if isinstance(linalg_mod, types.ModuleType) else None, "dask.array.linalg"),
            (fft_mod if isinstance(fft_mod, types.ModuleType) else None, "dask.array.fft"),
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
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            OSError,
            RuntimeError,
            LookupError,
            SyntaxError,
            AssertionError,
        ):
            _LOGGER.debug("Failed to collect dask top-level callables", exc_info=True)

    return results
