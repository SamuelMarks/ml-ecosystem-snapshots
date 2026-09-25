"""Dask API Snapshot Extractor."""

from typing import List
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo_ir.schema.ghost import GhostRef
from ..models import GhostInspector

from typing import Any

try:
    import dask.array as _da

    da: Any = _da
except (ImportError, Exception):
    da = None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect dask API.

    Args:
        category: The semantic tier to target.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of gathered API references.
    """
    results: List[GhostRef] = []

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
            for name in dir(submod_obj):
                if not include_nonpublic and name.startswith("_"):
                    continue
                obj = getattr(submod_obj, name)
                if callable(obj):
                    try:
                        res = GhostInspector.inspect(
                            obj, f"{prefix}.{name}", is_public=not name.startswith("_")
                        )
                        results.append(res)
                    except Exception:
                        pass

    elif category == SemanticTier.UTIL:
        try:
            import dask as _dask_top

            for name in dir(_dask_top):
                if not include_nonpublic and name.startswith("_"):
                    continue
                obj = getattr(_dask_top, name)
                if callable(obj):
                    try:
                        res = GhostInspector.inspect(
                            obj, f"dask.{name}", is_public=not name.startswith("_")
                        )
                        results.append(res)
                    except Exception:
                        pass
        except (ImportError, Exception):
            pass

    return results
