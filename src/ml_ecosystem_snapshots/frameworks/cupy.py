"""CuPy API Snapshot Extractor."""

from typing import List
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier
from ml_ecosystem_snapshots.models import GhostInspector

import typing

try:
    import cupy as _cp

    cp: typing.Any = _cp
except Exception:
    cp = None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect cupy API.

    Args:
        category: The semantic tier to target.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of gathered API references.
    """
    results: List[GhostRef] = []
    if not cp:
        return results

    if category == SemanticTier.ACTIVATION:
        for name in ["tanh", "exp", "maximum", "minimum"]:
            if hasattr(cp, name):
                obj = getattr(cp, name)
                if callable(obj):
                    try:
                        res = GhostInspector.inspect(
                            obj, f"cupy.{name}", is_public=True
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
        from ml_ecosystem_snapshots.frameworks._base import collect_dir_callables

        results.extend(
            collect_dir_callables(
                mod_obj=cp,
                api_prefix="cupy",
                include_nonpublic=include_nonpublic,
                inspector_fn=GhostInspector.inspect,
            )
        )

    return results
