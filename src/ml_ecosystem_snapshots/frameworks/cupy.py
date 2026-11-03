"""CuPy API Snapshot Extractor."""

from __future__ import annotations

import operator
import types

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import InspectableTarget

cp: types.ModuleType | None = None
try:
    import cupy as _cp

    cp = _cp
except Exception:
    cp = None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Collect cupy API.

    Args:
        category: The semantic tier to target.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of gathered API references.
    """
    results: list[GhostRef] = []
    if not cp:
        return results

    if category == SemanticTier.ACTIVATION:
        cp_names = set(dir(cp))
        for name in ["tanh", "exp", "maximum", "minimum"]:
            if name in cp_names:
                obj: InspectableTarget = operator.attrgetter(name)(cp)
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
