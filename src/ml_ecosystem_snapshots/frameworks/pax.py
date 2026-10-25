"""Pax API Snapshot Extractor."""

from __future__ import annotations

import inspect
import types

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.utils import get_all_members

paxml: types.ModuleType | None = None
praxis: types.ModuleType | None = None
layers: types.ModuleType | None = None
try:
    import paxml as _paxml_mod
    import praxis as _praxis_mod
    from praxis import layers as _layers_mod

    paxml = _paxml_mod
    praxis = _praxis_mod
    layers = _layers_mod
except Exception:
    paxml = None
    praxis = None
    layers = None


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Collect pax API.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of GhostRefs.
    """
    results: list[GhostRef] = []
    if not layers:
        return results

    if category == SemanticTier.NEURAL:
        for name, obj in get_all_members(layers):
            if not include_nonpublic and name.startswith("_"):
                continue
            if inspect.isclass(obj):
                try:
                    results.append(GhostInspector.inspect(obj, f"praxis.layers.{name}"))
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
