"""Orbax Checkpoint API Snapshot Extractor."""

from __future__ import annotations

import inspect
import types

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.utils import get_all_members

ocp: types.ModuleType | None = None
try:
    import orbax.checkpoint as _ocp_mod

    ocp = _ocp_mod
except Exception:
    pass


def _scan_orbax_checkpoint(include_nonpublic: bool) -> list[GhostRef]:
    """Dynamically scans orbax.checkpoint.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found APIs.
    """
    if not ocp:
        return []

    found: list[GhostRef] = []
    try:
        for name, obj in get_all_members(ocp):
            if inspect.isclass(obj) or inspect.isfunction(obj):
                if not include_nonpublic and name.startswith("_"):
                    continue
                try:
                    found.append(
                        GhostInspector.inspect(obj, f"orbax.checkpoint.{name}")
                    )
                except Exception:
                    pass
    except Exception:
        pass

    return found


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Entrypoint to collect the Orbax Checkpoint API signature.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.
    """
    if category == SemanticTier.ARRAY_API:
        return _scan_orbax_checkpoint(include_nonpublic)

    return []
