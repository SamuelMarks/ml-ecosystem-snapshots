"""Orbax Checkpoint API Snapshot Extractor."""

import inspect
from typing import List

from ml_ecosystem_snapshots.models import GhostInspector
from ml_ecosystem_snapshots.utils import get_all_members
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

try:
    import orbax.checkpoint as ocp
except Exception:
    ocp = None


def _scan_orbax_checkpoint(include_nonpublic: bool) -> List[GhostRef]:
    """Dynamically scans orbax.checkpoint.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found APIs.
    """
    if not ocp:
        return []

    found: List[GhostRef] = []
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
) -> List[GhostRef]:
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
