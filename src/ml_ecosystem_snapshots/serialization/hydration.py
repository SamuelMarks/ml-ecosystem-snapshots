"""Schema hydration and v1-to-v2 migration utilities for serialized GhostRef dictionaries."""

from __future__ import annotations

from typing import Any, Dict
from ml_switcheroo_ir.schema.ghost import ExtendedGhostRef
from ml_ecosystem_snapshots.models import (
    GhostInspector,
    migrate_ghost_ref_v2 as _migrate_ghost_ref_v2,
)


def hydrate_ghost_ref(data: Dict[str, Any]) -> ExtendedGhostRef:
    """Hydrate a raw snapshot dictionary into a polymorphic `ExtendedGhostRef` instance.

    Args:
        data: Dictionary representation of a serialized GhostRef, ISA instruction, or MLIR op.

    Returns:
        Hydrated `ExtendedGhostRef` subclass instance.
    """
    return GhostInspector.hydrate(data)


def migrate_ghost_ref_v2(data: Dict[str, Any]) -> ExtendedGhostRef:
    """Upgrade a v1.x or v2.x dictionary payload into a validated v2.0.0 `ExtendedGhostRef`.

    Args:
        data: Dictionary representation of a ghost reference.

    Returns:
        Validated `ExtendedGhostRef` instance.
    """
    return _migrate_ghost_ref_v2(data)


__all__ = ["hydrate_ghost_ref", "migrate_ghost_ref_v2"]
