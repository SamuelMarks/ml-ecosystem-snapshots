"""Serialization and schema hydration subpackage for Ghost snapshots."""

from __future__ import annotations

from ml_ecosystem_snapshots.serialization.hydration import (
    hydrate_ghost_ref,
    migrate_ghost_ref_v2,
)

__all__ = ["hydrate_ghost_ref", "migrate_ghost_ref_v2"]
