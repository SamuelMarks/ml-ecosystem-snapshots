"""Optax API Snapshot Extractor subpackage."""

from __future__ import annotations

import importlib
import logging
import os
import types

from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalPythonOpSpec,
    scan_griffe_module_members,
)
from ml_ecosystem_snapshots.frameworks.optax._array_api import (
    CANONICAL_OPTAX_ARRAY_API,
    KNOWN_OPTAX_ARRAY_API,
)
from ml_ecosystem_snapshots.frameworks.optax._losses import (
    CANONICAL_OPTAX_LOSSES,
    KNOWN_OPTAX_LOSSES,
)
from ml_ecosystem_snapshots.frameworks.optax._schedules import (
    CANONICAL_OPTAX_SCHEDULERS,
)
from ml_ecosystem_snapshots.frameworks.optax._transforms import (
    CANONICAL_OPTAX_OPTIMIZERS,
    KNOWN_OPTAX_OPTIMIZERS,
)
from ml_ecosystem_snapshots.frameworks.optax.scanner import (
    OptaxScanner,
    _get_optax_griffe_module,
    _hydrate_canonical_optax_tier,
    collect_api,
)
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.static_parsers import load_griffe_module_safely

_LOGGER = logging.getLogger(__name__)


def _try_import_live_optax() -> types.ModuleType | None:
    """Attempt to import live `optax` unless offline mode is active.

    Returns:
        Imported `optax` module or `None` when offline or unavailable.
    """
    if os.environ.get("ML_SNAPSHOTS_OFFLINE", "").lower() in ("1", "true", "yes"):
        return None
    try:
        return importlib.import_module("optax")
    except (
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
        LookupError,
        SyntaxError,
    ) as _optax_exc:
        _LOGGER.debug("Live optax import unavailable, using static/canonical tiers: %s", _optax_exc)
        return None


optax: types.ModuleType | None = _try_import_live_optax()

CANONICAL_OPTAX_OPS: list[CanonicalPythonOpSpec] = (
    CANONICAL_OPTAX_OPTIMIZERS + CANONICAL_OPTAX_LOSSES + CANONICAL_OPTAX_SCHEDULERS + CANONICAL_OPTAX_ARRAY_API
)

__all__ = [
    "CANONICAL_OPTAX_ARRAY_API",
    "CANONICAL_OPTAX_LOSSES",
    "CANONICAL_OPTAX_OPS",
    "CANONICAL_OPTAX_OPTIMIZERS",
    "CANONICAL_OPTAX_SCHEDULERS",
    "GhostInspector",
    "KNOWN_OPTAX_ARRAY_API",
    "KNOWN_OPTAX_LOSSES",
    "KNOWN_OPTAX_OPTIMIZERS",
    "OptaxScanner",
    "_get_optax_griffe_module",
    "_hydrate_canonical_optax_tier",
    "collect_api",
    "load_griffe_module_safely",
    "optax",
    "scan_griffe_module_members",
]
