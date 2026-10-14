"""Backward-compatibility shim for ml_framework_snapshots.frameworks.latex_dsl.

Transparently re-exports all members from ml_ecosystem_snapshots.frameworks.latex_dsl.
"""

from __future__ import annotations

from typing import Any, List
import ml_ecosystem_snapshots.frameworks.latex_dsl as _orig_mod
from ml_ecosystem_snapshots.frameworks.latex_dsl import *  # noqa: F401, F403

__all__: List[str] = getattr(
    _orig_mod,
    "__all__",
    [k for k in dir(_orig_mod) if not k.startswith("_")],
)


def __getattr__(name: str) -> Any:
    """Delegate attribute access to the canonical module.

    Args:
        name: Attribute name to look up.

    Returns:
        The attribute value from the canonical module.
    """
    return getattr(_orig_mod, name)


def __dir__() -> List[str]:
    """Return the list of attributes exposed by the canonical module.

    Returns:
        Sorted list of attribute names.
    """
    return dir(_orig_mod)
