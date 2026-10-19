"""Backward-compatibility shim for ml_framework_snapshots.cli.commands.capture.

Transparently re-exports all members from ml_ecosystem_snapshots.cli.commands.capture.
"""

from __future__ import annotations

from typing import Any, List
from ml_ecosystem_snapshots._alias import (
    register_alias_finder as _register_alias_finder,
)
import ml_ecosystem_snapshots.cli.commands.capture as _orig_mod
from ml_ecosystem_snapshots.cli.commands.capture import *  # noqa: F401, F403

_register_alias_finder()

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
