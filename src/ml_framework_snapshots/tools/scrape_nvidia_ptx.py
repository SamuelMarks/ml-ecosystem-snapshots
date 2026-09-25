"""Backward-compatibility shim for ml_framework_snapshots.tools.scrape_nvidia_ptx.

Transparently re-exports all members from ml_ecosystem_snapshots.tools.scrape_nvidia_ptx.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
import ml_ecosystem_snapshots.tools.scrape_nvidia_ptx as _orig_mod

if TYPE_CHECKING:
    from ml_ecosystem_snapshots.tools.scrape_nvidia_ptx import *  # noqa: F401, F403

# Re-export all attributes including internal and dunder methods
for _k in dir(_orig_mod):
    globals()[_k] = getattr(_orig_mod, _k)

__all__ = getattr(
    _orig_mod,
    "__all__",
    [k for k in dir(_orig_mod) if not k.startswith("_")],
)

sys.modules[__name__] = _orig_mod
