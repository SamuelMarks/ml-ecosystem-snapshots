"""Module aliasing and backward-compatibility machinery.

Provides import hooks that map `ml_framework_snapshots` submodule requests
directly to the canonical `ml_ecosystem_snapshots` singletons.
"""

from __future__ import annotations

import importlib
from importlib.abc import Loader
from importlib.machinery import ModuleSpec
import sys
from types import ModuleType
from typing import Optional, Sequence


class AliasLoader(Loader):
    """Import loader that returns an already-loaded canonical module instance.

    Attributes:
        target_mod: The canonical module instance to return.
    """

    def __init__(self, target_mod: ModuleType) -> None:
        """Initialize the alias loader with target canonical module.

        Args:
            target_mod: The canonical module instance.
        """
        self.target_mod = target_mod

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        """Return the target canonical module instance.

        Args:
            spec: Module specification descriptor.

        Returns:
            The canonical module instance.
        """
        return self.target_mod

    def exec_module(self, module: ModuleType) -> None:
        """Execute the module body (no-op since canonical module is already executed).

        Args:
            module: The module to execute.
        """
        pass


class AliasFinder:
    """Import finder intercepting `ml_framework_snapshots` lookups.

    Maps all imports under the legacy package name to their corresponding
    implementations in `ml_ecosystem_snapshots`.
    """

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[ModuleType] = None,
    ) -> Optional[ModuleSpec]:
        """Find the module specification for an import request.

        Args:
            fullname: Fully qualified name of the module to find.
            path: Search paths for package submodules.
            target: Target module object (if reloading).

        Returns:
            ModuleSpec pointing to AliasLoader, or None if not matching legacy package.
        """
        if fullname == "ml_framework_snapshots" or fullname.startswith(
            "ml_framework_snapshots."
        ):
            target_name = fullname.replace(
                "ml_framework_snapshots", "ml_ecosystem_snapshots", 1
            )
            target_mod = importlib.import_module(target_name)
            return ModuleSpec(fullname, AliasLoader(target_mod))
        return None


def register_alias_finder() -> None:
    """Register the AliasFinder into sys.meta_path if not already present."""
    for finder in sys.meta_path:
        if isinstance(finder, AliasFinder):
            return
    sys.meta_path.insert(0, AliasFinder())
