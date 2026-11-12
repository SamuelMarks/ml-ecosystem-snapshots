"""Live, static Griffe, and canonical fallback scanner for Optax."""

from __future__ import annotations

import inspect
import logging
import types
import typing

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

import ml_ecosystem_snapshots.frameworks.optax as _optax_pkg
from ml_ecosystem_snapshots.frameworks._base import (
    CanonicalPythonOpSpec,
    build_canonical_python_ref,
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
from ml_ecosystem_snapshots.introspection.runtime_boundary import get_member_dict
from ml_ecosystem_snapshots.introspection.static_parsers import GriffeCacheEntry
from ml_ecosystem_snapshots.network_env import is_offline_mode
from ml_ecosystem_snapshots.utils import get_all_members

_LOGGER = logging.getLogger(__name__)


def _get_optax_griffe_module() -> GriffeCacheEntry:
    """Statically load `optax` via the custom Griffe fork when not in offline mode.

    Returns:
        Loaded Griffe AST node for `optax` or `None` if unavailable or in offline mode.
    """
    if is_offline_mode():
        return None
    return _optax_pkg.load_griffe_module_safely("optax", framework_name="optax")


def _hydrate_canonical_optax_tier(
    specs: typing.Sequence[CanonicalPythonOpSpec],
    attach_root_alias: bool = False,
) -> list[GhostRef]:
    """Hydrate a sequence of `CanonicalPythonOpSpec` items into `GhostRef` records.

    Args:
        specs: Canonical Optax operation specifications.
        attach_root_alias: Whether to attach `optax.<name>` alias for `_optax_pkg.optax.losses.*` or `_optax_pkg.optax.schedules.*`.

    Returns:
        List of hydrated `GhostRef` objects.
    """
    refs: list[GhostRef] = []
    for spec in specs:
        ref = build_canonical_python_ref(
            spec,
            default_doc_prefix="Optax",
            environment_tags=("optax", "jax"),
        )
        if attach_root_alias and (
            ref.api_path.startswith("optax.losses.") or ref.api_path.startswith("optax.schedules.")
        ):
            ref.aliases = [f"optax.{ref.name}"]
        refs.append(ref)
    return refs


@typing.runtime_checkable
class _OptaxWithLossesProtocol(typing.Protocol):
    """Protocol for `optax` exposing `.losses`."""

    losses: types.ModuleType


@typing.runtime_checkable
class _OptaxWithSchedulesProtocol(typing.Protocol):
    """Protocol for `optax` exposing `.schedules`."""

    schedules: types.ModuleType


class OptaxScanner:
    """Helper to inspect Optax APIs for the Ghost Protocol discovery system."""

    @staticmethod
    def scan_optimizers(include_nonpublic: bool = False) -> list[GhostRef]:
        """Scan the Optax public API for optimizer and gradient transformation functions.

        Args:
            include_nonpublic: Whether to include non-public APIs.

        Returns:
            A list of `GhostRef` objects representing found optimizers.
        """
        if is_offline_mode():
            return _hydrate_canonical_optax_tier(CANONICAL_OPTAX_OPTIMIZERS)

        if not _optax_pkg.optax:
            return []

        results: list[GhostRef] = []
        for name, obj in get_all_members(_optax_pkg.optax):
            if not include_nonpublic and name.startswith("_"):
                continue

            if inspect.isfunction(obj) or inspect.isclass(obj):
                is_known = name.lower() in KNOWN_OPTAX_OPTIMIZERS
                is_suffixed = name.endswith("_optimizer")

                if is_known or is_suffixed:
                    try:
                        ref = _optax_pkg.GhostInspector.inspect(obj, f"optax.{name}")
                        results.append(ref)
                    except (
                        ImportError,
                        AttributeError,
                        TypeError,
                        ValueError,
                        OSError,
                        RuntimeError,
                        LookupError,
                        SyntaxError,
                        AssertionError,
                    ):
                        _LOGGER.debug("Failed to inspect optax.%s", name, exc_info=True)

        return results

    @staticmethod
    def scan_losses(include_nonpublic: bool = False) -> list[GhostRef]:
        """Scan `optax.losses` for loss functions.

        Args:
            include_nonpublic: Whether to include non-public APIs.

        Returns:
            A list of `GhostRef` objects representing found losses.
        """
        if is_offline_mode():
            return _hydrate_canonical_optax_tier(CANONICAL_OPTAX_LOSSES, attach_root_alias=True)

        if not isinstance(_optax_pkg.optax, _OptaxWithLossesProtocol):
            return []

        results: list[GhostRef] = []
        for name, obj in get_all_members(_optax_pkg.optax.losses):
            if not include_nonpublic and name.startswith("_"):
                continue

            if inspect.isfunction(obj):
                name_lower = name.lower()
                if (
                    "loss" in name_lower
                    or "error" in name_lower
                    or "entropy" in name_lower
                    or name_lower in KNOWN_OPTAX_LOSSES
                ):
                    try:
                        ref = _optax_pkg.GhostInspector.inspect(obj, f"optax.losses.{name}")
                        results.append(ref)
                    except (
                        ImportError,
                        AttributeError,
                        TypeError,
                        ValueError,
                        OSError,
                        RuntimeError,
                        LookupError,
                        SyntaxError,
                        AssertionError,
                    ):
                        _LOGGER.debug("Failed to inspect optax.losses.%s", name, exc_info=True)

        return results

    @staticmethod
    def scan_schedulers(include_nonpublic: bool = False) -> list[GhostRef]:
        """Scan `optax.schedules` for scheduler functions.

        Args:
            include_nonpublic: Whether to include non-public APIs.

        Returns:
            A list of `GhostRef` objects representing found schedulers.
        """
        if is_offline_mode():
            return _hydrate_canonical_optax_tier(CANONICAL_OPTAX_SCHEDULERS, attach_root_alias=True)

        if not isinstance(_optax_pkg.optax, _OptaxWithSchedulesProtocol):
            return []

        results: list[GhostRef] = []
        for name, obj in get_all_members(_optax_pkg.optax.schedules):
            if not include_nonpublic and name.startswith("_"):
                continue

            if inspect.isfunction(obj):
                try:
                    ref = _optax_pkg.GhostInspector.inspect(obj, f"optax.schedules.{name}")
                    results.append(ref)
                except (
                    ImportError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    OSError,
                    RuntimeError,
                    LookupError,
                    SyntaxError,
                    AssertionError,
                ):
                    _LOGGER.debug("Failed to inspect optax.schedules.%s", name, exc_info=True)

        return results

    @staticmethod
    def _scan_live_array_submodules(
        optax_mod: types.ModuleType,
        include_nonpublic: bool,
        results: list[GhostRef],
        seen_paths: set[str],
    ) -> None:
        """Scan Optax utility submodules for function members and append them to results.

        Args:
            optax_mod: Live imported `optax` module object.
            include_nonpublic: Whether to include private underscore-prefixed functions.
            results: Accumulator list of discovered `GhostRef` objects.
            seen_paths: Set of already recorded canonical API paths.
        """
        optax_members = get_member_dict(optax_mod)
        for submod_name in (
            "projections",
            "tree_utils",
            "second_order",
            "assignment",
            "perturbations",
        ):
            submod = optax_members.get(submod_name)
            if isinstance(submod, types.ModuleType):
                for name, obj in get_all_members(submod):
                    if not include_nonpublic and name.startswith("_"):
                        continue
                    if inspect.isfunction(obj):
                        api_path = f"optax.{submod_name}.{name}"
                        try:
                            ref = _optax_pkg.GhostInspector.inspect(obj, api_path)
                            results.append(ref)
                            seen_paths.add(api_path)
                        except (
                            ImportError,
                            AttributeError,
                            TypeError,
                            ValueError,
                            OSError,
                            RuntimeError,
                            LookupError,
                            SyntaxError,
                            AssertionError,
                        ):
                            _LOGGER.debug("Failed to inspect %s", api_path, exc_info=True)

    @staticmethod
    def _scan_live_array_root(
        optax_mod: types.ModuleType,
        include_nonpublic: bool,
        results: list[GhostRef],
        seen_paths: set[str],
    ) -> None:
        """Scan top-level `optax` module and utility submodules for array API functions.

        Args:
            optax_mod: Live imported `optax` module object.
            include_nonpublic: Whether to include private underscore-prefixed functions.
            results: Accumulator list of discovered `GhostRef` objects.
            seen_paths: Set of already recorded canonical API paths.
        """
        for name, obj in get_all_members(optax_mod):
            if not include_nonpublic and name.startswith("_"):
                continue
            if inspect.isfunction(obj) and name.lower() in KNOWN_OPTAX_ARRAY_API:
                try:
                    ref = _optax_pkg.GhostInspector.inspect(obj, f"optax.{name}")
                    results.append(ref)
                    seen_paths.add(ref.api_path)
                except (
                    ImportError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    OSError,
                    RuntimeError,
                    LookupError,
                    SyntaxError,
                    AssertionError,
                ):
                    _LOGGER.debug("Failed to inspect optax.%s", name, exc_info=True)
        OptaxScanner._scan_live_array_submodules(optax_mod, include_nonpublic, results, seen_paths)

    @staticmethod
    def scan_array_api(include_nonpublic: bool = False) -> list[GhostRef]:
        """Scan Optax parameter update, projection, tree_utils, second_order, assignment, and perturbation APIs.

        Args:
            include_nonpublic: Whether to include non-public APIs.

        Returns:
            A list of `GhostRef` objects representing Optax utility operations.
        """
        if is_offline_mode():
            return _hydrate_canonical_optax_tier(CANONICAL_OPTAX_ARRAY_API)

        results: list[GhostRef] = []
        seen_paths: set[str] = set()

        if _optax_pkg.optax is not None:
            OptaxScanner._scan_live_array_root(_optax_pkg.optax, include_nonpublic, results, seen_paths)

        if _optax_pkg._get_optax_griffe_module() is not None:
            for griffe_ref in _optax_pkg.scan_griffe_module_members(
                module_path="optax",
                api_prefix="optax",
                include_nonpublic=include_nonpublic,
                environment_tags=("optax", "jax"),
                framework_name="optax",
                inspector_fn=_optax_pkg.GhostInspector.inspect,
            ):
                if griffe_ref.name.lower() in KNOWN_OPTAX_ARRAY_API and griffe_ref.api_path not in seen_paths:
                    results.append(griffe_ref)
                    seen_paths.add(griffe_ref.api_path)

        for spec in CANONICAL_OPTAX_ARRAY_API:
            if spec.api_path not in seen_paths:
                results.append(
                    build_canonical_python_ref(
                        spec,
                        default_doc_prefix="Optax",
                        environment_tags=("optax", "jax"),
                    )
                )
                seen_paths.add(spec.api_path)

        return results


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect Optax APIs for the requested semantic tier.

    Args:
        category: The semantic tier to target (`OPTIMIZER`, `LOSS`, `SCHEDULER`, or `ARRAY_API`).
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        List of gathered `GhostRef` API references.
    """
    if category == SemanticTier.OPTIMIZER:
        return OptaxScanner.scan_optimizers(include_nonpublic)
    elif category == SemanticTier.LOSS:
        return OptaxScanner.scan_losses(include_nonpublic)
    elif category == SemanticTier.SCHEDULER:
        return OptaxScanner.scan_schedulers(include_nonpublic)
    elif category == SemanticTier.ARRAY_API:
        return OptaxScanner.scan_array_api(include_nonpublic)
    return []


__all__ = [
    "OptaxScanner",
    "_get_optax_griffe_module",
    "_hydrate_canonical_optax_tier",
    "collect_api",
]
