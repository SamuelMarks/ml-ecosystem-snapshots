"""JAX API Snapshot Extractor.

Provides functions to dynamically introspect the JAX and Optax libraries and
generate GhostRefs for activations, losses, and optimizers.
"""

from __future__ import annotations

import inspect
import logging
import types
import typing

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.frameworks.optax import OptaxScanner
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
    get_member_dict,
)
from ml_ecosystem_snapshots.utils import get_all_members

_LOGGER = logging.getLogger(__name__)


@typing.runtime_checkable
class _JaxPrivateStaticNumProtocol(typing.Protocol):
    """Protocol for JAX JIT functions exposing `._static_argnums`."""

    _static_argnums: typing.Sequence[int] | set[int] | frozenset[int] | None


@typing.runtime_checkable
class _JaxPublicStaticNumProtocol(typing.Protocol):
    """Protocol for JAX JIT functions exposing `.static_argnums`."""

    static_argnums: typing.Sequence[int] | set[int] | frozenset[int] | None


@typing.runtime_checkable
class _JaxPrivateStaticNameProtocol(typing.Protocol):
    """Protocol for JAX JIT functions exposing `._static_argnames`."""

    _static_argnames: typing.Sequence[str] | set[str] | frozenset[str] | None


@typing.runtime_checkable
class _JaxPublicStaticNameProtocol(typing.Protocol):
    """Protocol for JAX JIT functions exposing `.static_argnames`."""

    static_argnames: typing.Sequence[str] | set[str] | frozenset[str] | None


@typing.runtime_checkable
class _JaxWithLaxProtocol(typing.Protocol):
    """Protocol for `jax` exposing `.lax`."""

    lax: types.ModuleType


@typing.runtime_checkable
class _JaxWithRandomProtocol(typing.Protocol):
    """Protocol for `jax` exposing `.random`."""

    random: types.ModuleType


@typing.runtime_checkable
class _JaxWithArrayProtocol(typing.Protocol):
    """Protocol for `jax` exposing `.Array`."""

    Array: type


@typing.runtime_checkable
class _JaxWithShardingProtocol(typing.Protocol):
    """Protocol for `jax` exposing `.sharding`."""

    sharding: types.ModuleType


jax: types.ModuleType | None = None
try:
    import jax as _jax

    jax = _jax
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
    _LOGGER.debug("Optional jax module unavailable", exc_info=True)
    jax = None


def _scan_jax_activations(include_nonpublic: bool) -> list[GhostRef]:
    """Dynamically scans jax.nn for activation-like functions.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found activation functions.

    """
    if jax is None:
        return []
    found: list[GhostRef] = []
    try:
        import jax.nn as jax_nn

        for name, obj in get_all_members(jax_nn):
            if not include_nonpublic and name.startswith("_"):
                continue
            if inspect.isfunction(obj):
                found.append(GhostInspector.inspect(obj, f"jax.nn.{name}"))
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
        _LOGGER.debug("Failed to scan jax.nn activations", exc_info=True)
    return found


def _scan_jax_initializers(include_nonpublic: bool) -> list[GhostRef]:
    """Dynamically scans jax.nn.initializers for initializer functions.

    Args:
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef objects representing found initializer functions.

    """
    if jax is None:
        return []
    found: list[GhostRef] = []
    try:
        import jax.nn.initializers as jax_init

        for name, obj in get_all_members(jax_init):
            if not include_nonpublic and name.startswith("_"):
                continue
            if inspect.isfunction(obj):
                found.append(GhostInspector.inspect(obj, f"jax.nn.initializers.{name}"))
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
        _LOGGER.debug("Failed to scan jax.nn.initializers", exc_info=True)
    return found


def attach_jax_static_arg_metadata(ref: GhostRef, obj: LivePythonTarget) -> GhostRef:
    """Attach static argument numbers and names from JIT-compiled functions as metadata.

    Args:
        ref: GhostRef representation of function.
        obj: Live JAX function or JIT wrapper.

    Returns:
        Augmented GhostRef with static argument metadata.
    """
    priv_nums = obj._static_argnums if isinstance(obj, _JaxPrivateStaticNumProtocol) else None
    pub_nums = obj.static_argnums if isinstance(obj, _JaxPublicStaticNumProtocol) else None
    static_nums = priv_nums or pub_nums

    priv_names = obj._static_argnames if isinstance(obj, _JaxPrivateStaticNameProtocol) else None
    pub_names = obj.static_argnames if isinstance(obj, _JaxPublicStaticNameProtocol) else None
    static_names = priv_names or pub_names

    if isinstance(static_nums, (list, tuple, set, frozenset)) and static_nums:
        if ref.environment_tags is None:
            ref.environment_tags = []
        ref.environment_tags.append(f"static_argnums:{','.join(map(str, sorted(list(static_nums))))}")
        for idx, p in enumerate(ref.params):
            if idx in static_nums:
                p.standardized_name = "static_arg"

    if isinstance(static_names, (list, tuple, set, frozenset)) and static_names:
        if ref.environment_tags is None:
            ref.environment_tags = []
        ref.environment_tags.append(f"static_argnames:{','.join(map(str, sorted(list(static_names))))}")
        for p in ref.params:
            if p.name in static_names:
                p.standardized_name = "static_arg"

    return ref


_attach_jax_static_arg_metadata = attach_jax_static_arg_metadata


def _collect_jax_namespace(
    ns_obj: types.ModuleType | type,
    api_prefix: str,
    include_nonpublic: bool,
    attach_static_args: bool = False,
) -> list[GhostRef]:
    """Collect callable non-class members from a JAX submodule namespace.

    Args:
        ns_obj: Target JAX submodule object (e.g. `jax.numpy`, `jax.lax`, `jax.random`).
        api_prefix: Qualified API path prefix.
        include_nonpublic: Whether to include private underscore-prefixed symbols.
        attach_static_args: If True, attach static arg metadata via `_attach_jax_static_arg_metadata`.

    Returns:
        List of inspected `GhostRef` objects.
    """
    collected: list[GhostRef] = []
    for name, obj in get_all_members(ns_obj):
        if (include_nonpublic or not name.startswith("_")) and callable(obj) and not inspect.isclass(obj):
            try:
                ref: GhostRef = GhostInspector.inspect(obj, f"{api_prefix}.{name}")
                if attach_static_args:
                    ref = _attach_jax_static_arg_metadata(ref, obj)
                collected.append(ref)
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
                _LOGGER.debug("Failed to inspect JAX callable %s.%s", api_prefix, name, exc_info=True)
    return collected


_JAX_INSPECT_ERRORS: tuple[type[Exception], ...] = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    IndexError,
    SyntaxError,
    OSError,
    ImportError,
    RuntimeError,
)


def _collect_jax_member_callables(
    target: types.ModuleType | type,
    api_prefix: str,
    include_nonpublic: bool,
    exclude_classes: bool = False,
    kind: str | None = None,
) -> list[GhostRef]:
    """Inspect callable members of a JAX class or module into `GhostRef` records.

    Args:
        target: Module or type whose `inspect.getmembers` entries will be scanned.
        api_prefix: Qualified API path prefix (e.g. `'jax.Array'` or `'jax.sharding'`).
        include_nonpublic: Whether to include underscore-prefixed private members.
        exclude_classes: If True, skip members where `inspect.isclass(obj)` is True.
        kind: Optional override for `GhostInspector.inspect(..., kind=kind)`.

    Returns:
        List of inspected `GhostRef` items.
    """
    collected: list[GhostRef] = []
    for name, obj in get_member_dict(target).items():
        if not include_nonpublic and name.startswith("_"):
            continue
        if callable(obj) and (not exclude_classes or not inspect.isclass(obj)):
            try:
                if kind is not None:
                    collected.append(GhostInspector.inspect(obj, f"{api_prefix}.{name}", kind=kind))
                else:
                    collected.append(GhostInspector.inspect(obj, f"{api_prefix}.{name}"))
            except _JAX_INSPECT_ERRORS:
                pass
    return collected


def _collect_jax_pallas(include_nonpublic: bool) -> list[GhostRef]:
    """Collect custom GPU/TPU kernel primitives from `jax.experimental.pallas`.

    Args:
        include_nonpublic: Whether to include underscore-prefixed private members.

    Returns:
        List of `GhostRef` items from `jax.experimental.pallas`.
    """
    try:
        import jax.experimental.pallas as pallas

        return _collect_jax_member_callables(pallas, "jax.experimental.pallas", include_nonpublic)
    except _JAX_INSPECT_ERRORS:
        return []


def _scan_array_api(include_nonpublic: bool) -> list[GhostRef]:
    """Scan JAX for array API functions, primitives, random generation, and transforms.

    Args:
        include_nonpublic: Include non-public APIs.

    Returns:
        List of GhostRefs.
    """
    found: list[GhostRef] = []
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np

        # 1. Exhaustive jax.numpy.*
        found.extend(_collect_jax_namespace(jnp, "jax.numpy", include_nonpublic, attach_static_args=True))

        # 2. Core primitive operations in jax.lax.*
        if isinstance(jax, _JaxWithLaxProtocol):
            found.extend(_collect_jax_namespace(jax.lax, "jax.lax", include_nonpublic, attach_static_args=True))

        # 3. Random generation APIs in jax.random.*
        if isinstance(jax, _JaxWithRandomProtocol):
            found.extend(
                _collect_jax_namespace(
                    jax.random,
                    "jax.random",
                    include_nonpublic,
                    attach_static_args=False,
                )
            )

        # 4. Transformations (jax.jit, jax.grad, jax.vmap, jax.pmap, jax.checkpoint)
        jax_members: dict[str, LivePythonTarget] = get_member_dict(jax)
        for transform_name in ("jit", "grad", "vmap", "pmap", "checkpoint"):
            obj = jax_members.get(transform_name)
            if obj and callable(obj):
                try:
                    found.append(GhostInspector.inspect(obj, f"jax.{transform_name}"))
                except _JAX_INSPECT_ERRORS:
                    pass

        jnp_members: dict[str, LivePythonTarget] = get_member_dict(jnp)
        obj = jnp_members.get("transpose")
        if obj and not any(r.name == "transpose" for r in found):
            found.append(GhostInspector.inspect(obj, "jnp.transpose"))

        found.append(GhostInspector.inspect(np.float32, "jax.numpy.float32"))

        # 5. jax.Array member methods
        if isinstance(jax, _JaxWithArrayProtocol):
            found.extend(
                _collect_jax_member_callables(
                    jax.Array,
                    "jax.Array",
                    include_nonpublic,
                    exclude_classes=True,
                    kind="method",
                )
            )

        # 6. jax.sharding APIs
        if isinstance(jax, _JaxWithShardingProtocol):
            found.extend(_collect_jax_member_callables(jax.sharding, "jax.sharding", include_nonpublic))

        # 7. jax.experimental.pallas custom GPU/TPU kernel primitives
        found.extend(_collect_jax_pallas(include_nonpublic))
    except ImportError:
        pass

    return found


def _collect_live(category: SemanticTier, include_nonpublic: bool) -> list[GhostRef]:
    """Scan the live JAX library for a specific API category.

    Args:
        category: The SemanticTier enum value specifying what to scan.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of populated GhostRef objects.

    """
    results = []
    if category == SemanticTier.LOSS:
        results.extend(OptaxScanner.scan_losses(include_nonpublic))
    elif category == SemanticTier.OPTIMIZER:
        results.extend(OptaxScanner.scan_optimizers(include_nonpublic))
    elif category == SemanticTier.ACTIVATION:
        results.extend(_scan_jax_activations(include_nonpublic))
    elif category == SemanticTier.SCHEDULER:
        results.extend(OptaxScanner.scan_schedulers(include_nonpublic))
    elif category == SemanticTier.INITIALIZER:
        results.extend(_scan_jax_initializers(include_nonpublic))
    elif category == SemanticTier.ARRAY_API:
        results.extend(_scan_array_api(include_nonpublic))
    return results


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Entrypoint to collect the JAX API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.

    """
    return _collect_live(category, include_nonpublic)
