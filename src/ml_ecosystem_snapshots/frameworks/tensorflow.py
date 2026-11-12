"""TensorFlow API Snapshot Extractor.

Provides functions to dynamically introspect the TensorFlow library and generate
GhostRefs for layers, losses, optimizers, and activations.
"""

from __future__ import annotations

import inspect
import logging
import types

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
    get_member_dict,
)
from ml_ecosystem_snapshots.utils import get_all_members

_LOGGER = logging.getLogger(__name__)

tf: types.ModuleType | None = None


def _get_tf() -> types.ModuleType | None:
    """Lazily import TensorFlow to prevent static C++ registry conflict on worker fork.

    Returns:
        TensorFlow module or None if not installed or fails to initialize.
    """
    global tf
    if tf is not None:
        return tf
    try:
        import tensorflow as tf_mod

        tf = tf_mod
        return tf
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
        _LOGGER.debug("Optional tensorflow module unavailable", exc_info=True)
        return None


def _get_submodule(
    parent: types.ModuleType | types.SimpleNamespace, name: str
) -> types.ModuleType | types.SimpleNamespace | None:
    """Resolve a submodule attribute on `parent` via `inspect.getmembers`.

    Args:
        parent: Parent module object.
        name: Submodule attribute name.

    Returns:
        Submodule `ModuleType` / `SimpleNamespace` or None if absent.
    """
    members: dict[str, LivePythonTarget] = get_member_dict(parent)
    sub = members.get(name)
    return sub if isinstance(sub, (types.ModuleType, types.SimpleNamespace)) else None


def _collect_tf_submodules(
    tf_mod: types.ModuleType,
    submodule_names: list[str],
    include_nonpublic: bool,
    allow_classes: bool = False,
) -> list[GhostRef]:
    """Collect callable or class members across a list of TensorFlow submodules.

    Args:
        tf_mod: Imported TensorFlow root module.
        submodule_names: List of submodule attribute names on `tf_mod`.
        include_nonpublic: Whether to include private underscore-prefixed symbols.
        allow_classes: Whether classes should also be collected alongside callables.

    Returns:
        List of inspected `GhostRef` objects.
    """
    collected: list[GhostRef] = []
    for sub_name in submodule_names:
        sub_mod = _get_submodule(tf_mod, sub_name)
        if sub_mod is not None:
            for name, obj in get_member_dict(sub_mod).items():
                if (callable(obj) or (allow_classes and inspect.isclass(obj))) and (
                    include_nonpublic or not name.startswith("_")
                ):
                    try:
                        collected.append(GhostInspector.inspect(obj, f"tf.{sub_name}.{name}"))
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
                        _LOGGER.debug("Failed to inspect tf.%s.%s", sub_name, name, exc_info=True)
    return collected


def _collect_tf_classes_from_mod(
    mod: types.ModuleType | types.SimpleNamespace | None,
    prefix: str,
    include_nonpublic: bool,
) -> list[GhostRef]:
    """Collect class symbols from a TensorFlow or Keras submodule.

    Args:
        mod: Target submodule or None.
        prefix: Qualified API path prefix.
        include_nonpublic: Whether to include underscore-prefixed classes.

    Returns:
        List of inspected `GhostRef` items.
    """
    if mod is None:
        return []
    return [
        GhostInspector.inspect(
            obj,
            f"{prefix}.{name}",
            is_public=not name.startswith("_"),
        )
        for name, obj in get_all_members(mod)
        if inspect.isclass(obj) and (include_nonpublic or not name.startswith("_"))
    ]


def _collect_tf_activations(tf_mod: types.ModuleType) -> list[GhostRef]:
    """Collect TensorFlow activation functions from `tf.nn`.

    Args:
        tf_mod: Loaded TensorFlow root module.

    Returns:
        List of inspected `GhostRef` items.
    """
    nn_mod = _get_submodule(tf_mod, "nn")
    if nn_mod is None:
        return []
    target_names = {
        "relu",
        "sigmoid",
        "tanh",
        "softmax",
        "leaky_relu",
        "elu",
        "selu",
    }
    nn_members: dict[str, LivePythonTarget] = get_member_dict(nn_mod)
    return [GhostInspector.inspect(nn_members[name], f"tf.nn.{name}") for name in target_names if name in nn_members]


def _collect_tf_keras_tier(tf_mod: types.ModuleType, category: SemanticTier, include_nonpublic: bool) -> list[GhostRef]:
    """Collect Keras-backed TensorFlow tiers (`LAYER`, `LOSS`, `OPTIMIZER`, `SCHEDULER`, `INITIALIZER`, `METRIC`).

    Args:
        tf_mod: Loaded TensorFlow root module.
        category: Target `SemanticTier`.
        include_nonpublic: Whether to include private classes.

    Returns:
        List of inspected `GhostRef` items.
    """
    keras_mod = _get_submodule(tf_mod, "keras")
    if keras_mod is None:
        return []

    tier_submod_map: dict[SemanticTier, tuple[str, str]] = {
        SemanticTier.LAYER: ("layers", "tf.keras.layers"),
        SemanticTier.LOSS: ("losses", "tf.keras.losses"),
        SemanticTier.OPTIMIZER: ("optimizers", "tf.keras.optimizers"),
        SemanticTier.INITIALIZER: ("initializers", "tf.keras.initializers"),
        SemanticTier.METRIC: ("metrics", "tf.keras.metrics"),
    }
    if category in tier_submod_map:
        attr_name, prefix = tier_submod_map[category]
        return _collect_tf_classes_from_mod(_get_submodule(keras_mod, attr_name), prefix, include_nonpublic)

    opt_mod = _get_submodule(keras_mod, "optimizers")
    sched_mod = _get_submodule(opt_mod, "schedules") if opt_mod else None
    return _collect_tf_classes_from_mod(sched_mod, "tf.keras.optimizers.schedules", include_nonpublic)


def _collect_tf_type_methods(tf_mod: types.ModuleType, type_name: str, include_nonpublic: bool) -> list[GhostRef]:
    """Collect instance methods from `tf.Tensor` or `tf.Variable`.

    Args:
        tf_mod: Loaded TensorFlow root module.
        type_name: Class name on `tf_mod` (`'Tensor'` or `'Variable'`).
        include_nonpublic: Whether to include private methods.

    Returns:
        List of inspected `GhostRef` items.
    """
    tf_members: dict[str, LivePythonTarget] = get_member_dict(tf_mod)
    cls_obj = tf_members.get(type_name)
    if cls_obj is None:
        return []
    results: list[GhostRef] = []
    for name, obj in get_member_dict(cls_obj).items():
        if (include_nonpublic or not name.startswith("_")) and callable(obj) and not inspect.isclass(obj):
            try:
                results.append(GhostInspector.inspect(obj, f"tf.{type_name}.{name}", kind="method"))
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
                _LOGGER.debug("Failed to inspect tf.%s.%s", type_name, name, exc_info=True)
    return results


def _collect_tf_raw_ops(tf_mod: types.ModuleType, include_nonpublic: bool, add_tag: bool) -> list[GhostRef]:
    """Collect functions from `tf.raw_ops`.

    Args:
        tf_mod: Loaded TensorFlow root module.
        include_nonpublic: Whether to include private functions.
        add_tag: Whether to append `'tf.raw_op'` to `environment_tags`.

    Returns:
        List of inspected `GhostRef` items.
    """
    raw_ops_mod = _get_submodule(tf_mod, "raw_ops")
    if raw_ops_mod is None:
        return []
    results: list[GhostRef] = []
    for name, obj in get_all_members(raw_ops_mod):
        if (include_nonpublic or not name.startswith("_")) and (callable(obj) and not inspect.isclass(obj)):
            try:
                ref = GhostInspector.inspect(obj, f"tf.raw_ops.{name}")
                if add_tag:
                    if ref.environment_tags is None:
                        ref.environment_tags = []
                    ref.environment_tags.append("tf.raw_op")
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
                _LOGGER.debug("Failed to inspect tf.raw_ops.%s", name, exc_info=True)
    return results


def _collect_tf_array_api(tf_mod: types.ModuleType, include_nonpublic: bool) -> list[GhostRef]:
    """Collect `SemanticTier.ARRAY_API` symbols across TensorFlow modules and classes.

    Args:
        tf_mod: Loaded TensorFlow root module.
        include_nonpublic: Whether to include private symbols.

    Returns:
        List of inspected `GhostRef` items.
    """
    results: list[GhostRef] = []
    results.extend(
        _collect_tf_submodules(
            tf_mod,
            ["math", "linalg", "nn", "image"],
            include_nonpublic=include_nonpublic,
            allow_classes=False,
        )
    )
    results.extend(
        _collect_tf_submodules(
            tf_mod,
            ["lookup"],
            include_nonpublic=include_nonpublic,
            allow_classes=True,
        )
    )

    top_level_tensor_ops = [
        "concat",
        "stack",
        "unstack",
        "split",
        "gather",
        "gather_nd",
        "scatter_nd",
        "reshape",
        "squeeze",
        "expand_dims",
        "transpose",
        "tile",
        "slice",
        "strided_slice",
        "where",
    ]
    seen_ops = set(top_level_tensor_ops)
    tf_top_members: dict[str, LivePythonTarget] = get_member_dict(tf_mod)
    for name in top_level_tensor_ops:
        if name in tf_top_members and callable(tf_top_members[name]):
            try:
                results.append(GhostInspector.inspect(tf_top_members[name], f"tf.{name}"))
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
                _LOGGER.debug("Failed to inspect tf.%s", name, exc_info=True)

    for name, obj in get_all_members(tf_mod):
        if (
            (not name.startswith("_") or include_nonpublic)
            and name not in seen_ops
            and callable(obj)
            and not inspect.isclass(obj)
        ):
            try:
                results.append(GhostInspector.inspect(obj, f"tf.{name}"))
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
                _LOGGER.debug("Failed to inspect tf.%s", name, exc_info=True)

    results.extend(_collect_tf_raw_ops(tf_mod, include_nonpublic, add_tag=False))
    results.extend(_collect_tf_type_methods(tf_mod, "Tensor", include_nonpublic))
    results.extend(_collect_tf_type_methods(tf_mod, "Variable", include_nonpublic))
    results.extend(_collect_tf_raw_ops(tf_mod, include_nonpublic, add_tag=True))
    return results


def _collect_live(category: SemanticTier, include_nonpublic: bool) -> list[GhostRef]:
    """Scan the live TensorFlow library for a specific API category.

    Args:
        category: The SemanticTier enum value specifying what to scan.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of populated GhostRef objects.
    """
    tf_mod = _get_tf()
    if not tf_mod:
        return []

    results: list[GhostRef] = []
    try:
        if category == SemanticTier.ACTIVATION:
            results = _collect_tf_activations(tf_mod)
        elif category in (
            SemanticTier.LAYER,
            SemanticTier.LOSS,
            SemanticTier.OPTIMIZER,
            SemanticTier.SCHEDULER,
            SemanticTier.INITIALIZER,
            SemanticTier.METRIC,
        ):
            results = _collect_tf_keras_tier(tf_mod, category, include_nonpublic)
        elif category == SemanticTier.DATALOADER:
            results = _collect_tf_classes_from_mod(_get_submodule(tf_mod, "data"), "tf.data", include_nonpublic)
        elif category == SemanticTier.ARRAY_API:
            results = _collect_tf_array_api(tf_mod, include_nonpublic)
        elif category == SemanticTier.UTIL:
            results = _collect_tf_raw_ops(tf_mod, include_nonpublic, add_tag=True)
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
        _LOGGER.debug("Failed to collect live TensorFlow API for %s", category, exc_info=True)

    return results


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Entrypoint to collect the TensorFlow API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.
    """
    return _collect_live(category, include_nonpublic)
