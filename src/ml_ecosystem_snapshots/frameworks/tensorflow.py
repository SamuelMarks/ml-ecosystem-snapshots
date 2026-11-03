"""TensorFlow API Snapshot Extractor.

Provides functions to dynamically introspect the TensorFlow library and generate
GhostRefs for layers, losses, optimizers, and activations.
"""

from __future__ import annotations

import inspect
import operator
import types

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import InspectableTarget
from ml_ecosystem_snapshots.utils import get_all_members

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
    except Exception:
        return None


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
    tf_names = set(dir(tf_mod))
    for sub_name in submodule_names:
        if sub_name in tf_names:
            sub_mod: types.ModuleType = operator.attrgetter(sub_name)(tf_mod)
            for name, obj in get_all_members(sub_mod):
                if (callable(obj) or (allow_classes and inspect.isclass(obj))) and (
                    include_nonpublic or not name.startswith("_")
                ):
                    try:
                        collected.append(
                            GhostInspector.inspect(obj, f"tf.{sub_name}.{name}")
                        )
                    except Exception:
                        pass
    return collected


def _collect_live(category: SemanticTier, include_nonpublic: bool) -> list[GhostRef]:
    """Scan the live TensorFlow library for a specific API category.

    Args:
        category: The SemanticTier enum value specifying what to scan.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of populated GhostRef objects.

    """
    results: list[GhostRef] = []
    tf_mod = _get_tf()
    if not tf_mod:
        return results

    try:
        if category == SemanticTier.ACTIVATION:
            target_names = {
                "relu",
                "sigmoid",
                "tanh",
                "softmax",
                "leaky_relu",
                "elu",
                "selu",
            }
            if "nn" in dir(tf_mod):
                nn_names = set(dir(tf_mod.nn))
                for name in target_names:
                    if name in nn_names:
                        act_obj: InspectableTarget = operator.attrgetter(name)(
                            tf_mod.nn
                        )
                        results.append(GhostInspector.inspect(act_obj, f"tf.nn.{name}"))

        elif category == SemanticTier.LAYER:
            is_keras_available = "keras" in dir(tf_mod) and "layers" in dir(
                tf_mod.keras
            )
            if is_keras_available:
                for name, obj in get_all_members(tf_mod.keras.layers):
                    if inspect.isclass(obj) and (
                        include_nonpublic or not name.startswith("_")
                    ):
                        results.append(
                            GhostInspector.inspect(obj, f"tf.keras.layers.{name}")
                        )

        elif category == SemanticTier.LOSS:
            is_keras_available = "keras" in dir(tf_mod) and "losses" in dir(
                tf_mod.keras
            )
            if is_keras_available:
                for name, obj in get_all_members(tf_mod.keras.losses):
                    if inspect.isclass(obj) and (
                        include_nonpublic or not name.startswith("_")
                    ):
                        results.append(
                            GhostInspector.inspect(obj, f"tf.keras.losses.{name}")
                        )

        elif category == SemanticTier.OPTIMIZER:
            is_keras_available = "keras" in dir(tf_mod) and "optimizers" in dir(
                tf_mod.keras
            )
            if is_keras_available:
                for name, obj in get_all_members(tf_mod.keras.optimizers):
                    if inspect.isclass(obj) and (
                        include_nonpublic or not name.startswith("_")
                    ):
                        results.append(
                            GhostInspector.inspect(obj, f"tf.keras.optimizers.{name}")
                        )

        elif category == SemanticTier.SCHEDULER:
            is_keras_available = (
                "keras" in dir(tf_mod)
                and "optimizers" in dir(tf_mod.keras)
                and "schedules" in dir(tf_mod.keras.optimizers)
            )
            if is_keras_available:
                for name, obj in get_all_members(tf_mod.keras.optimizers.schedules):
                    if inspect.isclass(obj) and (
                        include_nonpublic or not name.startswith("_")
                    ):
                        results.append(
                            GhostInspector.inspect(
                                obj, f"tf.keras.optimizers.schedules.{name}"
                            )
                        )

        elif category == SemanticTier.INITIALIZER:
            is_keras_available = "keras" in dir(tf_mod) and "initializers" in dir(
                tf_mod.keras
            )
            if is_keras_available:
                for name, obj in get_all_members(tf_mod.keras.initializers):
                    if inspect.isclass(obj) and (
                        include_nonpublic or not name.startswith("_")
                    ):
                        results.append(
                            GhostInspector.inspect(obj, f"tf.keras.initializers.{name}")
                        )

        elif category == SemanticTier.METRIC:
            is_keras_available = "keras" in dir(tf_mod) and "metrics" in dir(
                tf_mod.keras
            )
            if is_keras_available:
                for name, obj in get_all_members(tf_mod.keras.metrics):
                    if inspect.isclass(obj) and (
                        include_nonpublic or not name.startswith("_")
                    ):
                        results.append(
                            GhostInspector.inspect(obj, f"tf.keras.metrics.{name}")
                        )

        elif category == SemanticTier.DATALOADER:
            if "data" in dir(tf_mod):
                for name, obj in get_all_members(tf_mod.data):
                    if inspect.isclass(obj) and (
                        include_nonpublic or not name.startswith("_")
                    ):
                        results.append(GhostInspector.inspect(obj, f"tf.data.{name}"))

        elif category == SemanticTier.ARRAY_API:
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
            tf_top_names = set(dir(tf_mod))
            for name in top_level_tensor_ops:
                if name in tf_top_names:
                    obj_top: InspectableTarget = operator.attrgetter(name)(tf_mod)
                    if callable(obj_top):
                        try:
                            results.append(
                                GhostInspector.inspect(obj_top, f"tf.{name}")
                            )
                        except Exception:
                            pass

            for name, obj in get_all_members(tf_mod):
                if (
                    not name.startswith("_") or include_nonpublic
                ) and name not in seen_ops:
                    if callable(obj) and not inspect.isclass(obj):
                        try:
                            results.append(GhostInspector.inspect(obj, f"tf.{name}"))
                        except Exception:
                            pass

            if "raw_ops" in dir(tf_mod):
                for name, obj in get_all_members(tf_mod.raw_ops):
                    if not name.startswith("_") or include_nonpublic:
                        if callable(obj) and not inspect.isclass(obj):
                            try:
                                results.append(
                                    GhostInspector.inspect(obj, f"tf.raw_ops.{name}")
                                )
                            except Exception:
                                pass

            if "Tensor" in dir(tf_mod):
                for name, obj in inspect.getmembers(tf_mod.Tensor):
                    if (
                        (include_nonpublic or not name.startswith("_"))
                        and callable(obj)
                        and not inspect.isclass(obj)
                    ):
                        try:
                            results.append(
                                GhostInspector.inspect(
                                    obj, f"tf.Tensor.{name}", kind="method"
                                )
                            )
                        except Exception:
                            pass

            if "Variable" in dir(tf_mod):
                for name, obj in inspect.getmembers(tf_mod.Variable):
                    if (
                        (include_nonpublic or not name.startswith("_"))
                        and callable(obj)
                        and not inspect.isclass(obj)
                    ):
                        try:
                            results.append(
                                GhostInspector.inspect(
                                    obj, f"tf.Variable.{name}", kind="method"
                                )
                            )
                        except Exception:
                            pass

        if category in (SemanticTier.ARRAY_API, SemanticTier.UTIL) and "raw_ops" in dir(
            tf_mod
        ):
            for name, obj in get_all_members(tf_mod.raw_ops):
                if not include_nonpublic and name.startswith("_"):
                    continue
                if callable(obj) and not inspect.isclass(obj):
                    try:
                        ref = GhostInspector.inspect(obj, f"tf.raw_ops.{name}")
                        if ref.environment_tags is None:
                            ref.environment_tags = []
                        ref.environment_tags.append("tf.raw_op")
                        results.append(ref)
                    except Exception:
                        pass

    except Exception:
        pass

    return results


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Entrypoint to collect the TensorFlow API signature for a given category.

    Args:
        category: The category of API to collect.
        include_nonpublic: Whether to include non-public APIs.

    Returns:
        A list of GhostRef items discovered for the requested category.

    """
    return _collect_live(category, include_nonpublic)
