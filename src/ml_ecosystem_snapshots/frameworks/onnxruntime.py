"""ONNX Runtime API Snapshot Extractor."""

from __future__ import annotations

import importlib
import logging

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
    get_member_dict,
)

_LOGGER = logging.getLogger(__name__)


def _inspect_ort_symbol(obj: LivePythonTarget, name: str, inspector: GhostInspector) -> GhostRef | None:
    """Inspect a single ONNX Runtime symbol and inject `providers` on `InferenceSession`.

    Args:
        obj: Target ONNX Runtime symbol.
        name: Unqualified symbol name.
        inspector: Active `GhostInspector` instance.

    Returns:
        Populated `GhostRef` or `None` if inspection fails.
    """
    try:
        ref: GhostRef | None = inspector.inspect(obj, f"onnxruntime.{name}")
    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        SyntaxError,
        OSError,
        ImportError,
        RuntimeError,
    ) as exc:
        _LOGGER.debug("Skipping onnxruntime.%s: %s", name, exc)
        return None

    if ref is not None and name == "InferenceSession" and not any(p.name == "providers" for p in ref.params):
        ref.params.append(
            GhostParam(
                name="providers",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                default="None",
                annotation="Sequence[str | tuple[str, dict[str, JsonValue]]] | None",
            )
        )
    return ref


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect ONNX Runtime API symbols for the requested semantic tier.

    Args:
        category: Target `SemanticTier` category (`MODEL` or `UTIL`).
        include_nonpublic: Whether to include private (`_`-prefixed) symbols.

    Returns:
        List of extracted `GhostRef` definitions.
    """
    results: list[GhostRef] = []

    target_cats = {SemanticTier.MODEL, SemanticTier.UTIL}
    if category not in target_cats:
        return results

    try:
        mod = importlib.import_module("onnxruntime")
    except ImportError:
        return results

    inspector = GhostInspector()

    for name, obj in get_member_dict(mod).items():
        if (not include_nonpublic and name.startswith("_")) or obj is None:
            continue

        obj_cat = SemanticTier.MODEL if "Session" in name else SemanticTier.UTIL

        if obj_cat == category:
            ref = _inspect_ort_symbol(obj, name, inspector)
            if ref is not None:
                results.append(ref)

    return results
