"""ONNX Runtime API Snapshot Extractor."""

from __future__ import annotations

import importlib
import logging
from typing import List

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)

from ml_ecosystem_snapshots.models import GhostInspector

_LOGGER = logging.getLogger(__name__)


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect ONNX Runtime API symbols for the requested semantic tier.

    Args:
        category: Target `SemanticTier` category (`MODEL` or `UTIL`).
        include_nonpublic: Whether to include private (`_`-prefixed) symbols.

    Returns:
        List of extracted `GhostRef` definitions.
    """
    results: List[GhostRef] = []

    target_cats = {SemanticTier.MODEL, SemanticTier.UTIL}
    if category not in target_cats:
        return results

    try:
        mod = importlib.import_module("onnxruntime")
    except ImportError:
        return results

    inspector = GhostInspector()

    for name in dir(mod):
        if not include_nonpublic and name.startswith("_"):
            continue

        obj = getattr(mod, name, None)
        if obj is None:
            continue

        obj_cat = SemanticTier.MODEL if "Session" in name else SemanticTier.UTIL

        if obj_cat == category:
            try:
                ref = inspector.inspect(obj, f"onnxruntime.{name}")
                if ref:
                    if name == "InferenceSession":
                        has_providers = any(p.name == "providers" for p in ref.params)
                        if not has_providers:
                            ref.params.append(
                                GhostParam(
                                    name="providers",
                                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                                    default="None",
                                    annotation="Sequence[str | tuple[str, dict[str, Any]]] | None",
                                )
                            )
                    results.append(ref)
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

    return results
