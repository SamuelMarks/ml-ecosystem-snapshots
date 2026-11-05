"""DeepSpeed API Snapshot Extractor."""

from __future__ import annotations

import importlib
import inspect
import logging

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import InspectableTarget

_LOGGER = logging.getLogger(__name__)


def _inspect_deepspeed_symbol(obj: InspectableTarget, name: str, inspector: GhostInspector) -> GhostRef | None:
    """Inspect a single DeepSpeed symbol and inject `config_params` on `initialize`.

    Args:
        obj: Target DeepSpeed symbol.
        name: Unqualified symbol name.
        inspector: Active `GhostInspector` instance.

    Returns:
        Populated `GhostRef` or `None` if inspection fails.
    """
    try:
        ref: GhostRef | None = inspector.inspect(obj, f"deepspeed.{name}")
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
        _LOGGER.debug("Skipping deepspeed.%s: %s", name, exc)
        return None

    if ref is not None and name == "initialize" and not any(p.name == "config_params" for p in ref.params):
        ref.params.append(
            GhostParam(
                name="config_params",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                default="None",
                annotation="dict | str | None",
            )
        )
    return ref


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect DeepSpeed API symbols for the requested semantic tier.

    Args:
        category: Target `SemanticTier` category (`MODEL` or `UTIL`).
        include_nonpublic: Whether to include private (`_`-prefixed) symbols.

    Returns:
        List of extracted `GhostRef` definitions.
    """
    results: list[GhostRef] = []

    if category not in (SemanticTier.MODEL, SemanticTier.UTIL):
        return results

    try:
        mod = importlib.import_module("deepspeed")
    except ImportError:
        return results

    inspector = GhostInspector()

    for name, obj in inspect.getmembers(mod):
        if (not include_nonpublic and name.startswith("_")) or obj is None:
            continue

        obj_cat = SemanticTier.MODEL if name in ("initialize", "DeepSpeedEngine") else SemanticTier.UTIL

        if obj_cat == category:
            ref = _inspect_deepspeed_symbol(obj, name, inspector)
            if ref is not None:
                results.append(ref)

    return results
