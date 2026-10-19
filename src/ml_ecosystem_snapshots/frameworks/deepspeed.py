"""DeepSpeed API Snapshot Extractor."""

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

_LOGGER = logging.getLogger(__name__)


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
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

    for name in dir(mod):
        if not include_nonpublic and name.startswith("_"):
            continue

        obj = getattr(mod, name, None)
        if obj is None:
            continue

        obj_cat = (
            SemanticTier.MODEL
            if name in ("initialize", "DeepSpeedEngine")
            else SemanticTier.UTIL
        )

        if obj_cat == category:
            try:
                ref = inspector.inspect(obj, f"deepspeed.{name}")
                if ref is not None:
                    if name == "initialize":
                        has_config = any(p.name == "config_params" for p in ref.params)
                        if not has_config:
                            ref.params.append(
                                GhostParam(
                                    name="config_params",
                                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                                    default="None",
                                    annotation="dict | str | None",
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
                _LOGGER.debug("Skipping deepspeed.%s: %s", name, exc)

    return results
