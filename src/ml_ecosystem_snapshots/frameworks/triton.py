"""Triton Language and Kernel API Snapshot Extractor."""

from __future__ import annotations

import importlib
import inspect

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    InspectableTarget,
    has_runtime_attr,
    read_runtime_attr,
)


def _extract_triton_kernel(
    obj: InspectableTarget, name: str, module_name: str, inspector: GhostInspector
) -> GhostRef:
    """Extract a `GhostRef` for a Triton JIT kernel or language primitive.

    Args:
        obj: Target Triton callable or JITFunction wrapper.
        name: Unqualified symbol name.
        module_name: Parent module dotted path.
        inspector: Active `GhostInspector` instance.

    Returns:
        The extracted `GhostRef` with `tl.constexpr` annotations normalized.
    """
    fn = read_runtime_attr(obj, "fn", obj) if has_runtime_attr(obj, "fn") else obj

    ref = inspector.inspect(fn, f"{module_name}.{name}")
    raw_annos = read_runtime_attr(fn, "__annotations__", None)
    if isinstance(raw_annos, dict):
        for param in ref.params:
            if param.name in raw_annos:
                anno = raw_annos[param.name]
                if "constexpr" in str(anno):
                    param.annotation = "tl.constexpr"
    return ref


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> list[GhostRef]:
    """Collect Triton language (`triton.language` and `triton.language.math`) API symbols.

    Args:
        category: Target `SemanticTier` category (`UTIL`).
        include_nonpublic: Whether to include private (`_`-prefixed) symbols.

    Returns:
        List of extracted `GhostRef` definitions.
    """
    results: list[GhostRef] = []
    if category != SemanticTier.UTIL:
        return results

    try:
        importlib.import_module("triton")
    except ImportError:
        return results

    inspector = GhostInspector()

    try:
        tl = importlib.import_module("triton.language")
        for name in dir(tl):
            if not include_nonpublic and name.startswith("_"):
                continue
            obj = getattr(tl, name, None)
            if obj and callable(obj):
                results.append(
                    _extract_triton_kernel(obj, name, "triton.language", inspector)
                )

        tl_math = getattr(tl, "math", None)
        if inspect.ismodule(tl_math):
            for name in dir(tl_math):
                if not include_nonpublic and name.startswith("_"):
                    continue
                obj = getattr(tl_math, name, None)
                if obj and callable(obj):
                    results.append(
                        _extract_triton_kernel(
                            obj, name, "triton.language.math", inspector
                        )
                    )
    except ImportError:
        pass

    return results
