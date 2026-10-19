"""Triton Language and Kernel API Snapshot Extractor."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, List

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.models import GhostInspector


def _extract_triton_kernel(
    obj: Any, name: str, module_name: str, inspector: GhostInspector
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
    fn = obj.fn if hasattr(obj, "fn") else obj

    ref = inspector.inspect(fn, f"{module_name}.{name}")
    if ref:
        for param in ref.params:
            if hasattr(fn, "__annotations__") and param.name in fn.__annotations__:
                anno = fn.__annotations__[param.name]
                if "constexpr" in str(anno):
                    param.annotation = "tl.constexpr"
    return ref


def collect_api(
    category: SemanticTier, include_nonpublic: bool = False
) -> List[GhostRef]:
    """Collect Triton language (`triton.language` and `triton.language.math`) API symbols.

    Args:
        category: Target `SemanticTier` category (`UTIL`).
        include_nonpublic: Whether to include private (`_`-prefixed) symbols.

    Returns:
        List of extracted `GhostRef` definitions.
    """
    results: List[GhostRef] = []
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
                ref = _extract_triton_kernel(obj, name, "triton.language", inspector)
                if ref:
                    results.append(ref)

        tl_math = getattr(tl, "math", None)
        if inspect.ismodule(tl_math):
            for name in dir(tl_math):
                if not include_nonpublic and name.startswith("_"):
                    continue
                obj = getattr(tl_math, name, None)
                if obj and callable(obj):
                    ref = _extract_triton_kernel(
                        obj, name, "triton.language.math", inspector
                    )
                    if ref:
                        results.append(ref)
    except ImportError:
        pass

    return results
