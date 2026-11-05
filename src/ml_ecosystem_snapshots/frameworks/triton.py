"""Triton Language and Kernel API Snapshot Extractor."""

from __future__ import annotations

import importlib
import inspect
import types
import typing

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import InspectableTarget


@typing.runtime_checkable
class _TritonJitWrapperProtocol(typing.Protocol):
    """Protocol for Triton JITFunction wrappers exposing an underlying `.fn` callable."""

    fn: InspectableTarget


@typing.runtime_checkable
class _TritonLanguageWithMathProtocol(typing.Protocol):
    """Protocol for `triton.language` exposing a `.math` submodule."""

    math: types.ModuleType


def _extract_triton_kernel(obj: InspectableTarget, name: str, module_name: str, inspector: GhostInspector) -> GhostRef:
    """Extract a `GhostRef` for a Triton JIT kernel or language primitive.

    Args:
        obj: Target Triton callable or JITFunction wrapper.
        name: Unqualified symbol name.
        module_name: Parent module dotted path.
        inspector: Active `GhostInspector` instance.

    Returns:
        The extracted `GhostRef` with `tl.constexpr` annotations normalized.
    """
    fn: InspectableTarget = obj.fn if isinstance(obj, _TritonJitWrapperProtocol) and obj.fn is not None else obj

    ref = inspector.inspect(fn, f"{module_name}.{name}")
    fn_members = dict(inspect.getmembers(fn))
    raw_annos = fn_members.get("__annotations__")
    if isinstance(raw_annos, dict):
        for param in ref.params:
            if param.name in raw_annos:
                anno = raw_annos[param.name]
                if "constexpr" in str(anno):
                    param.annotation = "tl.constexpr"
    return ref


def _collect_triton_module_callables(
    mod: types.ModuleType,
    module_name: str,
    include_nonpublic: bool,
    inspector: GhostInspector,
) -> list[GhostRef]:
    """Collect callable Triton primitives from a module.

    Args:
        mod: Loaded `triton.language` or `triton.language.math` module.
        module_name: Dotted module name prefix.
        include_nonpublic: Whether to include private symbols.
        inspector: Active `GhostInspector` instance.

    Returns:
        List of extracted `GhostRef` items.
    """
    collected: list[GhostRef] = []
    for name, obj in inspect.getmembers(mod):
        if not include_nonpublic and name.startswith("_"):
            continue
        if obj and callable(obj):
            collected.append(_extract_triton_kernel(obj, name, module_name, inspector))
    return collected


def collect_api(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
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
        results.extend(_collect_triton_module_callables(tl, "triton.language", include_nonpublic, inspector))

        tl_members = dict(inspect.getmembers(tl))
        tl_math = tl_members.get("math")
        if isinstance(tl_math, types.ModuleType):
            results.extend(_collect_triton_module_callables(tl_math, "triton.language.math", include_nonpublic, inspector))
    except ImportError:
        pass

    return results
