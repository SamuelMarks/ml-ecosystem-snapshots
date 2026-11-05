"""Huggingface module."""

from __future__ import annotations

import inspect
import typing

from ml_switcheroo_ir.schema.ghost import (
    GhostParam,
    GhostRef,
    ParameterKind,
    SemanticTier,
)

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    InspectableTarget,
    get_runtime_signature,
)
from ml_ecosystem_snapshots.introspection.sanitize import sanitize_type_str


@typing.runtime_checkable
class _GeneratingModelProtocol(typing.Protocol):
    """Protocol for Hugging Face model classes exposing a `generate` method."""

    generate: InspectableTarget


def _extract_generation_kwargs(obj: InspectableTarget, ref: GhostRef) -> None:
    """Extract generation keyword arguments from a model's `generate` method signature.

    Args:
        obj: Target Hugging Face model class or callable exposing `generate`.
        ref: Populated `GhostRef` to augment with generation keyword arguments.
    """
    gen_method: InspectableTarget = obj.generate if isinstance(obj, _GeneratingModelProtocol) else None
    try:
        sig = get_runtime_signature(gen_method)
        for param in sig.parameters.values():
            if param.name == "self" or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if not any(p.name == param.name for p in ref.params):
                default_str = str(param.default) if param.default is not inspect.Parameter.empty else None
                anno_str = str(param.annotation) if param.annotation is not inspect.Parameter.empty else None
                anno_str = sanitize_type_str(anno_str) if anno_str else None
                ref.params.append(
                    GhostParam(
                        name=param.name,
                        kind=ParameterKind.KEYWORD_ONLY,
                        default=default_str,
                        annotation=anno_str,
                    )
                )
    except (AttributeError, TypeError, ValueError, KeyError):
        pass


def _parse_pretrained_config(obj: InspectableTarget, ref: GhostRef) -> None:
    """Extract class-level `__annotations__` fields from a PreTrainedConfig subclass.

    Args:
        obj: Target configuration class symbol.
        ref: Populated `GhostRef` to augment with configuration attributes.
    """
    obj_members = dict(inspect.getmembers(obj))
    raw_annos = obj_members.get("__annotations__")
    if not isinstance(raw_annos, dict):
        return

    for attr_name, attr_type in raw_annos.items():
        if not any(p.name == attr_name for p in ref.params):
            anno_str = str(attr_type) if not isinstance(attr_type, str) else attr_type
            final_anno = sanitize_type_str(anno_str) if anno_str else None
            ref.params.append(
                GhostParam(
                    name=str(attr_name),
                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=final_anno,
                )
            )


def _handle_automodel_factory(obj: InspectableTarget, name: str, ref: GhostRef) -> None:
    """Ensure `AutoModel*` factory classes expose a `config: PreTrainedConfig` parameter.

    Args:
        obj: Target AutoModel factory class.
        name: Symbol name of the AutoModel class.
        ref: Populated `GhostRef` to augment with `config`.
    """
    if not any(p.name == "config" for p in ref.params):
        ref.params.append(
            GhostParam(
                name="config",
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation="PreTrainedConfig",
            )
        )


def _is_transformers_fast_filter_match(name: str) -> bool:
    """Check whether a symbol in a large `transformers` module matches core prefixes/suffixes.

    Args:
        name: Unqualified symbol name.

    Returns:
        True if the symbol matches core `transformers` API patterns.
    """
    prefixes = ("Auto", "TFAuto", "FlaxAuto", "pipeline", "PreTrained")
    suffixes = ("Pipeline", "TokenizerFast", "Tokenizer")
    return name.startswith(prefixes) or name.endswith(suffixes)


def _classify_hf_symbol(name: str) -> SemanticTier:
    """Classify a Hugging Face symbol into its `SemanticTier`.

    Args:
        name: Unqualified symbol name.

    Returns:
        Target `SemanticTier`.
    """
    if any(k in name for k in ("Config", "Model", "Pipeline", "Tokenizer")):
        return SemanticTier.MODEL
    if "Scheduler" in name:
        return SemanticTier.OPTIMIZER
    return SemanticTier.UTIL


def _inspect_and_augment_hf_symbol(
    obj: InspectableTarget,
    name: str,
    module_name: str,
    inspector: GhostInspector,
) -> GhostRef | None:
    """Inspect a Hugging Face symbol and augment Config, AutoModel, and generate kwargs.

    Args:
        obj: Target symbol object.
        name: Unqualified symbol name.
        module_name: Top-level package name.
        inspector: Active `GhostInspector` instance.

    Returns:
        Populated `GhostRef` or `None` if inspection fails.
    """
    try:
        ref = inspector.inspect(obj, f"{module_name}.{name}")
        if ref is not None:
            if "Config" in name:
                _parse_pretrained_config(obj, ref)
            if name.startswith("AutoModel"):
                _handle_automodel_factory(obj, name, ref)
            if isinstance(obj, _GeneratingModelProtocol):
                _extract_generation_kwargs(obj, ref)
            return ref
    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        ImportError,
        OSError,
        RuntimeError,
    ):
        pass
    return None


def collect_huggingface(
    module_name: str,
    category_mapping: dict[SemanticTier, list[str]],
    category: SemanticTier,
    include_nonpublic: bool = False,
) -> list[GhostRef]:
    """Collect Hugging Face ecosystem APIs for a given package and semantic tier.

    Args:
        module_name: Top-level package name ('transformers', 'diffusers', or 'tokenizers').
        category_mapping: Mapping of `SemanticTier` to matched symbol name substrings.
        category: Target `SemanticTier` to collect.
        include_nonpublic: Whether to include private (`_`-prefixed) symbols.

    Returns:
        List of `GhostRef` objects.
    """
    import importlib

    results: list[GhostRef] = []

    if category not in category_mapping:
        return results

    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return results

    inspector = GhostInspector()
    mod_members: list[tuple[str, InspectableTarget]] = inspect.getmembers(mod)
    is_large_module = len(mod_members) > 100

    for name, obj in mod_members:
        if (not include_nonpublic and name.startswith("_")) or obj is None:
            continue

        if is_large_module and module_name == "transformers" and not _is_transformers_fast_filter_match(name):
            continue

        if _classify_hf_symbol(name) == category:
            ref = _inspect_and_augment_hf_symbol(obj, name, module_name, inspector)
            if ref is not None:
                results.append(ref)

    return results


def collect_transformers(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect Hugging Face `transformers` APIs.

    Args:
        category: Target `SemanticTier` to collect.
        include_nonpublic: Whether to include non-public symbols.

    Returns:
        List of `GhostRef` objects.
    """
    mapping = {
        SemanticTier.MODEL: ["Config", "Model", "Pipeline", "Tokenizer"],
        SemanticTier.UTIL: ["other"],
    }
    return collect_huggingface("transformers", mapping, category, include_nonpublic)


def collect_diffusers(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect Hugging Face `diffusers` APIs.

    Args:
        category: Target `SemanticTier` to collect.
        include_nonpublic: Whether to include non-public symbols.

    Returns:
        List of `GhostRef` objects.
    """
    mapping = {
        SemanticTier.MODEL: ["Model", "Pipeline"],
        SemanticTier.OPTIMIZER: ["Scheduler"],
        SemanticTier.UTIL: ["other"],
    }
    return collect_huggingface("diffusers", mapping, category, include_nonpublic)


def collect_tokenizers(category: SemanticTier, include_nonpublic: bool = False) -> list[GhostRef]:
    """Collect Hugging Face `tokenizers` APIs.

    Args:
        category: Target `SemanticTier` to collect.
        include_nonpublic: Whether to include non-public symbols.

    Returns:
        List of `GhostRef` objects.
    """
    mapping = {
        SemanticTier.MODEL: ["Tokenizer", "Model"],
        SemanticTier.UTIL: ["other"],
    }
    return collect_huggingface("tokenizers", mapping, category, include_nonpublic)
