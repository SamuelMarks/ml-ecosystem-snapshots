"""Shared declarative collector utilities for ML ecosystem framework extractors.

Consolidates canonical-or-live Python symbol collection, submodule-backed
canonical augmentation, and SSA/operation specification hydration across
framework collector modules.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    GhostOperationRef,
    GhostParam,
    GhostPythonRef,
    GhostRef,
    GhostResult,
    IRParameterRole,
    OperandDirection,
    ParameterKind,
)
from ml_ecosystem_snapshots.models import GhostInspector


def build_canonical_python_ref(
    op: Mapping[str, Any],
    default_doc_prefix: str,
    environment_tags: Sequence[str],
    domain_metadata: Optional[Dict[str, Any]] = None,
) -> GhostPythonRef:
    """Construct a `GhostPythonRef` from a canonical operation specification dictionary.

    Args:
        op: Dictionary describing `name`, `api_path`, `kind`, `params`, and `docstring`.
        default_doc_prefix: Prefix used when `docstring` is omitted in `op`.
        environment_tags: Sequence of environment tag strings to attach.
        domain_metadata: Optional dictionary of domain-specific metadata.

    Returns:
        Hydrated `GhostPythonRef` instance.
    """
    params = [
        GhostParam(
            name=p["name"],
            kind=getattr(ParameterKind, p.get("kind", "POSITIONAL_OR_KEYWORD")),
            default=p.get("default"),
        )
        for p in op.get("params", [])
    ]
    return GhostPythonRef(
        name=op["name"],
        api_path=op["api_path"],
        kind=op.get("kind", "function"),
        params=params,
        docstring=op.get("docstring", f"{default_doc_prefix} {op['name']} API."),
        environment_tags=list(environment_tags),
        domain_metadata=dict(domain_metadata) if domain_metadata is not None else {},
    )


def collect_from_canonical_or_live(
    live_module: Any,
    module_prefix: str,
    target_names: Sequence[str],
    canonical_ops: Sequence[Mapping[str, Any]],
    default_doc_prefix: str,
    environment_tags: Sequence[str],
    inspector_fn: Optional[Callable[..., GhostRef]] = None,
) -> List[GhostRef]:
    """Collect GhostRefs from a live module when installed, or fall back to canonical specs.

    Args:
        live_module: Imported runtime module or None when unavailable.
        module_prefix: Canonical dot-prefix (e.g. 'awkward', 'dpnp', 'numba').
        target_names: Attribute names to inspect on `live_module`.
        canonical_ops: Canonical fallback operation specifications.
        default_doc_prefix: Fallback docstring label when building canonical refs.
        environment_tags: Environment tags appended to each returned ref.
        inspector_fn: Optional custom inspection callable; defaults to `GhostInspector.inspect`.

    Returns:
        List of `GhostRef` objects.
    """
    if live_module is None:
        return [
            build_canonical_python_ref(
                op,
                default_doc_prefix=default_doc_prefix,
                environment_tags=environment_tags,
            )
            for op in canonical_ops
        ]

    inspect_callable = inspector_fn or GhostInspector.inspect
    results: List[GhostRef] = []
    for name in target_names:
        if hasattr(live_module, name):
            obj = getattr(live_module, name)
            try:
                ref = inspect_callable(obj, f"{module_prefix}.{name}", is_public=True)
                ref.environment_tags = list(ref.environment_tags or ()) + list(
                    environment_tags
                )
                results.append(ref)
            except (AttributeError, TypeError, ValueError, RuntimeError, Exception):
                pass
    return results


def collect_from_submodules_or_canonical(
    loaded_submods: Mapping[str, Any],
    canonical_ops: Sequence[Mapping[str, Any]],
    framework_tag: str,
    default_submodule: str,
    default_doc_prefix: str,
    inspector_fn: Optional[Callable[..., GhostRef]] = None,
) -> List[GhostRef]:
    """Collect GhostRefs across loaded submodules, backfilling missed ops from canonical specs.

    Args:
        loaded_submods: Mapping from submodule short name to imported module object.
        canonical_ops: Canonical fallback operation specifications.
        framework_tag: Primary environment tag (e.g. 'safetensors', 'scipy', 'torchaudio').
        default_submodule: Default submodule key when `op` omits `'submodule'`.
        default_doc_prefix: Prefix used when `docstring` is omitted in `op`.
        inspector_fn: Optional custom inspection callable; defaults to `GhostInspector.inspect`.

    Returns:
        List of `GhostRef` objects.
    """
    if not loaded_submods:
        return [
            build_canonical_python_ref(
                op,
                default_doc_prefix=default_doc_prefix,
                environment_tags=(
                    framework_tag,
                    str(op.get("submodule", default_submodule)),
                ),
                domain_metadata={
                    "submodule": str(op.get("submodule", default_submodule))
                },
            )
            for op in canonical_ops
        ]

    inspect_callable = inspector_fn or GhostInspector.inspect
    results: List[GhostRef] = []
    seen: set[str] = set()

    for op in canonical_ops:
        sub = str(op.get("submodule", default_submodule))
        if sub in loaded_submods:
            mod = loaded_submods[sub]
            name = str(op["name"])
            if hasattr(mod, name):
                obj = getattr(mod, name)
                try:
                    ref = inspect_callable(obj, str(op["api_path"]), is_public=True)
                    ref.environment_tags = list(ref.environment_tags or ()) + [
                        framework_tag,
                        sub,
                    ]
                    results.append(ref)
                    seen.add(str(op["api_path"]))
                except (AttributeError, TypeError, ValueError, RuntimeError, Exception):
                    pass

    for op in canonical_ops:
        if str(op["api_path"]) not in seen:
            results.append(
                build_canonical_python_ref(
                    op,
                    default_doc_prefix=default_doc_prefix,
                    environment_tags=(
                        framework_tag,
                        str(op.get("submodule", default_submodule)),
                    ),
                    domain_metadata={
                        "submodule": str(op.get("submodule", default_submodule))
                    },
                )
            )

    return results


def build_ssa_operation_refs(
    ops: Sequence[Mapping[str, Any]],
    api_prefix: str,
    name_key: str,
    operand_annotation_fn: Callable[[Mapping[str, Any], str], str],
    operand_description_fn: Callable[[Mapping[str, Any], str], str],
    operand_direction_fn: Callable[[Mapping[str, Any], str], OperandDirection],
    attr_description_prefix: str,
    result_type: str,
    empty_return_type: str,
    provenance_fn: Callable[[Mapping[str, Any]], Dict[str, Any]],
    default_doc_fn: Callable[[str], str],
    environment_tags: Sequence[str],
) -> List[GhostRef]:
    """Construct `GhostOperationRef` objects from SSA / hardware operation specs.

    Args:
        ops: Sequence of operation dictionary records.
        api_prefix: Prefix for `api_path` (e.g. 'cpp', 'metal', 'wasm', 'webgl').
        name_key: Dictionary key for operation identifier ('name' or 'mnemonic').
        operand_annotation_fn: Callback returning type annotation for `(op, inp)`.
        operand_description_fn: Callback returning description for `(op, inp)`.
        operand_direction_fn: Callback returning `OperandDirection` for `(op, inp)`.
        attr_description_prefix: Prefix for attribute descriptions (e.g. 'C++ attribute').
        result_type: Result type string when outputs exist (e.g. 'auto', 'T', 'v128', 'vec4').
        empty_return_type: Return type string when no outputs exist (e.g. 'void', 'None').
        provenance_fn: Callback returning provenance metadata dictionary for `op`.
        default_doc_fn: Callback returning fallback docstring for `op_name`.
        environment_tags: Sequence of environment tag strings.

    Returns:
        List of populated `GhostOperationRef` items.
    """
    refs: List[GhostRef] = []
    for op in ops:
        op_name = str(op[name_key])
        api_path = f"{api_prefix}.{op_name}"
        params: List[GhostParam] = []

        for inp in op.get("inputs", []):
            inp_str = str(inp)
            params.append(
                ExtendedGhostParam(
                    name=inp_str,
                    kind=ParameterKind.POSITIONAL_ONLY,
                    annotation=operand_annotation_fn(op, inp_str),
                    standardized_name=inp_str,
                    description=operand_description_fn(op, inp_str),
                    direction=operand_direction_fn(op, inp_str),
                    role=IRParameterRole.OPERAND,
                )
            )

        attrs_meta: Dict[str, Any] = {}
        for attr in op.get("attributes", []):
            attr_name = str(attr.get("name", "attr"))
            attr_type = str(attr.get("type", "Any"))
            attr_def = attr.get("default")
            params.append(
                ExtendedGhostParam(
                    name=attr_name,
                    kind=ParameterKind.KEYWORD_ONLY,
                    annotation=attr_type,
                    default=str(attr_def) if attr_def is not None else None,
                    standardized_name=attr_name,
                    description=f"{attr_description_prefix} {attr_name}",
                    role=IRParameterRole.ATTRIBUTE,
                )
            )
            attrs_meta[attr_name] = attr

        returns: List[GhostResult] = [
            GhostResult(name=str(out), type=result_type)
            for out in op.get("outputs", [])
        ]
        returns_type = result_type if returns else empty_return_type

        refs.append(
            GhostOperationRef(
                name=op_name,
                api_path=api_path,
                kind="function",
                params=params,
                operands=[
                    p
                    for p in params
                    if getattr(p, "role", None) == IRParameterRole.OPERAND
                ],
                returns=returns,
                returns_type=returns_type,
                docstring=str(op.get("description", default_doc_fn(op_name))),
                environment_tags=list(environment_tags),
                domain_metadata=provenance_fn(op),
                attributes=attrs_meta if attrs_meta else None,
            )
        )
    return refs


def collect_dir_callables(
    mod_obj: Any,
    api_prefix: str,
    include_nonpublic: bool = False,
    inspector_fn: Optional[Callable[..., GhostRef]] = None,
) -> List[GhostRef]:
    """Iterate `dir(mod_obj)` and inspect callable attributes into `GhostRef` items.

    Args:
        mod_obj: Target module or namespace object.
        api_prefix: Qualified dot prefix (e.g. 'cupy', 'dask.array').
        include_nonpublic: Whether to include underscore-prefixed symbols.
        inspector_fn: Optional custom inspection callable; defaults to `GhostInspector.inspect`.

    Returns:
        List of `GhostRef` objects.
    """
    inspect_callable = inspector_fn or GhostInspector.inspect
    results: List[GhostRef] = []
    for name in dir(mod_obj):
        if not include_nonpublic and name.startswith("_"):
            continue
        obj = getattr(mod_obj, name)
        if callable(obj):
            try:
                res = inspect_callable(
                    obj, f"{api_prefix}.{name}", is_public=not name.startswith("_")
                )
                results.append(res)
            except (AttributeError, TypeError, ValueError, RuntimeError, Exception):
                pass
    return results
