"""Live Python runtime signature, MRO, and `ExtendedGhostParam` construction helpers."""

from __future__ import annotations

import inspect
import logging
import re
import types
import typing

from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    GhostParam,
)

import ml_ecosystem_snapshots.introspection.inspector as _insp_mod
from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.introspection.context import (
    ExtractedParamRecord,
    OverloadExtractionContext,
    ParamExtractionContext,
    RawParamTuple,
    _get_target_module_name,
    to_parameter_kind,
)
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    GRIFFE_DOCSTRING_TYPE,
    GRIFFE_ERROR_TYPE,
    GRIFFE_FUNCTION_TYPES,
    STATIC_GRIFFE_TYPES,
    LivePythonTarget,
    StaticGriffeNode,
    get_member_dict,
)
from ml_ecosystem_snapshots.introspection.sanitize import (
    sanitize_param_default,
)

_LOGGER = logging.getLogger(__name__)


def _format_annotation_obj(ann: LivePythonTarget | StaticGriffeNode) -> str:
    """Format a runtime parameter annotation into its string representation.

    Args:
        ann: Parameter annotation object from `inspect.Parameter`.

    Returns:
        String representation of the annotation.
    """
    if isinstance(ann, type):
        return repr(ann)[8:-2].split(".")[-1]
    return str(ann)


def _supplement_cdd_params_from_signature(
    extracted_params: list[RawParamTuple],
    cdd_ast_params: typing.Mapping[str, ExtractedParamRecord],
    target: LivePythonTarget | StaticGriffeNode,
    resolved_hints: typing.Mapping[str, type | types.GenericAlias | str],
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
) -> bool:
    """Supplement CDD AST parameters with parameter kinds and missing entries from `inspect.signature`.

    Args:
        extracted_params: Mutable list of `(name, kind, default, annotation)` tuples.
        cdd_ast_params: Mapping of parameter names already extracted via CDD AST.
        target: Runtime callable target (`obj` or `obj.__init__`).
        resolved_hints: Resolved runtime type hints mapping.
        active_sanitize_type_str: Type annotation string sanitizer callable.

    Returns:
        True if a `VAR_POSITIONAL` parameter was encountered, False otherwise.
    """
    has_varargs = False
    try:
        sig = _insp_mod.get_runtime_signature(target)
        for sig_param in sig.parameters.values():
            if sig_param.name == "self":
                continue
            if sig_param.kind == inspect.Parameter.VAR_POSITIONAL:
                has_varargs = True
            if sig_param.name not in cdd_ast_params:
                p_kind_str = str(sig_param.kind)
                anno_val = (
                    active_sanitize_type_str(str(resolved_hints.get(sig_param.name, sig_param.annotation)))
                    if sig_param.annotation is not inspect.Parameter.empty
                    else None
                )
                extracted_params.append((sig_param.name, p_kind_str, None, anno_val))
            else:
                for i, (pn, _pk, pd, pa) in enumerate(extracted_params):
                    if pn == sig_param.name:
                        extracted_params[i] = (pn, str(sig_param.kind), pd, pa)
    except (
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
        LookupError,
        SyntaxError,
        NameError,
        AssertionError,
        SnapshotError,
    ):
        _LOGGER.debug("Failed to supplement CDD parameters from runtime signature", exc_info=True)
    return has_varargs


def _extract_params_from_runtime_sig(
    obj: LivePythonTarget | StaticGriffeNode,
    target: LivePythonTarget | StaticGriffeNode,
    resolved_hints: typing.Mapping[str, type | types.GenericAlias | str],
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
) -> tuple[list[RawParamTuple], bool, bool]:
    """Extract raw parameter tuples from `inspect.signature(target)`.

    Args:
        obj: Unwrapped Python runtime symbol.
        target: Callable target (`obj` or `obj.__init__`).
        resolved_hints: Resolved runtime type hints mapping.
        active_sanitize_type_str: Type annotation string sanitizer callable.

    Returns:
        Tuple of `(extracted_params, has_varargs, is_c_ext)`.
    """
    sig = _insp_mod.get_runtime_signature(target)
    extracted_params: list[RawParamTuple] = []
    has_varargs = False
    for sig_param in sig.parameters.values():
        if sig_param.name == "self":
            continue
        if sig_param.kind == inspect.Parameter.VAR_POSITIONAL:
            has_varargs = True
        default_val, _factory, _is_mand = sanitize_param_default(sig_param.default)
        anno_val = None
        if sig_param.name in resolved_hints:
            anno_val = active_sanitize_type_str(str(resolved_hints[sig_param.name]))
        elif sig_param.annotation is not inspect.Parameter.empty:
            anno_val = active_sanitize_type_str(_format_annotation_obj(sig_param.annotation))
        extracted_params.append((sig_param.name, str(sig_param.kind), default_val, anno_val))

    is_c_ext = bool(
        inspect.isbuiltin(obj)
        or _get_target_module_name(target).startswith("_")
        or "torch._C" in str(target)
        or "builtin_function_or_method" in repr(type(target))
    )
    return extracted_params, has_varargs, is_c_ext


def _merge_mro_super_kwargs(
    obj: LivePythonTarget | StaticGriffeNode,
    extracted_params: list[RawParamTuple],
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
) -> None:
    """Merge parent `__init__` parameters across MRO when `super().__init__(**kwargs)` is used.

    Args:
        obj: Class target being inspected.
        extracted_params: Mutable list of `(name, kind, default, annotation)` tuples.
        active_sanitize_type_str: Type annotation string sanitizer callable.
    """
    mro_seq = inspect.getmro(obj) if isinstance(obj, type) else ()
    for parent in mro_seq[1:]:
        parent_members = get_member_dict(parent)
        if _format_annotation_obj(parent) == "object":
            continue
        try:
            parent_init = parent_members.get("__init__")
            parent_sig = _insp_mod.get_runtime_signature(parent_init)
            for param in parent_sig.parameters.values():
                if param.name == "self" or param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if not any(ep[0] == param.name for ep in extracted_params):
                    default_val = str(param.default) if param.default is not inspect.Parameter.empty else None
                    anno_val = (
                        _format_annotation_obj(param.annotation)
                        if param.annotation is not inspect.Parameter.empty
                        else None
                    )
                    anno_val = active_sanitize_type_str(anno_val)
                    extracted_params.append((param.name, str(param.kind), default_val, anno_val))
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            OSError,
            RuntimeError,
            LookupError,
            SyntaxError,
            NameError,
            AssertionError,
            SnapshotError,
        ):
            _LOGGER.debug("Failed to merge parent __init__ signature across MRO", exc_info=True)


def _promote_documented_var_kwargs(
    extracted_params: list[RawParamTuple],
    cdd_params: typing.Mapping[str, ExtractedParamRecord],
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
) -> list[RawParamTuple]:
    """Promote documented `**kwargs` entries into formal `KEYWORD_ONLY` parameter tuples.

    Args:
        extracted_params: Current list of `(name, kind, default, annotation)` tuples.
        cdd_params: Parsed docstring parameter dictionary.
        active_sanitize_type_str: Type annotation string sanitizer callable.

    Returns:
        Updated list of `(name, kind, default, annotation)` parameter tuples.
    """
    has_var_kw = any(ep[1] == "VAR_KEYWORD" for ep in extracted_params)
    if has_var_kw and cdd_params:
        existing_names = {ep[0] for ep in extracted_params}
        var_kw_idx = next(i for i, ep in enumerate(extracted_params) if ep[1] == "VAR_KEYWORD")
        promoted_params: list[RawParamTuple] = []
        for p_name, p_data in cdd_params.items():
            if p_name not in existing_names and p_name not in ("kwargs", "args", "self") and not p_name.startswith("_"):
                raw_typ = p_data.get("typ")
                p_typ = active_sanitize_type_str(raw_typ) if raw_typ else None
                p_def = p_data.get("default")
                promoted_params.append((p_name, "KEYWORD_ONLY", p_def, p_typ))
                existing_names.add(p_name)

        if promoted_params:
            return extracted_params[:var_kw_idx] + promoted_params + extracted_params[var_kw_idx:]
    return extracted_params


def _resolve_param_doc_and_anno(
    p_name: str,
    p_anno: str | None,
    cdd_params: typing.Mapping[str, ExtractedParamRecord],
    cdd_ast_params: typing.Mapping[str, ExtractedParamRecord],
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
) -> tuple[str | None, str | None]:
    """Resolve parameter description and fallback annotation from CDD docstring or AST records.

    Args:
        p_name: Parameter name.
        p_anno: Existing sanitized parameter type annotation, if any.
        cdd_params: Parsed docstring parameter metadata mapping.
        cdd_ast_params: Parsed AST parameter metadata mapping.
        active_sanitize_type_str: Type annotation sanitizer callable.

    Returns:
        Tuple of `(description, annotation)`.
    """
    p_desc = None
    if p_name in cdd_params and cdd_params[p_name].get("doc"):
        p_desc = cdd_params[p_name].get("doc")
    elif p_name in cdd_ast_params and cdd_ast_params[p_name].get("doc"):
        p_desc = cdd_ast_params[p_name].get("doc")

    if not p_anno or p_anno in ("Any", "object"):
        if p_name in cdd_params and cdd_params[p_name].get("typ"):
            p_anno = active_sanitize_type_str(cdd_params[p_name].get("typ"))
        elif p_name in cdd_ast_params and cdd_ast_params[p_name].get("typ"):
            p_anno = active_sanitize_type_str(cdd_ast_params[p_name].get("typ"))
    return p_desc, p_anno


def _build_extended_params(
    extracted_params: typing.Sequence[RawParamTuple],
    cdd_params: typing.Mapping[str, ExtractedParamRecord],
    cdd_ast_params: typing.Mapping[str, ExtractedParamRecord],
    is_torch_target: bool,
    name: str,
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
) -> list[GhostParam]:
    """Construct `ExtendedGhostParam` objects from raw parameter tuples.

    Args:
        extracted_params: Sequence of `(name, kind, default, annotation)` tuples.
        cdd_params: Parsed docstring parameter metadata.
        cdd_ast_params: Parsed AST parameter metadata.
        is_torch_target: Whether the symbol belongs to PyTorch.
        name: Unqualified symbol name.
        active_sanitize_type_str: Type annotation sanitizer callable.

    Returns:
        List of constructed `ExtendedGhostParam` objects.
    """
    params: list[GhostParam] = []
    for p_name, p_kind, p_default, p_anno in extracted_params:
        p_desc, p_anno = _resolve_param_doc_and_anno(
            p_name=p_name,
            p_anno=p_anno,
            cdd_params=cdd_params,
            cdd_ast_params=cdd_ast_params,
            active_sanitize_type_str=active_sanitize_type_str,
        )

        p_dtypes: list[str] | None = None
        p_rank: str | None = None
        if is_torch_target:
            try:
                from ml_ecosystem_snapshots.frameworks.torch import (
                    infer_torch_dtype_and_rank,
                )

                p_dtypes, p_rank = infer_torch_dtype_and_rank(name, p_name, p_anno or "")
            except (
                ImportError,
                AttributeError,
                TypeError,
                ValueError,
                OSError,
                RuntimeError,
                LookupError,
                SyntaxError,
                NameError,
                AssertionError,
                SnapshotError,
            ):
                _LOGGER.debug("Failed to infer torch dtype and rank for %s.%s", name, p_name, exc_info=True)

        p_factory = None
        if p_default and "<factory_default>" in str(p_default):
            p_factory = "<factory_default>"

        p_allowed_values = None
        if p_anno and "Literal[" in p_anno:
            p_allowed_values = re.findall(r"['\x22]([^'\x22]+)['\x22]", p_anno)
        elif p_name.lower() in _insp_mod.STANDARD_ENUM_MAP:
            p_allowed_values = _insp_mod.STANDARD_ENUM_MAP[p_name.lower()]

        params.append(
            ExtendedGhostParam(
                name=p_name,
                standardized_name=_insp_mod.STANDARD_ARG_MAP.get(p_name),
                kind=to_parameter_kind(p_kind),
                default=p_default,
                annotation=p_anno,
                description=p_desc,
                allowed_dtypes=p_dtypes,
                rank_constraint=p_rank,
                allowed_values=p_allowed_values,
                default_factory=p_factory,
                is_mandatory=p_default is None,
            )
        )
    return params


def _resolve_target_identity(
    obj: LivePythonTarget | StaticGriffeNode,
    obj_members: typing.Mapping[str, LivePythonTarget],
    api_path: str,
    kind: str | None,
) -> tuple[
    str,
    str,
    str | None,
    StaticGriffeNode | None,
    bool,
]:
    """Resolve symbol name, kind, docstring, and Griffe node status from an unwrapped target.

    Args:
        obj: Unwrapped inspectable target object or Griffe node.
        obj_members: Member mapping of `obj`.
        api_path: Canonical dotted API path.
        kind: Optional explicit symbol kind override.

    Returns:
        Tuple of `(name, determined_kind, doc, griffe_node, is_griffe_node)`.
    """
    if isinstance(obj, STATIC_GRIFFE_TYPES):
        name = str(obj.name or api_path.split(".")[-1])
        try:
            is_cls = bool(obj.is_class)
        except (AttributeError, RuntimeError, GRIFFE_ERROR_TYPE):
            is_cls = False
        determined_kind = kind if kind is not None else ("class" if is_cls else "function")
        try:
            doc_obj = obj.docstring
        except (AttributeError, RuntimeError, GRIFFE_ERROR_TYPE):
            doc_obj = None
        doc = doc_obj.value if isinstance(doc_obj, GRIFFE_DOCSTRING_TYPE) else None
        return name, determined_kind, doc, obj, True

    name = (
        _format_annotation_obj(obj)
        if isinstance(obj, type)
        else str(obj_members.get("__name__") or api_path.split(".")[-1])
    )
    if name == "<lambda>":
        name = api_path.split(".")[-1]
    determined_kind = kind if kind is not None else ("class" if isinstance(obj, type) else "function")
    return name, determined_kind, inspect.getdoc(obj), None, False


def _resolve_visibility_and_type_hints(
    target: LivePythonTarget | StaticGriffeNode,
    name: str,
    is_public: bool | None,
    is_griffe_node: bool,
    griffe_node: StaticGriffeNode | None,
    returns_type: str | None,
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
) -> tuple[bool, dict[str, type | types.GenericAlias | str], str | None]:
    """Resolve symbol public visibility, runtime type hints, and fallback return type.

    Args:
        target: Callable target (`obj` or `obj.__init__`).
        name: Unqualified symbol name.
        is_public: Optional explicit public visibility override.
        is_griffe_node: Whether the inspected object is a static Griffe node.
        griffe_node: Optional Griffe AST node.
        returns_type: Current docstring-inferred return type string.
        active_sanitize_type_str: Type annotation string sanitizer callable.

    Returns:
        Tuple of `(determined_is_public, resolved_hints, returns_type)`.
    """
    if is_public is not None:
        determined_is_public = is_public
    elif griffe_node is not None:
        determined_is_public = bool(griffe_node.is_public)
    else:
        determined_is_public = not name.startswith("_")

    resolved_hints: dict[str, type | types.GenericAlias | str] = {}
    if not is_griffe_node:
        try:
            resolved_hints = typing.get_type_hints(target)
            if "return" in resolved_hints and returns_type is None:
                returns_type = active_sanitize_type_str(str(resolved_hints["return"]))
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            OSError,
            RuntimeError,
            LookupError,
            SyntaxError,
            NameError,
            AssertionError,
            SnapshotError,
        ):
            _LOGGER.debug("Failed to resolve runtime type hints for %s", name, exc_info=True)

    if griffe_node is not None and returns_type is None:
        try:
            g_ret = (
                griffe_node.returns
                if isinstance(griffe_node, GRIFFE_FUNCTION_TYPES) and griffe_node.is_function
                else None
            )
        except (AttributeError, RuntimeError, GRIFFE_ERROR_TYPE):
            g_ret = None
        if g_ret:
            returns_type = active_sanitize_type_str(str(g_ret))

    return determined_is_public, resolved_hints, returns_type


__all__ = [
    "OverloadExtractionContext",
    "ParamExtractionContext",
    "RawParamTuple",
    "_build_extended_params",
    "_extract_params_from_runtime_sig",
    "_format_annotation_obj",
    "_merge_mro_super_kwargs",
    "_promote_documented_var_kwargs",
    "_resolve_param_doc_and_anno",
    "_resolve_target_identity",
    "_resolve_visibility_and_type_hints",
    "_supplement_cdd_params_from_signature",
    "to_parameter_kind",
]
