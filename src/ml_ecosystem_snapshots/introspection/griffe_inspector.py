"""Static Griffe AST and docstring inspection helpers for `GhostInspector`."""

from __future__ import annotations

import logging
import types
import typing

import typing_extensions
from ml_switcheroo_ir.schema.ghost import ExtendedGhostRef

from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.introspection.context import (
    DocstringIrResult,
    ExtractedParamRecord,
    OverloadExtractionContext,
    ParamDefaultValue,
    RawParamTuple,
)
from ml_ecosystem_snapshots.introspection.live_inspector import _build_extended_params
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    GRIFFE_CLASS_TYPES,
    GRIFFE_ERROR_TYPE,
    GRIFFE_FUNCTION_TYPES,
    STATIC_GRIFFE_TYPES,
    LivePythonTarget,
    StaticGriffeNode,
    get_member_dict,
)
from ml_ecosystem_snapshots.introspection.static_parsers import (
    GriffeCacheEntry,
    GriffeLoaderProto,
)
from ml_ecosystem_snapshots.models import JsonValue
from ml_ecosystem_snapshots.utils import (
    DocstringMetadata,
    DocstringParamMeta,
    DocstringReturnMeta,
    resolve_griffe_parser,
)

if typing.TYPE_CHECKING:
    import griffe

_LOGGER = logging.getLogger(__name__)


def _coerce_param_record(
    val: (ExtractedParamRecord | DocstringParamMeta | typing.Mapping[str, ParamDefaultValue] | ParamDefaultValue),
) -> ExtractedParamRecord:
    """Coerce a raw parameter dict or model into `ExtractedParamRecord`.

    Args:
        val: Raw dictionary or `ExtractedParamRecord` object.

    Returns:
        Populated `ExtractedParamRecord` instance.
    """
    if isinstance(val, (ExtractedParamRecord, DocstringParamMeta)):
        return ExtractedParamRecord(
            doc=str(val.doc) if val.doc is not None else None,
            typ=str(val.typ) if val.typ is not None else None,
            default=val.default,
        )
    if isinstance(val, typing.Mapping):
        raw_doc = val.get("doc")
        raw_typ = val.get("typ")
        return ExtractedParamRecord(
            doc=str(raw_doc) if raw_doc is not None else None,
            typ=str(raw_typ) if raw_typ is not None else None,
            default=val.get("default"),
        )
    return ExtractedParamRecord()


def _merge_griffe_return_meta(
    ret_meta: DocstringReturnMeta | typing.Mapping[str, JsonValue] | JsonValue,
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
    result: DocstringIrResult,
) -> None:
    """Merge Griffe return type and description metadata into `result` when missing.

    Args:
        ret_meta: Return metadata extracted from Griffe docstring parsing.
        active_sanitize_type_str: Type annotation string sanitizer callable.
        result: Mutable `DocstringIrResult` accumulator.
    """
    if isinstance(ret_meta, DocstringReturnMeta):
        ret_typ: JsonValue = ret_meta.typ
        ret_doc: JsonValue = ret_meta.doc
    elif isinstance(ret_meta, typing.Mapping):
        ret_typ = ret_meta.get("typ")
        ret_doc = ret_meta.get("doc")
    else:
        return
    if ret_typ and not result.returns_type:
        result.returns_type = active_sanitize_type_str(str(ret_typ))
    if ret_doc and not result.returns_description:
        result.returns_description = str(ret_doc)


def _merge_griffe_param_record(
    p_name: str,
    p_data: ExtractedParamRecord | DocstringParamMeta | typing.Mapping[str, ParamDefaultValue] | ParamDefaultValue,
    cdd_params: dict[str, ExtractedParamRecord],
) -> None:
    """Merge a single Griffe docstring parameter entry into `cdd_params`.

    Args:
        p_name: Parameter name from Griffe docstring metadata.
        p_data: Raw parameter metadata record or dictionary from Griffe.
        cdd_params: Mutable parameter dictionary keyed by parameter name.
    """
    clean_name = p_name.lstrip("*")
    rec = _coerce_param_record(p_data)
    if clean_name not in cdd_params:
        cdd_params[clean_name] = rec
    else:
        if rec.doc and not cdd_params[clean_name].doc:
            cdd_params[clean_name].doc = rec.doc
        if rec.typ and not cdd_params[clean_name].typ:
            cdd_params[clean_name].typ = rec.typ


def _populate_griffe_docstring_ir(
    doc: str,
    docstring_parser_name: str,
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
    active_extract_griffe_meta: typing.Callable[[str, str], DocstringMetadata | typing.Mapping[str, JsonValue]],
    result: DocstringIrResult,
) -> None:
    """Supplement `result` with docstring metadata extracted via Griffe.

    Args:
        doc: Raw docstring text to parse.
        docstring_parser_name: Parser style ('google', 'numpy', 'sphinx').
        active_sanitize_type_str: Type annotation string sanitizer callable.
        active_extract_griffe_meta: Griffe docstring metadata extractor callable.
        result: Mutable `DocstringIrResult` accumulator.
    """
    try:
        from ml_ecosystem_snapshots.models import ensure_json_dict, ensure_json_list

        griffe_meta = active_extract_griffe_meta(doc, docstring_parser_name)
        ret_meta = griffe_meta.returns if isinstance(griffe_meta, DocstringMetadata) else griffe_meta.get("returns")
        if ret_meta:
            _merge_griffe_return_meta(ret_meta, active_sanitize_type_str, result)

        raises_seq: typing.Sequence[JsonValue] = (
            griffe_meta.raises
            if isinstance(griffe_meta, DocstringMetadata)
            else ensure_json_list(griffe_meta.get("raises", []), "griffe raises")
        )
        for exc in raises_seq:
            exc_str = str(exc)
            if exc_str not in result.raises:
                result.raises.append(exc_str)
            if exc_str in result.cdd_params:
                del result.cdd_params[exc_str]

        params_map: typing.Mapping[str, DocstringParamMeta | JsonValue] = (
            griffe_meta.params
            if isinstance(griffe_meta, DocstringMetadata)
            else ensure_json_dict(griffe_meta.get("params", {}), "griffe params")
        )
        for p_name, p_data in params_map.items():
            _merge_griffe_param_record(p_name, p_data, result.cdd_params)
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
        _LOGGER.debug("Failed to extract Griffe docstring IR", exc_info=True)


def _get_griffe_param_kind_name(param: griffe.Parameter, default: str = "POSITIONAL_OR_KEYWORD") -> str:
    """Resolve the uppercase parameter kind string from a Griffe parameter node.

    Args:
        param: Griffe `Parameter` node.
        default: Fallback kind string when `.kind` is absent.

    Returns:
        Uppercase parameter kind string.
    """
    return str(param.kind.name).upper() if param.kind is not None else default


def _is_griffe_loader(
    fn: LivePythonTarget | StaticGriffeNode,
) -> typing_extensions.TypeGuard[GriffeLoaderProto]:
    """Narrow `fn` to `GriffeLoaderProto`.

    Args:
        fn: Candidate `griffe.load` callable.

    Returns:
        True if `fn` is callable.
    """
    return callable(fn)


def _resolve_griffe_node_from_cache(
    api_path: str,
    docstring_parser_name: str,
    griffe_mod: types.ModuleType,
    active_griffe_cache: dict[str, GriffeCacheEntry],
) -> StaticGriffeNode | None:
    """Load or look up a Griffe AST node for `api_path` using `active_griffe_cache`.

    Args:
        api_path: Canonical dotted API path.
        docstring_parser_name: Docstring style name ('google', 'numpy', 'sphinx').
        griffe_mod: Imported `griffe` module.
        active_griffe_cache: Module-level Griffe cache dictionary.

    Returns:
        Resolved `GriffeNodeProtocol` instance if found, otherwise `None`.
    """
    try:
        resolved_parser = resolve_griffe_parser(docstring_parser_name)
        loader_fn = get_member_dict(griffe_mod).get("load")
        if not _is_griffe_loader(loader_fn):
            return None
        if active_griffe_cache.get("__loader__") is not loader_fn:
            active_griffe_cache.clear()
            active_griffe_cache["__loader__"] = loader_fn
        top_level = api_path.split(".")[0]
        if top_level not in active_griffe_cache:
            active_griffe_cache[top_level] = loader_fn(top_level, docstring_parser=resolved_parser)

        parts = api_path.split(".")
        current: GriffeCacheEntry = active_griffe_cache[top_level]
        if len(parts) > 1 and isinstance(current, STATIC_GRIFFE_TYPES):
            for part in parts[1:]:
                if not current.members or part not in current.members:
                    return None
                current = current.members[part]
        if isinstance(current, STATIC_GRIFFE_TYPES):
            return current
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
        _LOGGER.debug("Failed to resolve Griffe node from cache for %s", api_path, exc_info=True)
    return None


def _get_griffe_node_params(
    griffe_node: StaticGriffeNode | None,
) -> list[griffe.Parameter] | None:
    """Extract non-empty Griffe parameter list from a Griffe function or class node.

    Args:
        griffe_node: Optional Griffe AST node for the inspected symbol.

    Returns:
        Non-empty list of `griffe.Parameter` items, or `None`.
    """
    if griffe_node is None:
        return None
    try:
        if (
            isinstance(griffe_node, GRIFFE_FUNCTION_TYPES)
            and griffe_node.is_function
            or (isinstance(griffe_node, GRIFFE_FUNCTION_TYPES) and griffe_node.is_function)
        ):
            params = list(griffe_node.parameters)
            return params if params else None
        if (
            isinstance(griffe_node, GRIFFE_CLASS_TYPES)
            and griffe_node.is_class
            or (isinstance(griffe_node, GRIFFE_FUNCTION_TYPES) and griffe_node.is_class)
        ):
            init_node = griffe_node.members.get("__init__") or griffe_node.members.get("__new__")
            if isinstance(init_node, GRIFFE_FUNCTION_TYPES) and init_node.is_function:
                params = list(init_node.parameters)
                return params if params else None
    except (AttributeError, TypeError, ValueError, RuntimeError, LookupError, GRIFFE_ERROR_TYPE):
        _LOGGER.debug("Failed to extract Griffe parameters", exc_info=True)
    return None


def _supplement_cdd_params_from_griffe(
    extracted_params: list[RawParamTuple],
    cdd_ast_params: typing.Mapping[str, ExtractedParamRecord],
    griffe_params: typing.Iterable[griffe.Parameter],
    resolved_hints: typing.Mapping[str, type | types.GenericAlias | str],
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
) -> bool:
    """Supplement CDD AST parameters with parameter kinds and missing entries from Griffe.

    Args:
        extracted_params: Mutable list of `(name, kind, default, annotation)` tuples.
        cdd_ast_params: Mapping of parameter names already extracted via CDD AST.
        griffe_params: Griffe parameter objects for the target symbol.
        resolved_hints: Resolved runtime type hints mapping.
        active_sanitize_type_str: Type annotation string sanitizer callable.

    Returns:
        True if a `VAR_POSITIONAL` parameter was encountered, False otherwise.
    """
    has_varargs = False
    for param in griffe_params:
        if param.name == "self":
            continue
        p_kind_str = _get_griffe_param_kind_name(param)
        if p_kind_str == "VAR_POSITIONAL":
            has_varargs = True
        if param.name not in cdd_ast_params:
            default_val = str(param.default) if param.default is not None else None
            anno_val = (
                active_sanitize_type_str(str(resolved_hints.get(param.name, param.annotation)))
                if param.annotation
                else None
            )
            extracted_params.append((param.name, p_kind_str, default_val, anno_val))
        else:
            for i, (pn, pk, pd, pa) in enumerate(extracted_params):
                if pn == param.name:
                    extracted_params[i] = (pn, _get_griffe_param_kind_name(param, default=pk), pd, pa)
    return has_varargs


def _extract_params_from_griffe_only(
    griffe_params: typing.Iterable[griffe.Parameter],
    resolved_hints: typing.Mapping[str, type | types.GenericAlias | str],
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
) -> tuple[list[RawParamTuple], bool]:
    """Extract raw parameter tuples directly from Griffe parameter nodes.

    Args:
        griffe_params: Iterable of Griffe parameter objects.
        resolved_hints: Resolved runtime type hints mapping.
        active_sanitize_type_str: Type annotation string sanitizer callable.

    Returns:
        Tuple of `(extracted_params, has_varargs)`.
    """
    extracted_params: list[RawParamTuple] = []
    has_varargs = False
    for param in griffe_params:
        if param.name == "self":
            continue
        p_kind_str = _get_griffe_param_kind_name(param)
        if p_kind_str == "VAR_POSITIONAL":
            has_varargs = True
        default_val = str(param.default) if param.default is not None else None
        if param.name in resolved_hints:
            anno_val = active_sanitize_type_str(str(resolved_hints[param.name]))
        else:
            anno_val = active_sanitize_type_str(str(param.annotation)) if param.annotation else None
        extracted_params.append((param.name, p_kind_str, default_val, anno_val))
    return extracted_params, has_varargs


def _get_griffe_node_overloads(
    griffe_node: StaticGriffeNode | None,
) -> list[griffe.Function] | None:
    """Resolve non-empty overload list from a Griffe function or class node.

    Args:
        griffe_node: Optional Griffe AST node.

    Returns:
        Non-empty list of `griffe.Function` overload objects, or `None`.
    """
    if griffe_node is None:
        return None
    try:
        if (
            isinstance(griffe_node, GRIFFE_FUNCTION_TYPES)
            and griffe_node.is_function
            and isinstance(griffe_node.overloads, list)
            and griffe_node.overloads
        ):
            return list(griffe_node.overloads)
        if isinstance(griffe_node, GRIFFE_CLASS_TYPES) and griffe_node.is_class:
            init_node = griffe_node.members.get("__init__")
            if (
                isinstance(init_node, GRIFFE_FUNCTION_TYPES)
                and init_node.is_function
                and isinstance(init_node.overloads, list)
                and init_node.overloads
            ):
                return list(init_node.overloads)
    except (AttributeError, TypeError, ValueError, RuntimeError, LookupError, GRIFFE_ERROR_TYPE):
        _LOGGER.debug("Failed to extract Griffe overloads", exc_info=True)
    return None


def _build_griffe_overload_ref(
    overload: griffe.Function,
    ctx: OverloadExtractionContext,
) -> ExtendedGhostRef:
    """Build an `ExtendedGhostRef` from a single Griffe overload node.

    Args:
        overload: Griffe `Function` overload node.
        ctx: Bundled overload extraction context.

    Returns:
        Constructed `ExtendedGhostRef` representing the overload signature.
    """
    ov_raw_tuples: list[RawParamTuple] = []
    overload_has_varargs = False
    for param in overload.parameters:
        if param.name == "self":
            continue
        p_kind_str = _get_griffe_param_kind_name(param)
        if p_kind_str == "VAR_POSITIONAL":
            overload_has_varargs = True
        default_val = str(param.default) if param.default is not None else None
        anno_val = ctx.active_sanitize_type_str(str(param.annotation)) if param.annotation else None
        ov_raw_tuples.append((param.name, p_kind_str, default_val, anno_val))

    overload_params = _build_extended_params(
        extracted_params=ov_raw_tuples,
        cdd_params={},
        cdd_ast_params={},
        is_torch_target=False,
        name=ctx.name,
        active_sanitize_type_str=ctx.active_sanitize_type_str,
    )
    ov_ret_raw = overload.returns
    overload_returns = ctx.active_sanitize_type_str(str(ov_ret_raw)) if ov_ret_raw else None

    return ExtendedGhostRef(
        name=ctx.name,
        api_path=ctx.api_path,
        kind=ctx.kind,
        params=overload_params,
        docstring=None,
        has_varargs=overload_has_varargs,
        aliases=[],
        is_public=ctx.determined_is_public,
        returns_type=overload_returns,
        returns_description=None,
        raises=[],
        environment_tags=ctx.env_tags,
        overloads=[],
    )


__all__ = [
    "DocstringIrResult",
    "ExtractedParamRecord",
    "_build_griffe_overload_ref",
    "_coerce_param_record",
    "_extract_params_from_griffe_only",
    "_get_griffe_node_overloads",
    "_get_griffe_node_params",
    "_get_griffe_param_kind_name",
    "_is_griffe_loader",
    "_merge_griffe_param_record",
    "_merge_griffe_return_meta",
    "_populate_griffe_docstring_ir",
    "_resolve_griffe_node_from_cache",
    "_supplement_cdd_params_from_griffe",
]
