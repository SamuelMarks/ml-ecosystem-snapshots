"""Core `GhostInspector` facade coordinating live, Griffe, CDD, and PyTorch inspection."""

from __future__ import annotations

import ast
import logging
import types
import typing

from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostRef,
    GhostParam,
    GhostPythonRef,
    GhostRef,
)

from ml_ecosystem_snapshots.c_extension_sig import (
    CExtensionSignature,
    extract_c_extension_signature,
)
from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.introspection.ast_kwargs import (
    extract_accepted_kwargs_from_ast,
)
from ml_ecosystem_snapshots.introspection.cdd_inspector import (
    _build_initial_cdd_ast_params,
    _detect_super_kwargs_call,
    _extract_docstring_ir,
    _parse_cdd_ast_params,
)
from ml_ecosystem_snapshots.introspection.griffe_inspector import (
    DocstringIrResult,
    ExtractedParamRecord,
    _build_griffe_overload_ref,
    _coerce_param_record,
    _extract_params_from_griffe_only,
    _get_griffe_node_overloads,
    _get_griffe_node_params,
    _get_griffe_param_kind_name,
    _merge_griffe_return_meta,
    _resolve_griffe_node_from_cache,
    _supplement_cdd_params_from_griffe,
)
from ml_ecosystem_snapshots.introspection.live_inspector import (
    OverloadExtractionContext,
    ParamExtractionContext,
    RawParamTuple,
    _build_extended_params,
    _extract_params_from_runtime_sig,
    _merge_mro_super_kwargs,
    _promote_documented_var_kwargs,
    _resolve_target_identity,
    _resolve_visibility_and_type_hints,
    _supplement_cdd_params_from_signature,
    to_parameter_kind,
)
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
    StaticGriffeNode,
    get_member_dict,
    get_runtime_signature,
    get_runtime_source,
)
from ml_ecosystem_snapshots.introspection.sanitize import sanitize_type_str
from ml_ecosystem_snapshots.introspection.static_parsers import (
    _GRIFFE_CACHE,
    GriffeCacheEntry,
)
from ml_ecosystem_snapshots.introspection.torch_inspector import (
    FRAMEWORK_CAPABILITIES,
    STANDARD_ARG_MAP,
    STANDARD_ENUM_MAP,
    _build_c_ext_overload_ref,
    _extract_params_from_c_ext,
    _is_torch_symbol,
    _resolve_environment_tags,
)
from ml_ecosystem_snapshots.introspection.unwrap import unwrap_decorated_target
from ml_ecosystem_snapshots.models import JsonValue, ScalarValue
from ml_ecosystem_snapshots.serialization.hydration import hydrate_ghost_ref
from ml_ecosystem_snapshots.utils import (
    extract_griffe_docstring_metadata,
    get_framework_docstring_parser,
)

_LOGGER = logging.getLogger(__name__)


def _extract_ast_and_griffe_node(
    obj: LivePythonTarget | StaticGriffeNode,
    kind: str,
    api_path: str,
    docstring_parser_name: str,
    griffe_node: StaticGriffeNode | None,
    cdd_params: typing.Mapping[str, ExtractedParamRecord],
    active_ast_mod: types.ModuleType,
    active_griffe_cache: dict[str, GriffeCacheEntry],
) -> tuple[
    dict[str, ExtractedParamRecord],
    bool,
    StaticGriffeNode | None,
]:
    """Extract CDD AST parameter definitions, `super().__init__(**kwargs)` usage, and Griffe node.

    Args:
        obj: Target Python runtime symbol.
        kind: Symbol kind ('class', 'function', 'method').
        api_path: Canonical dotted API path.
        docstring_parser_name: Docstring style name.
        griffe_node: Existing Griffe node if `obj` is already a Griffe node.
        cdd_params: Extracted docstring parameter dictionary.
        active_ast_mod: Active `ast` module reference.
        active_griffe_cache: Active Griffe module cache dictionary.

    Returns:
        Tuple of (cdd_ast_params, has_super_kwargs_call, griffe_node).
    """
    cdd_ast_params: dict[str, ExtractedParamRecord] = {}
    has_super_kwargs_call = False

    try:
        source = get_runtime_source(obj)
        parsed_ast = active_ast_mod.parse(source).body[0]
        if kind == "class":
            has_super_kwargs_call = _detect_super_kwargs_call(parsed_ast)
        cdd_ast_params = _parse_cdd_ast_params(parsed_ast, kind, cdd_params)
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
        _LOGGER.debug("Failed to parse runtime AST for %s", api_path, exc_info=True)

    griffe_mod: types.ModuleType | None
    try:
        import griffe as _griffe_mod

        griffe_mod = _griffe_mod
    except ImportError:
        griffe_mod = None

    if griffe_node is None and griffe_mod is not None:
        griffe_node = _resolve_griffe_node_from_cache(
            api_path=api_path,
            docstring_parser_name=docstring_parser_name,
            griffe_mod=griffe_mod,
            active_griffe_cache=active_griffe_cache,
        )

    return cdd_ast_params, has_super_kwargs_call, griffe_node


def _extract_raw_parameters(
    obj: LivePythonTarget | StaticGriffeNode,
    target: LivePythonTarget | StaticGriffeNode,
    ctx: ParamExtractionContext,
) -> tuple[
    list[RawParamTuple],
    CExtensionSignature | None,
    bool,
    bool,
    typing.Literal["exact", "heuristic", "opaque"],
    str | None,
    list[str] | None,
]:
    """Run the 4-stage parameter fallback cascade (CDD AST -> Griffe -> inspect -> C-Ext/ATen).

    Args:
        obj: Unwrapped Python runtime symbol.
        target: Callable target (`obj` or `obj.__init__`).
        ctx: Bundled parameter extraction inputs and active helper callables.

    Returns:
        Tuple of (extracted_params, c_ext_params, has_varargs, is_c_ext,
        sig_completeness, returns_type, environment_tags).
    """
    griffe_params = _get_griffe_node_params(ctx.griffe_node)

    if ctx.cdd_ast_params:
        extracted_params = _build_initial_cdd_ast_params(
            ctx.cdd_ast_params,
            ctx.resolved_hints,
            ctx.active_ast_mod,
            ctx.active_sanitize_type_str,
        )
        if griffe_params is not None:
            has_varargs = _supplement_cdd_params_from_griffe(
                extracted_params,
                ctx.cdd_ast_params,
                griffe_params,
                ctx.resolved_hints,
                ctx.active_sanitize_type_str,
            )
        else:
            has_varargs = _supplement_cdd_params_from_signature(
                extracted_params,
                ctx.cdd_ast_params,
                target,
                ctx.resolved_hints,
                ctx.active_sanitize_type_str,
            )
        return (
            extracted_params,
            None,
            has_varargs,
            False,
            "exact",
            ctx.returns_type,
            ctx.environment_tags,
        )

    if griffe_params is not None:
        extracted_params, has_varargs = _extract_params_from_griffe_only(
            griffe_params,
            ctx.resolved_hints,
            ctx.active_sanitize_type_str,
        )
        return (
            extracted_params,
            None,
            has_varargs,
            False,
            "exact",
            ctx.returns_type,
            ctx.environment_tags,
        )

    try:
        extracted_params, has_varargs, is_c_ext = _extract_params_from_runtime_sig(
            obj,
            target,
            ctx.resolved_hints,
            ctx.active_sanitize_type_str,
        )
        return (
            extracted_params,
            None,
            has_varargs,
            is_c_ext,
            "exact",
            ctx.returns_type,
            ctx.environment_tags,
        )
    except (ValueError, TypeError):
        (
            extracted_params,
            c_ext_params,
            has_varargs,
            sig_completeness,
            returns_type,
            environment_tags,
        ) = _extract_params_from_c_ext(target, ctx)
        return (
            extracted_params,
            c_ext_params,
            has_varargs,
            True,
            sig_completeness,
            returns_type,
            environment_tags,
        )


def _extract_overloads(ctx: OverloadExtractionContext) -> list[ExtendedGhostRef]:
    """Extract function/method overloads from Griffe AST nodes or C-extension docstrings.

    Args:
        ctx: Bundled overload extraction context.

    Returns:
        List of `ExtendedGhostRef` overload objects.
    """
    griffe_overloads = _get_griffe_node_overloads(ctx.griffe_node)
    if griffe_overloads is not None:
        return [_build_griffe_overload_ref(overload, ctx) for overload in griffe_overloads]
    if ctx.c_ext_params is not None and ctx.c_ext_params.overloads:
        return [_build_c_ext_overload_ref(ov, ctx) for ov in ctx.c_ext_params.overloads]
    return []


class GhostInspector:
    """Facade for API Inspection.

    Responsibility: Convert Live Objects -> JSON-serializable GhostDefs.
    Crucial for populating snapshots used by WASM/JS environments.
    """

    @staticmethod
    def inspect(
        obj: LivePythonTarget | StaticGriffeNode,
        api_path: str,
        is_public: bool | None = None,
        environment_tags: list[str] | None = None,
        kind: str | None = None,
    ) -> GhostPythonRef:
        """Create a GhostRef from a live Python runtime symbol.

        Gracefully handles C-Extensions and builtins that resist introspection.

        Args:
            obj: The live class or function to inspect.
            api_path: The canonical string path (e.g. 'torch.nn.ReLU').
            is_public: Optional override for public visibility.
            environment_tags: Optional explicit execution tags (e.g. ['cpu', 'cuda']).
            kind: Optional explicit kind override (e.g. 'function', 'class', 'method').

        Returns:
            A populated GhostPythonRef instance.
        """
        obj = unwrap_decorated_target(obj)
        obj_members = get_member_dict(obj)
        name, kind, doc, griffe_node, is_griffe_node = _resolve_target_identity(
            obj=obj,
            obj_members=obj_members,
            api_path=api_path,
            kind=kind,
        )

        active_sanitize_type_str = sanitize_type_str
        active_extract_griffe_meta = extract_griffe_docstring_metadata
        active_griffe_cache = _GRIFFE_CACHE
        active_ast_mod = ast

        top_level = api_path.split(".")[0] if api_path else ""
        docstring_parser_name = get_framework_docstring_parser(top_level, doc)

        cdd_params, returns_type, returns_description, raises = _extract_docstring_ir(
            doc=doc,
            docstring_parser_name=docstring_parser_name,
            active_sanitize_type_str=active_sanitize_type_str,
            active_extract_griffe_meta=active_extract_griffe_meta,
        )

        target = obj_members.get("__init__", obj) if kind == "class" else obj

        cdd_ast_params, has_super_kwargs_call, griffe_node = _extract_ast_and_griffe_node(
            obj=obj,
            kind=kind,
            api_path=api_path,
            docstring_parser_name=docstring_parser_name,
            griffe_node=griffe_node,
            cdd_params=cdd_params,
            active_ast_mod=active_ast_mod,
            active_griffe_cache=active_griffe_cache,
        )

        determined_is_public, resolved_hints, returns_type = _resolve_visibility_and_type_hints(
            target=target,
            name=name,
            is_public=is_public,
            is_griffe_node=is_griffe_node,
            griffe_node=griffe_node,
            returns_type=returns_type,
            active_sanitize_type_str=active_sanitize_type_str,
        )

        (
            extracted_params,
            c_ext_params,
            has_varargs,
            is_c_ext,
            sig_completeness,
            returns_type,
            environment_tags,
        ) = _extract_raw_parameters(
            obj=obj,
            target=target,
            ctx=ParamExtractionContext(
                name=name,
                kind=kind,
                api_path=api_path,
                cdd_ast_params=cdd_ast_params,
                griffe_node=griffe_node,
                resolved_hints=resolved_hints,
                returns_type=returns_type,
                environment_tags=environment_tags,
            ),
        )

        if has_super_kwargs_call:
            _merge_mro_super_kwargs(
                obj=obj,
                extracted_params=extracted_params,
                active_sanitize_type_str=active_sanitize_type_str,
            )

        extracted_params = _promote_documented_var_kwargs(
            extracted_params=extracted_params,
            cdd_params=cdd_params,
            active_sanitize_type_str=active_sanitize_type_str,
        )

        is_torch_target = _is_torch_symbol(name, target, api_path)

        params = _build_extended_params(
            extracted_params=extracted_params,
            cdd_params=cdd_params,
            cdd_ast_params=cdd_ast_params,
            is_torch_target=is_torch_target,
            name=name,
            active_sanitize_type_str=active_sanitize_type_str,
        )

        env_tags = _resolve_environment_tags(environment_tags, api_path)

        overloads_refs = _extract_overloads(
            OverloadExtractionContext(
                griffe_node=griffe_node,
                c_ext_params=c_ext_params,
                name=name,
                api_path=api_path,
                kind=kind,
                determined_is_public=determined_is_public,
                env_tags=env_tags,
                is_torch_target=is_torch_target,
            )
        )

        has_var_kwargs = any("VAR_KEYWORD" in str(p.kind) for p in params)
        discovered_kwargs = extract_accepted_kwargs_from_ast(target) if has_var_kwargs else None

        overloads_list: list[dict[str, JsonValue] | GhostRef] = list(overloads_refs)
        return GhostPythonRef(
            name=name,
            api_path=api_path,
            kind=kind,
            params=params,
            docstring=doc,
            has_varargs=has_varargs,
            aliases=[],
            is_public=determined_is_public,
            returns_type=returns_type,
            returns_description=returns_description,
            raises=raises,
            environment_tags=env_tags,
            overloads=overloads_list,
            signature_completeness=sig_completeness,
            is_c_extension=is_c_ext,
            accepted_kwargs=discovered_kwargs,
        )

    @staticmethod
    def hydrate(
        data: (typing.Mapping[str, JsonValue | typing.Sequence[JsonValue | GhostParam]] | ScalarValue),
    ) -> ExtendedGhostRef:
        """Create a GhostRef from a dictionary (JSON snapshot).

        Args:
            data: The dictionary data.

        Returns:
            The hydrated ExtendedGhostRef or specialized polymorphic GhostRef object.
        """
        return hydrate_ghost_ref(data)


build_extended_params = _build_extended_params
coerce_param_record = _coerce_param_record
get_griffe_node_overloads = _get_griffe_node_overloads
get_griffe_node_params = _get_griffe_node_params
get_griffe_param_kind_name = _get_griffe_param_kind_name
merge_griffe_return_meta = _merge_griffe_return_meta
resolve_griffe_node_from_cache = _resolve_griffe_node_from_cache

__all__ = [
    "DocstringIrResult",
    "ExtractedParamRecord",
    "FRAMEWORK_CAPABILITIES",
    "GhostInspector",
    "OverloadExtractionContext",
    "ParamExtractionContext",
    "RawParamTuple",
    "STANDARD_ARG_MAP",
    "STANDARD_ENUM_MAP",
    "ast",
    "build_extended_params",
    "coerce_param_record",
    "extract_c_extension_signature",
    "extract_griffe_docstring_metadata",
    "get_griffe_node_overloads",
    "get_griffe_node_params",
    "get_griffe_param_kind_name",
    "get_runtime_signature",
    "get_runtime_source",
    "merge_griffe_return_meta",
    "resolve_griffe_node_from_cache",
    "to_parameter_kind",
]
