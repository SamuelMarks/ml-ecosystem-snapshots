"""Core `GhostInspector` engine for live and static ML framework symbol extraction."""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import re
import types
import typing
import typing_extensions

from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostParam,
    GhostPythonRef,
    GhostRef,
    ParameterKind,
)
import pydantic

if typing.TYPE_CHECKING:
    import griffe

from ml_ecosystem_snapshots.c_extension_sig import (
    CExtensionSignature,
    extract_c_extension_signature,
)
from ml_ecosystem_snapshots.introspection.ast_kwargs import (
    extract_accepted_kwargs_from_ast,
)
from ml_ecosystem_snapshots.introspection.sanitize import (
    sanitize_param_default,
    sanitize_type_str,
)
from ml_ecosystem_snapshots.introspection.static_parsers import (
    GriffeCacheEntry,
    GriffeNodeProtocol,
    _GRIFFE_CACHE,
)
from ml_ecosystem_snapshots.introspection.unwrap import unwrap_decorated_target
from ml_ecosystem_snapshots.models import JsonValue, ScalarValue
from ml_ecosystem_snapshots.serialization.hydration import hydrate_ghost_ref
from ml_ecosystem_snapshots.utils import (
    DocstringMetadata,
    DocstringParamMeta,
    extract_griffe_docstring_metadata,
    get_framework_docstring_parser,
    resolve_griffe_parser,
)

RawParamTuple: typing_extensions.TypeAlias = tuple[str, str, "str | None", "str | None"]


class ExtractedParamRecord(pydantic.BaseModel):
    """Intermediate extracted parameter metadata from CDD or Griffe docstrings/AST."""

    model_config = pydantic.ConfigDict(extra="allow", arbitrary_types_allowed=True)

    doc: str | None = None
    typ: str | None = None
    default: JsonValue | ast.AST | None = None

    def get(self, key: str, default: str | None = None) -> str | None:
        """Retrieve a string field (`doc`, `typ`, or `default`) by key.

        Args:
            key: Field name.
            default: Default value when field is None or absent.

        Returns:
            String representation or `default`.
        """
        val: JsonValue | ast.AST | None = getattr(self, key, None)
        return str(val) if val is not None else default


class DocstringIrResult(pydantic.BaseModel):
    """Structured result returned by `_extract_docstring_ir`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    cdd_params: dict[str, ExtractedParamRecord] = pydantic.Field(default_factory=dict)
    returns_type: str | None = None
    returns_description: str | None = None
    raises: list[str] = pydantic.Field(default_factory=list)


def to_parameter_kind(
    val: ParameterKind | inspect._ParameterKind | str,
) -> ParameterKind:
    """Normalize a parameter kind string or enum to `ParameterKind`.

    Args:
        val: Parameter kind string, enum, or `inspect.Parameter._ParameterKind`.

    Returns:
        The corresponding `ParameterKind` enum member.
    """
    if isinstance(val, ParameterKind):
        return val
    s = str(getattr(val, "name", val)).upper()
    try:
        return ParameterKind[s]
    except KeyError:
        return ParameterKind.POSITIONAL_OR_KEYWORD


STANDARD_ARG_MAP: dict[str, str] = {
    "x": "input",
    "inputs": "input",
    "input_tensor": "input",
    "y": "other",
    "other_tensor": "other",
    "dim": "dim",
    "axis": "dim",
    "keepdim": "keepdims",
    "keep_dims": "keepdims",
    "keepdims": "keepdims",
}

FRAMEWORK_CAPABILITIES: dict[str, list[str]] = {
    "torch": ["cpu", "cuda", "rocm", "metal"],
    "tensorflow": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "tf": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "jax": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "chex": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "grain": ["cpu", "cuda", "rocm", "tpu"],
    "pygrain": ["cpu", "cuda", "rocm", "tpu"],
    "orbax": ["cpu", "cuda", "rocm", "tpu"],
    "orbax_checkpoint": ["cpu", "cuda", "rocm", "tpu"],
    "optax": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "mlx": ["cpu", "metal"],
    "triton": ["cuda", "rocm"],
    "cupy": ["cuda", "rocm"],
    "numpy": ["cpu"],
    "sklearn": ["cpu"],
    "scikit_learn": ["cpu"],
    "scipy": ["cpu"],
    "deepspeed": ["cpu", "cuda", "rocm"],
    "onnxruntime": ["cpu", "cuda", "rocm", "metal"],
    "nvidia_sass": ["cuda"],
    "amd_rdna": ["rocm"],
    "mlir": ["cpu", "cuda", "rocm", "tpu"],
    "stablehlo": ["cpu", "cuda", "rocm", "tpu"],
    "ir": ["cpu", "cuda", "rocm", "tpu", "metal", "webgpu"],
    "wgsl": ["webgpu"],
}

STANDARD_ENUM_MAP: dict[str, list[str]] = {
    "reduction": ["none", "mean", "sum"],
    "mode": [
        "nearest",
        "linear",
        "bilinear",
        "bicubic",
        "trilinear",
        "area",
        "nearest-exact",
    ],
    "padding": [
        "valid",
        "same",
        "zeros",
        "reflect",
        "replicate",
        "circular",
    ],
    "layout": ["NCHW", "NHWC", "NCDHW", "NDHWC"],
}


def _coerce_param_record(
    val: (
        ExtractedParamRecord
        | DocstringParamMeta
        | typing.Mapping[str, JsonValue | ast.AST]
        | JsonValue
    ),
) -> ExtractedParamRecord:
    """Coerce a raw parameter dict or model into `ExtractedParamRecord`.

    Args:
        val: Raw dictionary or `ExtractedParamRecord` object.

    Returns:
        Populated `ExtractedParamRecord` instance.
    """
    if isinstance(val, dict):
        raw_doc = val.get("doc")
        raw_typ = val.get("typ")
        return ExtractedParamRecord(
            doc=str(raw_doc) if raw_doc is not None else None,
            typ=str(raw_typ) if raw_typ is not None else None,
            default=val.get("default"),
        )
    raw_doc_attr = getattr(val, "doc", None)
    raw_typ_attr = getattr(val, "typ", None)
    return ExtractedParamRecord(
        doc=str(raw_doc_attr) if raw_doc_attr is not None else None,
        typ=str(raw_typ_attr) if raw_typ_attr is not None else None,
        default=getattr(val, "default", None),
    )


def _extract_docstring_ir(
    doc: str | None,
    docstring_parser_name: str,
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
    active_extract_griffe_meta: typing.Callable[
        [str, str], DocstringMetadata | typing.Mapping[str, JsonValue]
    ],
) -> tuple[dict[str, ExtractedParamRecord], str | None, str | None, list[str]]:
    """Extract parameter descriptions, return types, and raised exceptions from docstrings.

    Args:
        doc: Raw docstring text.
        docstring_parser_name: Parser style ('google', 'numpy', 'sphinx').
        active_sanitize_type_str: Type annotation string sanitizer callable.
        active_extract_griffe_meta: Griffe docstring metadata extractor callable.

    Returns:
        Tuple of (cdd_params, returns_type, returns_description, raises).
    """
    try:
        import cdd.docstring.parse as cdd_docstring_parsers
    except (ImportError, AttributeError):
        cdd_docstring_parsers = None

    cdd_params: dict[str, ExtractedParamRecord] = {}
    returns_type: str | None = None
    returns_description: str | None = None
    raises: list[str] = []

    if doc and cdd_docstring_parsers is not None:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                cdd_ir = cdd_docstring_parsers.docstring(doc)
            if cdd_ir.get("returns") and "return_type" in cdd_ir["returns"]:
                ret = cdd_ir["returns"]["return_type"]
                returns_type = active_sanitize_type_str(ret.get("typ"))
                returns_description = ret.get("doc")

            if cdd_ir.get("params"):
                for p_name, p_val in cdd_ir["params"].items():
                    if "Exception" in p_name or "Error" in p_name:
                        raises.append(p_name)
                    else:
                        cdd_params[p_name] = _coerce_param_record(p_val)

            if cdd_ir.get("raises"):
                for exc_dict in cdd_ir["raises"]:
                    if "typ" in exc_dict:
                        raises.append(exc_dict["typ"])
        except Exception:
            pass

        try:
            griffe_meta = active_extract_griffe_meta(doc, docstring_parser_name)
            ret_meta = griffe_meta.get("returns")
            if ret_meta:
                ret_typ = (
                    ret_meta.get("typ")
                    if isinstance(ret_meta, dict)
                    else getattr(ret_meta, "typ", None)
                )
                ret_doc = (
                    ret_meta.get("doc")
                    if isinstance(ret_meta, dict)
                    else getattr(ret_meta, "doc", None)
                )
                if ret_typ and not returns_type:
                    returns_type = active_sanitize_type_str(str(ret_typ))
                if ret_doc and not returns_description:
                    returns_description = str(ret_doc)

            raw_raises = griffe_meta.get("raises", [])
            assert isinstance(raw_raises, list)
            for exc in raw_raises:
                exc_str = str(exc)
                if exc_str not in raises:
                    raises.append(exc_str)
                if exc_str in cdd_params:
                    del cdd_params[exc_str]

            raw_g_params = griffe_meta.get("params", {})
            assert isinstance(raw_g_params, dict)
            for p_name, p_data in raw_g_params.items():
                rec = _coerce_param_record(p_data)
                if p_name not in cdd_params:
                    cdd_params[p_name] = rec
                else:
                    if rec.doc and not cdd_params[p_name].doc:
                        cdd_params[p_name].doc = rec.doc
                    if rec.typ and not cdd_params[p_name].typ:
                        cdd_params[p_name].typ = rec.typ
        except Exception:
            pass

    return cdd_params, returns_type, returns_description, raises


def _extract_ast_and_griffe_node(
    obj: object,
    kind: str,
    api_path: str,
    docstring_parser_name: str,
    griffe_node: griffe.Module
    | griffe.Object
    | griffe.Alias
    | GriffeNodeProtocol
    | None,
    cdd_params: typing.Mapping[str, ExtractedParamRecord],
    active_ast_mod: types.ModuleType,
    active_griffe_cache: dict[str, GriffeCacheEntry],
) -> tuple[
    dict[str, ExtractedParamRecord],
    bool,
    griffe.Module | griffe.Object | griffe.Alias | GriffeNodeProtocol | None,
]:
    """Extract CDD AST parameter definitions, `super().__init__(**kwargs)` usage, and Griffe node.

    Args:
        obj: Target Python object.
        kind: Symbol kind ('class', 'function', 'method').
        api_path: Canonical dotted API path.
        docstring_parser_name: Docstring style name.
        griffe_node: Existing Griffe node if `obj` is already a Griffe object.
        cdd_params: Extracted docstring parameter dictionary.
        active_ast_mod: Active `ast` module reference.
        active_griffe_cache: Active Griffe module cache dictionary.

    Returns:
        Tuple of (cdd_ast_params, has_super_kwargs_call, griffe_node).
    """
    cdd_ast_params: dict[str, ExtractedParamRecord] = {}
    has_super_kwargs_call = False

    try:
        getsource_fn: typing.Callable[[object], str] = getattr(inspect, "getsource")
        source = getsource_fn(obj)
        parsed_ast = active_ast_mod.parse(source).body[0]

        if kind == "class":
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "__init__"
                    ):
                        if (
                            isinstance(node.func.value, ast.Call)
                            and isinstance(node.func.value.func, ast.Name)
                            and node.func.value.func.id == "super"
                        ):
                            for keyword in node.keywords:
                                if keyword.arg is None:
                                    has_super_kwargs_call = True
                                    break

        if kind == "class":
            import cdd.class_.parse

            with contextlib.redirect_stderr(io.StringIO()):
                cdd_parsed_ir = cdd.class_.parse.class_(
                    parsed_ast, merge_inner_function="__init__"
                )
        else:
            import cdd.function.parse

            with contextlib.redirect_stderr(io.StringIO()):
                cdd_parsed_ir = cdd.function.parse.function(parsed_ast)

        if cdd_parsed_ir and "params" in cdd_parsed_ir:
            for p_name, p_val in cdd_parsed_ir["params"].items():
                rec = _coerce_param_record(p_val)
                if p_name in cdd_params and cdd_params[p_name].doc:
                    rec.doc = cdd_params[p_name].doc
                cdd_ast_params[p_name] = rec
    except Exception:
        pass

    griffe_mod: types.ModuleType | None
    try:
        import griffe as _griffe_mod

        griffe_mod = _griffe_mod
    except ImportError:
        griffe_mod = None

    if griffe_node is None and griffe_mod is not None:
        try:
            resolved_parser = resolve_griffe_parser(docstring_parser_name)
            loader_fn = getattr(griffe_mod, "load")
            if active_griffe_cache.get("__loader__") is not loader_fn:
                active_griffe_cache.clear()
                active_griffe_cache["__loader__"] = loader_fn
            top_level = api_path.split(".")[0]
            if top_level not in active_griffe_cache:
                active_griffe_cache[top_level] = loader_fn(
                    top_level, docstring_parser=resolved_parser
                )

            parts = api_path.split(".")
            current = active_griffe_cache[top_level]
            if len(parts) > 1 and hasattr(current, "members"):
                for part in parts[1:]:
                    current = getattr(current, "members")[part]
            if isinstance(current, GriffeNodeProtocol):
                griffe_node = current
        except Exception:
            pass

    return cdd_ast_params, has_super_kwargs_call, griffe_node


def _extract_raw_parameters(
    obj: object,
    target: object,
    name: str,
    kind: str,
    api_path: str,
    cdd_ast_params: typing.Mapping[str, ExtractedParamRecord],
    griffe_node: griffe.Module
    | griffe.Object
    | griffe.Alias
    | GriffeNodeProtocol
    | None,
    resolved_hints: typing.Mapping[str, type | types.GenericAlias | str],
    returns_type: str | None,
    environment_tags: list[str] | None,
    active_ast_mod: types.ModuleType,
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
    active_extract_c_ext_sig: typing.Callable[
        [object, str], CExtensionSignature | None
    ],
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
        obj: Unwrapped Python object.
        target: Callable target (`obj` or `obj.__init__`).
        name: Unqualified symbol name.
        kind: Symbol kind ('class', 'function', 'method').
        api_path: Canonical dotted API path.
        cdd_ast_params: CDD AST parameter dictionary.
        griffe_node: Optional Griffe AST node.
        resolved_hints: Resolved `typing.get_type_hints` dictionary.
        returns_type: Current inferred return type string.
        environment_tags: Optional explicit execution environment tags.
        active_ast_mod: Active `ast` module reference.
        active_sanitize_type_str: Type sanitizer callable.
        active_extract_c_ext_sig: C-extension signature extractor callable.

    Returns:
        Tuple of (extracted_params, c_ext_params, has_varargs, is_c_ext,
        sig_completeness, returns_type, environment_tags).
    """
    extracted_params: list[RawParamTuple] = []
    c_ext_params: CExtensionSignature | None = None
    has_varargs = False
    is_c_ext = False
    sig_completeness: typing.Literal["exact", "heuristic", "opaque"] = "exact"

    griffe_params = None
    has_griffe_params = False
    try:
        griffe_params = getattr(griffe_node, "parameters", None)
        if griffe_params is None and getattr(griffe_node, "is_class", False):
            members = getattr(griffe_node, "members", {})
            init_node = members.get("__init__") or members.get("__new__")
            griffe_params = getattr(init_node, "parameters", None)
        if griffe_params and len(griffe_params) > 0:
            has_griffe_params = True
    except Exception:
        pass

    if cdd_ast_params:
        for p_name, p_info in cdd_ast_params.items():
            if p_name == "self":
                continue
            p_kind_str = "POSITIONAL_OR_KEYWORD"

            default_val = None
            raw_def = p_info.default
            if raw_def is not None:
                try:
                    if isinstance(raw_def, ast.AST):
                        default_val = str(active_ast_mod.unparse(raw_def))
                    elif isinstance(raw_def, str):
                        default_val = repr(raw_def)
                    else:
                        default_val = str(raw_def)
                except Exception:
                    default_val = str(raw_def)

            anno_val = p_info.typ
            if not anno_val and p_name in resolved_hints:
                anno_val = active_sanitize_type_str(str(resolved_hints[p_name]))
            else:
                anno_val = active_sanitize_type_str(anno_val)

            extracted_params.append((p_name, p_kind_str, default_val, anno_val))

        if has_griffe_params and griffe_params is not None:
            for param in griffe_params:
                if param.name == "self":
                    continue
                if param.name not in cdd_ast_params:
                    p_kind_str = (
                        param.kind.name.upper()
                        if getattr(param, "kind", None)
                        else "POSITIONAL_OR_KEYWORD"
                    )
                    if p_kind_str == "VAR_POSITIONAL":
                        has_varargs = True
                    default_val = (
                        str(param.default) if param.default is not None else None
                    )
                    anno_val = (
                        active_sanitize_type_str(
                            str(resolved_hints.get(param.name, param.annotation))
                        )
                        if param.annotation
                        else None
                    )
                    extracted_params.append(
                        (param.name, p_kind_str, default_val, anno_val)
                    )
                else:
                    for i, (pn, pk, pd, pa) in enumerate(extracted_params):
                        if pn == param.name:
                            p_kind_str = (
                                param.kind.name.upper()
                                if getattr(param, "kind", None)
                                else pk
                            )
                            extracted_params[i] = (pn, p_kind_str, pd, pa)
        else:
            try:
                sig_fn1: typing.Callable[[object], inspect.Signature] = getattr(
                    inspect, "signature"
                )
                sig = sig_fn1(target)
                for param in sig.parameters.values():
                    if param.name == "self":
                        continue
                    if param.name not in cdd_ast_params:
                        p_kind_str = str(param.kind)
                        if param.kind == inspect.Parameter.VAR_POSITIONAL:
                            has_varargs = True

                        default_val = None
                        anno_val = (
                            active_sanitize_type_str(
                                str(resolved_hints.get(param.name, param.annotation))
                            )
                            if param.annotation is not inspect.Parameter.empty
                            else None
                        )
                        extracted_params.append(
                            (param.name, p_kind_str, default_val, anno_val)
                        )
                    else:
                        for i, (pn, _pk, pd, pa) in enumerate(extracted_params):
                            if pn == param.name:
                                extracted_params[i] = (pn, str(param.kind), pd, pa)
            except Exception:
                pass

    elif has_griffe_params and griffe_params is not None:
        for param in griffe_params:
            if param.name == "self":
                continue

            p_kind_str = (
                param.kind.name.upper()
                if getattr(param, "kind", None)
                else "POSITIONAL_OR_KEYWORD"
            )
            if p_kind_str == "VAR_POSITIONAL":
                has_varargs = True

            default_val = str(param.default) if param.default is not None else None
            if param.name in resolved_hints:
                anno_val = active_sanitize_type_str(str(resolved_hints[param.name]))
            else:
                anno_val = (
                    active_sanitize_type_str(str(param.annotation))
                    if param.annotation
                    else None
                )
            extracted_params.append((param.name, p_kind_str, default_val, anno_val))
    else:
        try:
            sig_fn2: typing.Callable[[object], inspect.Signature] = getattr(
                inspect, "signature"
            )
            sig = sig_fn2(target)

            for param in sig.parameters.values():
                if param.name == "self":
                    continue

                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    has_varargs = True

                default_val, _factory, _is_mand = sanitize_param_default(param.default)

                anno_val = None
                if param.name in resolved_hints:
                    anno_val = active_sanitize_type_str(str(resolved_hints[param.name]))
                elif param.annotation is not inspect.Parameter.empty:
                    if hasattr(param.annotation, "__name__"):
                        anno_val = param.annotation.__name__
                    else:
                        anno_val = str(param.annotation)
                    anno_val = active_sanitize_type_str(anno_val)

                extracted_params.append(
                    (param.name, str(param.kind), default_val, anno_val)
                )

            if (
                inspect.isbuiltin(obj)
                or getattr(target, "__module__", "").startswith("_")
                or "torch._C" in str(target)
                or type(target).__name__ == "builtin_function_or_method"
            ):
                is_c_ext = True

        except (ValueError, TypeError):
            is_c_ext = True
            c_ext_params = active_extract_c_ext_sig(target, name)

            is_torch_target = (
                (name and name.startswith("torch."))
                or (getattr(target, "__module__", "") or "").startswith("torch")
                or api_path.startswith("torch.")
            )
            is_exact_aten = False
            if is_torch_target:
                try:
                    from ml_ecosystem_snapshots.frameworks.torch import (
                        extract_aten_c_extension_signature,
                    )

                    aten_sig = extract_aten_c_extension_signature(
                        target,
                        name,
                        is_method=(kind == "method" or ".Tensor." in name),
                    )
                    if aten_sig is not None:
                        if c_ext_params is None:
                            c_ext_params = aten_sig
                            is_exact_aten = True
                        elif (
                            not getattr(c_ext_params, "overloads", None)
                            and aten_sig.overloads
                        ):
                            c_ext_params.overloads = aten_sig.overloads
                            is_exact_aten = True
                except Exception:
                    pass

            if c_ext_params is not None:
                sig_completeness = "exact" if is_exact_aten else "heuristic"
                if (returns_type is None or returns_type == "NoneType") and getattr(
                    c_ext_params, "returns_type", None
                ):
                    returns_type = active_sanitize_type_str(c_ext_params.returns_type)
                for idx in range(len(c_ext_params)):
                    p_item = c_ext_params[idx]
                    pn = str(p_item[0])
                    pk = str(p_item[1])
                    pd = p_item[2]
                    pa = p_item[3]
                    if pk == "VAR_POSITIONAL":
                        has_varargs = True

                    sanitized_pa = active_sanitize_type_str(pa) if pa else None
                    extracted_params.append((pn, pk, pd, sanitized_pa))
            elif kind == "function":
                sig_completeness = "opaque"
                has_varargs = True
                extracted_params.append(("args", "VAR_POSITIONAL", None, None))
                extracted_params.append(("kwargs", "VAR_KEYWORD", None, None))
                if environment_tags is None:
                    environment_tags = []
                if "inexact_signature" not in environment_tags:
                    environment_tags.append("inexact_signature")
                if "opaque_c_extension" not in environment_tags:
                    environment_tags.append("opaque_c_extension")

    return (
        extracted_params,
        c_ext_params,
        has_varargs,
        is_c_ext,
        sig_completeness,
        returns_type,
        environment_tags,
    )


def _merge_mro_super_kwargs(
    obj: object,
    extracted_params: list[RawParamTuple],
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
) -> None:
    """Merge parent `__init__` parameters across MRO when `super().__init__(**kwargs)` is used.

    Args:
        obj: Class object being inspected.
        extracted_params: Mutable list of `(name, kind, default, annotation)` tuples.
        active_sanitize_type_str: Type annotation string sanitizer callable.
    """
    for parent in getattr(obj, "__mro__", ())[1:]:
        if parent is object:
            continue
        try:
            parent_sig = inspect.signature(parent.__init__)
            for param in parent_sig.parameters.values():
                if param.name == "self" or param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if not any(ep[0] == param.name for ep in extracted_params):
                    default_val = (
                        str(param.default)
                        if param.default is not inspect.Parameter.empty
                        else None
                    )
                    anno_val = (
                        param.annotation.__name__
                        if hasattr(param.annotation, "__name__")
                        else (
                            str(param.annotation)
                            if param.annotation is not inspect.Parameter.empty
                            else None
                        )
                    )
                    anno_val = active_sanitize_type_str(anno_val)
                    extracted_params.append(
                        (param.name, str(param.kind), default_val, anno_val)
                    )
        except Exception:
            pass


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
        var_kw_idx = next(
            i for i, ep in enumerate(extracted_params) if ep[1] == "VAR_KEYWORD"
        )
        promoted_params: list[RawParamTuple] = []
        for p_name, p_data in cdd_params.items():
            if (
                p_name not in existing_names
                and p_name not in ("kwargs", "args", "self")
                and not p_name.startswith("_")
            ):
                raw_typ = p_data.get("typ")
                p_typ = active_sanitize_type_str(raw_typ) if raw_typ else None
                p_def = p_data.get("default")
                promoted_params.append((p_name, "KEYWORD_ONLY", p_def, p_typ))
                existing_names.add(p_name)

        if promoted_params:
            return (
                extracted_params[:var_kw_idx]
                + promoted_params
                + extracted_params[var_kw_idx:]
            )
    return extracted_params


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

        p_dtypes: list[str] | None = None
        p_rank: str | None = None
        if is_torch_target:
            try:
                from ml_ecosystem_snapshots.frameworks.torch import (
                    infer_torch_dtype_and_rank,
                )

                p_dtypes, p_rank = infer_torch_dtype_and_rank(
                    name, p_name, p_anno or ""
                )
            except Exception:
                pass

        p_factory = None
        if p_default and "<factory_default>" in str(p_default):
            p_factory = "<factory_default>"

        p_allowed_values = None
        if p_anno and "Literal[" in p_anno:
            p_allowed_values = re.findall(r"['\x22]([^'\x22]+)['\x22]", p_anno)
        elif p_name.lower() in STANDARD_ENUM_MAP:
            p_allowed_values = STANDARD_ENUM_MAP[p_name.lower()]

        params.append(
            ExtendedGhostParam(
                name=p_name,
                standardized_name=STANDARD_ARG_MAP.get(p_name),
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


def _extract_overloads(
    griffe_node: griffe.Module
    | griffe.Object
    | griffe.Alias
    | GriffeNodeProtocol
    | None,
    c_ext_params: CExtensionSignature | None,
    name: str,
    api_path: str,
    kind: str,
    determined_is_public: bool,
    env_tags: list[str],
    is_torch_target: bool,
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
) -> list[ExtendedGhostRef]:
    """Extract function/method overloads from Griffe AST nodes or C-extension docstrings.

    Args:
        griffe_node: Optional Griffe AST node.
        c_ext_params: Optional parsed `CExtensionSignature` with `.overloads`.
        name: Unqualified symbol name.
        api_path: Canonical dotted API path.
        kind: Symbol kind ('function', 'class', 'method').
        determined_is_public: Public visibility flag.
        env_tags: Execution environment tags.
        is_torch_target: Whether symbol belongs to PyTorch.
        active_sanitize_type_str: Type annotation string sanitizer callable.

    Returns:
        List of `ExtendedGhostRef` overload objects.
    """
    overloads_refs: list[ExtendedGhostRef] = []
    has_griffe_overloads = False
    griffe_overloads = None
    try:
        griffe_overloads = getattr(griffe_node, "overloads", None)
        if not griffe_overloads and getattr(griffe_node, "is_class", False):
            init_node = getattr(griffe_node, "members", {}).get("__init__")
            if init_node:
                griffe_overloads = getattr(init_node, "overloads", None)
        if griffe_overloads:
            has_griffe_overloads = True
    except Exception:
        pass

    if has_griffe_overloads and griffe_overloads is not None:
        for overload in griffe_overloads:
            if isinstance(overload, str):
                continue
            ov_raw_tuples: list[RawParamTuple] = []
            overload_has_varargs = False
            if hasattr(overload, "parameters"):
                for param in overload.parameters:
                    if param.name == "self":
                        continue
                    p_kind_str = (
                        param.kind.name.upper()
                        if getattr(param, "kind", None)
                        else "POSITIONAL_OR_KEYWORD"
                    )
                    if p_kind_str == "VAR_POSITIONAL":
                        overload_has_varargs = True
                    default_val = (
                        str(param.default) if param.default is not None else None
                    )
                    anno_val = (
                        active_sanitize_type_str(str(param.annotation))
                        if param.annotation
                        else None
                    )
                    ov_raw_tuples.append(
                        (param.name, p_kind_str, default_val, anno_val)
                    )

            overload_params = _build_extended_params(
                extracted_params=ov_raw_tuples,
                cdd_params={},
                cdd_ast_params={},
                is_torch_target=False,
                name=name,
                active_sanitize_type_str=active_sanitize_type_str,
            )
            overload_returns = (
                active_sanitize_type_str(str(overload.returns))
                if getattr(overload, "returns", None)
                else None
            )

            overloads_refs.append(
                ExtendedGhostRef(
                    name=name,
                    api_path=api_path,
                    kind=kind,
                    params=overload_params,
                    docstring=None,
                    has_varargs=overload_has_varargs,
                    aliases=[],
                    is_public=determined_is_public,
                    returns_type=overload_returns,
                    returns_description=None,
                    raises=[],
                    environment_tags=env_tags,
                    overloads=[],
                )
            )
    elif c_ext_params is not None and getattr(c_ext_params, "overloads", None):
        for ov in c_ext_params.overloads:
            ov_raw_tuples = []
            ov_has_varargs = False
            for idx in range(len(ov)):
                p_item = ov[idx]
                pn = str(p_item[0])
                pk = str(p_item[1])
                pd = p_item[2]
                pa = p_item[3]
                if pk == "VAR_POSITIONAL":
                    ov_has_varargs = True
                sanitized_pa = active_sanitize_type_str(pa) if pa else None
                ov_raw_tuples.append((pn, pk, pd, sanitized_pa))

            ov_params = _build_extended_params(
                extracted_params=ov_raw_tuples,
                cdd_params={},
                cdd_ast_params={},
                is_torch_target=is_torch_target,
                name=name,
                active_sanitize_type_str=active_sanitize_type_str,
            )
            ov_ret = (
                active_sanitize_type_str(ov.returns_type)
                if getattr(ov, "returns_type", None)
                else None
            )
            overloads_refs.append(
                ExtendedGhostRef(
                    name=name,
                    api_path=api_path,
                    kind=kind,
                    params=ov_params,
                    docstring=None,
                    has_varargs=ov_has_varargs,
                    aliases=[],
                    is_public=determined_is_public,
                    returns_type=ov_ret,
                    returns_description=None,
                    raises=[],
                    environment_tags=env_tags,
                    overloads=[],
                )
            )
    return overloads_refs


class GhostInspector:
    """Facade for API Inspection.

    Responsibility: Convert Live Objects -> JSON-serializable GhostDefs.
    Crucial for populating snapshots used by WASM/JS environments.
    """

    @staticmethod
    def inspect(
        obj: object,
        api_path: str,
        is_public: bool | None = None,
        environment_tags: list[str] | None = None,
        kind: str | None = None,
    ) -> GhostPythonRef:
        """Create a GhostRef from a live Python object.

        Gracefully handles C-Extensions and builtins that resist introspection.

        Args:
            obj: The live class or function to inspect.
            api_path: The canonical string path (e.g. 'torch.nn.ReLU').
            is_public: Optional override for public visibility.
            environment_tags: Optional explicit execution tags (e.g. ['cpu', 'cuda']).
            kind: Optional explicit kind override (e.g. 'function', 'class', 'method').

        Returns:
            A populated GhostRef object.
        """
        obj = unwrap_decorated_target(obj)

        is_griffe_node = hasattr(obj, "is_class") and hasattr(obj, "is_function")
        if is_griffe_node:
            name = str(getattr(obj, "name", api_path.split(".")[-1]))
            determined_kind = (
                kind
                if kind is not None
                else ("class" if getattr(obj, "is_class", False) else "function")
            )
            doc_obj = getattr(obj, "docstring", None)
            doc = (
                str(doc_obj.value)
                if doc_obj and hasattr(doc_obj, "value")
                else (str(doc_obj) if doc_obj else None)
            )
            griffe_node: (
                griffe.Module | griffe.Object | griffe.Alias | GriffeNodeProtocol | None
            ) = obj if isinstance(obj, GriffeNodeProtocol) else None
        else:
            name = str(getattr(obj, "__name__", api_path.split(".")[-1]))
            if name == "<lambda>":
                name = api_path.split(".")[-1]
            determined_kind = (
                kind
                if kind is not None
                else ("class" if inspect.isclass(obj) else "function")
            )
            doc = inspect.getdoc(obj)
            griffe_node = None
        kind = determined_kind

        active_sanitize_type_str = sanitize_type_str
        active_extract_griffe_meta = extract_griffe_docstring_metadata
        active_extract_c_ext_sig = extract_c_extension_signature
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

        target = getattr(obj, "__init__", obj) if kind == "class" else obj

        cdd_ast_params, has_super_kwargs_call, griffe_node = (
            _extract_ast_and_griffe_node(
                obj=obj,
                kind=kind,
                api_path=api_path,
                docstring_parser_name=docstring_parser_name,
                griffe_node=griffe_node,
                cdd_params=cdd_params,
                active_ast_mod=active_ast_mod,
                active_griffe_cache=active_griffe_cache,
            )
        )

        if is_public is not None:
            determined_is_public = is_public
        elif griffe_node is not None and hasattr(griffe_node, "is_public"):
            determined_is_public = bool(getattr(griffe_node, "is_public"))
        else:
            determined_is_public = not name.startswith("_")

        resolved_hints: dict[str, type | types.GenericAlias | str] = {}
        if not is_griffe_node:
            try:
                resolved_hints = typing.get_type_hints(target)
                if "return" in resolved_hints and returns_type is None:
                    returns_type = active_sanitize_type_str(
                        str(resolved_hints["return"])
                    )
            except Exception:
                pass

        if griffe_node is not None and returns_type is None:
            try:
                g_ret = getattr(griffe_node, "returns", None)
                if g_ret:
                    returns_type = active_sanitize_type_str(str(g_ret))
            except Exception:
                pass

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
            name=name,
            kind=kind,
            api_path=api_path,
            cdd_ast_params=cdd_ast_params,
            griffe_node=griffe_node,
            resolved_hints=resolved_hints,
            returns_type=returns_type,
            environment_tags=environment_tags,
            active_ast_mod=active_ast_mod,
            active_sanitize_type_str=active_sanitize_type_str,
            active_extract_c_ext_sig=active_extract_c_ext_sig,
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

        is_torch_target = bool(
            (name and name.startswith("torch."))
            or (getattr(target, "__module__", "") or "").startswith("torch")
            or api_path.startswith("torch.")
        )

        params = _build_extended_params(
            extracted_params=extracted_params,
            cdd_params=cdd_params,
            cdd_ast_params=cdd_ast_params,
            is_torch_target=is_torch_target,
            name=name,
            active_sanitize_type_str=active_sanitize_type_str,
        )

        if environment_tags is not None:
            env_tags = list(environment_tags)
        else:
            top_pkg = api_path.split(".")[0].lower() if api_path else ""
            env_tags = list(FRAMEWORK_CAPABILITIES.get(top_pkg, ["cpu"]))

        overloads_refs = _extract_overloads(
            griffe_node=griffe_node,
            c_ext_params=c_ext_params,
            name=name,
            api_path=api_path,
            kind=kind,
            determined_is_public=determined_is_public,
            env_tags=env_tags,
            is_torch_target=is_torch_target,
            active_sanitize_type_str=active_sanitize_type_str,
        )

        has_var_kwargs = any(
            "VAR_KEYWORD" in str(getattr(p, "kind", "")) for p in params
        )
        discovered_kwargs = (
            extract_accepted_kwargs_from_ast(target) if has_var_kwargs else None
        )

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
        data: (
            typing.Mapping[str, JsonValue | typing.Sequence[JsonValue | GhostParam]]
            | ScalarValue
        ),
    ) -> ExtendedGhostRef:
        """Create a GhostRef from a dictionary (JSON snapshot).

        Args:
            data: The dictionary data.

        Returns:
            The hydrated ExtendedGhostRef or specialized polymorphic GhostRef object.
        """
        return hydrate_ghost_ref(data)


__all__ = [
    "DocstringIrResult",
    "ExtractedParamRecord",
    "FRAMEWORK_CAPABILITIES",
    "GhostInspector",
    "RawParamTuple",
    "STANDARD_ARG_MAP",
    "STANDARD_ENUM_MAP",
    "_build_extended_params",
    "_extract_ast_and_griffe_node",
    "_extract_docstring_ir",
    "_extract_overloads",
    "_extract_raw_parameters",
    "_merge_mro_super_kwargs",
    "_promote_documented_var_kwargs",
    "to_parameter_kind",
]
