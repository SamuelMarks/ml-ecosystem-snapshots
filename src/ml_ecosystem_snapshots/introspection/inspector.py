"""Core `GhostInspector` engine for live and static ML framework symbol extraction."""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import re
import types
import typing

import pydantic
import typing_extensions
from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostParam,
    GhostPythonRef,
    GhostRef,
    ParameterKind,
)

if typing.TYPE_CHECKING:
    import griffe

from ml_ecosystem_snapshots.c_extension_sig import (
    CExtensionSignature,
    extract_c_extension_signature,
)
from ml_ecosystem_snapshots.introspection.ast_kwargs import (
    extract_accepted_kwargs_from_ast,
)
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    GriffeDocstringProtocol,
    GriffeMembersProtocol,
    GriffeParameterProtocol,
    GriffeParameterWithKindProtocol,
    InspectableTarget,
    NamedKindProtocol,
    get_member_dict,
    get_runtime_signature,
    get_runtime_source,
)
from ml_ecosystem_snapshots.introspection.sanitize import (
    sanitize_param_default,
    sanitize_type_str,
)
from ml_ecosystem_snapshots.introspection.static_parsers import (
    _GRIFFE_CACHE,
    GriffeCacheEntry,
    GriffeLoaderProto,
    GriffeNodeProtocol,
)
from ml_ecosystem_snapshots.introspection.unwrap import unwrap_decorated_target
from ml_ecosystem_snapshots.models import JsonValue, ScalarValue
from ml_ecosystem_snapshots.serialization.hydration import hydrate_ghost_ref
from ml_ecosystem_snapshots.utils import (
    DocstringMetadata,
    DocstringParamMeta,
    DocstringReturnMeta,
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
        field_map: dict[str, JsonValue | ast.AST | None] = {
            "doc": self.doc,
            "typ": self.typ,
            "default": self.default,
        }
        val = field_map.get(key)
        return str(val) if val is not None else default


class DocstringIrResult(pydantic.BaseModel):
    """Structured result returned by `_extract_docstring_ir`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    cdd_params: dict[str, ExtractedParamRecord] = pydantic.Field(default_factory=dict)
    returns_type: str | None = None
    returns_description: str | None = None
    raises: list[str] = pydantic.Field(default_factory=list)


class ParamExtractionContext:
    """Bundled metadata and helper inputs for `_extract_raw_parameters`.

    Attributes:
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
    """

    name: str
    kind: str
    api_path: str
    cdd_ast_params: typing.Mapping[str, ExtractedParamRecord]
    griffe_node: griffe.Module | griffe.Object | griffe.Alias | GriffeNodeProtocol | None
    resolved_hints: typing.Mapping[str, type | types.GenericAlias | str]
    returns_type: str | None
    environment_tags: list[str] | None
    active_ast_mod: types.ModuleType
    active_sanitize_type_str: typing.Callable[[str | None], str | None]
    active_extract_c_ext_sig: typing.Callable[[InspectableTarget, str], CExtensionSignature | None]

    def __init__(
        self,
        name: str,
        kind: str,
        api_path: str,
        cdd_ast_params: typing.Mapping[str, ExtractedParamRecord],
        griffe_node: griffe.Module | griffe.Object | griffe.Alias | GriffeNodeProtocol | None,
        resolved_hints: typing.Mapping[str, type | types.GenericAlias | str],
        **extra: str | list[str] | None,
    ) -> None:
        """Initialize parameter extraction context.

        Args:
            name: Unqualified symbol name.
            kind: Symbol kind ('class', 'function', 'method').
            api_path: Canonical dotted API path.
            cdd_ast_params: CDD AST parameter dictionary.
            griffe_node: Optional Griffe AST node.
            resolved_hints: Resolved `typing.get_type_hints` dictionary.
            **extra: Additional runtime context (`returns_type`, `environment_tags`).
        """
        self.name = name
        self.kind = kind
        self.api_path = api_path
        self.cdd_ast_params = cdd_ast_params
        self.griffe_node = griffe_node
        self.resolved_hints = resolved_hints
        raw_ret = extra.get("returns_type")
        self.returns_type = str(raw_ret) if isinstance(raw_ret, str) else None
        raw_env = extra.get("environment_tags")
        self.environment_tags = list(raw_env) if isinstance(raw_env, list) else None
        self.active_ast_mod = ast
        self.active_sanitize_type_str = sanitize_type_str
        self.active_extract_c_ext_sig = extract_c_extension_signature


class OverloadExtractionContext:
    """Bundled inputs for extracting function or method overloads in `_extract_overloads`.

    Attributes:
        griffe_node: Optional Griffe AST node.
        c_ext_params: Optional parsed `CExtensionSignature` with `.overloads`.
        name: Unqualified symbol name.
        api_path: Canonical dotted API path.
        kind: Symbol kind ('function', 'class', 'method').
        determined_is_public: Public visibility flag.
        env_tags: Execution environment tags.
        is_torch_target: Whether symbol belongs to PyTorch.
        active_sanitize_type_str: Type annotation string sanitizer callable.
    """

    griffe_node: griffe.Module | griffe.Object | griffe.Alias | GriffeNodeProtocol | None
    c_ext_params: CExtensionSignature | None
    name: str
    api_path: str
    kind: str
    determined_is_public: bool
    env_tags: list[str]
    is_torch_target: bool
    active_sanitize_type_str: typing.Callable[[str | None], str | None]

    def __init__(
        self,
        griffe_node: griffe.Module | griffe.Object | griffe.Alias | GriffeNodeProtocol | None,
        c_ext_params: CExtensionSignature | None,
        name: str,
        api_path: str,
        kind: str,
        determined_is_public: bool,
        **extra: list[str] | bool,
    ) -> None:
        """Initialize overload extraction context.

        Args:
            griffe_node: Optional Griffe AST node.
            c_ext_params: Optional parsed `CExtensionSignature` with `.overloads`.
            name: Unqualified symbol name.
            api_path: Canonical dotted API path.
            kind: Symbol kind ('function', 'class', 'method').
            determined_is_public: Public visibility flag.
            **extra: Additional runtime context (`env_tags`, `is_torch_target`).
        """
        self.griffe_node = griffe_node
        self.c_ext_params = c_ext_params
        self.name = name
        self.api_path = api_path
        self.kind = kind
        self.determined_is_public = determined_is_public
        raw_env = extra.get("env_tags")
        assert isinstance(raw_env, list)
        self.env_tags = list(raw_env)
        self.is_torch_target = bool(extra.get("is_torch_target", False))
        self.active_sanitize_type_str = sanitize_type_str


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
    s = val.name.upper() if isinstance(val, inspect._ParameterKind) else str(val).upper()
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
    val: (ExtractedParamRecord | DocstringParamMeta | typing.Mapping[str, JsonValue | ast.AST] | JsonValue),
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
    assert isinstance(val, (ExtractedParamRecord, DocstringParamMeta))
    return ExtractedParamRecord(
        doc=str(val.doc) if val.doc is not None else None,
        typ=str(val.typ) if val.typ is not None else None,
        default=val.default,
    )


def _populate_cdd_docstring_ir(
    doc: str,
    cdd_docstring_parsers: types.ModuleType,
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
    result: DocstringIrResult,
) -> None:
    """Parse docstring metadata using `cdd.docstring.parse` into `result`.

    Args:
        doc: Raw docstring text to parse.
        cdd_docstring_parsers: Imported `cdd.docstring.parse` module.
        active_sanitize_type_str: Type annotation string sanitizer callable.
        result: Mutable `DocstringIrResult` accumulator.
    """
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            cdd_ir = cdd_docstring_parsers.docstring(doc)
        if cdd_ir.get("returns") and "return_type" in cdd_ir["returns"]:
            ret = cdd_ir["returns"]["return_type"]
            result.returns_type = active_sanitize_type_str(ret.get("typ"))
            result.returns_description = ret.get("doc")

        if cdd_ir.get("params"):
            for p_name, p_val in cdd_ir["params"].items():
                if "Exception" in p_name or "Error" in p_name:
                    result.raises.append(p_name)
                else:
                    result.cdd_params[p_name] = _coerce_param_record(p_val)

        if cdd_ir.get("raises"):
            for exc_dict in cdd_ir["raises"]:
                if "typ" in exc_dict:
                    result.raises.append(exc_dict["typ"])
    except Exception:
        pass


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
    if isinstance(ret_meta, dict):
        ret_typ = ret_meta.get("typ")
        ret_doc = ret_meta.get("doc")
    else:
        assert isinstance(ret_meta, DocstringReturnMeta)
        ret_typ = ret_meta.typ
        ret_doc = ret_meta.doc
    if ret_typ and not result.returns_type:
        result.returns_type = active_sanitize_type_str(str(ret_typ))
    if ret_doc and not result.returns_description:
        result.returns_description = str(ret_doc)


def _merge_griffe_param_record(
    p_name: str,
    p_data: ExtractedParamRecord | DocstringParamMeta | typing.Mapping[str, JsonValue | ast.AST] | JsonValue,
    cdd_params: dict[str, ExtractedParamRecord],
) -> None:
    """Merge a single Griffe docstring parameter entry into `cdd_params`.

    Args:
        p_name: Parameter name from Griffe docstring metadata.
        p_data: Raw parameter metadata record or dictionary from Griffe.
        cdd_params: Mutable parameter dictionary keyed by parameter name.
    """
    rec = _coerce_param_record(p_data)
    if p_name not in cdd_params:
        cdd_params[p_name] = rec
    else:
        if rec.doc and not cdd_params[p_name].doc:
            cdd_params[p_name].doc = rec.doc
        if rec.typ and not cdd_params[p_name].typ:
            cdd_params[p_name].typ = rec.typ


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
        griffe_meta = active_extract_griffe_meta(doc, docstring_parser_name)
        ret_meta = griffe_meta.get("returns")
        if ret_meta:
            _merge_griffe_return_meta(ret_meta, active_sanitize_type_str, result)

        raw_raises = griffe_meta.get("raises", [])
        assert isinstance(raw_raises, list)
        for exc in raw_raises:
            exc_str = str(exc)
            if exc_str not in result.raises:
                result.raises.append(exc_str)
            if exc_str in result.cdd_params:
                del result.cdd_params[exc_str]

        raw_g_params = griffe_meta.get("params", {})
        assert isinstance(raw_g_params, dict)
        for p_name, p_data in raw_g_params.items():
            _merge_griffe_param_record(p_name, p_data, result.cdd_params)
    except Exception:
        pass


def _extract_docstring_ir(
    doc: str | None,
    docstring_parser_name: str,
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
    active_extract_griffe_meta: typing.Callable[[str, str], DocstringMetadata | typing.Mapping[str, JsonValue]],
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

    result = DocstringIrResult()
    if doc and cdd_docstring_parsers is not None:
        _populate_cdd_docstring_ir(doc, cdd_docstring_parsers, active_sanitize_type_str, result)
        _populate_griffe_docstring_ir(
            doc,
            docstring_parser_name,
            active_sanitize_type_str,
            active_extract_griffe_meta,
            result,
        )

    return result.cdd_params, result.returns_type, result.returns_description, result.raises


def _get_griffe_param_kind_name(param: GriffeParameterProtocol, default: str = "POSITIONAL_OR_KEYWORD") -> str:
    """Resolve the uppercase parameter kind string from a Griffe parameter node.

    Args:
        param: Griffe parameter node.
        default: Fallback kind string when `.kind` is absent.

    Returns:
        Uppercase parameter kind string.
    """
    assert isinstance(param, GriffeParameterWithKindProtocol) and param.kind is not None
    assert isinstance(param.kind, NamedKindProtocol)
    return str(param.kind.name).upper()


def _format_annotation_obj(ann: InspectableTarget) -> str:
    """Format a runtime parameter annotation into its string representation.

    Args:
        ann: Parameter annotation object from `inspect.Parameter`.

    Returns:
        String representation of the annotation.
    """
    if isinstance(ann, type):
        return repr(ann)[8:-2].split(".")[-1]
    return str(ann)


def _get_target_module_name(target: InspectableTarget) -> str:
    """Return the `__module__` string of a runtime target or empty string.

    Args:
        target: Runtime callable or class target.

    Returns:
        Module name string or `""`.
    """
    raw_mod = get_member_dict(target).get("__module__")
    return str(raw_mod) if raw_mod is not None else ""


def _is_griffe_loader(
    fn: InspectableTarget,
) -> typing_extensions.TypeGuard[GriffeLoaderProto]:
    """Narrow `fn` to `GriffeLoaderProto`.

    Args:
        fn: Candidate `griffe.load` callable.

    Returns:
        True if `fn` is callable.
    """
    return callable(fn)


def _is_super_init_call(node: ast.AST) -> bool:
    """Determine whether an AST node represents a `super().__init__(...)` call.

    Args:
        node: AST node to inspect.

    Returns:
        True if `node` is a call to `super().__init__`, False otherwise.
    """
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "__init__" and isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Name) and node.func.value.func.id == "super"


def _detect_super_kwargs_call(parsed_ast: ast.AST) -> bool:
    """Check whether a class AST contains a `super().__init__(**kwargs)` call.

    Args:
        parsed_ast: Parsed class AST node.

    Returns:
        True if any `super().__init__` call forwards `**kwargs`, False otherwise.
    """
    for node in ast.walk(parsed_ast):
        if _is_super_init_call(node):
            assert isinstance(node, ast.Call)
            for keyword in node.keywords:
                if keyword.arg is None:
                    return True
    return False


def _parse_cdd_ast_params(
    parsed_ast: ast.AST,
    kind: str,
    cdd_params: typing.Mapping[str, ExtractedParamRecord],
) -> dict[str, ExtractedParamRecord]:
    """Extract parameter definitions from a parsed class or function AST via CDD.

    Args:
        parsed_ast: Parsed class or function AST statement node.
        kind: Symbol kind ('class', 'function', 'method').
        cdd_params: Previously extracted docstring parameter records.

    Returns:
        Dictionary mapping parameter names to `ExtractedParamRecord` instances.
    """
    cdd_ast_params: dict[str, ExtractedParamRecord] = {}
    if kind == "class":
        import cdd.class_.parse

        with contextlib.redirect_stderr(io.StringIO()):
            cdd_parsed_ir = cdd.class_.parse.class_(parsed_ast, merge_inner_function="__init__")
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
    return cdd_ast_params


def _resolve_griffe_node_from_cache(
    api_path: str,
    docstring_parser_name: str,
    griffe_mod: types.ModuleType,
    active_griffe_cache: dict[str, GriffeCacheEntry],
) -> griffe.Module | griffe.Object | griffe.Alias | GriffeNodeProtocol | None:
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
        assert _is_griffe_loader(loader_fn)
        if active_griffe_cache.get("__loader__") is not loader_fn:
            active_griffe_cache.clear()
            active_griffe_cache["__loader__"] = loader_fn
        top_level = api_path.split(".")[0]
        if top_level not in active_griffe_cache:
            active_griffe_cache[top_level] = loader_fn(top_level, docstring_parser=resolved_parser)

        parts = api_path.split(".")
        current: GriffeCacheEntry | InspectableTarget = active_griffe_cache[top_level]
        if len(parts) > 1 and isinstance(current, GriffeMembersProtocol):
            for part in parts[1:]:
                assert isinstance(current, GriffeMembersProtocol)
                current = current.members[part]
        if isinstance(current, GriffeNodeProtocol):
            return current
    except Exception:
        pass
    return None


def _extract_ast_and_griffe_node(
    obj: InspectableTarget,
    kind: str,
    api_path: str,
    docstring_parser_name: str,
    griffe_node: griffe.Module | griffe.Object | griffe.Alias | GriffeNodeProtocol | None,
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
    except Exception:
        pass

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


def _get_griffe_node_params(
    griffe_node: griffe.Module | griffe.Object | griffe.Alias | GriffeNodeProtocol | None,
) -> typing.Iterable[GriffeParameterProtocol] | None:
    """Extract non-empty Griffe parameter iterable from a Griffe function or class node.

    Args:
        griffe_node: Optional Griffe AST node for the inspected symbol.

    Returns:
        Non-empty iterable of `GriffeParameterProtocol` items, or `None`.
    """
    g_node_members = get_member_dict(griffe_node)
    raw_g_params = g_node_members.get("parameters")
    if raw_g_params is None and bool(g_node_members.get("is_class", False)):
        members = g_node_members.get("members", {})
        assert isinstance(members, typing.Mapping)
        init_node = members.get("__init__") or members.get("__new__")
        raw_g_params = get_member_dict(init_node).get("parameters")
    if raw_g_params and isinstance(raw_g_params, typing.Iterable):
        return list(raw_g_params)
    return None


def _format_cdd_param_default(
    raw_def: JsonValue | ast.AST | None,
    active_ast_mod: types.ModuleType,
) -> str | None:
    """Format a CDD AST parameter default into a string representation.

    Args:
        raw_def: Raw default value or AST node from CDD parameter metadata.
        active_ast_mod: Active `ast` module reference used for `unparse`.

    Returns:
        Formatted default value string, or `None` when no default exists.
    """
    if raw_def is None:
        return None
    try:
        if isinstance(raw_def, ast.AST):
            return str(active_ast_mod.unparse(raw_def))
        if isinstance(raw_def, str):
            return repr(raw_def)
        return str(raw_def)
    except Exception:
        return str(raw_def)


def _build_initial_cdd_ast_params(
    cdd_ast_params: typing.Mapping[str, ExtractedParamRecord],
    resolved_hints: typing.Mapping[str, type | types.GenericAlias | str],
    active_ast_mod: types.ModuleType,
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
) -> list[RawParamTuple]:
    """Convert CDD AST parameter records into initial raw parameter tuples.

    Args:
        cdd_ast_params: Mapping of parameter names to `ExtractedParamRecord` instances.
        resolved_hints: Resolved runtime type hints mapping.
        active_ast_mod: Active `ast` module reference.
        active_sanitize_type_str: Type annotation string sanitizer callable.

    Returns:
        List of `(name, kind, default, annotation)` tuples excluding `self`.
    """
    extracted_params: list[RawParamTuple] = []
    for p_name, p_info in cdd_ast_params.items():
        if p_name == "self":
            continue
        default_val = _format_cdd_param_default(p_info.default, active_ast_mod)
        anno_val = p_info.typ
        if not anno_val and p_name in resolved_hints:
            anno_val = active_sanitize_type_str(str(resolved_hints[p_name]))
        else:
            anno_val = active_sanitize_type_str(anno_val)
        extracted_params.append((p_name, "POSITIONAL_OR_KEYWORD", default_val, anno_val))
    return extracted_params


def _supplement_cdd_params_from_griffe(
    extracted_params: list[RawParamTuple],
    cdd_ast_params: typing.Mapping[str, ExtractedParamRecord],
    griffe_params: typing.Iterable[GriffeParameterProtocol],
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
        if param.name not in cdd_ast_params:
            p_kind_str = _get_griffe_param_kind_name(param)
            if p_kind_str == "VAR_POSITIONAL":
                has_varargs = True
            default_val = str(param.default) if param.default is not None else None
            anno_val = active_sanitize_type_str(str(resolved_hints.get(param.name, param.annotation))) if param.annotation else None
            extracted_params.append((param.name, p_kind_str, default_val, anno_val))
        else:
            for i, (pn, pk, pd, pa) in enumerate(extracted_params):
                if pn == param.name:
                    p_kind_str = _get_griffe_param_kind_name(param, default=pk)
                    extracted_params[i] = (pn, p_kind_str, pd, pa)
    return has_varargs


def _supplement_cdd_params_from_signature(
    extracted_params: list[RawParamTuple],
    cdd_ast_params: typing.Mapping[str, ExtractedParamRecord],
    target: InspectableTarget,
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
        sig = get_runtime_signature(target)
        for sig_param in sig.parameters.values():
            if sig_param.name == "self":
                continue
            if sig_param.name not in cdd_ast_params:
                p_kind_str = str(sig_param.kind)
                if sig_param.kind == inspect.Parameter.VAR_POSITIONAL:
                    has_varargs = True
                anno_val = active_sanitize_type_str(str(resolved_hints.get(sig_param.name, sig_param.annotation))) if sig_param.annotation is not inspect.Parameter.empty else None
                extracted_params.append((sig_param.name, p_kind_str, None, anno_val))
            else:
                for i, (pn, _pk, pd, pa) in enumerate(extracted_params):
                    if pn == sig_param.name:
                        extracted_params[i] = (pn, str(sig_param.kind), pd, pa)
    except Exception:
        pass
    return has_varargs


def _extract_params_from_griffe_only(
    griffe_params: typing.Iterable[GriffeParameterProtocol],
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


def _extract_params_from_runtime_sig(
    obj: InspectableTarget,
    target: InspectableTarget,
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
    sig = get_runtime_signature(target)
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

    is_c_ext = bool(inspect.isbuiltin(obj) or _get_target_module_name(target).startswith("_") or "torch._C" in str(target) or "builtin_function_or_method" in repr(type(target)))
    return extracted_params, has_varargs, is_c_ext


def _is_torch_symbol(name: str, target: InspectableTarget, api_path: str) -> bool:
    """Determine whether the inspected target belongs to the PyTorch namespace.

    Args:
        name: Unqualified or partially qualified symbol name.
        target: Runtime callable target.
        api_path: Canonical dotted API path.

    Returns:
        True if the symbol belongs to `torch`, False otherwise.
    """
    return bool((name and name.startswith("torch.")) or _get_target_module_name(target).startswith("torch") or api_path.startswith("torch."))


def _resolve_c_ext_and_aten_sig(
    target: InspectableTarget,
    ctx: ParamExtractionContext,
) -> tuple[CExtensionSignature | None, bool]:
    """Extract C-extension signature and merge PyTorch ATen schema when applicable.

    Args:
        target: Callable target (`obj` or `obj.__init__`).
        ctx: Bundled parameter extraction context.

    Returns:
        Tuple of `(c_ext_params, is_exact_aten)`.
    """
    c_ext_params = ctx.active_extract_c_ext_sig(target, ctx.name)
    is_exact_aten = False
    if _is_torch_symbol(ctx.name, target, ctx.api_path):
        try:
            from ml_ecosystem_snapshots.frameworks.torch import (
                extract_aten_c_extension_signature,
            )

            aten_sig = extract_aten_c_extension_signature(
                target,
                ctx.name,
                is_method=(ctx.kind == "method" or ".Tensor." in ctx.name),
            )
            if aten_sig is not None:
                if c_ext_params is None:
                    c_ext_params = aten_sig
                    is_exact_aten = True
                elif not c_ext_params.overloads and aten_sig.overloads:
                    c_ext_params.overloads = aten_sig.overloads
                    is_exact_aten = True
        except Exception:
            pass
    return c_ext_params, is_exact_aten


def _append_opaque_env_tags(environment_tags: list[str] | None) -> list[str]:
    """Append opaque C-extension tags to the environment tag list if not already present.

    Args:
        environment_tags: Existing environment tag list or `None`.

    Returns:
        Updated environment tag list including `'inexact_signature'` and `'opaque_c_extension'`.
    """
    tags = [] if environment_tags is None else environment_tags
    if "inexact_signature" not in tags:
        tags.append("inexact_signature")
    if "opaque_c_extension" not in tags:
        tags.append("opaque_c_extension")
    return tags


def _extract_params_from_c_ext(
    target: InspectableTarget,
    ctx: ParamExtractionContext,
) -> tuple[
    list[RawParamTuple],
    CExtensionSignature | None,
    bool,
    typing.Literal["exact", "heuristic", "opaque"],
    str | None,
    list[str] | None,
]:
    """Fallback parameter extractor for C-extensions and opaque builtins.

    Args:
        target: Callable target (`obj` or `obj.__init__`).
        ctx: Bundled parameter extraction context.

    Returns:
        Tuple of `(extracted_params, c_ext_params, has_varargs, sig_completeness,
        returns_type, environment_tags)`.
    """
    extracted_params: list[RawParamTuple] = []
    has_varargs = False
    sig_completeness: typing.Literal["exact", "heuristic", "opaque"] = "exact"
    returns_type = ctx.returns_type
    environment_tags = ctx.environment_tags

    c_ext_params, is_exact_aten = _resolve_c_ext_and_aten_sig(target, ctx)
    if c_ext_params is not None:
        sig_completeness = "exact" if is_exact_aten else "heuristic"
        if (returns_type is None or returns_type == "NoneType") and c_ext_params.returns_type:
            returns_type = ctx.active_sanitize_type_str(c_ext_params.returns_type)
        for idx in range(len(c_ext_params)):
            p_item = c_ext_params[idx]
            pn = str(p_item[0])
            pk = str(p_item[1])
            pd = p_item[2]
            pa = p_item[3]
            if pk == "VAR_POSITIONAL":
                has_varargs = True
            sanitized_pa = ctx.active_sanitize_type_str(pa) if pa else None
            extracted_params.append((pn, pk, pd, sanitized_pa))
    elif ctx.kind == "function":
        sig_completeness = "opaque"
        has_varargs = True
        extracted_params.append(("args", "VAR_POSITIONAL", None, None))
        extracted_params.append(("kwargs", "VAR_KEYWORD", None, None))
        environment_tags = _append_opaque_env_tags(environment_tags)

    return (
        extracted_params,
        c_ext_params,
        has_varargs,
        sig_completeness,
        returns_type,
        environment_tags,
    )


def _extract_raw_parameters(
    obj: InspectableTarget,
    target: InspectableTarget,
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


def _merge_mro_super_kwargs(
    obj: InspectableTarget,
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
            parent_sig = get_runtime_signature(parent_init)
            for param in parent_sig.parameters.values():
                if param.name == "self" or param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if not any(ep[0] == param.name for ep in extracted_params):
                    default_val = str(param.default) if param.default is not inspect.Parameter.empty else None
                    anno_val = _format_annotation_obj(param.annotation) if param.annotation is not inspect.Parameter.empty else None
                    anno_val = active_sanitize_type_str(anno_val)
                    extracted_params.append((param.name, str(param.kind), default_val, anno_val))
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


def _get_griffe_node_overloads(
    griffe_node: griffe.Module | griffe.Object | griffe.Alias | GriffeNodeProtocol | None,
) -> list[InspectableTarget] | None:
    """Resolve non-empty overload list from a Griffe function or class node.

    Args:
        griffe_node: Optional Griffe AST node.

    Returns:
        Non-empty list of Griffe overload objects, or `None`.
    """
    g_node_members = get_member_dict(griffe_node)
    raw_overloads = g_node_members.get("overloads")
    if not raw_overloads and bool(g_node_members.get("is_class", False)):
        members = g_node_members.get("members", {})
        assert isinstance(members, typing.Mapping)
        init_node = members.get("__init__")
        if init_node:
            raw_overloads = get_member_dict(init_node).get("overloads")
    if isinstance(raw_overloads, list) and raw_overloads:
        return raw_overloads
    return None


def _build_griffe_overload_ref(
    overload: InspectableTarget,
    ctx: OverloadExtractionContext,
) -> ExtendedGhostRef:
    """Build an `ExtendedGhostRef` from a single Griffe overload node.

    Args:
        overload: Griffe overload node.
        ctx: Bundled overload extraction context.

    Returns:
        Constructed `ExtendedGhostRef` representing the overload signature.
    """
    ov_raw_tuples: list[RawParamTuple] = []
    overload_has_varargs = False
    ov_members = get_member_dict(overload)
    if "parameters" in ov_members:
        ov_params_seq = ov_members.get("parameters", ())
        assert isinstance(ov_params_seq, typing.Iterable)
        for param in ov_params_seq:
            assert isinstance(param, GriffeParameterProtocol)
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
    ov_ret_raw = ov_members.get("returns")
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


def _build_c_ext_overload_ref(
    ov: CExtensionSignature,
    ctx: OverloadExtractionContext,
) -> ExtendedGhostRef:
    """Build an `ExtendedGhostRef` from a single C-extension overload signature.

    Args:
        ov: Parsed `CExtensionSignature` overload entry.
        ctx: Bundled overload extraction context.

    Returns:
        Constructed `ExtendedGhostRef` representing the C-extension overload signature.
    """
    ov_raw_tuples: list[RawParamTuple] = []
    ov_has_varargs = False
    for idx in range(len(ov)):
        p_item = ov[idx]
        pn = str(p_item[0])
        pk = str(p_item[1])
        pd = p_item[2]
        pa = p_item[3]
        if pk == "VAR_POSITIONAL":
            ov_has_varargs = True
        sanitized_pa = ctx.active_sanitize_type_str(pa) if pa else None
        ov_raw_tuples.append((pn, pk, pd, sanitized_pa))

    ov_params = _build_extended_params(
        extracted_params=ov_raw_tuples,
        cdd_params={},
        cdd_ast_params={},
        is_torch_target=ctx.is_torch_target,
        name=ctx.name,
        active_sanitize_type_str=ctx.active_sanitize_type_str,
    )
    ov_ret_val = ov.returns_type
    ov_ret = ctx.active_sanitize_type_str(str(ov_ret_val)) if ov_ret_val else None
    return ExtendedGhostRef(
        name=ctx.name,
        api_path=ctx.api_path,
        kind=ctx.kind,
        params=ov_params,
        docstring=None,
        has_varargs=ov_has_varargs,
        aliases=[],
        is_public=ctx.determined_is_public,
        returns_type=ov_ret,
        returns_description=None,
        raises=[],
        environment_tags=ctx.env_tags,
        overloads=[],
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


def _resolve_target_identity(
    obj: InspectableTarget,
    obj_members: typing.Mapping[str, InspectableTarget],
    api_path: str,
    kind: str | None,
) -> tuple[
    str,
    str,
    str | None,
    griffe.Module | griffe.Object | griffe.Alias | GriffeNodeProtocol | None,
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
    if isinstance(obj, GriffeNodeProtocol) and "is_class" in obj_members and "is_function" in obj_members:
        name = str(obj_members.get("name") or api_path.split(".")[-1])
        determined_kind = kind if kind is not None else ("class" if bool(obj_members.get("is_class", False)) else "function")
        doc_obj = obj_members.get("docstring")
        doc = str(doc_obj.value) if isinstance(doc_obj, GriffeDocstringProtocol) else (str(doc_obj) if doc_obj is not None else None)
        return name, determined_kind, doc, obj, True

    name = _format_annotation_obj(obj) if isinstance(obj, type) else str(obj_members.get("__name__") or api_path.split(".")[-1])
    if name == "<lambda>":
        name = api_path.split(".")[-1]
    determined_kind = kind if kind is not None else ("class" if isinstance(obj, type) else "function")
    return name, determined_kind, inspect.getdoc(obj), None, False


def _resolve_visibility_and_type_hints(
    target: InspectableTarget,
    name: str,
    is_public: bool | None,
    is_griffe_node: bool,
    griffe_node: griffe.Module | griffe.Object | griffe.Alias | GriffeNodeProtocol | None,
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
    g_node_members = get_member_dict(griffe_node)
    if is_public is not None:
        determined_is_public = is_public
    elif griffe_node is not None and "is_public" in g_node_members:
        determined_is_public = bool(g_node_members.get("is_public", True))
    else:
        determined_is_public = not name.startswith("_")

    resolved_hints: dict[str, type | types.GenericAlias | str] = {}
    if not is_griffe_node:
        try:
            resolved_hints = typing.get_type_hints(target)
            if "return" in resolved_hints and returns_type is None:
                returns_type = active_sanitize_type_str(str(resolved_hints["return"]))
        except Exception:
            pass

    if griffe_node is not None and returns_type is None:
        g_ret = g_node_members.get("returns")
        if g_ret:
            returns_type = active_sanitize_type_str(str(g_ret))

    return determined_is_public, resolved_hints, returns_type


def _resolve_environment_tags(
    environment_tags: list[str] | None,
    api_path: str,
) -> list[str]:
    """Resolve execution environment capability tags for an inspected symbol.

    Args:
        environment_tags: Explicit environment tags list, or `None` for framework defaults.
        api_path: Canonical dotted API path used to infer the top-level package.

    Returns:
        List of execution environment tag strings.
    """
    if environment_tags is not None:
        return list(environment_tags)
    top_pkg = api_path.split(".")[0].lower() if api_path else ""
    return list(FRAMEWORK_CAPABILITIES.get(top_pkg, ["cpu"]))


class GhostInspector:
    """Facade for API Inspection.

    Responsibility: Convert Live Objects -> JSON-serializable GhostDefs.
    Crucial for populating snapshots used by WASM/JS environments.
    """

    @staticmethod
    def inspect(
        obj: InspectableTarget,
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
    "_build_extended_params",
    "_extract_ast_and_griffe_node",
    "_extract_docstring_ir",
    "_extract_overloads",
    "_extract_raw_parameters",
    "_merge_mro_super_kwargs",
    "_promote_documented_var_kwargs",
    "to_parameter_kind",
]
