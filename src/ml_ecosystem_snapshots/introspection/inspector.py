"""Core `GhostInspector` engine for live and static ML framework symbol extraction."""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import re
import typing
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from ml_switcheroo_ir.schema.ghost import (
    ExtendedGhostParam,
    ExtendedGhostRef,
    GhostParam,
    GhostPythonRef,
    ParameterKind,
)

from ml_ecosystem_snapshots.introspection.ast_kwargs import (
    extract_accepted_kwargs_from_ast,
)
from ml_ecosystem_snapshots.introspection.sanitize import (
    sanitize_param_default,
    sanitize_type_str,
)
from ml_ecosystem_snapshots.introspection.static_parsers import _GRIFFE_CACHE
from ml_ecosystem_snapshots.introspection.unwrap import unwrap_decorated_target
from ml_ecosystem_snapshots.serialization.hydration import hydrate_ghost_ref
from ml_ecosystem_snapshots.utils import (
    extract_c_extension_signature,
    extract_griffe_docstring_metadata,
    get_framework_docstring_parser,
    resolve_griffe_parser,
)


def to_parameter_kind(val: Any) -> ParameterKind:
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


STANDARD_ARG_MAP: Dict[str, str] = {
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

FRAMEWORK_CAPABILITIES: Dict[str, List[str]] = {
    "torch": ["cpu", "cuda", "rocm", "metal"],
    "tensorflow": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "tf": ["cpu", "cuda", "rocm", "tpu", "metal"],
    "jax": ["cpu", "cuda", "rocm", "tpu", "metal"],
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

STANDARD_ENUM_MAP: Dict[str, List[str]] = {
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


class GhostInspector:
    """Facade for API Inspection.

    Responsibility: Convert Live Objects -> JSON-serializable GhostDefs.
    Crucial for populating snapshots used by WASM/JS environments.
    """

    @staticmethod
    def inspect(
        obj: Union[Any, Callable[..., Any]],
        api_path: str,
        is_public: Optional[bool] = None,
        environment_tags: Optional[List[str]] = None,
        kind: Optional[str] = None,
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
        import ml_ecosystem_snapshots.models as _models_mod

        try:
            import cdd.docstring.parse as cdd_docstring_parsers
        except ImportError:
            cdd_docstring_parsers = None

        griffe_mod: Optional[Any]
        try:
            import griffe as _griffe_mod

            griffe_mod = _griffe_mod
        except ImportError:
            griffe_mod = None

        obj = unwrap_decorated_target(obj)

        is_griffe_node = hasattr(obj, "is_class") and hasattr(obj, "is_function")
        if is_griffe_node:
            name = getattr(obj, "name", api_path.split(".")[-1])
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
            griffe_node = obj
        else:
            name = getattr(obj, "__name__", api_path.split(".")[-1])
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
        params: List[GhostParam] = []
        has_varargs = False
        is_c_ext = False
        sig_completeness: Literal["exact", "heuristic", "opaque"] = "exact"

        determined_is_public = True
        if is_public is not None:
            determined_is_public = is_public

        # Resolve patchable hooks via models module if overridden in tests
        active_sanitize_type_str = getattr(
            _models_mod, "sanitize_type_str", sanitize_type_str
        )
        active_extract_griffe_meta = getattr(
            _models_mod,
            "extract_griffe_docstring_metadata",
            extract_griffe_docstring_metadata,
        )
        active_extract_c_ext_sig = getattr(
            _models_mod,
            "extract_c_extension_signature",
            extract_c_extension_signature,
        )
        active_griffe_cache = getattr(_models_mod, "_GRIFFE_CACHE", _GRIFFE_CACHE)

        # 1. Try to load docstring information via cdd and griffe reST/Sphinx enumeration
        cdd_params = {}
        returns_type = None
        returns_description = None
        raises = []

        top_level = api_path.split(".")[0] if api_path else ""
        docstring_parser_name = get_framework_docstring_parser(top_level, doc)

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
                            cdd_params[p_name] = p_val

                if cdd_ir.get("raises"):
                    for exc_dict in cdd_ir["raises"]:
                        if "typ" in exc_dict:
                            raises.append(exc_dict["typ"])

            except Exception:
                pass

            # Supplement with Griffe's structured docstring enumeration
            try:
                griffe_meta = active_extract_griffe_meta(
                    doc, parser_name=docstring_parser_name
                )
                if griffe_meta.get("returns"):
                    ret_meta = griffe_meta["returns"]
                    if ret_meta.get("typ") and not returns_type:
                        returns_type = active_sanitize_type_str(ret_meta["typ"])
                    if ret_meta.get("doc") and not returns_description:
                        returns_description = ret_meta["doc"]

                for exc in griffe_meta.get("raises", []):
                    if exc not in raises:
                        raises.append(exc)
                    if exc in cdd_params:
                        del cdd_params[exc]

                for p_name, p_data in griffe_meta.get("params", {}).items():
                    if p_name not in cdd_params:
                        cdd_params[p_name] = p_data
                    else:
                        if p_data.get("doc") and not cdd_params[p_name].get("doc"):
                            cdd_params[p_name]["doc"] = p_data["doc"]
                        if p_data.get("typ") and not cdd_params[p_name].get("typ"):
                            cdd_params[p_name]["typ"] = p_data["typ"]
            except Exception:
                pass

        # 2. Try to load AST information via cdd.parse as primary zero-import static analysis engine
        cdd_parsed_ir = None
        cdd_ast_params: Dict[str, Any] = {}

        target = obj
        if kind == "class":
            target = getattr(obj, "__init__", obj)

        has_super_kwargs_call = False

        try:
            source = inspect.getsource(obj)
            parsed_ast = _models_mod.ast.parse(source).body[0]

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
                    cdd_ast_params[p_name] = p_val
                    if p_name in cdd_params and "doc" in cdd_params[p_name]:
                        cdd_ast_params[p_name]["doc"] = cdd_params[p_name]["doc"]
        except Exception:
            pass

        # 2.5 Try to load AST information via griffe (fallback)
        if griffe_node is None and griffe_mod is not None:
            try:
                resolved_parser = resolve_griffe_parser(docstring_parser_name)
                loader_fn = griffe_mod.load
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
                        current = current.members[part]
                griffe_node = current
            except Exception:
                pass

        if is_public is None:
            if griffe_node is not None and hasattr(griffe_node, "is_public"):
                determined_is_public = griffe_node.is_public
            else:
                determined_is_public = not name.startswith("_")

        resolved_hints = {}
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

        # 3. Parameter Extraction Strategy (CDD -> Griffe -> Standard -> C-Extension Fallback)
        extracted_params = []
        c_ext_params = None

        griffe_params = None
        has_griffe_params = False
        try:
            if griffe_node is not None:
                griffe_params = getattr(griffe_node, "parameters", None)
                if griffe_params is None and getattr(griffe_node, "is_class", False):
                    members = getattr(griffe_node, "members", {})
                    init_node = members.get("__init__") or members.get("__new__")
                    if init_node:
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
                if "default" in p_info and p_info["default"] is not None:
                    try:
                        if isinstance(p_info["default"], ast.AST):
                            default_val = _models_mod.ast.unparse(p_info["default"])
                        elif isinstance(p_info["default"], str):
                            default_val = repr(p_info["default"])
                        else:
                            default_val = str(p_info["default"])
                    except Exception:
                        default_val = str(p_info["default"])

                anno_val = p_info.get("typ")
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
                    sig = inspect.signature(target)
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
                                    str(
                                        resolved_hints.get(param.name, param.annotation)
                                    )
                                )
                                if param.annotation is not inspect.Parameter.empty
                                else None
                            )
                            extracted_params.append(
                                (param.name, p_kind_str, default_val, anno_val)
                            )
                        else:
                            for i, (pn, pk, pd, pa) in enumerate(extracted_params):
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

                default_val = None
                if param.default is not None:
                    default_val = str(param.default)

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
                sig = inspect.signature(target)

                for param in sig.parameters.values():
                    if param.name == "self":
                        continue

                    if param.kind == inspect.Parameter.VAR_POSITIONAL:
                        has_varargs = True

                    default_val, _factory, _is_mand = sanitize_param_default(
                        param.default
                    )

                    anno_val = None
                    if param.name in resolved_hints:
                        anno_val = active_sanitize_type_str(
                            str(resolved_hints[param.name])
                        )
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
                            is_method=(
                                kind == "method"
                                or (name is not None and ".Tensor." in name)
                            ),
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
                        returns_type = active_sanitize_type_str(
                            c_ext_params.returns_type
                        )
                    for pn, pk, pd, pa in c_ext_params:
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

        if has_super_kwargs_call and hasattr(obj, "__mro__") and len(obj.__mro__) > 1:
            for parent in obj.__mro__[1:]:
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

        # 3.5 Promote documented **kwargs into formal KEYWORD_ONLY GhostParams
        has_var_kw = any(ep[1] == "VAR_KEYWORD" for ep in extracted_params)
        if has_var_kw and cdd_params:
            existing_names = {ep[0] for ep in extracted_params}
            var_kw_idx = next(
                i for i, ep in enumerate(extracted_params) if ep[1] == "VAR_KEYWORD"
            )
            promoted_params = []
            for p_name, p_data in cdd_params.items():
                if (
                    p_name not in existing_names
                    and p_name not in ("kwargs", "args", "self")
                    and not p_name.startswith("_")
                ):
                    p_typ = (
                        active_sanitize_type_str(p_data.get("typ"))
                        if p_data.get("typ")
                        else None
                    )
                    p_def = p_data.get("default")
                    promoted_params.append((p_name, "KEYWORD_ONLY", p_def, p_typ))
                    existing_names.add(p_name)

            if promoted_params:
                extracted_params = (
                    extracted_params[:var_kw_idx]
                    + promoted_params
                    + extracted_params[var_kw_idx:]
                )

        # 4. Finalize GhostParams by merging in CDD docstring descriptions
        is_torch_target = (
            (name and name.startswith("torch."))
            or (getattr(target, "__module__", "") or "").startswith("torch")
            or api_path.startswith("torch.")
        )

        for p_name, p_kind, p_default, p_anno in extracted_params:
            p_desc = None
            if p_name in cdd_params and cdd_params[p_name].get("doc"):
                p_desc = cdd_params[p_name].get("doc")
            elif p_name in cdd_ast_params and cdd_ast_params[p_name].get("doc"):
                p_desc = cdd_ast_params[p_name].get("doc")

            if not p_anno or p_anno == "Any":
                if p_name in cdd_params and cdd_params[p_name].get("typ"):
                    p_anno = active_sanitize_type_str(cdd_params[p_name]["typ"])
                elif p_name in cdd_ast_params and cdd_ast_params[p_name].get("typ"):
                    p_anno = active_sanitize_type_str(cdd_ast_params[p_name]["typ"])

            p_dtypes, p_rank = None, None
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

        if environment_tags is not None:
            env_tags = list(environment_tags)
        else:
            top_pkg = api_path.split(".")[0].lower() if api_path else ""
            env_tags = list(FRAMEWORK_CAPABILITIES.get(top_pkg, ["cpu"]))

        # Extract overloads via Griffe
        overloads_refs = []
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
                overload_params: List[GhostParam] = []
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

                        ov_allowed_values = None
                        if anno_val and "Literal[" in anno_val:
                            ov_allowed_values = re.findall(
                                r"['\x22]([^'\x22]+)['\x22]", anno_val
                            )
                        elif param.name.lower() in STANDARD_ENUM_MAP:
                            ov_allowed_values = STANDARD_ENUM_MAP[param.name.lower()]

                        overload_params.append(
                            ExtendedGhostParam(
                                name=param.name,
                                standardized_name=STANDARD_ARG_MAP.get(param.name),
                                kind=to_parameter_kind(p_kind_str),
                                default=default_val,
                                annotation=anno_val,
                                description=None,
                                allowed_values=ov_allowed_values,
                            )
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
                ov_params: List[GhostParam] = []
                ov_has_varargs = False
                for pn, pk, pd, pa in ov:
                    if pk == "VAR_POSITIONAL":
                        ov_has_varargs = True
                    sanitized_pa = active_sanitize_type_str(pa) if pa else None

                    ov_dtypes, ov_rank = None, None
                    if is_torch_target:
                        try:
                            from ml_ecosystem_snapshots.frameworks.torch import (
                                infer_torch_dtype_and_rank,
                            )

                            ov_dtypes, ov_rank = infer_torch_dtype_and_rank(
                                name, pn, sanitized_pa or ""
                            )
                        except Exception:
                            pass

                    ov_factory = None
                    if pd and "<factory_default>" in str(pd):
                        ov_factory = "<factory_default>"

                    ov_allowed_values = None
                    if sanitized_pa and "Literal[" in sanitized_pa:
                        ov_allowed_values = re.findall(
                            r"['\x22]([^'\x22]+)['\x22]", sanitized_pa
                        )
                    elif pn.lower() in STANDARD_ENUM_MAP:
                        ov_allowed_values = STANDARD_ENUM_MAP[pn.lower()]

                    ov_params.append(
                        ExtendedGhostParam(
                            name=pn,
                            standardized_name=STANDARD_ARG_MAP.get(pn),
                            kind=to_parameter_kind(pk),
                            default=pd,
                            annotation=sanitized_pa,
                            description=None,
                            allowed_dtypes=ov_dtypes,
                            rank_constraint=ov_rank,
                            allowed_values=ov_allowed_values,
                            default_factory=ov_factory,
                            is_mandatory=pd is None,
                        )
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

        has_var_kwargs = any(
            "VAR_KEYWORD" in str(getattr(p, "kind", "")) for p in params
        )
        discovered_kwargs = (
            extract_accepted_kwargs_from_ast(target) if has_var_kwargs else None
        )

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
            overloads=overloads_refs,
            signature_completeness=sig_completeness,
            is_c_extension=is_c_ext,
            accepted_kwargs=discovered_kwargs,
        )

    @staticmethod
    def hydrate(data: Dict[str, Any]) -> ExtendedGhostRef:
        """Create a GhostRef from a dictionary (JSON snapshot).

        Args:
            data: The dictionary data.

        Returns:
            The hydrated ExtendedGhostRef or specialized polymorphic GhostRef object.
        """
        return hydrate_ghost_ref(data)


__all__ = [
    "FRAMEWORK_CAPABILITIES",
    "GhostInspector",
    "STANDARD_ARG_MAP",
    "STANDARD_ENUM_MAP",
    "to_parameter_kind",
]
