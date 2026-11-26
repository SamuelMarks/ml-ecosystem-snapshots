"""Static Python AST extraction and pure namespace alignment for compliance verification."""

from __future__ import annotations

import ast
import logging
import os
import typing
from pathlib import Path

import griffe
from ml_switcheroo_ir.compliance import (
    FILE_COMPLIANCE_MONOID,
    FileComplianceResult,
    get_dialect_ops,
)
from ml_switcheroo_ir.schema.ghost import (
    GhostOverload,
    GhostParam,
    GhostPythonRef,
    GhostRef,
    ParameterKind,
)
from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result

from ml_ecosystem_snapshots.compliance.models import NamespaceAlignmentRule
from ml_ecosystem_snapshots.errors import (
    IntrospectionFault,
    SnapshotDomainError,
    SnapshotNotFoundFault,
)
from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.models import JsonValue
from ml_ecosystem_snapshots.utils import (
    get_framework_docstring_parser,
    resolve_griffe_parser,
)

_LOGGER = logging.getLogger(__name__)

_BSLASH: typing.Final[str] = chr(92)
_IR_REQUIRED_METHODS: typing.Final[frozenset[str]] = frozenset({"to_ir", "from_ir", "compile", "lower", "emit"})
_FW_REQUIRED_METHODS: typing.Final[frozenset[str]] = frozenset(
    {"convert_tensor", "map_dtype", "resolve_op", "emit_call"}
)


def get_module_info_from_path(
    file_path: str,
    target_prefix: str = "",
) -> Result[SnapshotDomainError, tuple[str, str]]:
    """Resolve a file path totally into `(root_search_path, derived_module_name)`.

    Args:
        file_path: Path to a Python file or directory.
        target_prefix: Optional module prefix fallback.

    Returns:
        `Ok((search_path, module_name))` on success, `Err(SnapshotNotFoundFault)` when
        `file_path` does not exist, or `Err(IntrospectionFault)` when module name cannot be derived.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        return Err(SnapshotNotFoundFault(target=file_path, version=None, searched_paths=()))

    is_dir = path.is_dir()
    current = path if is_dir else path.parent

    while Path(os.path.join(current, "__init__.py")).exists() and current.parent != current:
        current = current.parent

    search_path = str(current)
    rel_path = path.relative_to(current)
    if is_dir:
        mod_name = ".".join(rel_path.parts)
    else:
        parts = list(rel_path.parts)
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
            if parts[-1] == "__init__":
                parts.pop()
        mod_name = ".".join(parts)

    if not mod_name:
        if target_prefix:
            mod_name = target_prefix.split(".")[0]
            if Path(os.path.join(Path(search_path), "src")).is_dir():
                search_path = str(Path(os.path.join(Path(search_path), "src")))
        else:
            return Err(
                IntrospectionFault(
                    module_path=file_path,
                    symbol_name=None,
                    reason=f"Could not derive module name from path: {file_path}",
                )
            )

    return Ok((search_path, mod_name))


def extract_target_ast(
    file_path: str,
    target_prefix: str = "",
) -> Result[SnapshotDomainError, griffe.Module | griffe.Object | griffe.Alias]:
    """Extract the static Griffe AST totally for `file_path` without code execution.

    Args:
        file_path: Path to a Python file or directory.
        target_prefix: Prefix of the target module.

    Returns:
        `Ok(griffe_node)` on success, or `Err(SnapshotDomainError)` on missing/unparseable target.
    """
    info_res = get_module_info_from_path(file_path, target_prefix)
    if isinstance(info_res, Err):
        return info_res
    search_path, mod_name = info_res.value
    parser_choice = get_framework_docstring_parser(mod_name.split(".")[0])
    resolved_parser = resolve_griffe_parser(parser_choice)
    try:
        loaded = griffe.load(
            mod_name,
            search_paths=[search_path],
            docstring_parser=resolved_parser,
            allow_inspection=False,
        )
        return Ok(loaded)
    except (OSError, ValueError, TypeError, SyntaxError, griffe.GriffeError) as exc:
        return Err(
            IntrospectionFault(
                module_path=file_path,
                symbol_name=mod_name,
                reason=f"Failed to load static AST for '{file_path}': {exc}",
            )
        )


def align_namespace(
    api_path: str,
    target_prefix: str,
    reference_prefix: str,
    alignment_rule: NamespaceAlignmentRule | None = None,
) -> str:
    """Align a target namespace path to a reference namespace path via `make_namespace_align_endo`.

    Args:
        api_path: API path to align.
        target_prefix: Prefix in the target namespace.
        reference_prefix: Prefix in the reference namespace.
        alignment_rule: Optional `NamespaceAlignmentRule`.

    Returns:
        Aligned canonical API path string.
    """
    from ml_ecosystem_snapshots.typeclasses import make_namespace_align_endo

    return make_namespace_align_endo(target_prefix, reference_prefix, alignment_rule).app(api_path)


def normalize_source_module_name(module_key: str, target_prefix: str = "") -> str:
    """Normalize an in-memory module key or virtual file path into a dotted module name.

    Args:
        module_key: Dotted module identifier (`'zero_jax.nn'`) or virtual path (`'src/zero_jax/nn/__init__.py'`).
        target_prefix: Optional fallback module prefix when `module_key` resolves to root `'__init__.py'`.

    Returns:
        Normalized dotted Python module name.
    """
    cleaned = module_key.strip().replace(_BSLASH, "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if cleaned.startswith("src/"):
        cleaned = cleaned[4:]
    if cleaned.endswith(".py"):
        cleaned = cleaned[:-3]
        parts = [seg for seg in cleaned.split("/") if seg]
    elif "/" in cleaned:
        parts = [seg for seg in cleaned.split("/") if seg]
    else:
        parts = [seg for seg in cleaned.split(".") if seg]

    if parts and parts[-1] == "__init__":
        parts.pop()
    if not parts:
        return target_prefix.strip() if target_prefix.strip() else "module"
    return ".".join(parts)


def _format_ast_default(node: ast.expr) -> str:
    """Render an AST parameter default expression into a normalized string.

    Args:
        node: Python `ast.expr` default node.

    Returns:
        Normalized string representation matching `GhostParam.default` conventions.
    """
    if isinstance(node, ast.Constant) and node.value is None:
        return "None"
    return ast.unparse(node)


def _extract_ast_func_params(args: ast.arguments, *, is_method: bool) -> list[GhostParam]:
    """Extract ordered `GhostParam` definitions from a Python `ast.arguments` node.

    Args:
        args: Parsed function or method `ast.arguments`.
        is_method: Whether to strip a leading `self` or `cls` receiver parameter.

    Returns:
        Ordered list of `GhostParam` instances.
    """
    params: list[GhostParam] = []
    pos_all = list(args.posonlyargs) + list(args.args)
    posonly_count = len(args.posonlyargs)
    default_offset = len(pos_all) - len(args.defaults)

    for idx, arg in enumerate(pos_all):
        if is_method and idx == 0 and arg.arg in ("self", "cls"):
            continue
        kind = ParameterKind.POSITIONAL_ONLY if idx < posonly_count else ParameterKind.POSITIONAL_OR_KEYWORD
        default_str = _format_ast_default(args.defaults[idx - default_offset]) if idx >= default_offset else None
        ann_str = ast.unparse(arg.annotation) if arg.annotation is not None else None
        params.append(
            GhostParam(
                name=arg.arg,
                kind=kind,
                default=default_str,
                annotation=ann_str,
            )
        )

    if args.vararg is not None:
        var_ann = ast.unparse(args.vararg.annotation) if args.vararg.annotation is not None else None
        params.append(
            GhostParam(
                name=args.vararg.arg,
                kind=ParameterKind.VAR_POSITIONAL,
                annotation=var_ann,
            )
        )

    for kw_idx, kw_arg in enumerate(args.kwonlyargs):
        kw_def_node = args.kw_defaults[kw_idx] if kw_idx < len(args.kw_defaults) else None
        kw_def = _format_ast_default(kw_def_node) if kw_def_node is not None else None
        kw_ann = ast.unparse(kw_arg.annotation) if kw_arg.annotation is not None else None
        params.append(
            GhostParam(
                name=kw_arg.arg,
                kind=ParameterKind.KEYWORD_ONLY,
                default=kw_def,
                annotation=kw_ann,
            )
        )

    if args.kwarg is not None:
        kwarg_ann = ast.unparse(args.kwarg.annotation) if args.kwarg.annotation is not None else None
        params.append(
            GhostParam(
                name=args.kwarg.arg,
                kind=ParameterKind.VAR_KEYWORD,
                annotation=kwarg_ann,
            )
        )

    return params


def _is_overload_decorated(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return `True` when `func_node` carries `@overload` or `@typing.overload`.

    Args:
        func_node: Function AST node to check.

    Returns:
        `True` if decorated with `overload`.
    """
    for dec in func_node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "overload":
            return True
    return False


def analyze_source_ir_fw_compliance(tree: ast.Module, module_key: str) -> FileComplianceResult:
    """Analyze an in-memory `ast.Module` purely via `ml_switcheroo_ir.compliance` `FILE_COMPLIANCE_MONOID`.

    Args:
        tree: Parsed Python `ast.Module`.
        module_key: Source module name or virtual filepath.

    Returns:
        `FileComplianceResult` capturing IR, FrameworkAdapter, and dialect op coverage.
    """
    del module_key
    per_class_results: list[FileComplianceResult] = []
    known_dialect_ops = get_dialect_ops()
    found_ops: set[str] = {
        ast_node.value
        for ast_node in ast.walk(tree)
        if isinstance(ast_node, ast.Constant)
        and isinstance(ast_node.value, str)
        and ast_node.value in known_dialect_ops
    }
    if found_ops:
        per_class_results.append(FileComplianceResult(implemented_ops=found_ops))

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            base_names = {ast.unparse(b).split(".")[-1] for b in node.bases}
            method_names = {
                stmt.name for stmt in node.body if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            is_ir = bool(base_names & {"CompilerBackend", "BaseFrontend", "GraphFrontend"})
            is_fw = bool(base_names & {"FrameworkAdapter", "FrameworkImporter"})
            ir_met = len(method_names & _IR_REQUIRED_METHODS)
            fw_met = len(method_names & _FW_REQUIRED_METHODS)
            if is_ir or is_fw or ir_met > 0 or fw_met > 0:
                per_class_results.append(
                    FileComplianceResult(
                        is_ir=is_ir,
                        is_fw=is_fw,
                        ir_score=(ir_met, 5),
                        fw_score=(fw_met, 4),
                    )
                )

    return FILE_COMPLIANCE_MONOID.mconcat(per_class_results)


def _build_callable_ghost_ref(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    aligned_path: str,
    *,
    kind: str,
    is_method: bool,
    overloads: typing.Sequence[GhostOverload] = (),
) -> GhostPythonRef:
    """Construct a `GhostPythonRef` from a parsed function or method AST node.

    Args:
        func_node: Function or async-function AST node.
        aligned_path: Canonical aligned dotted API path.
        kind: `'function'` or `'method'`.
        is_method: Whether receiver `self`/`cls` should be omitted.
        overloads: Collected `@overload` signatures for this callable.

    Returns:
        Hydrated `GhostPythonRef` instance.
    """
    params = _extract_ast_func_params(func_node.args, is_method=is_method)
    ret_type = ast.unparse(func_node.returns) if func_node.returns is not None else None
    has_varargs = any(p.kind in (ParameterKind.VAR_POSITIONAL, ParameterKind.VAR_KEYWORD) for p in params)
    doc = ast.get_docstring(func_node)
    ov_list: list[dict[str, JsonValue] | GhostOverload | GhostRef] = list(overloads)
    return GhostPythonRef(
        name=func_node.name,
        api_path=aligned_path,
        kind=kind,
        params=params,
        docstring=doc,
        has_varargs=has_varargs,
        returns_type=ret_type,
        is_public=not func_node.name.startswith("_"),
        overloads=ov_list,
    )


def _extract_class_refs(
    stmt: ast.ClassDef,
    mod_name: str,
    align_fn: typing.Callable[[str], str],
) -> list[GhostRef]:
    """Extract aligned `GhostRef` entries for a public `ast.ClassDef` and its methods.

    Args:
        stmt: Parsed Python `ast.ClassDef`.
        mod_name: Normalized module name containing `stmt`.
        align_fn: Pure namespace alignment callable.

    Returns:
        Ordered list starting with the class `GhostRef` followed by public method `GhostRef`s.
    """
    aligned_cls_path = align_fn(f"{mod_name}.{stmt.name}")
    init_params: list[GhostParam] = []
    cls_methods: list[GhostRef] = []
    method_overloads: dict[str, list[GhostOverload]] = {}
    for body_item in stmt.body:
        if not isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if body_item.name == "__init__":
            if not _is_overload_decorated(body_item):
                init_params = _extract_ast_func_params(body_item.args, is_method=True)
        elif not body_item.name.startswith("_"):
            if _is_overload_decorated(body_item):
                m_ov_params = _extract_ast_func_params(body_item.args, is_method=True)
                m_ov_ret = ast.unparse(body_item.returns) if body_item.returns is not None else None
                method_overloads.setdefault(body_item.name, []).append(
                    GhostOverload(params=m_ov_params, returns_type=m_ov_ret)
                )
            else:
                aligned_m_path = align_fn(f"{mod_name}.{stmt.name}.{body_item.name}")
                cls_methods.append(
                    _build_callable_ghost_ref(
                        body_item,
                        aligned_m_path,
                        kind="method",
                        is_method=True,
                        overloads=method_overloads.pop(body_item.name, ()),
                    )
                )
    has_cls_varargs = any(p.kind in (ParameterKind.VAR_POSITIONAL, ParameterKind.VAR_KEYWORD) for p in init_params)
    cls_ref = GhostPythonRef(
        name=stmt.name,
        api_path=aligned_cls_path,
        kind="class",
        params=init_params,
        docstring=ast.get_docstring(stmt),
        has_varargs=has_cls_varargs,
        is_public=True,
    )
    return [cls_ref, *cls_methods]


def extract_refs_from_source(
    module_key: str,
    source: str,
    target_prefix: str,
    reference_prefix: str,
    alignment_rule: NamespaceAlignmentRule | None = None,
) -> Result[IntrospectionFault, tuple[tuple[GhostRef, ...], FileComplianceResult]]:
    """Purely extract aligned `GhostRef`s and `FileComplianceResult` from in-memory Python source.

    Performs zero filesystem I/O.

    Args:
        module_key: Dotted module name or virtual file path for `source`.
        source: Raw Python source string.
        target_prefix: Prefix of the target package namespace.
        reference_prefix: Prefix of the reference snapshot namespace.
        alignment_rule: Optional `NamespaceAlignmentRule`.

    Returns:
        `Ok((ghost_refs_tuple, file_compliance))` on valid Python syntax,
        else `Err(IntrospectionFault)`.
    """
    try:
        tree = ast.parse(source, filename=module_key)
    except SyntaxError as exc:
        return Err(
            IntrospectionFault(
                module_path=module_key,
                symbol_name=None,
                reason=f"SyntaxError in '{module_key}': {exc}",
            )
        )

    from ml_ecosystem_snapshots.typeclasses import make_namespace_align_endo

    mod_name = normalize_source_module_name(module_key, target_prefix=target_prefix)
    align_endo = make_namespace_align_endo(target_prefix, reference_prefix, alignment_rule)
    refs: list[GhostRef] = []

    pending_overloads: dict[str, list[GhostOverload]] = {}
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and not stmt.name.startswith("_"):
            if _is_overload_decorated(stmt):
                ov_params = _extract_ast_func_params(stmt.args, is_method=False)
                ov_ret = ast.unparse(stmt.returns) if stmt.returns is not None else None
                pending_overloads.setdefault(stmt.name, []).append(GhostOverload(params=ov_params, returns_type=ov_ret))
            else:
                aligned_fn_path = align_endo.app(f"{mod_name}.{stmt.name}")
                refs.append(
                    _build_callable_ghost_ref(
                        stmt,
                        aligned_fn_path,
                        kind="function",
                        is_method=False,
                        overloads=pending_overloads.pop(stmt.name, ()),
                    )
                )
        elif isinstance(stmt, ast.ClassDef) and not stmt.name.startswith("_"):
            refs.extend(_extract_class_refs(stmt, mod_name, align_endo.app))

    file_comp = analyze_source_ir_fw_compliance(tree, module_key)
    return Ok((tuple(refs), file_comp))


def _apply_node_path_rule(
    node: griffe.Function | griffe.Class | griffe.Alias,
    current_path: str,
    target_prefix: str,
    alignment_rule: NamespaceAlignmentRule | None,
) -> str:
    """Apply optional `NamespaceAlignmentRule` path rewrites to a static Griffe node path.

    Args:
        node: Griffe AST node being inspected.
        current_path: Current dotted traversal path.
        target_prefix: Prefix of the target module.
        alignment_rule: Optional `NamespaceAlignmentRule`.

    Returns:
        Normalized dotted symbol path string.
    """
    if alignment_rule is None:
        return current_path

    node_path = str(node.path or "")
    for src_prefix, template in alignment_rule.node_prefix_rewrites:
        rendered = template.format(target_prefix=target_prefix)
        if node_path.startswith(src_prefix):
            rewritten = node_path.replace(src_prefix, rendered, 1)
            for old_seg, new_seg in alignment_rule.path_rewrites:
                rewritten = rewritten.replace(old_seg, new_seg)
            return rewritten
        if src_prefix in current_path:
            return current_path.replace(src_prefix, rendered, 1)

    return current_path


def extract_target_refs_single(
    file_path: str,
    target_prefix: str,
    reference_prefix: str,
    alignment_rule: NamespaceAlignmentRule | None = None,
) -> Result[SnapshotDomainError, tuple[GhostRef, ...]]:
    """Statically extract an immutable `GhostRef` tuple from `file_path` wrapped in `Result`.

    Args:
        file_path: Path to a Python file or directory.
        target_prefix: Prefix of the target module.
        reference_prefix: Prefix of the reference module.
        alignment_rule: Optional `NamespaceAlignmentRule` for namespace normalization.

    Returns:
        `Ok(tuple[GhostRef, ...])` on success, or `Err(SnapshotDomainError)` on path/load failure.
    """
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
    from ml_ecosystem_snapshots.typeclasses import make_namespace_align_endo

    info_res = get_module_info_from_path(file_path, target_prefix)
    if isinstance(info_res, Err):
        return info_res
    search_path, mod_name = info_res.value

    ast_res = extract_target_ast(file_path, target_prefix)
    if isinstance(ast_res, Err):
        return ast_res
    del search_path
    mod_ast = ast_res.value

    refs: list[GhostRef] = []
    align_endo = make_namespace_align_endo(target_prefix, reference_prefix, alignment_rule)

    def walk(node: griffe.Module | griffe.Object | griffe.Alias, current_path: str) -> None:
        """Recursively traverse a static Griffe AST node and inspect public symbols.

        Args:
            node: Griffe AST node (`Module`, `Class`, `Function`, or `Alias`).
            current_path: Dotted Python module/symbol path of `node`.
        """
        if isinstance(node, (griffe.Function, griffe.Class, griffe.Alias)):
            try:
                inspect_target: griffe.Function | griffe.Class | griffe.Object = (
                    node.final_target if isinstance(node, griffe.Alias) else node
                )
                normalized_path = _apply_node_path_rule(node, current_path, target_prefix, alignment_rule)
                aligned_path = align_endo.app(normalized_path)
                refs.append(GhostInspector.inspect(inspect_target, aligned_path))
            except (
                SnapshotError,
                ImportError,
                AttributeError,
                TypeError,
                ValueError,
                OSError,
                RuntimeError,
                LookupError,
                SyntaxError,
                griffe.GriffeError,
            ) as exc:
                _LOGGER.warning("Exception in walk for %s: %s", current_path, exc)

        if isinstance(node, (griffe.Module, griffe.Class)):
            for name, member in node.members.items():
                if not name.startswith("_"):
                    walk(member, f"{current_path}.{name}")

    walk(mod_ast, mod_name)
    return Ok(tuple(refs))


def extract_target_refs(
    file_paths: typing.Sequence[str] | str,
    target_prefix: str,
    reference_prefix: str,
    alignment_rule: NamespaceAlignmentRule | None = None,
) -> Result[SnapshotDomainError, tuple[GhostRef, ...]]:
    """Extract a deduplicated `GhostRef` tuple statically from one or more target paths.

    Args:
        file_paths: Single path string or sequence of paths to Python files or directories.
        target_prefix: Prefix of the target module.
        reference_prefix: Prefix of the reference module.
        alignment_rule: Optional `NamespaceAlignmentRule` for namespace normalization.

    Returns:
        `Ok(tuple[GhostRef, ...])` on success, or `Err(SnapshotDomainError)` if any path fails.
    """
    paths = (file_paths,) if isinstance(file_paths, str) else tuple(file_paths)
    seen_paths: set[str] = set()
    refs: list[GhostRef] = []
    for fpath in paths:
        single_res = extract_target_refs_single(
            fpath,
            target_prefix,
            reference_prefix,
            alignment_rule=alignment_rule,
        )
        if isinstance(single_res, Err):
            return single_res
        for ref in single_res.value:
            if ref.api_path not in seen_paths:
                seen_paths.add(ref.api_path)
                refs.append(ref)
    return Ok(tuple(refs))


__all__ = [
    "NamespaceAlignmentRule",
    "align_namespace",
    "analyze_source_ir_fw_compliance",
    "extract_refs_from_source",
    "extract_target_ast",
    "extract_target_refs",
    "extract_target_refs_single",
    "get_module_info_from_path",
    "normalize_source_module_name",
]
