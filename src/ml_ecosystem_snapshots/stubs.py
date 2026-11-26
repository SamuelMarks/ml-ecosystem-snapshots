"""Pure Ghost `.pyi` stub renderer and total AST verifier."""

from __future__ import annotations

import ast
import os
import typing
from pathlib import Path

import typing_extensions
from ml_switcheroo_ir.schema.ghost import SnapshotEnvelope
from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result, traverse_result

from ml_ecosystem_snapshots.errors import SnapshotDomainError, SnapshotEnvelopeFault
from ml_ecosystem_snapshots.models import JsonValue, ScalarValue, WireSerializableProtocol

if typing.TYPE_CHECKING:
    from ml_ecosystem_snapshots.proofs import EnvelopeAndHydratedPhase, ProvenSnapshot

_NL: typing.Final[str] = chr(10)

_SAFE_DEFAULT_AST_NODES = (
    ast.List,
    ast.Dict,
    ast.Tuple,
    ast.Set,
    ast.Name,
    ast.Attribute,
    ast.UnaryOp,
    ast.BinOp,
)

StubSnapshotInput: typing_extensions.TypeAlias = (
    "ProvenSnapshot[EnvelopeAndHydratedPhase]"
    " | SnapshotEnvelope"
    " | WireSerializableProtocol"
    " | typing.Mapping[str, JsonValue]"
    " | typing.Sequence[typing.Mapping[str, JsonValue]]"
    " | ScalarValue"
)


def _sanitize_default(default_str: str) -> str:
    """Sanitize complex or un-importable default values for `.pyi` stubs.

    Args:
        default_str: Default value string to sanitize.

    Returns:
        Sanitized Python expression string safe for `.pyi` stubs.
    """
    if default_str == "<unrepresentable>":
        return "..."
    try:
        node = ast.parse(default_str, mode="eval").body
    except SyntaxError:
        return "..."
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, _SAFE_DEFAULT_AST_NODES):
        return default_str
    return "..."


def _param_field(p: typing.Mapping[str, JsonValue], key: str) -> str | None:
    """Extract a string field from a parameter dictionary.

    Args:
        p: Parameter mapping.
        key: Field name (`'name'`, `'annotation'`, `'default'`, or `'kind'`).

    Returns:
        String value or `None`.
    """
    val = p.get(key)
    return str(val) if val is not None else None


def _format_single_param(p: typing.Mapping[str, JsonValue]) -> str:
    """Format a single parameter dictionary into a `.pyi` parameter declaration.

    Args:
        p: Parameter dictionary definition.

    Returns:
        Formatted parameter snippet for a `.pyi` signature.
    """
    p_name = _param_field(p, "name")
    p_anno = _param_field(p, "annotation")
    p_default = _param_field(p, "default")
    p_kind = _param_field(p, "kind")

    if p_kind == "VAR_POSITIONAL":
        return f"*{p_name}"
    if p_kind == "VAR_KEYWORD":
        return f"**{p_name}"

    p_str = str(p_name)
    p_str += f": {p_anno}" if p_anno and p_anno != "Any" else ": Incomplete"
    if p_default is not None:
        p_str += f" = {_sanitize_default(str(p_default))}"
    return p_str


def _format_param_list(
    params: typing.Sequence[typing.Mapping[str, JsonValue]],
    has_varargs: bool = False,
    is_method: bool = False,
) -> str:
    """Format a sequence of parameter definitions into a Python `.pyi` signature string.

    Preserves `/` after `POSITIONAL_ONLY` parameters and `*` before `KEYWORD_ONLY`
    parameters so generated `.pyi` stubs remain syntactically valid Python.

    Args:
        params: Sequence of parameter dictionary definitions.
        has_varargs: Whether to append generic varargs if none are present.
        is_method: Whether this signature belongs to a method requiring `'self'`.

    Returns:
        Comma-separated parameter string for `def` statements.
    """
    param_strs: list[str] = []
    if is_method and not any(_param_field(p, "name") in ("self", "cls") for p in params):
        param_strs.append("self")

    in_pos_only = False
    seen_star_or_vararg = False

    for p in params:
        p_kind = _param_field(p, "kind")
        if in_pos_only and p_kind != "POSITIONAL_ONLY":
            param_strs.append("/")
            in_pos_only = False

        if p_kind == "POSITIONAL_ONLY":
            in_pos_only = True
        elif p_kind == "VAR_POSITIONAL":
            seen_star_or_vararg = True
        elif p_kind == "KEYWORD_ONLY" and not seen_star_or_vararg:
            param_strs.append("*")
            seen_star_or_vararg = True

        param_strs.append(_format_single_param(p))

    if in_pos_only:
        param_strs.append("/")

    if has_varargs and not seen_star_or_vararg:
        param_strs.append("*args: Incomplete")

    return ", ".join(param_strs)


def _extract_param_dicts(raw_params: JsonValue) -> list[dict[str, JsonValue]]:
    """Extract a typed list of parameter dictionaries from a JSON value.

    Args:
        raw_params: Raw `params` field from a snapshot item or overload.

    Returns:
        List of parameter dictionaries.
    """
    if not isinstance(raw_params, (list, tuple)):
        return []
    return [dict(p) for p in raw_params if isinstance(p, dict)]


def _extract_overload_dicts(raw_overloads: JsonValue) -> list[dict[str, JsonValue]]:
    """Extract a typed list of overload dictionaries from a JSON value.

    Args:
        raw_overloads: Raw `overloads` field from a snapshot item.

    Returns:
        List of overload dictionaries.
    """
    if not isinstance(raw_overloads, (list, tuple)):
        return []
    return [dict(ov) for ov in raw_overloads if isinstance(ov, dict)]


def _extract_stub_categories(
    snapshot_data: StubSnapshotInput,
) -> dict[str, list[dict[str, JsonValue]]]:
    """Normalize snapshot input purely into a dictionary of category item lists.

    Args:
        snapshot_data: Snapshot envelope, `ProvenSnapshot`, dictionary, or sequence of items.

    Returns:
        Mapping from category name to list of symbol dictionaries.
    """
    categories_dict: dict[str, list[dict[str, JsonValue]]] = {}
    if isinstance(snapshot_data, (list, tuple)):
        return {"all": [dict(x) for x in snapshot_data if isinstance(x, typing.Mapping)]}

    raw_snap: typing.Mapping[str, JsonValue]
    if isinstance(snapshot_data, SnapshotEnvelope):
        raw_snap = snapshot_data.model_dump(mode="json")
    elif isinstance(snapshot_data, WireSerializableProtocol):
        raw_snap = snapshot_data.to_wire_dict()
    elif isinstance(snapshot_data, typing.Mapping):
        raw_snap = snapshot_data
    else:
        return categories_dict

    raw_cats = raw_snap.get("categories", {})
    if isinstance(raw_cats, dict):
        for cat_k, cat_v in raw_cats.items():
            if isinstance(cat_v, (list, tuple)):
                categories_dict[str(cat_k)] = [dict(x) for x in cat_v if isinstance(x, dict)]
    return categories_dict


def _partition_stub_symbols(
    categories_dict: typing.Mapping[str, typing.Sequence[dict[str, JsonValue]]],
    include_nonpublic: bool,
) -> tuple[
    dict[str, list[tuple[str, dict[str, JsonValue]]]],
    dict[str, dict[str, list[tuple[str, dict[str, JsonValue]]]]],
]:
    """Group snapshot symbols into module-level objects and class methods.

    Args:
        categories_dict: Mapping of category names to symbol dictionaries.
        include_nonpublic: Whether to retain symbols marked `is_public=False`.

    Returns:
        Tuple of `(modules, class_methods)` mappings keyed by module name.
    """
    modules: dict[str, list[tuple[str, dict[str, JsonValue]]]] = {}
    class_methods: dict[str, dict[str, list[tuple[str, dict[str, JsonValue]]]]] = {}

    for items in categories_dict.values():
        for item in items:
            if not include_nonpublic and item.get("is_public") is False:
                continue

            api_path = str(item.get("api_path") or item.get("name") or item.get("mnemonic") or "")
            if not api_path:
                continue

            parts = api_path.split(".")
            kind = str(item.get("kind") or "function")
            fallback_mod = str(item.get("framework") or item.get("dialect") or "")

            if kind in ("method", "property") and len(parts) >= 2:
                class_name = parts[-2]
                method_name = parts[-1]
                module_name = fallback_mod if len(parts) == 2 else ".".join(parts[:-2])
                class_methods.setdefault(module_name, {}).setdefault(class_name, []).append((method_name, item))
            else:
                module_name = fallback_mod if len(parts) == 1 else ".".join(parts[:-1])
                obj_name = parts[0] if len(parts) == 1 else parts[-1]
                modules.setdefault(module_name, []).append((obj_name, item))

    return modules, class_methods


def _format_return_annotation(ret_val: JsonValue, fallback: str = "Incomplete") -> str:
    """Format a return type annotation for a `.pyi` stub signature.

    Args:
        ret_val: Raw `returns_type` value from the symbol or overload dictionary.
        fallback: Fallback return type name when `ret_val` is absent or `'Any'`.

    Returns:
        Formatted `' -> <type>'` string.
    """
    return f" -> {ret_val}" if ret_val and ret_val != "Any" else f" -> {fallback}"


def _render_callable_stub_lines(
    func_name: str,
    item: typing.Mapping[str, JsonValue],
    indent: str = "",
    is_method: bool = False,
    overload_fallback_ret: str = "Incomplete",
) -> list[str]:
    """Render `.pyi` stub lines (including `@typing.overload` variants) for a function or method.

    Args:
        func_name: Function or method identifier (e.g. `'__init__'`, `'forward'`).
        item: Symbol dictionary containing `params`, `overloads`, `has_varargs`, and `returns_type`.
        indent: Leading indentation prefix string.
        is_method: Whether to prepend `self` to parameter lists.
        overload_fallback_ret: Fallback return type for overloads without an explicit return type.

    Returns:
        List of formatted `.pyi` definition lines.
    """
    out: list[str] = []
    for ov in _extract_overload_dicts(item.get("overloads")):
        ov_sig = _format_param_list(_extract_param_dicts(ov.get("params")), is_method=is_method)
        ov_ret = _format_return_annotation(ov.get("returns_type"), fallback=overload_fallback_ret)
        out.append(f"{indent}@typing.overload")
        out.append(f"{indent}def {func_name}({ov_sig}){ov_ret}: ...")

    sig = _format_param_list(
        _extract_param_dicts(item.get("params")),
        has_varargs=bool(item.get("has_varargs", False)),
        is_method=is_method,
    )
    ret_str = _format_return_annotation(item.get("returns_type"), fallback="Incomplete")
    out.append(f"{indent}def {func_name}({sig}){ret_str}: ...")
    return out


def _render_module_stub_lines(
    mod_items: typing.Sequence[tuple[str, dict[str, JsonValue]]],
    mod_class_methods: typing.Mapping[str, typing.Sequence[tuple[str, dict[str, JsonValue]]]],
) -> tuple[str, ...]:
    """Purely render the complete tuple of `.pyi` lines for a single module.

    Args:
        mod_items: Top-level `(obj_name, item)` pairs belonging to the module.
        mod_class_methods: Mapping of class names to `(method_name, item)` pairs in the module.

    Returns:
        Immutable tuple of `.pyi` source lines for the module.
    """
    lines = ["import typing", "from _typeshed import Incomplete", ""]
    handled_classes: set[str] = set()

    for obj_name, item in mod_items:
        kind = str(item.get("kind") or "function")
        if kind == "class":
            handled_classes.add(obj_name)
            lines.append(f"class {obj_name}:")
            lines.extend(
                _render_callable_stub_lines(
                    "__init__",
                    item,
                    indent="    ",
                    is_method=True,
                    overload_fallback_ret="None",
                )
            )
            for m_name, m_item in mod_class_methods.get(obj_name, ()):
                lines.extend(_render_callable_stub_lines(m_name, m_item, indent="    ", is_method=True))
            lines.append("")
        else:
            lines.extend(_render_callable_stub_lines(obj_name, item))
            lines.append("")

    for cls_name, methods in mod_class_methods.items():
        if cls_name not in handled_classes:
            lines.append(f"class {cls_name}:")
            for m_name, m_item in methods:
                lines.extend(_render_callable_stub_lines(m_name, m_item, indent="    ", is_method=True))
            lines.append("")

    return tuple(lines)


def validate_pyi_stub(
    stub_content: str,
    context: str = "<generated stub>",
) -> Result[SnapshotEnvelopeFault, ast.Module]:
    """Validate totally that a `.pyi` stub compiles into a valid Python `ast.Module`.

    Args:
        stub_content: Source content of the `.pyi` stub file.
        context: Optional module/file label included in error diagnostics.

    Returns:
        `Ok(ast.Module)` if syntactically valid Python, else `Err(SnapshotEnvelopeFault)`.
    """
    try:
        return Ok(ast.parse(stub_content, filename=context))
    except SyntaxError as exc:
        return Err(SnapshotEnvelopeFault(reason=f"Invalid .pyi stub syntax in {context}: {exc}"))


def render_stub_modules(
    snapshot_data: StubSnapshotInput,
    include_nonpublic: bool = False,
) -> Result[SnapshotEnvelopeFault, tuple[tuple[str, str], ...]]:
    """Purely render and AST-validate `(module_name, pyi_source)` pairs without touching disk.

    Args:
        snapshot_data: `ProvenSnapshot[EnvelopeAndHydratedPhase]`, `SnapshotEnvelope`, dict, or sequence.
        include_nonpublic: Whether to generate stubs for non-public symbols.

    Returns:
        `Ok(tuple[(module_name, stub_text), ...])` if every module compiles cleanly,
        or `Err(SnapshotEnvelopeFault)` on invalid syntax.
    """
    categories_dict = _extract_stub_categories(snapshot_data)
    modules, class_methods = _partition_stub_symbols(categories_dict, include_nonpublic)
    all_modules = sorted(m for m in (set(modules.keys()) | set(class_methods.keys())) if m)

    def _build_and_validate(mod_name: str) -> Result[SnapshotEnvelopeFault, tuple[str, str]]:
        """Render and validate a single module stub.

        Args:
            mod_name: Dotted module name.

        Returns:
            `Ok((mod_name, stub_content))` or `Err(SnapshotEnvelopeFault)`.
        """
        lines = _render_module_stub_lines(
            modules.get(mod_name, ()),
            class_methods.get(mod_name, {}),
        )
        stub_content = _NL.join(lines) + _NL
        val_res = validate_pyi_stub(stub_content, context=f"{mod_name}.__init__.pyi")
        if isinstance(val_res, Err):
            return val_res
        return Ok((mod_name, stub_content))

    return traverse_result(all_modules, _build_and_validate)


def generate_stubs(
    snapshot_data: StubSnapshotInput,
    output_dir: str | Path,
    include_nonpublic: bool = False,
) -> Result[SnapshotDomainError, tuple[str, ...]]:
    """Generate and write `.pyi` stub files totally into `Result[SnapshotDomainError, tuple[str, ...]]`.

    Args:
        snapshot_data: `ProvenSnapshot[EnvelopeAndHydratedPhase]`, `SnapshotEnvelope`, dict, or sequence.
        output_dir: Base directory where `.pyi` stub packages should be written.
        include_nonpublic: Whether to generate stubs for non-public components.

    Returns:
        `Ok(tuple_of_written_paths)` on success, or `Err(SnapshotDomainError)` on syntax or I/O error.
    """
    rendered_res = render_stub_modules(snapshot_data, include_nonpublic=include_nonpublic)
    if isinstance(rendered_res, Err):
        return rendered_res

    try:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for module_name, stub_content in rendered_res.value:
            module_path = Path(os.path.join(out_path, module_name.replace(".", os.sep)))
            module_path.mkdir(parents=True, exist_ok=True)
            init_file = Path(os.path.join(module_path, "__init__.pyi"))
            with open(init_file, "w", encoding="utf-8", newline=_NL) as f:
                f.write(stub_content)
            written.append(str(init_file))
        return Ok(tuple(written))
    except OSError as exc:
        return Err(SnapshotEnvelopeFault(reason=f"Failed to write .pyi stubs to '{output_dir}': {exc}"))


__all__ = [
    "StubSnapshotInput",
    "generate_stubs",
    "render_stub_modules",
    "validate_pyi_stub",
]
