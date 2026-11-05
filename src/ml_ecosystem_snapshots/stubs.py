"""Ghost Type Stubs Generator.

Generate .pyi stub files from snapshot JSON data.
"""

from __future__ import annotations

import ast
import os
import typing
from pathlib import Path

from ml_switcheroo_ir.schema.ghost import SnapshotEnvelope

from ml_ecosystem_snapshots.models import JsonValue

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


def _sanitize_default(default_str: str) -> str:
    """Sanitize complex or un-importable default values for stubs.

    Args:
        default_str: The default value string to sanitize.

    Returns:
        The sanitized string.
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
        key: Field name ('name', 'annotation', 'default', or 'kind').

    Returns:
        String value or None.
    """
    val = p.get(key)
    return str(val) if val is not None else None


def _format_param_list(
    params: typing.Sequence[typing.Mapping[str, JsonValue]],
    has_varargs: bool = False,
    is_method: bool = False,
) -> str:
    """Format a list of parameter definitions into a Python signature string.

    Args:
        params: Sequence of parameter dictionary definitions.
        has_varargs: Whether to append generic varargs if none are present.
        is_method: Whether this signature belongs to a method requiring 'self'.

    Returns:
        A comma-separated parameter string for def statements.
    """
    param_strs: list[str] = []
    if is_method and not any(_param_field(p, "name") in ("self", "cls") for p in params):
        param_strs.append("self")

    for p in params:
        p_name = _param_field(p, "name")
        p_anno = _param_field(p, "annotation")
        p_default = _param_field(p, "default")
        p_kind = _param_field(p, "kind")

        if p_kind == "VAR_POSITIONAL":
            p_str = f"*{p_name}"
        elif p_kind == "VAR_KEYWORD":
            p_str = f"**{p_name}"
        else:
            p_str = str(p_name)
            if p_anno and p_anno != "Any":
                p_str += f": {p_anno}"
            else:
                p_str += ": object"

            if p_default is not None:
                sanitized = _sanitize_default(str(p_default))
                p_str += f" = {sanitized}"

        param_strs.append(p_str)

    if has_varargs and not any(_param_field(p, "kind") == "VAR_POSITIONAL" for p in params):
        param_strs.append("*args: object")

    return ", ".join(param_strs)


def _extract_param_dicts(raw_params: JsonValue) -> list[dict[str, JsonValue]]:
    """Extract a typed list of parameter dictionaries from a JSON value.

    Args:
        raw_params: Raw `params` field from a snapshot item or overload.

    Returns:
        List of parameter dictionaries.
    """
    raw = raw_params or []
    assert isinstance(raw, list)
    return [dict(p) for p in raw if isinstance(p, dict)]


def _extract_overload_dicts(raw_overloads: JsonValue) -> list[dict[str, JsonValue]]:
    """Extract a typed list of overload dictionaries from a JSON value.

    Args:
        raw_overloads: Raw `overloads` field from a snapshot item.

    Returns:
        List of overload dictionaries.
    """
    if not isinstance(raw_overloads, list):
        return []
    return [dict(ov) for ov in raw_overloads if isinstance(ov, dict)]


def _extract_stub_categories(
    snapshot_data: (SnapshotEnvelope | typing.Mapping[str, JsonValue] | typing.Sequence[typing.Mapping[str, JsonValue]]),
) -> dict[str, list[dict[str, JsonValue]]]:
    """Normalize snapshot input into a dictionary of category item lists.

    Args:
        snapshot_data: The snapshot dictionary, envelope, or list of items.

    Returns:
        Mapping from category name to list of symbol dictionaries.
    """
    categories_dict: dict[str, list[dict[str, JsonValue]]] = {}
    if isinstance(snapshot_data, list):
        return {"all": [dict(x) for x in snapshot_data if isinstance(x, dict)]}
    if not isinstance(snapshot_data, (typing.Mapping, SnapshotEnvelope)):
        return categories_dict
    raw_snap: typing.Mapping[str, JsonValue] = snapshot_data.model_dump() if isinstance(snapshot_data, SnapshotEnvelope) else snapshot_data
    raw_cats = raw_snap.get("categories", {})
    assert isinstance(raw_cats, dict)
    for cat_k, cat_v in raw_cats.items():
        assert isinstance(cat_v, list)
        categories_dict[str(cat_k)] = [dict(x) for x in cat_v if isinstance(x, dict)]
    return categories_dict


def _partition_stub_symbols(
    categories_dict: dict[str, list[dict[str, JsonValue]]],
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
            if not include_nonpublic and not item.get("is_public", True):
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


def _format_return_annotation(ret_val: JsonValue, fallback: str = "object") -> str:
    """Format a return type annotation for a `.pyi` stub signature.

    Args:
        ret_val: Raw `returns_type` value from the symbol or overload dictionary.
        fallback: Fallback return type name when `ret_val` is absent or `'Any'`.

    Returns:
        Formatted ` -> <type>` string.
    """
    return f" -> {ret_val}" if ret_val and ret_val != "Any" else f" -> {fallback}"


def _render_callable_stub_lines(
    func_name: str,
    item: dict[str, JsonValue],
    indent: str = "",
    is_method: bool = False,
    overload_fallback_ret: str = "object",
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
    ret_str = _format_return_annotation(item.get("returns_type"), fallback="object")
    out.append(f"{indent}def {func_name}({sig}){ret_str}: ...")
    return out


def _render_module_stub_lines(
    mod_items: list[tuple[str, dict[str, JsonValue]]],
    mod_class_methods: dict[str, list[tuple[str, dict[str, JsonValue]]]],
) -> list[str]:
    """Render the complete list of `.pyi` lines for a single module.

    Args:
        mod_items: Top-level `(obj_name, item)` pairs belonging to the module.
        mod_class_methods: Mapping of class names to `(method_name, item)` pairs in the module.

    Returns:
        List of `.pyi` source lines for the module.
    """
    lines = ["import typing", ""]
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
            for m_name, m_item in mod_class_methods.get(obj_name, []):
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

    return lines


def generate_stubs(
    snapshot_data: (SnapshotEnvelope | typing.Mapping[str, JsonValue] | typing.Sequence[typing.Mapping[str, JsonValue]]),
    output_dir: str,
    include_nonpublic: bool = False,
) -> None:
    """Generate .pyi stub files from a snapshot dictionary.

    Args:
        snapshot_data: The snapshot dictionary, envelope, or list.
        output_dir: The base directory where stubs should be written.
        include_nonpublic: Whether to generate stubs for non-public components.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    categories_dict = _extract_stub_categories(snapshot_data)
    modules, class_methods = _partition_stub_symbols(categories_dict, include_nonpublic)
    all_modules = sorted(set(modules.keys()) | set(class_methods.keys()))

    for module_name in all_modules:
        if not module_name:
            continue

        module_path = Path(os.path.join(out_path, module_name.replace(".", os.sep)))
        module_path.mkdir(parents=True, exist_ok=True)
        init_file = Path(os.path.join(module_path, "__init__.pyi"))

        lines = _render_module_stub_lines(
            modules.get(module_name, []),
            class_methods.get(module_name, {}),
        )
        stub_content = "\n".join(lines) + "\n"
        validate_pyi_stub(stub_content)

        with open(init_file, "w", encoding="utf-8") as f:
            f.write(stub_content)


def validate_pyi_stub(stub_content: str) -> bool:
    """Validate that a generated .pyi stub compiles into valid Python AST.

    Args:
        stub_content: The string content of the .pyi stub file.

    Returns:
        True if the stub compiles without syntax errors.

    Raises:
        SyntaxError: If the stub content is syntactically invalid.
    """
    try:
        ast.parse(stub_content)
    except SyntaxError as e:
        raise SyntaxError(str(e)) from e
    return True
