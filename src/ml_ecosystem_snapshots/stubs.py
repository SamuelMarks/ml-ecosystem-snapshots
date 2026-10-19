"""Ghost Type Stubs Generator.

Generate .pyi stub files from snapshot JSON data.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import typing

from ml_switcheroo_ir.schema.ghost import SnapshotEnvelope

from ml_ecosystem_snapshots.models import JsonValue


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
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, (ast.List, ast.Dict, ast.Tuple, ast.Set)):
            return default_str
        elif isinstance(node, ast.Name):
            return default_str
        elif isinstance(node, ast.Attribute):
            return default_str
        elif isinstance(node, ast.UnaryOp):
            return default_str
        elif isinstance(node, ast.BinOp):
            return default_str
        return "..."
    except SyntaxError:
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
    if is_method and not any(
        _param_field(p, "name") in ("self", "cls") for p in params
    ):
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
            if p_anno:
                p_str += f": {p_anno}"
            else:
                p_str += ": Any"

            if p_default is not None:
                sanitized = _sanitize_default(str(p_default))
                p_str += f" = {sanitized}"

        param_strs.append(p_str)

    if has_varargs and not any(
        _param_field(p, "kind") == "VAR_POSITIONAL" for p in params
    ):
        param_strs.append("*args: Any")

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


def generate_stubs(
    snapshot_data: (
        SnapshotEnvelope
        | typing.Mapping[str, object]
        | typing.Sequence[typing.Mapping[str, object]]
    ),
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

    modules: dict[str, list[tuple[str, dict[str, JsonValue]]]] = {}
    class_methods: dict[str, dict[str, list[tuple[str, dict[str, JsonValue]]]]] = {}

    categories_dict: dict[str, list[dict[str, JsonValue]]] = {}
    if isinstance(snapshot_data, list):
        categories_dict = {
            "all": [dict(x) for x in snapshot_data if isinstance(x, dict)]
        }
    elif isinstance(snapshot_data, (dict, SnapshotEnvelope)):
        raw_snap = (
            snapshot_data.model_dump()
            if isinstance(snapshot_data, SnapshotEnvelope)
            else snapshot_data
        )
        raw_cats = raw_snap.get("categories", {})
        assert isinstance(raw_cats, dict)
        for cat_k, cat_v in raw_cats.items():
            assert isinstance(cat_v, list)
            categories_dict[str(cat_k)] = [
                dict(x) for x in cat_v if isinstance(x, dict)
            ]

    for _cat, items in categories_dict.items():
        for item in items:
            if not include_nonpublic and not item.get("is_public", True):
                continue

            api_path = str(
                item.get("api_path") or item.get("name") or item.get("mnemonic") or ""
            )
            if not api_path:
                continue

            parts = api_path.split(".")
            kind = str(item.get("kind") or "function")

            if kind in ("method", "property") and len(parts) >= 2:
                class_name = parts[-2]
                method_name = parts[-1]
                if len(parts) == 2:
                    module_name = str(
                        item.get("framework") or item.get("dialect") or ""
                    )
                else:
                    module_name = ".".join(parts[:-2])

                if module_name not in class_methods:
                    class_methods[module_name] = {}
                if class_name not in class_methods[module_name]:
                    class_methods[module_name][class_name] = []
                class_methods[module_name][class_name].append((method_name, item))
            else:
                if len(parts) == 1:
                    module_name = str(
                        item.get("framework") or item.get("dialect") or ""
                    )
                    obj_name = parts[0]
                else:
                    module_name = ".".join(parts[:-1])
                    obj_name = parts[-1]

                if module_name not in modules:
                    modules[module_name] = []
                modules[module_name].append((obj_name, item))

    all_modules = sorted(set(modules.keys()) | set(class_methods.keys()))

    for module_name in all_modules:
        if not module_name:
            continue

        module_path = Path(os.path.join(out_path, module_name.replace(".", os.sep)))
        module_path.mkdir(parents=True, exist_ok=True)
        init_file = Path(os.path.join(module_path, "__init__.pyi"))

        lines = [
            "from typing import Any, Optional, Union, Tuple, List, Callable, Dict, overload",
            "",
        ]

        handled_classes = set()
        mod_items = modules.get(module_name, [])

        for obj_name, item in mod_items:
            kind = str(item.get("kind") or "function")

            if kind == "class":
                handled_classes.add(obj_name)
                lines.append(f"class {obj_name}:")

                for ov in _extract_overload_dicts(item.get("overloads")):
                    ov_sig = _format_param_list(
                        _extract_param_dicts(ov.get("params")), is_method=True
                    )
                    ov_ret = (
                        f" -> {ov.get('returns_type')}"
                        if ov.get("returns_type")
                        else " -> None"
                    )
                    lines.append("    @overload")
                    lines.append(f"    def __init__({ov_sig}){ov_ret}: ...")

                init_sig = _format_param_list(
                    _extract_param_dicts(item.get("params")),
                    has_varargs=bool(item.get("has_varargs", False)),
                    is_method=True,
                )
                ret_type = item.get("returns_type")
                ret_str = f" -> {ret_type}" if ret_type else " -> Any"
                lines.append(f"    def __init__({init_sig}){ret_str}: ...")

                methods = class_methods.get(module_name, {}).get(obj_name, [])
                for m_name, m_item in methods:
                    for ov in _extract_overload_dicts(m_item.get("overloads")):
                        ov_sig = _format_param_list(
                            _extract_param_dicts(ov.get("params")), is_method=True
                        )
                        ov_ret = (
                            f" -> {ov.get('returns_type')}"
                            if ov.get("returns_type")
                            else " -> Any"
                        )
                        lines.append("    @overload")
                        lines.append(f"    def {m_name}({ov_sig}){ov_ret}: ...")

                    m_sig = _format_param_list(
                        _extract_param_dicts(m_item.get("params")),
                        has_varargs=bool(m_item.get("has_varargs", False)),
                        is_method=True,
                    )
                    m_ret = (
                        f" -> {m_item.get('returns_type')}"
                        if m_item.get("returns_type")
                        else " -> Any"
                    )
                    lines.append(f"    def {m_name}({m_sig}){m_ret}: ...")

                lines.append("")
            else:
                for ov in _extract_overload_dicts(item.get("overloads")):
                    ov_sig = _format_param_list(_extract_param_dicts(ov.get("params")))
                    ov_ret = (
                        f" -> {ov.get('returns_type')}"
                        if ov.get("returns_type")
                        else " -> Any"
                    )
                    lines.append("@overload")
                    lines.append(f"def {obj_name}({ov_sig}){ov_ret}: ...")

                sig = _format_param_list(
                    _extract_param_dicts(item.get("params")),
                    has_varargs=bool(item.get("has_varargs", False)),
                )
                ret_type = item.get("returns_type")
                ret_str = f" -> {ret_type}" if ret_type else " -> Any"
                lines.append(f"def {obj_name}({sig}){ret_str}: ...")
                lines.append("")

        for cls_name, methods in class_methods.get(module_name, {}).items():
            if cls_name not in handled_classes:
                lines.append(f"class {cls_name}:")
                for m_name, m_item in methods:
                    for ov in _extract_overload_dicts(m_item.get("overloads")):
                        ov_sig = _format_param_list(
                            _extract_param_dicts(ov.get("params")), is_method=True
                        )
                        ov_ret = (
                            f" -> {ov.get('returns_type')}"
                            if ov.get("returns_type")
                            else " -> Any"
                        )
                        lines.append("    @overload")
                        lines.append(f"    def {m_name}({ov_sig}){ov_ret}: ...")

                    m_sig = _format_param_list(
                        _extract_param_dicts(m_item.get("params")),
                        has_varargs=bool(m_item.get("has_varargs", False)),
                        is_method=True,
                    )
                    m_ret = (
                        f" -> {m_item.get('returns_type')}"
                        if m_item.get("returns_type")
                        else " -> Any"
                    )
                    lines.append(f"    def {m_name}({m_sig}){m_ret}: ...")
                lines.append("")

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
