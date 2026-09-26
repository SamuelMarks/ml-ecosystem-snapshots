"""Ghost Type Stubs Generator.

Generate .pyi stub files from snapshot JSON data.
"""

import os

from typing import Dict, Any, List, Tuple
from pathlib import Path
import ast


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
        # Check if the default string parses as valid Python
        node = ast.parse(default_str, mode="eval").body
        # We can selectively let things through
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


def _format_param_list(
    params: List[Dict[str, Any]],
    has_varargs: bool = False,
    is_method: bool = False,
) -> str:
    """Format a list of parameter dictionaries into a Python signature string.

    Args:
        params: List of parameter dictionary definitions.
        has_varargs: Whether to append generic varargs if none are present.
        is_method: Whether this signature belongs to a method requiring 'self'.

    Returns:
        A comma-separated parameter string for def statements.
    """
    param_strs: List[str] = []
    if is_method and not any(p.get("name") in ("self", "cls") for p in params):
        param_strs.append("self")

    for p in params:
        p_name = p.get("name")
        p_anno = p.get("annotation")
        p_default = p.get("default")
        p_kind = p.get("kind")

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

    if has_varargs and not any(p.get("kind") == "VAR_POSITIONAL" for p in params):
        param_strs.append("*args: Any")

    return ", ".join(param_strs)


def generate_stubs(
    snapshot_data: Any, output_dir: str, include_nonpublic: bool = False
) -> None:
    """Generate .pyi stub files from a snapshot dictionary.

    Args:
        snapshot_data: The snapshot dictionary or list.
        output_dir: The base directory where stubs should be written.
        include_nonpublic: Whether to generate stubs for non-public components.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    modules: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    class_methods: Dict[str, Dict[str, List[Tuple[str, Dict[str, Any]]]]] = {}

    if isinstance(snapshot_data, list):
        categories_dict = {"all": snapshot_data}
    elif isinstance(snapshot_data, dict):
        categories_dict = snapshot_data.get("categories", {})
    else:
        categories_dict = {}

    for _cat, items in categories_dict.items():
        for item in items:
            if not include_nonpublic and not item.get("is_public", True):
                continue

            api_path = (
                item.get("api_path") or item.get("name") or item.get("mnemonic") or ""
            )
            if not api_path:
                continue

            parts = api_path.split(".")
            kind = item.get("kind", "function")

            if kind in ("method", "property") and len(parts) >= 2:
                class_name = parts[-2]
                method_name = parts[-1]
                if len(parts) == 2:
                    module_name = item.get("framework") or item.get("dialect") or ""
                else:
                    module_name = ".".join(parts[:-2])

                if module_name not in class_methods:
                    class_methods[module_name] = {}
                if class_name not in class_methods[module_name]:
                    class_methods[module_name][class_name] = []
                class_methods[module_name][class_name].append((method_name, item))
            else:
                if len(parts) == 1:
                    module_name = item.get("framework") or item.get("dialect") or ""
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
        items = modules.get(module_name, [])

        for obj_name, item in items:
            kind = item.get("kind", "function")

            if kind == "class":
                handled_classes.add(obj_name)
                lines.append(f"class {obj_name}:")

                # Generate @overload signatures for __init__
                for ov in item.get("overloads", []):
                    ov_sig = _format_param_list(ov.get("params", []), is_method=True)
                    ov_ret = (
                        f" -> {ov.get('returns_type')}"
                        if ov.get("returns_type")
                        else " -> None"
                    )
                    lines.append("    @overload")
                    lines.append(f"    def __init__({ov_sig}){ov_ret}: ...")

                init_sig = _format_param_list(
                    item.get("params", []),
                    has_varargs=item.get("has_varargs", False),
                    is_method=True,
                )
                ret_type = item.get("returns_type")
                ret_str = f" -> {ret_type}" if ret_type else " -> Any"
                lines.append(f"    def __init__({init_sig}){ret_str}: ...")

                # Emit associated methods
                methods = class_methods.get(module_name, {}).get(obj_name, [])
                for m_name, m_item in methods:
                    for ov in m_item.get("overloads", []):
                        ov_sig = _format_param_list(
                            ov.get("params", []), is_method=True
                        )
                        ov_ret = (
                            f" -> {ov.get('returns_type')}"
                            if ov.get("returns_type")
                            else " -> Any"
                        )
                        lines.append("    @overload")
                        lines.append(f"    def {m_name}({ov_sig}){ov_ret}: ...")

                    m_sig = _format_param_list(
                        m_item.get("params", []),
                        has_varargs=m_item.get("has_varargs", False),
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
                # Generate @overload signatures for function
                for ov in item.get("overloads", []):
                    ov_sig = _format_param_list(ov.get("params", []))
                    ov_ret = (
                        f" -> {ov.get('returns_type')}"
                        if ov.get("returns_type")
                        else " -> Any"
                    )
                    lines.append("@overload")
                    lines.append(f"def {obj_name}({ov_sig}){ov_ret}: ...")

                sig = _format_param_list(
                    item.get("params", []),
                    has_varargs=item.get("has_varargs", False),
                )
                ret_type = item.get("returns_type")
                ret_str = f" -> {ret_type}" if ret_type else " -> Any"
                lines.append(f"def {obj_name}({sig}){ret_str}: ...")
                lines.append("")

        # Emit classes that only had methods
        for cls_name, methods in class_methods.get(module_name, {}).items():
            if cls_name not in handled_classes:
                lines.append(f"class {cls_name}:")
                for m_name, m_item in methods:
                    for ov in m_item.get("overloads", []):
                        ov_sig = _format_param_list(
                            ov.get("params", []), is_method=True
                        )
                        ov_ret = (
                            f" -> {ov.get('returns_type')}"
                            if ov.get("returns_type")
                            else " -> Any"
                        )
                        lines.append("    @overload")
                        lines.append(f"    def {m_name}({ov_sig}){ov_ret}: ...")

                    m_sig = _format_param_list(
                        m_item.get("params", []),
                        has_varargs=m_item.get("has_varargs", False),
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
