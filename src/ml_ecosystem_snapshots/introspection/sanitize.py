"""Default value and type annotation sanitization utilities for Ghost introspection."""

from __future__ import annotations

import ast
import inspect
import re

from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    DefaultValueInput,
    read_runtime_attr,
)
from ml_ecosystem_snapshots.utils import strip_sphinx_roles


def sanitize_param_default(
    val: DefaultValueInput,
) -> tuple[str | None, str | None, bool]:
    """Sanitize parameter default value while preserving semantic literals and scrubbing memory addresses.

    Args:
        val: Raw default value from `inspect.Parameter` or AST.

    Returns:
        Tuple of `(default_value_string, default_factory_string, is_mandatory_bool)`.
    """
    if val is inspect.Parameter.empty:
        return (None, None, True)
    if val is Ellipsis:
        return ("...", None, False)
    if val is None:
        return ("None", None, False)
    if isinstance(val, bool):
        return ("True" if val else "False", None, False)
    if isinstance(val, (int, float)):
        return (str(val), None, False)
    if isinstance(val, str):
        return (repr(val), None, False)

    try:
        if callable(val):
            val_repr = repr(val)
            if val_repr.startswith("<class '") and val_repr.endswith("'>"):
                cls_name = val_repr[8:-2]
                return (cls_name, None, False)
            func_name = read_runtime_attr(val, "__name__", None)
            factory_str = (
                f"<function {func_name}>" if func_name else "<factory_default>"
            )
            return ("<factory_default>", factory_str, False)

        val_repr = repr(val)
        if re.match(r"^<dtype:\s*'([^']+)'\s*>$", val_repr):
            m_dt = re.match(r"^<dtype:\s*'([^']+)'\s*>$", val_repr)
            return (f"tf.{m_dt.group(1)}" if m_dt else val_repr, None, False)

        if "device(type=" in val_repr:
            m_dev = re.search(r"type='([^']+)'", val_repr)
            dev_type = m_dev.group(1) if m_dev else "cpu"
            return (repr(dev_type), val_repr, False)

        if " at 0x" in val_repr or re.search(r"\b0x[0-9a-fA-F]{4,}\b", val_repr):
            scrubbed = re.sub(r"\s*at\s*0x[0-9a-fA-F]+", "", val_repr)
            scrubbed = re.sub(r"\b0x[0-9a-fA-F]{4,}\b", "<addr>", scrubbed)
            if scrubbed.startswith("<") and scrubbed.endswith(">"):
                return ("<factory_default>", scrubbed, False)
            return (scrubbed, None, False)

        val_str = str(val)
        if " at 0x" in val_str or re.search(r"\b0x[0-9a-fA-F]{4,}\b", val_str):
            return ("<factory_default>", "<factory_default>", False)
        return (val_str, None, False)
    except Exception:
        return ("<unrepresentable>", None, False)


def sanitize_type_str(typ_str: str | None) -> str | None:
    """Normalize type hint strings to PEP-585 and PEP-604 syntax.

    Args:
        typ_str: Raw type annotation string or None.

    Returns:
        Sanitized type string or None.
    """
    if not typ_str:
        return typ_str

    typ_str = strip_sphinx_roles(typ_str)
    if not typ_str:
        return typ_str

    typ_str = re.sub(r"<class '([^']+)'>", r"\1", typ_str)

    class TypeHintSanitizer(ast.NodeTransformer):
        """AST transformer to sanitize and format type hints."""

        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            """Visit subscript nodes and format Unions and Optionals.

            Args:
                node: The node to visit.

            Returns:
                The transformed node.
            """
            self.generic_visit(node)
            is_union = False
            if isinstance(node.value, ast.Name) and node.value.id == "Union":
                is_union = True

            if is_union:
                slice_val = node.slice
                if isinstance(slice_val, ast.Tuple):
                    elts = slice_val.elts
                else:
                    elts = [slice_val]

                if len(elts) >= 2:
                    new_node: ast.expr = elts[0]
                    for elt in elts[1:]:
                        new_node = ast.BinOp(left=new_node, op=ast.BitOr(), right=elt)
                    return ast.copy_location(new_node, node)

            is_optional = False
            if isinstance(node.value, ast.Name) and node.value.id == "Optional":
                is_optional = True

            if is_optional:
                slice_val = node.slice
                return ast.copy_location(
                    ast.BinOp(
                        left=slice_val, op=ast.BitOr(), right=ast.Constant(value=None)
                    ),
                    node,
                )

            return node

        def visit_Name(self, node: ast.Name) -> ast.AST:
            """Clean up common Name nodes like List -> list.

            Args:
                node: The node to visit.

            Returns:
                The transformed node.
            """
            pep585_map = {
                "List": "list",
                "Dict": "dict",
                "Tuple": "tuple",
                "Set": "set",
                "Type": "type",
            }
            if node.id in pep585_map:
                node.id = pep585_map[node.id]
            return node

        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            """Flatten attribute access for simplified type names.

            Args:
                node: AST Attribute node to normalize.

            Returns:
                The transformed node.
            """
            if isinstance(node.value, ast.Name) and node.value.id == "typing":
                pep585_map = {
                    "List": "list",
                    "Dict": "dict",
                    "Tuple": "tuple",
                    "Set": "set",
                    "Type": "type",
                }
                if node.attr in pep585_map:
                    return ast.copy_location(
                        ast.Name(id=pep585_map[node.attr], ctx=ast.Load()), node
                    )
                return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
            self.generic_visit(node)
            return node

    try:
        node = ast.parse(typ_str, mode="eval")
        node = TypeHintSanitizer().visit(node)
        return ast.unparse(node)
    except Exception:
        return typ_str


__all__ = ["sanitize_param_default", "sanitize_type_str"]
