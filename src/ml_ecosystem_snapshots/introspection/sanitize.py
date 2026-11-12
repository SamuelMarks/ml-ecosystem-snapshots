"""Default value and type annotation sanitization utilities for Ghost introspection."""

from __future__ import annotations

import ast
import inspect
import re

from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    DefaultValueInput,
    get_member_dict,
)
from ml_ecosystem_snapshots.utils import strip_sphinx_roles


def _sanitize_callable_default(
    val: DefaultValueInput,
) -> tuple[str | None, str | None, bool]:
    """Sanitize a callable parameter default value into class name or factory metadata.

    Args:
        val: Callable default value from `inspect.Parameter`.

    Returns:
        Tuple of `(default_value_string, default_factory_string, is_mandatory_bool)`.
    """
    val_repr = repr(val)
    if val_repr.startswith("<class '") and val_repr.endswith("'>"):
        cls_name = val_repr[8:-2]
        return (cls_name, None, False)
    func_name = get_member_dict(val).get("__name__")
    factory_str = f"<function {func_name}>" if func_name else "<factory_default>"
    return ("<factory_default>", factory_str, False)


def _sanitize_repr_default(
    val: DefaultValueInput,
) -> tuple[str | None, str | None, bool]:
    """Sanitize a non-primitive, non-callable default value by scrubbing memory addresses.

    Args:
        val: Non-primitive default value object.

    Returns:
        Tuple of `(default_value_string, default_factory_string, is_mandatory_bool)`.
    """
    val_repr = repr(val)
    m_dt = re.match(r"^<dtype:\s*'([^']+)'\s*>$", val_repr)
    if m_dt:
        return (f"tf.{m_dt.group(1)}", None, False)

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
    if val is Ellipsis or val is None or isinstance(val, (bool, int, float)):
        return ("..." if val is Ellipsis else str(val), None, False)
    if isinstance(val, str):
        return (repr(val), None, False)

    try:
        if callable(val):
            return _sanitize_callable_default(val)
        return _sanitize_repr_default(val)
    except (
        SnapshotError,
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        LookupError,
        OSError,
        RuntimeError,
        SyntaxError,
    ):
        return ("<unrepresentable>", None, False)


_PEP585_TYPE_MAP: dict[str, str] = {
    "List": "list",
    "Dict": "dict",
    "Tuple": "tuple",
    "Set": "set",
    "Type": "type",
}


class _TypeHintSanitizer(ast.NodeTransformer):
    """AST transformer to normalize `Union`, `Optional`, and `typing.*` nodes to PEP-585/604."""

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Transform `Union[...]` and `Optional[...]` subscripts into bitwise-or (`|`) expressions.

        Args:
            node: The AST Subscript node to transform.

        Returns:
            The transformed AST node.
        """
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "Union":
            slice_val = node.slice
            elts = slice_val.elts if isinstance(slice_val, ast.Tuple) else [slice_val]
            if len(elts) >= 2:
                new_node: ast.expr = elts[0]
                for elt in elts[1:]:
                    new_node = ast.BinOp(left=new_node, op=ast.BitOr(), right=elt)
                return ast.copy_location(new_node, node)

        if isinstance(node.value, ast.Name) and node.value.id == "Optional":
            return ast.copy_location(
                ast.BinOp(left=node.slice, op=ast.BitOr(), right=ast.Constant(value=None)),
                node,
            )

        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Normalize capitalized typing generic names (`List` -> `list`, `Dict` -> `dict`).

        Args:
            node: The AST Name node to normalize.

        Returns:
            The normalized AST Name node.
        """
        if node.id in _PEP585_TYPE_MAP:
            node.id = _PEP585_TYPE_MAP[node.id]
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Flatten `typing.<Name>` attribute accesses into unqualified PEP-585 identifiers.

        Args:
            node: AST Attribute node to normalize.

        Returns:
            The transformed AST node.
        """
        if isinstance(node.value, ast.Name) and node.value.id == "typing":
            target_id = _PEP585_TYPE_MAP.get(node.attr, node.attr)
            return ast.copy_location(ast.Name(id=target_id, ctx=ast.Load()), node)
        self.generic_visit(node)
        return node


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

    try:
        node = ast.parse(typ_str, mode="eval")
        node = _TypeHintSanitizer().visit(node)
        return ast.unparse(node)
    except (
        SnapshotError,
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        LookupError,
        OSError,
        RuntimeError,
        SyntaxError,
    ):
        return typ_str


__all__ = ["sanitize_param_default", "sanitize_type_str"]
