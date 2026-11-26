"""AST analysis utilities for discovering accepted `**kwargs` keys in Python functions."""

from __future__ import annotations

import ast
import textwrap

from ml_switcheroo_ir.typeclasses.outcome import Err, Ok, Result

from ml_ecosystem_snapshots.errors import IntrospectionFault
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    LivePythonTarget,
    get_member_dict,
    get_runtime_source,
)


def parse_callable_ast(func: LivePythonTarget) -> Result[IntrospectionFault, ast.AST]:
    """Retrieve and parse source code of `func` totally into `Result[IntrospectionFault, ast.AST]`.

    Args:
        func: Live function or method target to inspect.

    Returns:
        `Ok(ast.AST)` on success, or `Err(IntrospectionFault)` when source cannot be read or parsed.
    """
    members = get_member_dict(func)
    mod_name = str(members.get("__module__") or "<runtime>")
    sym_name = str(members.get("__qualname__") or members.get("__name__") or repr(func))
    try:
        src = textwrap.dedent(get_runtime_source(func))
        return Ok(ast.parse(src))
    except (OSError, TypeError, SyntaxError, IndentationError, ValueError) as exc:
        return Err(
            IntrospectionFault(
                module_path=mod_name,
                symbol_name=sym_name,
                reason=f"Unable to parse AST for {func!r}: {exc}",
            )
        )


class _KwargAccessVisitor(ast.NodeVisitor):
    """AST visitor collecting string keys queried on a `**kwargs` dictionary."""

    def __init__(self, kwarg_name: str) -> None:
        """Initialize the visitor with the target `**kwargs` parameter name.

        Args:
            kwarg_name: Formal identifier of the `**kwargs` parameter.
        """
        self._kwarg_name = kwarg_name
        self.discovered: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        """Collect string keys passed to `kwargs.get`, `kwargs.pop`, or `kwargs.setdefault`.

        Args:
            node: AST call node to inspect.
        """
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == self._kwarg_name
            and node.func.attr in ("get", "pop", "setdefault")
        ):
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                self.discovered.add(node.args[0].value)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Collect string keys accessed via `kwargs['key']` subscripting.

        Args:
            node: AST subscript node to inspect.
        """
        if isinstance(node.value, ast.Name) and node.value.id == self._kwarg_name:
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                self.discovered.add(node.slice.value)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Collect string keys tested with `'key' in kwargs` or `'key' not in kwargs`.

        Args:
            node: AST comparison node to inspect.
        """
        for idx in range(min(len(node.ops), len(node.comparators))):
            op = node.ops[idx]
            comparator = node.comparators[idx]
            if isinstance(op, (ast.In, ast.NotIn)):
                if isinstance(comparator, ast.Name) and comparator.id == self._kwarg_name:
                    if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                        self.discovered.add(node.left.value)
        self.generic_visit(node)


def extract_accepted_kwargs_from_ast(
    func: LivePythonTarget,
) -> list[str] | None:
    """Statically analyze a function AST to extract keyword arguments accessed via `**kwargs`.

    Detects patterns like:
        kwargs.get("foo")
        kwargs.pop("bar", default)
        kwargs["baz"]
        "qux" in kwargs

    Args:
        func: Live function or method target to inspect.

    Returns:
        Sorted list of discovered keyword argument names, or None when unavailable.
    """
    tree_res = parse_callable_ast(func)
    if isinstance(tree_res, Err):
        return None
    tree = tree_res.value

    kwarg_name = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.args.kwarg:
                kwarg_name = node.args.kwarg.arg
                break

    if not kwarg_name:
        return None

    visitor = _KwargAccessVisitor(kwarg_name)
    visitor.visit(tree)
    return sorted(list(visitor.discovered)) if visitor.discovered else None


__all__ = ["extract_accepted_kwargs_from_ast", "parse_callable_ast"]
