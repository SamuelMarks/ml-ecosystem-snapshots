"""AST analysis utilities for discovering accepted `**kwargs` keys in Python functions."""

from __future__ import annotations

import ast
import textwrap

from ml_ecosystem_snapshots.exceptions import IntrospectionError
from ml_ecosystem_snapshots.introspection.runtime_boundary import (
    InspectableTarget,
    get_runtime_source,
)


def _parse_callable_ast(
    func: InspectableTarget,
) -> ast.AST:
    """Retrieve and parse the Python AST of a live callable target.

    Args:
        func: Live function or method target to inspect.

    Returns:
        Parsed module `ast.AST` tree.

    Raises:
        IntrospectionError: If source code cannot be retrieved or parsed.
    """
    try:
        src = get_runtime_source(func)
        src = textwrap.dedent(src)
        return ast.parse(src)
    except (OSError, TypeError, SyntaxError, IndentationError, ValueError) as exc:
        raise IntrospectionError(
            f"Unable to parse AST for callable {func!r}: {exc}"
        ) from exc


def extract_accepted_kwargs_from_ast(
    func: InspectableTarget,
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
    try:
        tree = _parse_callable_ast(func)
    except IntrospectionError:
        return None

    kwarg_name = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.args.kwarg:
                kwarg_name = node.args.kwarg.arg
                break

    if not kwarg_name:
        return None

    discovered: set[str] = set()

    class KwargAccessVisitor(ast.NodeVisitor):
        """AST visitor collecting string keys queried on the var_keyword dictionary."""

        def visit_Call(self, node: ast.Call) -> None:
            """Inspect method calls on the kwargs dict.

            Args:
                node: The AST Call node.
            """
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == kwarg_name
                and node.func.attr in ("get", "pop", "setdefault")
            ):
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    discovered.add(node.args[0].value)
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            """Inspect subscript accesses like kwargs['key'].

            Args:
                node: The AST Subscript node.
            """
            if isinstance(node.value, ast.Name) and node.value.id == kwarg_name:
                slice_node = node.slice
                if isinstance(slice_node, ast.Constant) and isinstance(
                    slice_node.value, str
                ):
                    discovered.add(slice_node.value)
                elif hasattr(ast, "Index") and isinstance(
                    slice_node, getattr(ast, "Index")
                ):
                    idx_val = getattr(slice_node, "value", None)
                    if isinstance(idx_val, ast.Constant) and isinstance(
                        idx_val.value, str
                    ):
                        discovered.add(idx_val.value)
            self.generic_visit(node)

        def visit_Compare(self, node: ast.Compare) -> None:
            """Inspect membership checks like 'key' in kwargs.

            Args:
                node: The AST Compare node.
            """
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    if isinstance(comparator, ast.Name) and comparator.id == kwarg_name:
                        if isinstance(node.left, ast.Constant) and isinstance(
                            node.left.value, str
                        ):
                            discovered.add(node.left.value)
            self.generic_visit(node)

    KwargAccessVisitor().visit(tree)
    return sorted(list(discovered)) if discovered else None


__all__ = ["extract_accepted_kwargs_from_ast"]
