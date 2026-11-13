"""AST docstring completeness and quality linter across `src/`, `scripts/`, and `tests/`.

Verifies 100% file, module, class, method, function, parameter (`Args:` / `:param`),
return (`Returns:` / `Yields:` / `:returns:`), and exception (`Raises:` / `:raises:`)
documentation coverage, with zero robotic placeholder text.
"""

from __future__ import annotations

import ast
import glob
import re
import sys
from collections.abc import Sequence

BANNED_PLACEHOLDER_PHRASES: tuple[str, ...] = (
    "Execute helper ",
    "Computed result.",
    ": Exception.",
    "Mock or helper class ",
)


def _is_overload_decorated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a function or method is decorated with `@overload`.

    Args:
        node: Function or async function AST node.

    Returns:
        True if any decorator is `overload` or `<mod>.overload`.
    """
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "overload":
            return True
    return False


def _collect_own_scope_raises(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether `node` contains an explicit `raise Exc(...)` in its own scope.

    Skips nested function and class definitions so inner helper raises do not
    require `Raises:` on outer enclosing functions.

    Args:
        node: Function or async function AST node to inspect.

    Returns:
        True if an explicit non-bare `raise` statement exists in `node`'s own scope.
    """
    stack: list[ast.AST] = list(node.body)
    while stack:
        current = stack.pop()
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(current, ast.Raise) and current.exc is not None:
            return True
        stack.extend(ast.iter_child_nodes(current))
    return False


def _is_param_documented(doc: str, param_name: str) -> bool:
    """Return whether `param_name` is documented in Google or Sphinx format inside `doc`.

    Args:
        doc: Cleaned docstring text.
        param_name: Parameter identifier (including leading `*` or `**` for variadics).

    Returns:
        True if `param_name` appears in `Args:` or `:param <name>:` format.
    """
    escaped = re.escape(param_name)
    bare_escaped = re.escape(param_name.lstrip("*"))
    nl = chr(10)
    pattern = rf"(?:^|{nl})\s*(?:{escaped}\s*(?:\([^{nl})]*\))?\s*:|:param\s+(?:{escaped}|{bare_escaped})\s*:)"
    return re.search(pattern, doc) is not None


def _collect_required_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Extract parameter names requiring documentation (excluding `self` and `cls`).

    Args:
        node: Function or async function AST node.

    Returns:
        Ordered list of parameter identifiers (with `*`/`**` prefixes on variadics).
    """
    names: list[str] = [
        arg.arg
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        if arg.arg not in {"self", "cls"}
    ]
    if node.args.vararg is not None:
        names.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg is not None:
        names.append(f"**{node.args.kwarg.arg}")
    return names


def _requires_return_section(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether `node`'s return annotation requires a `Returns:` or `Yields:` section.

    Args:
        node: Function or async function AST node.

    Returns:
        True if `node` is not `__init__` and has a non-`None` / non-`NoReturn` return annotation.
    """
    if node.name == "__init__" or node.returns is None:
        return False
    ret_text = ast.unparse(node.returns).strip()
    return ret_text not in {"None", "NoReturn", "typing.NoReturn", "typing_extensions.NoReturn"}


def _check_placeholder_phrases(doc: str, filepath: str, lineno: int, symbol: str) -> list[str]:
    """Check a docstring for banned robotic placeholder phrases.

    Args:
        doc: Cleaned docstring text.
        filepath: File path for violation messages.
        lineno: Line number of the documented AST node.
        symbol: Symbol name for violation messages.

    Returns:
        List of formatted violation strings.
    """
    return [
        f"{filepath}:{lineno}: banned robotic docstring phrase {phrase!r} in '{symbol}'"
        for phrase in BANNED_PLACEHOLDER_PHRASES
        if phrase in doc
    ]


def _check_function_docstring(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    filepath: str,
) -> list[str]:
    """Validate docstring presence, `Args:`, `Returns:`/`Yields:`, and `Raises:` for a function.

    Args:
        node: Function or async function AST node.
        filepath: File path for violation messages.

    Returns:
        List of formatted violation strings.
    """
    if _is_overload_decorated(node):
        return []

    doc = ast.get_docstring(node)
    if not doc or not doc.strip():
        return [f"{filepath}:{node.lineno}: missing function docstring on '{node.name}'"]

    errors = _check_placeholder_phrases(doc, filepath, node.lineno, node.name)
    params = _collect_required_params(node)
    if params:
        has_section = "Args:" in doc or ":param " in doc
        missing = [p for p in params if not _is_param_documented(doc, p)]
        if not has_section or missing:
            errors.append(f"{filepath}:{node.lineno}: undocumented parameter(s) {missing or params} on '{node.name}'")

    if _requires_return_section(node):
        if not any(tok in doc for tok in ("Returns:", "Yields:", ":returns:", ":return:")):
            errors.append(f"{filepath}:{node.lineno}: missing Returns:/Yields: docstring section on '{node.name}'")

    if _collect_own_scope_raises(node):
        if "Raises:" not in doc and ":raises " not in doc:
            errors.append(f"{filepath}:{node.lineno}: missing Raises: docstring section on '{node.name}'")

    return errors


def check_file_docstrings(filepath: str) -> list[str]:
    """Parse `filepath` and return all module, class, function, arg, return, and raises doc violations.

    Args:
        filepath: Path to the Python source file to inspect.

    Returns:
        List of formatted violation messages.
    """
    with open(filepath, encoding="utf-8") as handle:
        source = handle.read()

    tree = ast.parse(source, filename=filepath)
    violations: list[str] = []

    mod_doc = ast.get_docstring(tree)
    if not mod_doc or not mod_doc.strip():
        violations.append(f"{filepath}:1: missing module docstring")
    else:
        violations.extend(_check_placeholder_phrases(mod_doc, filepath, 1, "<module>"))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            cls_doc = ast.get_docstring(node)
            if not cls_doc or not cls_doc.strip():
                violations.append(f"{filepath}:{node.lineno}: missing class docstring on '{node.name}'")
            else:
                violations.extend(_check_placeholder_phrases(cls_doc, filepath, node.lineno, node.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_check_function_docstring(node, filepath))

    return violations


def lint_docstring_directories(
    roots: Sequence[str] = ("src", "scripts", "tests"),
) -> list[str]:
    """Scan all `.py` files under `roots` and return all docstring violations.

    Args:
        roots: Sequence of directory paths to scan recursively.

    Returns:
        List of all docstring violation strings discovered.
    """
    all_violations: list[str] = []
    for root in roots:
        for filepath in sorted(glob.glob(f"{root}/**/*.py", recursive=True)):
            all_violations.extend(check_file_docstrings(filepath))
    return all_violations


def main() -> int:
    """Execute the AST docstring completeness linter across `src/`, `scripts/`, and `tests/`.

    Returns:
        Exit code `0` when 0 violations are found, `1` otherwise.
    """
    violations = lint_docstring_directories()
    if violations:
        for item in violations:
            print(item, file=sys.stderr)
        print(f"Found {len(violations)} docstring violation(s).", file=sys.stderr)
        return 1
    print("scripts/lint_docstrings.py: 0 docstring violations across src/, scripts/, and tests/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
