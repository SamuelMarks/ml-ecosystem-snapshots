"""AST linter enforcing zero reflection builtins, `Any`, `object`, `cast`, `attrgetter`, `dir`, and dunders.

Scans `src/`, `scripts/`, and `tests/` to guarantee 0 occurrences of:
- Builtin calls: `getattr`, `setattr`, `hasattr`, `delattr`, `vars`, `globals`, `locals`, `dir`, `eval`, `exec`, `compile`
- Operator/inspect/runtime reflection calls & imports: `attrgetter`, `itemgetter`, `methodcaller`, `getattr_static`, `read_runtime_attr`, `has_runtime_attr`, `set_runtime_attr`
- Typing escapes: `Any`, `object`, `cast`
- All dunder attribute accesses (`obj.__foo__`) and direct dunder method invocations (`obj.__foo__(...)`)
"""

from __future__ import annotations

import ast
import glob
import sys
from collections.abc import Sequence

BANNED_CALL_NAMES: frozenset[str] = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "vars",
        "globals",
        "locals",
        "dir",
        "eval",
        "exec",
        "compile",
        "cast",
        "attrgetter",
        "itemgetter",
        "methodcaller",
        "getattr_static",
        "read_runtime_attr",
        "has_runtime_attr",
        "set_runtime_attr",
    }
)

BANNED_TYPE_NAMES: frozenset[str] = frozenset({"Any", "object"})

BANNED_IMPORT_NAMES: frozenset[str] = BANNED_CALL_NAMES | BANNED_TYPE_NAMES


BANNED_MODULE_NAMES: frozenset[str] = frozenset({"builtins", "typing", "typing_extensions", "operator", "inspect"})


def _is_dunder_name(name: str) -> bool:
    """Return whether `name` is a double-underscore dunder identifier (`__foo__`).

    Args:
        name: Identifier string to check.

    Returns:
        True if `name` starts and ends with `'__'` and has length > 4.
    """
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _check_call_node(node: ast.Call, filepath: str) -> list[str]:
    """Inspect an `ast.Call` node for banned builtin/reflection calls and dunder invocations.

    Args:
        node: AST Call node.
        filepath: Source file path for error reporting.

    Returns:
        List of formatted violation messages.
    """
    errors: list[str] = []
    if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALL_NAMES:
        errors.append(f"{filepath}:{node.lineno}:{node.col_offset}: banned call '{node.func.id}(...)'")
    elif isinstance(node.func, ast.Attribute):
        attr = node.func.attr
        if attr in BANNED_CALL_NAMES and isinstance(node.func.value, ast.Name) and node.func.value.id in BANNED_MODULE_NAMES:
            errors.append(f"{filepath}:{node.lineno}:{node.col_offset}: banned attribute call '.{attr}(...)'")
        elif _is_dunder_name(attr):
            errors.append(f"{filepath}:{node.lineno}:{node.col_offset}: banned direct dunder method call '.{attr}(...)'")
    return errors


def _check_attribute_node(node: ast.Attribute, filepath: str) -> list[str]:
    """Inspect an `ast.Attribute` node for banned dunder attribute access or `typing.Any`/`object`.

    Args:
        node: AST Attribute node.
        filepath: Source file path for error reporting.

    Returns:
        List of formatted violation messages.
    """
    if _is_dunder_name(node.attr):
        return [f"{filepath}:{node.lineno}:{node.col_offset}: banned dunder attribute access '.{node.attr}'"]
    if node.attr in BANNED_TYPE_NAMES and isinstance(node.value, ast.Name) and node.value.id in BANNED_MODULE_NAMES:
        return [f"{filepath}:{node.lineno}:{node.col_offset}: banned '.{node.attr}' type usage"]
    return []


def _check_import_from_node(node: ast.ImportFrom, filepath: str) -> list[str]:
    """Inspect an `ast.ImportFrom` node for imports of banned typing or reflection symbols.

    Args:
        node: AST ImportFrom node.
        filepath: Source file path for error reporting.

    Returns:
        List of formatted violation messages.
    """
    return [f"{filepath}:{node.lineno}:{node.col_offset}: banned import of '{alias.name}'" for alias in node.names if alias.name in BANNED_IMPORT_NAMES]


def check_file(filepath: str) -> list[str]:
    """Parse a Python source file into an AST and return all typing/reflection violations.

    Args:
        filepath: Path to the Python source file to inspect.

    Returns:
        List of formatted violation strings (`filepath:lineno:col: message`).
    """
    with open(filepath, encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=filepath)
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            violations.extend(_check_call_node(node, filepath))
        elif isinstance(node, ast.Attribute):
            violations.extend(_check_attribute_node(node, filepath))
        elif isinstance(node, ast.Name) and node.id in BANNED_TYPE_NAMES:
            violations.append(f"{filepath}:{node.lineno}:{node.col_offset}: banned '{node.id}' usage")
        elif isinstance(node, ast.ImportFrom):
            violations.extend(_check_import_from_node(node, filepath))

    return violations


def lint_directories(
    roots: Sequence[str] = ("src", "scripts", "tests"),
) -> list[str]:
    """Scan all `.py` files under the given root directories and collect AST violations.

    Args:
        roots: Sequence of directory paths to scan recursively.

    Returns:
        Sorted list of all AST violation strings discovered.
    """
    all_violations: list[str] = []
    for root in roots:
        for filepath in sorted(glob.glob(f"{root}/**/*.py", recursive=True)):
            all_violations.extend(check_file(filepath))
    return all_violations


def main() -> int:
    """Run the typing and reflection AST linter across `src/`, `scripts/`, and `tests/`.

    Returns:
        Exit code `0` if zero violations are found, `1` otherwise.
    """
    violations = lint_directories()
    if violations:
        for v in violations:
            print(v, file=sys.stderr)
        print(
            f"Found {len(violations)} typing/reflection violation(s).",
            file=sys.stderr,
        )
        return 1
    print("scripts/lint_typing.py: 0 violations across src/, scripts/, and tests/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
