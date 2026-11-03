"""AST linter enforcing zero reflection builtins, `Any`, `cast`, and direct dunder calls.

Scans `src/`, `scripts/`, and `tests/` to guarantee 0 occurrences of:
- Builtin calls: `getattr`, `setattr`, `hasattr`, `delattr`, `vars`, `globals`, `locals`, `cast`
- Qualified calls: `typing.cast`, `typing_extensions.cast`
- `Any` annotations or references (`Any`, `typing.Any`, `typing_extensions.Any`)
- Reflection attributes: `.__dict__`, `.__mro__`
- Direct dunder method invocations (excluding `super().__<method>__(...)` and `.__init__(...)`)
"""

from __future__ import annotations

import ast
import glob
from pathlib import Path
import sys

_BANNED_BUILTIN_CALLS: frozenset[str] = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "vars",
        "globals",
        "locals",
        "cast",
    }
)

_BANNED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__dict__",
        "__mro__",
    }
)

_ALLOWED_DUNDER_CALLS: frozenset[str] = frozenset(
    {
        "__init__",
        "__import__",
    }
)


def _is_super_call(node: ast.AST) -> bool:
    """Check whether an AST expression is a `super()` call.

    Args:
        node: AST expression node to inspect.

    Returns:
        True if `node` is `super(...)`, False otherwise.
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "super"
    )


def check_file(file_path: str) -> list[str]:
    """Scan a single Python source file for banned reflection and typing constructs.

    Args:
        file_path: Path to the Python file to scan.

    Returns:
        List of formatted violation messages found in the file.
    """
    source = Path(file_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=file_path)
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in _BANNED_BUILTIN_CALLS
            ):
                violations.append(
                    f"{file_path}:{node.lineno}:{node.col_offset}: banned call '{node.func.id}()'"
                )
            elif isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if (
                    attr == "cast"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in {"typing", "typing_extensions"}
                ):
                    violations.append(
                        f"{file_path}:{node.lineno}:{node.col_offset}: banned call '{node.func.value.id}.cast()'"
                    )
                elif (
                    attr.startswith("__")
                    and attr.endswith("__")
                    and attr not in _ALLOWED_DUNDER_CALLS
                    and not _is_super_call(node.func.value)
                ):
                    violations.append(
                        f"{file_path}:{node.lineno}:{node.col_offset}: banned direct dunder call '.{attr}()'"
                    )
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTRIBUTES:
                violations.append(
                    f"{file_path}:{node.lineno}:{node.col_offset}: banned attribute '.{node.attr}'"
                )
            elif node.attr == "Any":
                violations.append(
                    f"{file_path}:{node.lineno}:{node.col_offset}: banned 'Any' usage"
                )
        elif isinstance(node, ast.Name) and node.id in {"Any", "object"}:
            violations.append(
                f"{file_path}:{node.lineno}:{node.col_offset}: banned '{node.id}' usage"
            )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in {"Any", "cast"}:
                    violations.append(
                        f"{file_path}:{node.lineno}:{node.col_offset}: banned import '{alias.name}'"
                    )

    return violations


def lint_directories(
    directories: tuple[str, ...] = ("src", "scripts", "tests"),
) -> list[str]:
    """Scan all Python files across the target directories.

    Args:
        directories: Tuple of root directory paths to scan recursively.

    Returns:
        Combined list of all violation strings across all scanned files.
    """
    all_violations: list[str] = []
    for root in directories:
        for file_path in sorted(glob.glob(f"{root}/**/*.py", recursive=True)):
            all_violations.extend(check_file(file_path))
    return all_violations


def main() -> int:
    """Execute the typing and anti-reflection AST linter CLI.

    Returns:
        Exit code (0 if clean, 1 if violations were detected).
    """
    violations = lint_directories()
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        print(
            f"Found {len(violations)} typing/reflection violation(s).",
            file=sys.stderr,
        )
        return 1
    print("scripts/lint_typing.py: 0 violations across src/, scripts/, and tests/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
