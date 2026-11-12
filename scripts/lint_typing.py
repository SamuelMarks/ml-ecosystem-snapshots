"""AST linter enforcing zero reflection builtins, `Any`, `object`, `cast`, `attrgetter`, `dir`, and dunders.

Scans `src/`, `scripts/`, and `tests/` to guarantee 0 occurrences of:
- Builtin calls: `getattr`, `setattr`, `hasattr`, `delattr`, `vars`, `globals`, `locals`, `dir`, `eval`, `exec`, `compile`
- Operator/inspect/runtime reflection calls & imports: `attrgetter`, `itemgetter`, `methodcaller`, `getattr_static`, `read_runtime_attr`, `has_runtime_attr`, `set_runtime_attr`, `_get_attr`, `_has_attr`, `_set_attr`, `_del_attr`, `_obj_set_attr`, `_obj_del_attr`, `_obj_vars`, `_caller_locals`, `_cast_val`, `lookup_backend_attr`, `has_backend_attr`, `get_backend_attr`
- Dynamic type references & casts: `Any`, `object`, `cast`, `FunctionType`, `any_schema`, `as_*` disguised cast functions
- Structural bypasses: empty `Protocol` classes, tautological `TypeGuard` functions, `if TYPE_CHECKING: ... else: ...` type-erasure blocks, and type-ignore comments
- All dunder attribute accesses (`obj.__foo__`) and direct dunder method invocations (`obj.__foo__(...)`)
"""

from __future__ import annotations

import ast
import glob
import io
import sys
import tokenize
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
        "__import__",
        "cast",
        "attrgetter",
        "itemgetter",
        "methodcaller",
        "getattr_static",
        "read_runtime_attr",
        "has_runtime_attr",
        "set_runtime_attr",
        "_get_attr",
        "_has_attr",
        "_set_attr",
        "_del_attr",
        "_obj_set_attr",
        "_obj_del_attr",
        "_obj_vars",
        "_caller_locals",
        "_cast_val",
        "lookup_backend_attr",
        "has_backend_attr",
        "get_backend_attr",
        "any_schema",
    }
)

BANNED_TYPE_NAMES: frozenset[str] = frozenset({"Any", "object", "FunctionType"})

BANNED_IMPORT_NAMES: frozenset[str] = BANNED_CALL_NAMES | BANNED_TYPE_NAMES

BANNED_MODULE_NAMES: frozenset[str] = frozenset(
    {"builtins", "typing", "typing_extensions", "operator", "inspect", "types", "core_schema"}
)


def is_dunder_name(name: str) -> bool:
    """Return whether `name` is a double-underscore dunder identifier (`__foo__`).

    Args:
        name: Identifier string to check.

    Returns:
        True if `name` starts and ends with `__` and has length greater than 4.
    """
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


_is_dunder_name = is_dunder_name


BANNED_DISGUISED_CASTS: frozenset[str] = frozenset(
    {
        "as_tensor",
        "as_callable",
        "as_int",
        "as_float",
        "as_shape",
        "as_mod",
        "as_ndarray",
        "as_attrs",
        "as_attr_val",
        "as_graph",
        "as_node",
        "as_node_dict",
        "as_tensor_dict",
        "as_tensor_list",
        "as_tensor_obj",
        "as_module_type",
        "as_type_obj",
        "as_int_list",
        "as_str",
        "as_opt_str",
        "as_opt_int",
        "as_bool",
        "as_bytes",
        "as_float_list",
        "as_eager_list",
        "as_tensor_pair",
        "as_tensor_triple",
        "as_tensor_quad",
        "as_opt_int_seq",
        "as_int_or_seq",
        "as_opt_callable",
    }
)


def _is_disguised_cast_name(name: str) -> bool:
    """Return whether `name` is a disguised `as_*` type-erasure cast function.

    Args:
        name: Function or call identifier string.

    Returns:
        True if `name` (stripped of leading underscores) is in `BANNED_DISGUISED_CASTS`.
    """
    return name.lstrip("_") in BANNED_DISGUISED_CASTS


BANNED_INSPECT_ATTRS_OUTSIDE_BOUNDARY: frozenset[str] = frozenset(
    {"getmembers", "getattr_static", "signature", "getsource"}
)


def _check_call_node(node: ast.Call, filepath: str) -> list[str]:
    """Inspect an `ast.Call` node for banned builtin/reflection calls and dunder invocations.

    Args:
        node: AST Call node.
        filepath: Source file path for error reporting.

    Returns:
        List of formatted violation messages.
    """
    errors: list[str] = []
    if isinstance(node.func, ast.Name):
        if node.func.id in BANNED_CALL_NAMES:
            errors.append(f"{filepath}:{node.lineno}:{node.col_offset}: banned call '{node.func.id}(...)'")
        elif _is_disguised_cast_name(node.func.id):
            errors.append(
                f"{filepath}:{node.lineno}:{node.col_offset}: banned disguised cast call '{node.func.id}(...)'"
            )
    elif isinstance(node.func, ast.Attribute):
        attr = node.func.attr
        if (
            attr in BANNED_CALL_NAMES
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in BANNED_MODULE_NAMES
        ):
            errors.append(f"{filepath}:{node.lineno}:{node.col_offset}: banned attribute call '.{attr}(...)'")
        elif attr in BANNED_INSPECT_ATTRS_OUTSIDE_BOUNDARY and not filepath.replace("\\", "/").endswith(
            "introspection/runtime_boundary.py"
        ):
            errors.append(
                f"{filepath}:{node.lineno}:{node.col_offset}: banned reflection call '.{attr}(...)' outside runtime_boundary.py"
            )
        elif _is_dunder_name(attr):
            errors.append(
                f"{filepath}:{node.lineno}:{node.col_offset}: banned direct dunder method call '.{attr}(...)'"
            )
    return errors


def _check_attribute_node(node: ast.Attribute, filepath: str) -> list[str]:
    """Inspect an `ast.Attribute` node for banned dunder attribute access or `typing.Any`/`object`/`FunctionType`.

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
    return [
        f"{filepath}:{node.lineno}:{node.col_offset}: banned import of '{alias.name}'"
        for alias in node.names
        if alias.name in BANNED_IMPORT_NAMES
    ]


BANNED_CLASS_DUNDERS: frozenset[str] = frozenset(
    {
        "__getitem__",
        "__setitem__",
        "__contains__",
        "__getattr__",
        "__getattribute__",
        "__setattr__",
        "__dir__",
        "__call__",
    }
)

WHITELISTED_CONTAINER_DUNDERS: frozenset[tuple[str, str]] = frozenset(
    {
        ("CExtensionSignature", "__getitem__"),
    }
)


def _is_protocol_base(base: ast.expr) -> bool:
    """Return whether `base` is `Protocol` or `typing.Protocol`.

    Args:
        base: Base class AST expression.

    Returns:
        True if the expression references `Protocol`.
    """
    return (isinstance(base, ast.Name) and base.id == "Protocol") or (
        isinstance(base, ast.Attribute) and base.attr == "Protocol"
    )


def _is_type_checking_test(test: ast.expr) -> bool:
    """Return whether an `if` condition tests `TYPE_CHECKING` or `<module>.TYPE_CHECKING`.

    Args:
        test: AST expression from an `ast.If` node.

    Returns:
        True if `test` references `TYPE_CHECKING`.
    """
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _is_src_path(filepath: str) -> bool:
    """Return whether `filepath` resides under `src/`.

    Args:
        filepath: File path string to classify.

    Returns:
        True if `filepath` is inside `src/`.
    """
    norm_path = filepath.replace("\\", "/")
    return norm_path.startswith("src/") or "/src/" in norm_path


def _check_class_node(node: ast.ClassDef, filepath: str) -> list[str]:
    """Inspect an `ast.ClassDef` node for empty `Protocol`s, phantom `if TYPE_CHECKING:` bodies, and non-protocol dunders.

    Args:
        node: AST ClassDef node.
        filepath: Source file path for error reporting.

    Returns:
        List of formatted violation messages.
    """
    errors: list[str] = []
    is_protocol = any(_is_protocol_base(b) for b in node.bases)

    for stmt in node.body:
        if isinstance(stmt, ast.If) and _is_type_checking_test(stmt.test):
            errors.append(
                f"{filepath}:{stmt.lineno}:{stmt.col_offset}: banned phantom 'if TYPE_CHECKING:' block inside class '{node.name}'"
            )
        if (
            not is_protocol
            and _is_src_path(filepath)
            and isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            and stmt.name in BANNED_CLASS_DUNDERS
            and (node.name, stmt.name) not in WHITELISTED_CONTAINER_DUNDERS
        ):
            errors.append(
                f"{filepath}:{stmt.lineno}:{stmt.col_offset}: banned non-protocol dunder method '{stmt.name}' on class '{node.name}'"
            )

    if is_protocol:
        non_doc_stmts = [
            stmt
            for stmt in node.body
            if not isinstance(stmt, ast.Pass)
            and not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
        ]
        if not non_doc_stmts:
            errors.append(f"{filepath}:{node.lineno}:{node.col_offset}: banned empty Protocol class '{node.name}'")
    return errors


def _is_typeguard_return(returns: ast.expr | None) -> bool:
    """Return whether a return annotation is `TypeGuard[...]` or `TypeIs[...]`.

    Args:
        returns: Return annotation AST node or None.

    Returns:
        True if the return annotation uses `TypeGuard` or `TypeIs`.
    """
    if not isinstance(returns, ast.Subscript):
        return False
    val = returns.value
    return (isinstance(val, ast.Name) and val.id in {"TypeGuard", "TypeIs"}) or (
        isinstance(val, ast.Attribute) and val.attr in {"TypeGuard", "TypeIs"}
    )


def _is_trivial_non_none_compare(expr: ast.expr) -> bool:
    """Return whether `expr` is a trivial `x is not None` comparison.

    Args:
        expr: Return expression AST node.

    Returns:
        True if `expr` is `<name> is not None`.
    """
    return (
        isinstance(expr, ast.Compare)
        and len(expr.ops) == 1
        and isinstance(expr.ops[0], ast.IsNot)
        and len(expr.comparators) == 1
        and isinstance(expr.comparators[0], ast.Constant)
        and expr.comparators[0].value is None
    )


def _references_first_param_cleanly(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether `node` has a first parameter loaded in its body without `del <param>`.

    Args:
        node: AST FunctionDef or AsyncFunctionDef node.

    Returns:
        True if the first parameter exists, is never deleted, and is referenced in a `Load` context.
    """
    all_args = [*node.args.posonlyargs, *node.args.args]
    if not all_args:
        return False
    target_name = all_args[0].arg
    has_load = False
    for child in ast.walk(node):
        if isinstance(child, ast.Delete):
            for del_target in child.targets:
                if isinstance(del_target, ast.Name) and del_target.id == target_name:
                    return False
        elif isinstance(child, ast.Name) and child.id == target_name and isinstance(child.ctx, ast.Load):
            has_load = True
    return has_load


def _check_function_node(node: ast.FunctionDef | ast.AsyncFunctionDef, filepath: str) -> list[str]:
    """Inspect a function definition for disguised `as_*` casts or unsound/tautological `TypeGuard` bodies.

    Args:
        node: AST FunctionDef or AsyncFunctionDef node.
        filepath: Source file path for error reporting.

    Returns:
        List of formatted violation messages.
    """
    errors: list[str] = []
    if node.name in BANNED_CALL_NAMES or _is_disguised_cast_name(node.name):
        errors.append(f"{filepath}:{node.lineno}:{node.col_offset}: banned wrapper function definition '{node.name}'")
    if _is_typeguard_return(node.returns):
        if not _references_first_param_cleanly(node):
            errors.append(
                f"{filepath}:{node.lineno}:{node.col_offset}: banned unsound TypeGuard '{node.name}' ignoring or deleting its narrowed parameter"
            )
        for stmt in node.body:
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                if isinstance(stmt.value, ast.Constant) and stmt.value.value is True:
                    errors.append(
                        f"{filepath}:{node.lineno}:{node.col_offset}: banned tautological TypeGuard '{node.name}' returning True"
                    )
                elif _is_trivial_non_none_compare(stmt.value):
                    errors.append(
                        f"{filepath}:{node.lineno}:{node.col_offset}: banned tautological TypeGuard '{node.name}' returning 'is not None'"
                    )
    return errors


def _check_if_node(node: ast.If, filepath: str) -> list[str]:
    """Inspect an `ast.If` node for `if TYPE_CHECKING: ... else: ...` type-erasure blocks.

    Args:
        node: AST If node.
        filepath: Source file path for error reporting.

    Returns:
        List of formatted violation messages.
    """
    if _is_type_checking_test(node.test) and node.orelse:
        return [f"{filepath}:{node.lineno}:{node.col_offset}: banned 'if TYPE_CHECKING: ... else:' block"]
    return []


def _check_except_handler_node(node: ast.ExceptHandler, filepath: str) -> list[str]:
    """Inspect an `ast.ExceptHandler` node for bare `except:`, `BaseException`, or silent `pass` / `del exc`.

    Args:
        node: AST ExceptHandler node.
        filepath: Source file path for error reporting.

    Returns:
        List of formatted violation messages.
    """
    if node.type is None:
        return [f"{filepath}:{node.lineno}:{node.col_offset}: banned bare 'except:' handler"]
    if isinstance(node.type, ast.Name) and node.type.id == "BaseException":
        has_reraise = any(isinstance(n, ast.Raise) for n in ast.walk(node))
        if not has_reraise:
            return [f"{filepath}:{node.lineno}:{node.col_offset}: banned 'except BaseException:' without re-raise"]
    is_broad_exc = isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}
    if is_broad_exc:
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            return [f"{filepath}:{node.lineno}:{node.col_offset}: banned silent 'except Exception: pass' block"]
        if any(isinstance(stmt, ast.Delete) for stmt in node.body):
            return [
                f"{filepath}:{node.lineno}:{node.col_offset}: banned linter-evading 'except Exception: del exc' block"
            ]
        if isinstance(node.type, ast.Name) and node.type.id == "Exception" and _is_prod_or_script_path(filepath):
            return [
                f"{filepath}:{node.lineno}:{node.col_offset}: banned broad 'except Exception:' handler in production/script code"
            ]
    return []


def _check_comments(source: str, filepath: str) -> list[str]:
    """Inspect source tokens for type-ignore, noqa, and pragma-no-cover comments.

    Args:
        source: Raw Python source text.
        filepath: Source file path for error reporting.

    Returns:
        List of formatted violation messages.
    """
    errors: list[str] = []
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for tok_type, tok_str, (lineno, col), _, _ in tokens:
        if tok_type == tokenize.COMMENT:
            if "type:" in tok_str and "ignore" in tok_str:
                errors.append(f"{filepath}:{lineno}:{col}: banned type-ignore comment")
            if "noqa" in tok_str:
                errors.append(f"{filepath}:{lineno}:{col}: banned noqa suppression comment")
            if "pragma:" in tok_str and "no cover" in tok_str:
                errors.append(f"{filepath}:{lineno}:{col}: banned pragma: no cover comment")
    return errors


def _is_prod_or_script_path(filepath: str) -> bool:
    """Return whether `filepath` resides under `src/` or `scripts/`.

    Args:
        filepath: File path string to classify.

    Returns:
        True if `filepath` is inside `src/` or `scripts/`.
    """
    norm_path = filepath.replace("\\", "/")
    return (
        norm_path.startswith("src/")
        or norm_path.startswith("scripts/")
        or "/src/" in norm_path
        or "/scripts/" in norm_path
    )


def _check_single_ast_node(node: ast.AST, filepath: str) -> list[str]:
    """Dispatch AST violation checks for a single visited AST node.

    Args:
        node: Visited AST node.
        filepath: Source file path for error reporting.

    Returns:
        List of formatted violation messages for `node`.
    """
    result: list[str] = []
    if isinstance(node, ast.Call):
        result = _check_call_node(node, filepath)
    elif isinstance(node, ast.Attribute):
        result = _check_attribute_node(node, filepath)
    elif isinstance(node, ast.Name) and node.id in BANNED_TYPE_NAMES:
        result = [f"{filepath}:{node.lineno}:{node.col_offset}: banned '{node.id}' usage"]
    elif isinstance(node, ast.ImportFrom):
        result = _check_import_from_node(node, filepath)
    elif isinstance(node, ast.ClassDef):
        result = _check_class_node(node, filepath)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        result = _check_function_node(node, filepath)
    elif isinstance(node, ast.If):
        result = _check_if_node(node, filepath)
    elif isinstance(node, ast.ExceptHandler):
        result = _check_except_handler_node(node, filepath)
    elif isinstance(node, ast.Assert) and _is_prod_or_script_path(filepath):
        result = [
            f"{filepath}:{node.lineno}:{node.col_offset}: banned runtime 'assert' statement in production/script code"
        ]
    return result


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
    violations: list[str] = _check_comments(source, filepath)
    for node in ast.walk(tree):
        violations.extend(_check_single_ast_node(node, filepath))
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
