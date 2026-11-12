"""CDD docstring and Python AST parameter extraction helpers for `GhostInspector`."""

from __future__ import annotations

import ast
import contextlib
import io
import logging
import types
import typing

import typing_extensions

from ml_ecosystem_snapshots.exceptions import SnapshotError
from ml_ecosystem_snapshots.introspection.context import (
    DocstringIrResult,
    ExtractedParamRecord,
    RawParamTuple,
)
from ml_ecosystem_snapshots.introspection.griffe_inspector import (
    _coerce_param_record,
    _populate_griffe_docstring_ir,
)
from ml_ecosystem_snapshots.models import JsonValue
from ml_ecosystem_snapshots.utils import DocstringMetadata

_LOGGER = logging.getLogger(__name__)


def _populate_cdd_docstring_ir(
    doc: str,
    cdd_docstring_parsers: types.ModuleType,
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
    result: DocstringIrResult,
) -> None:
    """Parse docstring metadata using `cdd.docstring.parse` into `result`.

    Args:
        doc: Raw docstring text to parse.
        cdd_docstring_parsers: Imported `cdd.docstring.parse` module.
        active_sanitize_type_str: Type annotation string sanitizer callable.
        result: Mutable `DocstringIrResult` accumulator.
    """
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            cdd_ir = cdd_docstring_parsers.docstring(doc)
        if cdd_ir.get("returns") and "return_type" in cdd_ir["returns"]:
            ret = cdd_ir["returns"]["return_type"]
            result.returns_type = active_sanitize_type_str(ret.get("typ"))
            result.returns_description = ret.get("doc")

        if cdd_ir.get("params"):
            for p_name, p_val in cdd_ir["params"].items():
                if "Exception" in p_name or "Error" in p_name:
                    result.raises.append(p_name)
                else:
                    result.cdd_params[p_name] = _coerce_param_record(p_val)

        if cdd_ir.get("raises"):
            for exc_dict in cdd_ir["raises"]:
                if "typ" in exc_dict:
                    result.raises.append(exc_dict["typ"])
    except (
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
        LookupError,
        SyntaxError,
        NameError,
        AssertionError,
        SnapshotError,
    ):
        _LOGGER.debug("Failed to extract CDD docstring IR", exc_info=True)


def _extract_docstring_ir(
    doc: str | None,
    docstring_parser_name: str,
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
    active_extract_griffe_meta: typing.Callable[[str, str], DocstringMetadata | typing.Mapping[str, JsonValue]],
) -> tuple[dict[str, ExtractedParamRecord], str | None, str | None, list[str]]:
    """Extract parameter descriptions, return types, and raised exceptions from docstrings.

    Args:
        doc: Raw docstring text.
        docstring_parser_name: Parser style ('google', 'numpy', 'sphinx').
        active_sanitize_type_str: Type annotation string sanitizer callable.
        active_extract_griffe_meta: Griffe docstring metadata extractor callable.

    Returns:
        Tuple of (cdd_params, returns_type, returns_description, raises).
    """
    try:
        import cdd.docstring.parse as cdd_docstring_parsers
    except (ImportError, AttributeError):
        cdd_docstring_parsers = None

    result = DocstringIrResult()
    if doc and cdd_docstring_parsers is not None:
        _populate_cdd_docstring_ir(doc, cdd_docstring_parsers, active_sanitize_type_str, result)
        _populate_griffe_docstring_ir(
            doc,
            docstring_parser_name,
            active_sanitize_type_str,
            active_extract_griffe_meta,
            result,
        )

    return result.cdd_params, result.returns_type, result.returns_description, result.raises


def _is_super_init_call(node: ast.AST) -> typing_extensions.TypeGuard[ast.Call]:
    """Determine whether an AST node represents a `super().__init__(...)` call.

    Args:
        node: AST node to inspect.

    Returns:
        True if `node` is a call to `super().__init__`, False otherwise.
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__init__"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
    )


def _detect_super_kwargs_call(parsed_ast: ast.AST) -> bool:
    """Check whether a class AST contains a `super().__init__(**kwargs)` call.

    Args:
        parsed_ast: Parsed class AST node.

    Returns:
        True if any `super().__init__` call forwards `**kwargs`, False otherwise.
    """
    for node in ast.walk(parsed_ast):
        if _is_super_init_call(node):
            for keyword in node.keywords:
                if keyword.arg is None:
                    return True
    return False


def _parse_cdd_ast_params(
    parsed_ast: ast.AST,
    kind: str,
    cdd_params: typing.Mapping[str, ExtractedParamRecord],
) -> dict[str, ExtractedParamRecord]:
    """Extract parameter definitions from a parsed class or function AST via CDD.

    Args:
        parsed_ast: Parsed class or function AST statement node.
        kind: Symbol kind ('class', 'function', 'method').
        cdd_params: Previously extracted docstring parameter records.

    Returns:
        Dictionary mapping parameter names to `ExtractedParamRecord` instances.
    """
    cdd_ast_params: dict[str, ExtractedParamRecord] = {}
    if kind == "class":
        import cdd.class_.parse

        with contextlib.redirect_stderr(io.StringIO()):
            cdd_parsed_ir = cdd.class_.parse.class_(parsed_ast, merge_inner_function="__init__")
    else:
        import cdd.function.parse

        with contextlib.redirect_stderr(io.StringIO()):
            cdd_parsed_ir = cdd.function.parse.function(parsed_ast)

    if cdd_parsed_ir and "params" in cdd_parsed_ir:
        for p_name, p_val in cdd_parsed_ir["params"].items():
            rec = _coerce_param_record(p_val)
            if p_name in cdd_params and cdd_params[p_name].doc:
                rec.doc = cdd_params[p_name].doc
            cdd_ast_params[p_name] = rec
    return cdd_ast_params


def _format_cdd_param_default(
    raw_def: JsonValue | ast.AST | None,
    active_ast_mod: types.ModuleType,
) -> str | None:
    """Format a CDD AST parameter default into a string representation.

    Args:
        raw_def: Raw default value or AST node from CDD parameter metadata.
        active_ast_mod: Active `ast` module reference used for `unparse`.

    Returns:
        Formatted default value string, or `None` when no default exists.
    """
    if raw_def is None:
        return None
    try:
        if isinstance(raw_def, ast.AST):
            return str(active_ast_mod.unparse(raw_def))
        if isinstance(raw_def, str):
            return repr(raw_def)
        return str(raw_def)
    except (
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
        LookupError,
        SyntaxError,
        NameError,
        AssertionError,
        SnapshotError,
    ):
        _LOGGER.debug("Failed to format CDD parameter default", exc_info=True)
        return str(raw_def)


def _build_initial_cdd_ast_params(
    cdd_ast_params: typing.Mapping[str, ExtractedParamRecord],
    resolved_hints: typing.Mapping[str, type | types.GenericAlias | str],
    active_ast_mod: types.ModuleType,
    active_sanitize_type_str: typing.Callable[[str | None], str | None],
) -> list[RawParamTuple]:
    """Convert CDD AST parameter records into initial raw parameter tuples.

    Args:
        cdd_ast_params: Mapping of parameter names to `ExtractedParamRecord` instances.
        resolved_hints: Resolved runtime type hints mapping.
        active_ast_mod: Active `ast` module reference.
        active_sanitize_type_str: Type annotation string sanitizer callable.

    Returns:
        List of `(name, kind, default, annotation)` tuples excluding `self`.
    """
    extracted_params: list[RawParamTuple] = []
    for p_name, p_info in cdd_ast_params.items():
        if p_name == "self":
            continue
        default_val = _format_cdd_param_default(p_info.default, active_ast_mod)
        anno_val = p_info.typ
        if not anno_val and p_name in resolved_hints:
            anno_val = active_sanitize_type_str(str(resolved_hints[p_name]))
        else:
            anno_val = active_sanitize_type_str(anno_val)
        extracted_params.append((p_name, "POSITIONAL_OR_KEYWORD", default_val, anno_val))
    return extracted_params


__all__ = [
    "_build_initial_cdd_ast_params",
    "_detect_super_kwargs_call",
    "_extract_docstring_ir",
    "_format_cdd_param_default",
    "_is_super_init_call",
    "_parse_cdd_ast_params",
    "_populate_cdd_docstring_ir",
]
