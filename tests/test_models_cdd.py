"""Unit tests for CDD-based docstring and AST parameter parsing in GhostInspector."""

from __future__ import annotations

import sys

import pytest
from ml_switcheroo_ir.schema.ghost import ParameterKind

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.models import JsonValue


def some_function(a: int = 1, b: str = "test", *args: JsonValue, **kwargs: JsonValue) -> None:
    """Execute a sample function with typed parameters and variadic arguments.

    Args:
        a: The first integer parameter.
        b: The second string parameter.
        *args: Variadic positional arguments.
        **kwargs: Variadic keyword arguments.
    """
    pass


def container_default_function(
    pair: tuple[int, int] = (1, 2),
    tags: set[str] | None = None,
) -> None:
    """Execute a sample function with a Python tuple default value.

    Args:
        pair: Two-element integer coordinate tuple.
        tags: Optional set of tag identifiers.
    """
    del pair, tags


class SomeClass:
    """Sample class with Sphinx-style docstring for CDD class inspection."""

    def __init__(self, x: float, y: int = 5) -> None:
        """Initialize SomeClass.

        :param x: The x coordinate.
        :type x: float
        :param y: The y count.
        :type y: int
        """
        pass


def test_ghost_inspector_cdd_function() -> None:
    """Verify GhostInspector extracts parameter types, defaults, variadics, and descriptions via CDD."""
    ref = GhostInspector.inspect(some_function, "some_function")
    assert ref.name == "some_function"
    assert ref.kind == "function"
    assert ref.has_varargs is True
    assert len(ref.params) == 4  # a, b, args, kwargs

    a_param = next(p for p in ref.params if p.name == "a")
    assert a_param.annotation == "int"
    assert a_param.default == "1"
    assert a_param.description == "The first integer parameter."

    b_param = next(p for p in ref.params if p.name == "b")
    assert b_param.annotation == "str"
    assert b_param.default == "'test'"
    assert b_param.description == "The second string parameter."

    args_param = next(p for p in ref.params if p.name == "args")
    assert args_param.kind == ParameterKind.VAR_POSITIONAL
    assert args_param.description == "Variadic positional arguments."

    kwargs_param = next(p for p in ref.params if p.name == "kwargs")
    assert kwargs_param.kind == ParameterKind.VAR_KEYWORD
    assert kwargs_param.description == "Variadic keyword arguments."


def test_ghost_inspector_cdd_container_defaults() -> None:
    """Verify GhostInspector preserves CDD AST extraction on Python tuple and container defaults."""
    import ast

    from ml_ecosystem_snapshots.introspection.cdd_inspector import collect_variadic_ast_names
    from ml_ecosystem_snapshots.introspection.context import (
        ExtractedParamRecord,
        ParamDefaultAtom,
    )

    assert collect_variadic_ast_names(ast.Pass()) == set()
    ref = GhostInspector.inspect(container_default_function, "container_default_function")
    pair_param = next(p for p in ref.params if p.name == "pair")
    assert pair_param.default == "(1, 2)"
    assert pair_param.description == "Two-element integer coordinate tuple."

    list_default: list[ParamDefaultAtom] = [(1, 2)]
    dict_default: dict[str, ParamDefaultAtom] = {"shape": (1, 2)}
    assert ExtractedParamRecord(default=()).default == ()
    assert ExtractedParamRecord(default=((1, 2), "x")).default == ((1, 2), "x")
    assert ExtractedParamRecord(default=b"abc").default == b"abc"
    assert ExtractedParamRecord(default={"a"}).default == {"a"}
    assert ExtractedParamRecord(default=frozenset({"b"})).default == frozenset({"b"})
    assert ExtractedParamRecord(default=list_default).default == [(1, 2)]
    assert ExtractedParamRecord(default=dict_default).default == {"shape": (1, 2)}


def test_ghost_inspector_cdd_class() -> None:
    """Verify GhostInspector extracts class __init__ parameters via CDD."""
    ref = GhostInspector.inspect(SomeClass, "SomeClass")
    assert ref.name == "SomeClass"
    assert ref.kind == "class"

    x_param = next(p for p in ref.params if p.name == "x")
    assert x_param.annotation == "float"
    assert x_param.description == "The x coordinate."

    y_param = next(p for p in ref.params if p.name == "y")
    assert y_param.annotation == "int"
    assert y_param.default == "5"
    assert y_param.description == "The y count."


def test_ghost_inspector_cdd_docstring_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify GhostInspector.inspect falls back cleanly when cdd.docstring.parse raises ImportError.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setitem(sys.modules, "cdd.docstring.parse", None)
    ref = GhostInspector.inspect(some_function, "some_function")
    assert ref.name == "some_function"
    assert len(ref.params) == 4
