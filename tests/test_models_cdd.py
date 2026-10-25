"""Unit tests for CDD-based docstring and AST parameter parsing in GhostInspector."""

from __future__ import annotations

import builtins
import types
from collections.abc import Mapping, Sequence
from unittest.mock import patch

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.runtime_boundary import InspectableTarget
from ml_ecosystem_snapshots.models import JsonValue


def some_function(
    a: int = 1, b: str = "test", *args: JsonValue, **kwargs: JsonValue
) -> None:
    """Execute a sample function with typed parameters and variadic arguments.

    Args:
        a: The first integer parameter.
        b: The second string parameter.
        args: Variadic positional arguments.
        kwargs: Variadic keyword arguments.
    """
    pass


class SomeClass:
    """Sample class with Sphinx-style docstring for CDD class inspection."""

    def __init__(self, x: float, y: list[int] = []) -> None:
        """:param x: The x coordinate.

        :type x: float
        :param y: The y list.
        :type y: list
        """  # noqa: D417
        pass


def test_ghost_inspector_cdd_function() -> None:
    """Verify GhostInspector extracts parameter types, defaults, and descriptions via CDD."""
    ref = GhostInspector.inspect(some_function, "some_function")
    assert ref.name == "some_function"
    assert ref.kind == "function"
    assert len(ref.params) == 4  # a, b, args, kwargs

    a_param = next(p for p in ref.params if p.name == "a")
    assert a_param.annotation == "int"
    assert a_param.default == "1"
    assert a_param.description == "The first integer parameter."

    b_param = next(p for p in ref.params if p.name == "b")
    assert b_param.annotation == "str"
    assert b_param.default == "'test'"
    assert b_param.description == "The second string parameter."


def test_ghost_inspector_cdd_class() -> None:
    """Verify GhostInspector extracts class __init__ parameters via CDD."""
    ref = GhostInspector.inspect(SomeClass, "SomeClass")
    assert ref.name == "SomeClass"
    assert ref.kind == "class"

    x_param = next(p for p in ref.params if p.name == "x")
    assert x_param.annotation == "float"
    assert x_param.description == "The x coordinate."

    y_param = next(p for p in ref.params if p.name == "y")
    assert y_param.annotation == "list"
    assert y_param.default == "[]"
    assert y_param.description == "The y list."


def test_ghost_inspector_cdd_docstring_import_error() -> None:
    """Verify GhostInspector.inspect falls back cleanly when cdd.docstring.parse raises ImportError."""
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Mapping[str, InspectableTarget] | None = None,
        locals: Mapping[str, InspectableTarget] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> types.ModuleType:
        """Raise ImportError when importing cdd.docstring.parse.

        Args:
            name: Target module name to import.
            globals: Globals mapping forwarded to __import__.
            locals: Locals mapping forwarded to __import__.
            fromlist: Fromlist sequence forwarded to __import__.
            level: Relative import level forwarded to __import__.

        Returns:
            Imported module object when not blocked.

        Raises:
            ImportError: When name starts with cdd.docstring.
        """
        if name.startswith("cdd.docstring"):
            raise ImportError("Simulated missing cdd.docstring.parse")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=fake_import):
        ref = GhostInspector.inspect(some_function, "some_function")
        assert ref.name == "some_function"
        assert len(ref.params) == 4
