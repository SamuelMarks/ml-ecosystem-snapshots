"""Tests for GhostInspector overload extraction."""

from __future__ import annotations

import pathlib
import sys
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.static_parsers import _GRIFFE_CACHE
from ml_switcheroo_ir.schema.ghost import GhostRef


def test_ghost_inspector_overloads(tmp_path: pathlib.Path) -> None:
    """Test extracting function overloads via Griffe.

    Args:
        tmp_path: Temporary directory fixture.
    """
    p = tmp_path / "test_overload_mock_unique.py"
    p.write_text(
        """
import typing

@typing.overload
def my_overloaded_func(a: int) -> int: ...

@typing.overload
def my_overloaded_func(a: str, b: str) -> str: ...

def my_overloaded_func(a: object, b: object = None) -> object:
    pass
"""
    )

    _GRIFFE_CACHE.clear()

    sys.path.insert(0, str(tmp_path))
    if "test_overload_mock_unique" in sys.modules:
        del sys.modules["test_overload_mock_unique"]
    import test_overload_mock_unique

    ref = GhostInspector.inspect(
        test_overload_mock_unique.my_overloaded_func,
        "test_overload_mock_unique.my_overloaded_func",
    )

    assert hasattr(ref, "overloads")
    assert ref.overloads is not None
    assert len(ref.overloads) == 2
    ov0 = ref.overloads[0]
    assert isinstance(ov0, GhostRef)
    assert ov0.params[0].name == "a"
    assert ov0.params[0].annotation == "int"
    assert ov0.returns_type == "int"

    ov1 = ref.overloads[1]
    assert isinstance(ov1, GhostRef)
    assert ov1.params[1].name == "b"
    assert ov1.params[1].annotation == "str"
    assert ov1.returns_type == "str"

    p.unlink(missing_ok=True)
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))


def test_ghost_inspector_class_constructor_overloads(tmp_path: pathlib.Path) -> None:
    """Test extracting overloads on class constructors via Griffe.

    Args:
        tmp_path: Temporary directory fixture.
    """
    p = tmp_path / "test_class_overload_mock.py"
    p.write_text(
        """
import typing

class OverloadedClass:
    @typing.overload
    def __init__(self, dim: int) -> None: ...

    @typing.overload
    def __init__(self, dim: str, mode: str) -> None: ...

    def __init__(self, dim: object, mode: object = None) -> None:
        pass
"""
    )

    _GRIFFE_CACHE.clear()

    sys.path.insert(0, str(tmp_path))
    if "test_class_overload_mock" in sys.modules:
        del sys.modules["test_class_overload_mock"]
    import test_class_overload_mock

    ref = GhostInspector.inspect(
        test_class_overload_mock.OverloadedClass,
        "test_class_overload_mock.OverloadedClass",
        kind="class",
    )

    assert hasattr(ref, "overloads")
    assert ref.overloads is not None
    assert len(ref.overloads) == 2
    ov0 = ref.overloads[0]
    assert isinstance(ov0, GhostRef)
    assert ov0.params[0].name == "dim"
    assert ov0.params[0].annotation == "int"

    ov1 = ref.overloads[1]
    assert isinstance(ov1, GhostRef)
    assert ov1.params[0].name == "dim"
    assert ov1.params[1].name == "mode"
    assert ov1.params[1].annotation == "str"

    p.unlink(missing_ok=True)
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))
