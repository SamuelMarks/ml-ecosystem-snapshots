"""Tests for GhostInspector overload extraction."""

from __future__ import annotations

import pathlib
import sys

import pytest
from ml_switcheroo_ir.schema.ghost import GhostRef

from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.introspection.static_parsers import clear_griffe_cache


def test_ghost_inspector_overloads(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test extracting function overloads via Griffe.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    p = tmp_path / "test_overload_mock_unique.py"
    p.write_text(
        chr(10).join(
            [
                "from __future__ import annotations",
                "import typing",
                "@typing.overload",
                "def my_overloaded_func(a: int) -> int: ...",
                "@typing.overload",
                "def my_overloaded_func(a: str, b: str) -> str: ...",
                "def my_overloaded_func(a: int | str, b: str | None = None) -> int | str:",
                "    return a",
            ]
        ),
        encoding="utf-8",
    )

    clear_griffe_cache()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "test_overload_mock_unique", raising=False)
    import test_overload_mock_unique

    ref = GhostInspector.inspect(
        test_overload_mock_unique.my_overloaded_func,
        "test_overload_mock_unique.my_overloaded_func",
    )

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


def test_ghost_inspector_class_constructor_overloads(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test extracting overloads on class constructors via Griffe.

    Args:
        tmp_path: Temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    p = tmp_path / "test_class_overload_mock.py"
    p.write_text(
        chr(10).join(
            [
                "from __future__ import annotations",
                "import typing",
                "class OverloadedClass:",
                "    @typing.overload",
                "    def __init__(self, dim: int) -> None: ...",
                "    @typing.overload",
                "    def __init__(self, dim: str, mode: str) -> None: ...",
                "    def __init__(self, dim: int | str, mode: str | None = None) -> None:",
                "        pass",
            ]
        ),
        encoding="utf-8",
    )

    clear_griffe_cache()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "test_class_overload_mock", raising=False)
    import test_class_overload_mock

    ref = GhostInspector.inspect(
        test_class_overload_mock.OverloadedClass,
        "test_class_overload_mock.OverloadedClass",
        kind="class",
    )

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
