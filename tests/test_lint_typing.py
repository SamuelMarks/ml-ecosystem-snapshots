"""Unit test suite for `scripts/lint_typing.py`."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

import pytest
import pytest_mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scripts.lint_typing as lint_mod
from scripts.lint_typing import check_file, lint_directories, main


def test_check_file_clean_and_super_and_init(tmp_path: Path) -> None:
    """Verify `check_file` allows clean code, `super().__init__()`, `super().__repr__()`, and `.__init__()`.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    sample = tmp_path / "clean.py"
    code = """class Base:
    def __init__(self) -> None:
        pass
class Sub(Base):
    def __init__(self) -> None:
        super().__init__()
        super().__repr__()
        Base.__init__(self)
        x = int('1')
        y = [1, 2].append(x)
"""
    sample.write_text(code, encoding="utf-8")
    assert check_file(str(sample)) == []


def test_check_file_all_violation_branches(tmp_path: Path) -> None:
    """Verify `check_file` detects every banned reflection, typing, and dunder construct.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    bad = tmp_path / "bad.py"
    lines = [
        "from typing import " + "Any, " + "cast",
        "from typing import Sequence",
        "import typing",
        "x: " + "Any = 1",
        "y: typing." + "Any = 2",
        "z: " + "object = 3",
        "a = " + "getattr(x, 'a')",
        "b = typing." + "cast(int, x)",
        "c = x." + "__dict__",
        "d = x." + "__mro__",
        "e = x." + "__getitem__(0)",
        "f = x.normal_attr",
    ]
    bad.write_text(os.linesep.join(lines), encoding="utf-8")
    violations = check_file(str(bad))
    assert any("banned import 'Any'" in v for v in violations)
    assert any("banned import 'cast'" in v for v in violations)
    assert any("banned 'Any' usage" in v for v in violations)
    assert any("banned 'object' usage" in v for v in violations)
    assert any("banned call 'getattr()'" in v for v in violations)
    assert any("banned call 'typing.cast()'" in v for v in violations)
    assert any("banned attribute '.__dict__'" in v for v in violations)
    assert any("banned attribute '.__mro__'" in v for v in violations)
    assert any("banned direct dunder call '.__getitem__()'" in v for v in violations)


def test_lint_directories_and_main(
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `lint_directories`, `main`, and `__main__` entrypoint for both passing and failing states.

    Args:
        tmp_path: Pytest temporary directory fixture.
        mocker: Pytest mocker fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    pkg = tmp_path / "sub"
    pkg.mkdir()
    f_ok = pkg / "ok.py"
    f_ok.write_text("val: int = 42", encoding="utf-8")

    assert lint_directories((str(pkg),)) == []

    mocker.patch("scripts.lint_typing.lint_directories", return_value=[])
    assert main() == 0
    out_ok = capsys.readouterr()
    assert "0 violations" in out_ok.out

    mocker.patch(
        "scripts.lint_typing.lint_directories",
        return_value=["bad.py:1:0: banned 'Any' usage"],
    )
    assert main() == 1
    out_err = capsys.readouterr()
    assert "Found 1 typing/reflection violation(s)." in out_err.err

    # Test __main__ execution
    mocker.patch("glob.glob", return_value=[str(f_ok)])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(lint_mod.__file__, run_name="__main__")
    assert exc_info.value.code == 0
