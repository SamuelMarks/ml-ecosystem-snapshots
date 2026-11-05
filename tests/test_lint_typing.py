"""Unit test suite for `scripts/lint_typing.py`."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import pytest
import pytest_mock

from scripts.lint_typing import (
    _is_dunder_name,
    check_file,
    lint_directories,
    main,
)


def test_is_dunder_name_helper() -> None:
    """Verify `_is_dunder_name` identifies dunder identifiers and rejects short/non-dunder names."""
    assert _is_dunder_name("__init__") is True
    assert _is_dunder_name("__dict__") is True
    assert _is_dunder_name("____") is False
    assert _is_dunder_name("_private") is False
    assert _is_dunder_name("normal") is False


def test_check_file_detects_all_violation_categories(tmp_path: Path) -> None:
    """Verify `check_file` flags all banned builtins, reflection helpers, Any, object, cast, and dunders.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    bad_file = tmp_path / "bad_code.py"
    lines = [
        "from typing import Any, cast",
        "from operator import attrgetter",
        "import typing, builtins, operator",
        "x: Any = 1",
        "y: typing.Any = 2",
        "o: object = 3",
        "o2: builtins.object = 4",
        "z = getattr(x, 'a')",
        "builtins.setattr(x, 'a', 1)",
        "operator.attrgetter('a')(x)",
        "d_names = dir(x)",
        "r_attr = read_runtime_attr(x, 'a')",
        "d = x.__dict__",
        "m = x.__mro__",
        "n = x.__name__",
        "c = x.__call__()",
        "v = x.normal_method()",
        "s = super().__init__()",
    ]
    bad_file.write_text(chr(10).join(lines), encoding="utf-8")
    violations = check_file(str(bad_file))
    joined = chr(10).join(violations)
    assert "banned import of 'Any'" in joined
    assert "banned import of 'cast'" in joined
    assert "banned import of 'attrgetter'" in joined
    assert "banned 'Any' usage" in joined
    assert "banned '.Any' type usage" in joined
    assert "banned 'object' usage" in joined
    assert "banned '.object' type usage" in joined
    assert "banned call 'getattr(...)'" in joined
    assert "banned attribute call '.setattr(...)'" in joined
    assert "banned attribute call '.attrgetter(...)'" in joined
    assert "banned call 'dir(...)'" in joined
    assert "banned call 'read_runtime_attr(...)'" in joined
    assert "banned dunder attribute access '.__dict__'" in joined
    assert "banned dunder attribute access '.__mro__'" in joined
    assert "banned dunder attribute access '.__name__'" in joined
    assert "banned direct dunder method call '.__call__(...)'" in joined
    assert "banned direct dunder method call '.__init__(...)'" in joined
    assert "normal_method" not in joined


def test_lint_directories_and_main(
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `lint_directories` and `main` return 0 on clean code and 1 on violations.

    Args:
        tmp_path: Pytest temporary directory fixture.
        mocker: Pytest mocker fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    f_ok = pkg / "clean.py"
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

    spec = importlib.util.find_spec("scripts.lint_typing")
    assert spec is not None and spec.origin is not None
    mocker.patch("glob.glob", return_value=[str(f_ok)])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(spec.origin, run_name="__main__")
    assert exc_info.value.code == 0
