"""Unit test suite for `scripts/lint_docstrings.py`."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import pytest
import pytest_mock

from scripts.lint_docstrings import (
    check_file_docstrings,
    lint_docstring_directories,
    main,
)


def test_check_file_docstrings_detects_missing_and_placeholder_docs(tmp_path: Path) -> None:
    """Verify `check_file_docstrings` flags module, class, function, arg, return, raises, and robotic docs.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    nl = chr(10)
    missing_mod = tmp_path / "missing_mod.py"
    missing_mod.write_text(f"x: int = 1{nl}", encoding="utf-8")
    assert any("missing module docstring" in v for v in check_file_docstrings(str(missing_mod)))

    placeholder_Phrase = "Execute " + "helper foo."
    bad_cls_phrase = "Mock or " + "helper class Foo"
    sample = tmp_path / "sample.py"
    lines = [
        f'"""Module doc with {placeholder_Phrase}"""',
        "import typing",
        "",
        "class UndocumentedClass:",
        "    x: int = 1",
        "",
        "class PlaceholderClass:",
        f'    """{bad_cls_phrase}."""',
        "    def __init__(self, a: int) -> None:",
        '        """Initialize instance.',
        "",
        "        Args:",
        "            a: Value.",
        '        """',
        "        self.a = a",
        "",
        "@typing.overload",
        "def overloaded_fn(x: int) -> int: ...",
        "",
        "@overload",
        "def overloaded_fn(x: str) -> str: ...",
        "",
        "def overloaded_fn(x: int | str) -> int | str:",
        '    """Process input.',
        "",
        "    Args:",
        "        x: Input value.",
        "",
        "    Returns:",
        "        Same value.",
        '    """',
        "    return x",
        "",
        "@typing.final",
        "def undoc_fn() -> None:",
        "    pass",
        "",
        "def bad_sig(a: int, *args: int, **kwargs: str) -> int:",
        '    """Missing args, returns, and raises."""',
        "    def inner_raiser() -> None:",
        '        """Inner helper.',
        "",
        "        Raises:",
        "            RuntimeError: Always.",
        '        """',
        "        raise RuntimeError('inner')",
        "    if a < 0:",
        "        raise ValueError('bad')",
        "    return a",
        "",
        "def outer_only_inner_raise() -> None:",
        '    """Outer function that does not raise in own scope."""',
        "    class NestedCls:",
        '        """Nested class."""',
        "    def nested() -> None:",
        '        """Nested raiser.',
        "",
        "        Raises:",
        "            KeyError: Always.",
        '        """',
        "        raise KeyError('k')",
        "",
        "def sphinx_ok(x: int) -> int:",
        '    """Sphinx style doc.',
        "",
        "    :param x: Input.",
        "    :returns: Output.",
        "    :raises ValueError: When negative.",
        '    """',
        "    if x < 0:",
        "        raise ValueError('neg')",
        "    return x",
    ]
    sample.write_text(nl.join(lines) + nl, encoding="utf-8")
    violations = check_file_docstrings(str(sample))
    joined = nl.join(violations)

    assert "banned robotic docstring phrase" in joined
    assert "missing class docstring on 'UndocumentedClass'" in joined
    assert "missing function docstring on 'undoc_fn'" in joined
    assert "undocumented parameter(s) ['a', '*args', '**kwargs'] on 'bad_sig'" in joined
    assert "missing Returns:/Yields: docstring section on 'bad_sig'" in joined
    assert "missing Raises: docstring section on 'bad_sig'" in joined
    assert "overloaded_fn" not in joined
    assert "outer_only_inner_raise" not in joined
    assert "sphinx_ok" not in joined


def test_lint_docstring_directories_and_cli_main(
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `lint_docstring_directories` and `main` exit codes on clean and violating trees.

    Args:
        tmp_path: Pytest temporary directory fixture.
        mocker: Pytest mocker fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    nl = chr(10)
    pkg = tmp_path / "clean_pkg"
    pkg.mkdir()
    clean_file = pkg / "mod.py"
    clean_file.write_text(f'"""Clean module docstring."""{nl}{nl}VAL: int = 1{nl}', encoding="utf-8")

    assert lint_docstring_directories((str(pkg),)) == []

    mocker.patch("scripts.lint_docstrings.lint_docstring_directories", return_value=[])
    assert main() == 0
    assert "0 docstring violations" in capsys.readouterr().out

    mocker.patch(
        "scripts.lint_docstrings.lint_docstring_directories",
        return_value=["mod.py:1: missing module docstring"],
    )
    assert main() == 1
    assert "Found 1 docstring violation(s)." in capsys.readouterr().err

    spec = importlib.util.find_spec("scripts.lint_docstrings")
    assert spec is not None and spec.origin is not None
    mocker.patch("glob.glob", return_value=[str(clean_file)])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(spec.origin, run_name="__main__")
    assert exc_info.value.code == 0
