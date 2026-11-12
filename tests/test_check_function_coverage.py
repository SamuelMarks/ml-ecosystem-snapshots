"""Unit test suite for `scripts/check_function_coverage.py`."""

from __future__ import annotations

import importlib.util
import os
import runpy
from pathlib import Path

import coverage
import pytest
import pytest_mock

from scripts.check_function_coverage import (
    check_function_coverage,
    main,
)


def test_check_function_coverage_missing_file(tmp_path: Path) -> None:
    """Verify `check_function_coverage` raises `FileNotFoundError` and `main` returns 1 when `.coverage` is missing.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    missing = str(tmp_path / "nonexistent.coverage")
    with pytest.raises(FileNotFoundError):
        check_function_coverage(data_file=missing, roots=())
    assert main(data_file=missing) == 1


def test_check_function_coverage_detects_covered_and_uncovered_functions(
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `check_function_coverage` ignores Protocols/overloads/TYPE_CHECKING and flags unexecuted functions.

    Args:
        tmp_path: Pytest temporary directory fixture.
        mocker: Pytest mocker fixture.
        capsys: Pytest stdout/stderr capture fixture.
    """
    src_dir = tmp_path / "pkg"
    src_dir.mkdir()
    mod_file = src_dir / "sample.py"
    lines = [
        '"""Sample module."""',
        "import typing",
        "",
        "if typing.TYPE_CHECKING:",
        "    def tc_only() -> int:",
        '        """Type-checking stub."""',
        "        return 1",
        "else:",
        "    VAL = 1",
        "",
        "class MyProto(typing.Protocol):",
        '    """Sample protocol."""',
        "    def proto_doc_only(self) -> int:",
        '        """Docstring-only protocol method."""',
        "    def proto_pass(self) -> int:",
        '        """Pass protocol method."""',
        "        pass",
        "    def proto_concrete(self) -> int:",
        '        """Concrete protocol helper."""',
        "        return 7",
        "",
        "@typing.overload",
        "def ov_fn(x: int) -> int: ...",
        "",
        "@typing.final",
        "def ov_fn(x: int) -> int:",
        '    """Covered implementation."""',
        "    def nested_only() -> int:",
        "        return 99",
        "    return x + 1",
        "",
        "def only_nested_wrapper() -> None:",
        '    """Function containing only a nested function and docstring."""',
        "    def child() -> int:",
        '        """Covered child."""',
        "        return 2",
    ]
    nl = chr(10)
    mod_file.write_text(nl.join(lines) + nl, encoding="utf-8")

    cov_path = str(tmp_path / ".coverage")
    cov_data = coverage.CoverageData(basename=cov_path)
    real_mod = os.path.realpath(str(mod_file))
    # Cover proto_concrete (line 20), ov_fn return (line 30), and child return (line 36), leaving nested_only (line 29) uncovered
    cov_data.add_lines({real_mod: [20, 30, 36]})
    cov_data.write()

    uncovered = check_function_coverage(data_file=cov_path, roots=(str(src_dir),))
    assert len(uncovered) == 1
    assert "nested_only" in uncovered[0]

    mocker.patch("scripts.check_function_coverage.check_function_coverage", return_value=uncovered)
    assert main(data_file=cov_path) == 1
    assert "Found 1 uncovered function(s)." in capsys.readouterr().err

    # Now cover line 29 as well and verify 100% pass
    cov_data_all = coverage.CoverageData(basename=cov_path)
    cov_data_all.add_lines({real_mod: [20, 29, 30, 36]})
    cov_data_all.write()
    assert check_function_coverage(data_file=cov_path, roots=(str(src_dir),)) == []

    mocker.patch("scripts.check_function_coverage.check_function_coverage", return_value=[])
    assert main(data_file=cov_path) == 0
    assert "100.0% function coverage" in capsys.readouterr().out

    spec = importlib.util.find_spec("scripts.check_function_coverage")
    assert spec is not None and spec.origin is not None
    mocker.patch("coverage.CoverageData.read", return_value=None)
    mocker.patch("coverage.CoverageData.measured_files", return_value=[])
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("glob.glob", return_value=[])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(spec.origin, run_name="__main__")
    assert exc_info.value.code == 0
