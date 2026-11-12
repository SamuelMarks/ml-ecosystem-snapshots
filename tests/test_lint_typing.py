"""Unit test suite for `scripts/lint_typing.py`."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import pytest
import pytest_mock

from scripts.lint_typing import (
    check_file,
    is_dunder_name,
    lint_directories,
    main,
)


def test_is_dunder_name_helper() -> None:
    """Verify `is_dunder_name` identifies dunder identifiers and rejects short/non-dunder names."""
    assert is_dunder_name("__init__") is True
    assert is_dunder_name("__dict__") is True
    assert is_dunder_name("____") is False
    assert is_dunder_name("_private") is False
    assert is_dunder_name("normal") is False


def test_check_file_allows_test_asserts_and_boundary_inspect(tmp_path: Path) -> None:
    """Verify `check_file` permits `assert` in test files and `inspect.getmembers` in `runtime_boundary.py`.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    test_only_file = tmp_path / "test_sample.py"
    test_only_file.write_text(
        chr(10).join(["assert 1 == 1", "try:", "    x = 1", "except Exception:", "    x = 2"]),
        encoding="utf-8",
    )
    assert check_file(str(test_only_file)) == []

    boundary_dir = tmp_path / "src" / "introspection"
    boundary_dir.mkdir(parents=True)
    boundary_file = boundary_dir / "runtime_boundary.py"
    boundary_file.write_text(f"import inspect{chr(10)}m = inspect.getmembers(inspect){chr(10)}", encoding="utf-8")
    assert check_file(str(boundary_file)) == []


def test_check_file_detects_all_violation_categories(tmp_path: Path) -> None:
    """Verify `check_file` flags all banned builtins, reflection helpers, Any, object, cast, and dunders.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    bad_src_dir = tmp_path / "src"
    bad_src_dir.mkdir()
    bad_file = bad_src_dir / "bad_code.py"
    lines = [
        "from typing import Any, cast, Protocol, TypeGuard, TYPE_CHECKING",
        "from operator import attrgetter",
        "import typing, builtins, operator, types, inspect",
        "x: Any = 1  #" + " type: ignore[assignment]",
        "y: typing.Any = 2",
        "o: object = 3",
        "o2: builtins.object = 4",
        "ft: types.FunctionType = None",
        "z = getattr(x, 'a')",
        "builtins.setattr(x, 'a', 1)",
        "operator.attrgetter('a')(x)",
        "m_leak = inspect.getmembers(x)",
        "assert x == 1",
        "d_names = dir(x)",
        "r_attr = read_runtime_attr(x, 'a')",
        "c_cast = _as_tensor(x)",
        "d = x.__dict__",
        "m = x.__mro__",
        "n = x.__name__",
        "c = x.__call__()",
        "v = x.normal_method()",
        "s = super().__init__()",
        "class EmptyProto(Protocol):",
        "    '''Docstring only.'''",
        "    pass",
        "class NonEmptyProto(typing.Protocol):",
        "    def __call__(self, x: int) -> int: ...",
        "class NormalEmptyClass:",
        "    pass",
        "def _as_int(v: int) -> int:",
        "    return v",
        "def bad_guard(v: int) -> TypeGuard[int]:",
        "    return True",
        "def noarg_guard() -> TypeGuard[int]:",
        "    return False",
        "def del_guard(v: int, other: bool) -> TypeGuard[int]:",
        "    other = bool(v)",
        "    del other",
        "    del v",
        "    return other",
        "def bad_none_guard(v: int | None) -> TypeGuard[int]:",
        "    return v is not None",
        "def ok_guard(v: int) -> typing.TypeGuard[int]:",
        "    flag = isinstance(v, int)",
        "    return flag",
        "imp = __import__('os')",
        "try:",
        "    pass",
        "except:",
        "    x = 1",
        "try:",
        "    pass",
        "except BaseException:",
        "    x = 2",
        "try:",
        "    pass",
        "except BaseException:",
        "    raise",
        "try:",
        "    pass",
        "except Exception as exc:",
        "    x = 3",
        "    del exc",
        "noqa_val: int = 1  #" + " noqa: F401",
        "nocov_val: int = 2  #" + " pragma: no cover",
        "if TYPE_CHECKING:",
        "    T = int",
        "else:",
        "    T = str",
        "if typing.TYPE_CHECKING:",
        "    U = int",
        "try:",
        "    pass",
        "except Exception:",
        "    x = 4",
        "try:",
        "    pass",
        "except Exception:",
        "    pass",
        "try:",
        "    pass",
        "except ValueError:",
        "    pass",
    ]
    bad_file.write_text(chr(10).join(lines), encoding="utf-8")
    violations = check_file(str(bad_file))
    joined = chr(10).join(violations)
    assert "banned reflection call '.getmembers(...)' outside runtime_boundary.py" in joined
    assert "banned runtime 'assert' statement in production/script code" in joined
    assert "banned broad 'except Exception:' handler in production/script code" in joined
    assert "banned type-ignore comment" in joined
    assert "banned noqa suppression comment" in joined
    assert "banned pragma: no cover comment" in joined
    assert "banned call '__import__(...)'" in joined
    assert "banned unsound TypeGuard 'noarg_guard'" in joined
    assert "banned unsound TypeGuard 'del_guard'" in joined
    assert "banned bare 'except:' handler" in joined
    assert "banned 'except BaseException:' without re-raise" in joined
    assert "banned linter-evading 'except Exception: del exc' block" in joined
    assert "banned import of 'Any'" in joined
    assert "banned import of 'cast'" in joined
    assert "banned import of 'attrgetter'" in joined
    assert "banned 'Any' usage" in joined
    assert "banned '.Any' type usage" in joined
    assert "banned 'object' usage" in joined
    assert "banned '.object' type usage" in joined
    assert "banned '.FunctionType' type usage" in joined
    assert "banned call 'getattr(...)'" in joined
    assert "banned attribute call '.setattr(...)'" in joined
    assert "banned attribute call '.attrgetter(...)'" in joined
    assert "banned call 'dir(...)'" in joined
    assert "banned call 'read_runtime_attr(...)'" in joined
    assert "banned disguised cast call '_as_tensor(...)'" in joined
    assert "banned dunder attribute access '.__dict__'" in joined
    assert "banned dunder attribute access '.__mro__'" in joined
    assert "banned dunder attribute access '.__name__'" in joined
    assert "banned direct dunder method call '.__call__(...)'" in joined
    assert "banned direct dunder method call '.__init__(...)'" in joined
    assert "banned empty Protocol class 'EmptyProto'" in joined
    assert "NonEmptyProto" not in joined
    assert "NormalEmptyClass" not in joined
    assert "banned wrapper function definition '_as_int'" in joined
    assert "banned tautological TypeGuard 'bad_guard' returning True" in joined
    assert "banned tautological TypeGuard 'bad_none_guard' returning 'is not None'" in joined
    assert "ok_guard" not in joined
    assert "banned 'if TYPE_CHECKING: ... else:' block" in joined
    assert "banned silent 'except Exception: pass' block" in joined
    assert "normal_method" not in joined


def test_check_file_detects_phantom_type_checking_and_dunders(tmp_path: Path) -> None:
    """Verify `check_file` flags phantom `if TYPE_CHECKING:` class bodies and non-protocol dunders in `src/`.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    sample = src_dir / "models_check.py"
    sample.write_text(
        chr(10).join(
            [
                "import typing",
                "class PhantomDict(dict[str, int]):",
                "    if typing.TYPE_CHECKING:",
                "        def __getitem__(self, k: str) -> int: ...",
                "class CosplayModel:",
                "    def __getitem__(self, k: str) -> int:",
                "        return 1",
                "    def helper(self) -> int:",
                "        return 2",
                "class CExtensionSignature:",
                "    def __getitem__(self, idx: int) -> int:",
                "        return idx",
            ]
        ),
        encoding="utf-8",
    )
    joined = chr(10).join(check_file(str(sample)))
    assert "banned phantom 'if TYPE_CHECKING:' block inside class 'PhantomDict'" in joined
    assert "banned non-protocol dunder method '__getitem__' on class 'CosplayModel'" in joined
    assert "CExtensionSignature" not in joined


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
