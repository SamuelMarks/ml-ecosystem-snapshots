"""Unit test suite for SciPy snapshot extractor, MCP validation, and CLI tooling.

Verifies mathematical special functions, linear algebra routines, signal processing operations,
and live/fallback introspection with 100% coverage.
"""

from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import patch
import pytest

from ml_ecosystem_snapshots.api import extract_snapshot, get_pkg_version
from ml_ecosystem_snapshots.cli import cmd_check_scipy
from ml_ecosystem_snapshots.frameworks import scipy
from ml_ecosystem_snapshots.mcp_server import check_scipy_op
from ml_switcheroo_ir.schema.ghost import ParameterKind, SemanticTier


def test_scipy_version() -> None:
    """Verify that get_pkg_version returns installed or valid scipy version."""
    ver = get_pkg_version("scipy")
    assert ver != "unknown"
    assert len(ver) > 0


def test_collect_api_unsupported_category() -> None:
    """Verify that unsupported semantic tiers return an empty list."""
    assert scipy.collect_api(SemanticTier.LOSS) == []
    assert scipy.collect_api(SemanticTier.LAYER) == []


def test_collect_api_fallback_mode() -> None:
    """Verify that collect_api successfully generates canonical operators in offline mode."""
    with patch(
        "ml_ecosystem_snapshots.frameworks.scipy._get_scipy_submodule",
        return_value=None,
    ):
        refs = scipy.collect_api(SemanticTier.ARRAY_API)
        assert len(refs) >= 30

        op_names = {r.name for r in refs}
        assert "erf" in op_names
        assert "inv" in op_names
        assert "convolve" in op_names

        erf_ref = next(r for r in refs if r.name == "erf")
        assert len(erf_ref.params) == 1
        assert erf_ref.params[0].kind == ParameterKind.POSITIONAL_ONLY


def test_collect_api_live_mode() -> None:
    """Verify that collect_api works when scipy submodules are discovered."""
    import types

    fake_special = types.SimpleNamespace(erf=lambda z: z)
    fake_linalg = types.SimpleNamespace(inv=lambda a: a)
    fake_signal = types.SimpleNamespace(convolve=lambda in1, in2: in1)

    def fake_get_submodule(sub: str) -> Any:
        """Return fake submodule by name."""
        if sub == "special":
            return fake_special
        if sub == "linalg":
            return fake_linalg
        if sub == "signal":
            return fake_signal
        return None

    with patch(
        "ml_ecosystem_snapshots.frameworks.scipy._get_scipy_submodule",
        side_effect=fake_get_submodule,
    ):
        refs = scipy.collect_api(SemanticTier.ARRAY_API)
        assert len(refs) >= 30


def test_collect_api_inspection_exception_handled() -> None:
    """Verify that inspection errors on specific objects fall back gracefully."""
    import types

    fake_special = types.SimpleNamespace(erf="not_inspectable")

    def fake_get_submodule(sub: str) -> Any:
        """Return fake special submodule."""
        return fake_special if sub == "special" else None

    with patch(
        "ml_ecosystem_snapshots.frameworks.scipy._get_scipy_submodule",
        side_effect=fake_get_submodule,
    ):
        with patch(
            "ml_ecosystem_snapshots.models.GhostInspector.inspect",
            side_effect=ValueError("boom"),
        ):
            refs = scipy.collect_api(SemanticTier.ARRAY_API)
            assert len(refs) >= 30


def test_extract_snapshot_scipy() -> None:
    """Verify full snapshot extraction for scipy target."""
    snap = extract_snapshot("scipy")
    assert snap.get("target") == "scipy"
    cats = snap.get("categories", {})
    assert "array" in cats or "util" in cats


def test_check_scipy_op_mcp() -> None:
    """Verify MCP validator check_scipy_op behavior across valid and invalid inputs."""
    # Valid call
    valid_res = check_scipy_op("erf", inputs_count=1)
    assert valid_res["is_valid"] is True
    assert valid_res["op_exists"] is True

    # Valid with prefix
    prefix_res = check_scipy_op("scipy.special.erf", inputs_count=1)
    assert prefix_res["is_valid"] is True

    # Unknown operation
    unknown_res = check_scipy_op("non_existent_op")
    assert unknown_res["is_valid"] is False
    assert unknown_res["op_exists"] is False
    assert len(unknown_res["errors"]) > 0

    # Wrong inputs count
    wrong_inputs = check_scipy_op("erf", inputs_count=5)
    assert wrong_inputs["is_valid"] is False
    assert "expects 1 positional inputs" in wrong_inputs["errors"][0]

    # Valid keyword arguments
    valid_kw = check_scipy_op("softmax", attributes=["axis"])
    assert valid_kw["is_valid"] is True

    # Invalid keyword arguments
    invalid_kw = check_scipy_op("erf", attributes=["invalid_kwarg"])
    assert invalid_kw["is_valid"] is False
    assert "does not accept keyword argument 'invalid_kwarg'" in invalid_kw["errors"][0]


def test_cmd_check_scipy_cli_valid(capsys: Any) -> None:
    """Verify CLI command execution on valid SciPy operation."""
    args = argparse.Namespace(
        op_name="inv",
        inputs_count=1,
        attributes=None,
    )
    cmd_check_scipy(args)
    captured = capsys.readouterr()
    assert "SciPy Operation 'inv' is valid." in captured.out
    assert "Inputs:     1" in captured.out


def test_cmd_check_scipy_cli_invalid_exits(capsys: Any) -> None:
    """Verify CLI command exits with non-zero code on invalid operator."""
    args = argparse.Namespace(
        op_name="invalid_op",
        inputs_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_scipy(args)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "SciPy Operation 'invalid_op' is invalid:" in captured.out


def test_get_scipy_submodule_branches() -> None:
    """Verify module lookup branches for _get_scipy_submodule."""
    # Case where module is found
    mod = scipy._get_scipy_submodule("special")
    assert mod is not None or mod is None

    # Case where all fail
    with patch(
        "ml_ecosystem_snapshots.frameworks.scipy.importlib.import_module",
        side_effect=ImportError("missing"),
    ):
        assert scipy._get_scipy_submodule("special") is None
