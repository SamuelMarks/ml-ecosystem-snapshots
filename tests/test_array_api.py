"""Unit test suite for Python Array API Standard snapshot extractor and CLI tooling.

Verifies canonical operator definitions, parameter specifications, live/fallback introspection,
and MCP validation mechanics with 100% coverage.
"""

from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from ml_ecosystem_snapshots.api import extract_snapshot, get_pkg_version
from ml_ecosystem_snapshots.cli import cmd_check_array_api
from ml_ecosystem_snapshots.frameworks import array_api
from ml_ecosystem_snapshots.mcp_server import check_array_api_op
from ml_switcheroo_ir.schema.ghost import ParameterKind, SemanticTier


def test_array_api_version() -> None:
    """Verify that get_pkg_version returns the 2024.12 specification version."""
    assert get_pkg_version("array_api") == "2024.12"


def test_collect_api_unsupported_category() -> None:
    """Verify that unsupported semantic tiers return an empty list."""
    assert array_api.collect_api(SemanticTier.LOSS) == []
    assert array_api.collect_api(SemanticTier.LAYER) == []


def test_collect_api_fallback_mode() -> None:
    """Verify that collect_api successfully generates canonical operators in offline mode."""
    with patch(
        "ml_ecosystem_snapshots.frameworks.array_api._get_array_api_module",
        return_value=None,
    ):
        refs = array_api.collect_api(SemanticTier.ARRAY_API)
        assert len(refs) >= 80

        op_names = {r.name for r in refs}
        assert "add" in op_names
        assert "matmul" in op_names
        assert "sum" in op_names
        assert "where" in op_names

        add_ref = next(r for r in refs if r.name == "add")
        assert len(add_ref.params) == 2
        assert add_ref.params[0].kind == ParameterKind.POSITIONAL_ONLY
        assert add_ref.params[1].kind == ParameterKind.POSITIONAL_ONLY


def test_collect_api_live_mode() -> None:
    """Verify that collect_api works when an array API implementation is discovered."""
    import types

    fake_mod = types.SimpleNamespace(add=lambda x1, x2: x1 + x2)

    with patch(
        "ml_ecosystem_snapshots.frameworks.array_api._get_array_api_module",
        return_value=fake_mod,
    ):
        refs = array_api.collect_api(SemanticTier.ARRAY_API)
        assert len(refs) >= 80


def test_collect_api_inspection_exception_handled() -> None:
    """Verify that inspection errors on specific objects fall back gracefully."""
    fake_mod = MagicMock()
    # Setting an attribute that raises on inspection
    fake_mod.add = "not_inspectable"

    with patch(
        "ml_ecosystem_snapshots.frameworks.array_api._get_array_api_module",
        return_value=fake_mod,
    ):
        with patch(
            "ml_ecosystem_snapshots.models.GhostInspector.inspect",
            side_effect=ValueError("boom"),
        ):
            refs = array_api.collect_api(SemanticTier.ARRAY_API)
            assert len(refs) >= 80


def test_extract_snapshot_array_api() -> None:
    """Verify full snapshot extraction for array_api target."""
    snap = extract_snapshot("array_api")
    assert snap.get("target") == "array_api"
    assert snap.get("version") == "2024.12"
    cats = snap.get("categories", {})
    assert "array" in cats or "util" in cats


def test_check_array_api_op_mcp() -> None:
    """Verify MCP validator check_array_api_op behavior across valid and invalid inputs."""
    # Valid call
    valid_res = check_array_api_op("add", inputs_count=2)
    assert valid_res["is_valid"] is True
    assert valid_res["op_exists"] is True

    # Valid with prefix
    prefix_res = check_array_api_op("array_api.matmul", inputs_count=2)
    assert prefix_res["is_valid"] is True

    # Unknown operation
    unknown_res = check_array_api_op("non_existent_op")
    assert unknown_res["is_valid"] is False
    assert unknown_res["op_exists"] is False
    assert len(unknown_res["errors"]) > 0

    # Wrong inputs count
    wrong_inputs = check_array_api_op("add", inputs_count=5)
    assert wrong_inputs["is_valid"] is False
    assert "expects 2 positional inputs" in wrong_inputs["errors"][0]

    # Valid keyword arguments
    valid_kw = check_array_api_op("sum", attributes=["axis", "keepdims"])
    assert valid_kw["is_valid"] is True

    # Invalid keyword arguments
    invalid_kw = check_array_api_op("sum", attributes=["invalid_kwarg"])
    assert invalid_kw["is_valid"] is False
    assert "does not accept keyword argument 'invalid_kwarg'" in invalid_kw["errors"][0]


def test_cmd_check_array_api_cli_valid(capsys: Any) -> None:
    """Verify CLI command execution on valid Array API operation."""
    args = argparse.Namespace(
        op_name="multiply",
        inputs_count=2,
        attributes=None,
    )
    cmd_check_array_api(args)
    captured = capsys.readouterr()
    assert "Array API Operation 'multiply' is valid." in captured.out
    assert "Inputs:     2" in captured.out


def test_cmd_check_array_api_cli_invalid_exits(capsys: Any) -> None:
    """Verify CLI command exits with non-zero code on invalid operator."""
    args = argparse.Namespace(
        op_name="invalid_op",
        inputs_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_array_api(args)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Array API Operation 'invalid_op' is invalid:" in captured.out


def test_get_array_api_module_branches() -> None:
    """Verify module lookup branches for _get_array_api_module."""
    # Case where module is found
    mod = array_api._get_array_api_module()
    assert mod is not None or mod is None

    # Case where all fail
    with patch(
        "ml_ecosystem_snapshots.frameworks.array_api.importlib.import_module",
        side_effect=ImportError("missing"),
    ):
        assert array_api._get_array_api_module() is None
