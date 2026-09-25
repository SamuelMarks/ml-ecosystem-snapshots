"""Unit test suite for PyTorch ATen dispatch extractor, MCP validation, and CLI tooling.

Verifies C++ dispatch signatures, argument mappings, overloads,
and live/fallback introspection with 100% coverage.
"""

from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from ml_ecosystem_snapshots.api import extract_snapshot, get_pkg_version
from ml_ecosystem_snapshots.cli import cmd_check_aten
from ml_ecosystem_snapshots.frameworks import aten
from ml_ecosystem_snapshots.mcp_server import check_aten_op
from ml_switcheroo_ir.schema.ghost import ParameterKind, SemanticTier
from torch_mock import ensure_torch


def test_aten_version() -> None:
    """Verify that get_pkg_version returns installed or valid torch version for aten."""
    ensure_torch()
    ver = get_pkg_version("aten")
    assert ver != "unknown"
    assert len(ver) > 0


def test_collect_api_unsupported_category() -> None:
    """Verify that unsupported semantic tiers return an empty list."""
    assert aten.collect_api(SemanticTier.LOSS) == []
    assert aten.collect_api(SemanticTier.OPTIMIZER) == []


def test_collect_api_fallback_mode() -> None:
    """Verify that collect_api successfully generates canonical operators in offline mode."""
    with patch(
        "ml_ecosystem_snapshots.frameworks.aten._get_torch_ops_aten", return_value=None
    ):
        refs = aten.collect_api(SemanticTier.NEURAL_OPS)
        assert len(refs) >= 20

        op_names = {r.name for r in refs}
        assert "add" in op_names
        assert "matmul" in op_names
        assert "relu" in op_names
        assert "softmax" in op_names

        add_ref = next(r for r in refs if r.name == "add")
        assert len(add_ref.params) == 3
        assert add_ref.params[0].kind == ParameterKind.POSITIONAL_ONLY


def test_collect_api_live_mode() -> None:
    """Verify that collect_api works when torch.ops.aten is discovered."""
    import types

    fake_ops = types.SimpleNamespace(add=MagicMock())

    fake_schema = [
        {
            "name": "add",
            "overload_name": "Tensor",
            "params": [
                {"name": "self", "kind": "POSITIONAL_ONLY"},
                {"name": "other", "kind": "POSITIONAL_ONLY"},
            ],
        }
    ]

    with patch(
        "ml_ecosystem_snapshots.frameworks.aten._get_torch_ops_aten",
        return_value=fake_ops,
    ):
        with patch(
            "ml_ecosystem_snapshots.frameworks.torch.get_aten_op_schema",
            return_value=fake_schema,
        ):
            refs = aten.collect_api(SemanticTier.NEURAL_OPS)
            assert len(refs) >= 20


def test_collect_api_schema_missing_fallback() -> None:
    """Verify that when schema is missing or empty, canonical definition is used."""
    fake_ops = MagicMock()
    fake_ops.add = MagicMock()

    with patch(
        "ml_ecosystem_snapshots.frameworks.aten._get_torch_ops_aten",
        return_value=fake_ops,
    ):
        with patch(
            "ml_ecosystem_snapshots.frameworks.torch.get_aten_op_schema",
            return_value=None,
        ):
            refs = aten.collect_api(SemanticTier.NEURAL_OPS)
            assert len(refs) >= 20


def test_extract_snapshot_aten() -> None:
    """Verify full snapshot extraction for aten target."""
    ensure_torch()
    snap = extract_snapshot("aten")
    assert snap.get("target") == "aten"
    cats = snap.get("categories", {})
    assert "neural_ops" in cats or "util" in cats


def test_check_aten_op_mcp() -> None:
    """Verify MCP validator check_aten_op behavior across valid and invalid inputs."""
    # Valid call
    valid_res = check_aten_op("add", inputs_count=2)
    assert valid_res["is_valid"] is True
    assert valid_res["op_exists"] is True

    # Valid with prefix
    prefix_res = check_aten_op("aten.matmul", inputs_count=2)
    assert prefix_res["is_valid"] is True

    # Unknown operation
    unknown_res = check_aten_op("non_existent_op")
    assert unknown_res["is_valid"] is False
    assert unknown_res["op_exists"] is False
    assert len(unknown_res["errors"]) > 0

    # Wrong inputs count
    wrong_inputs = check_aten_op("add", inputs_count=5)
    assert wrong_inputs["is_valid"] is False
    assert "expects 2 positional inputs" in wrong_inputs["errors"][0]

    # Valid keyword arguments
    valid_kw = check_aten_op("add", attributes=["alpha"])
    assert valid_kw["is_valid"] is True

    # Invalid keyword arguments
    invalid_kw = check_aten_op("add", attributes=["invalid_kwarg"])
    assert invalid_kw["is_valid"] is False
    assert "does not accept keyword argument 'invalid_kwarg'" in invalid_kw["errors"][0]


def test_cmd_check_aten_cli_valid(capsys: Any) -> None:
    """Verify CLI command execution on valid ATen operation."""
    args = argparse.Namespace(
        op_name="matmul",
        inputs_count=2,
        attributes=None,
    )
    cmd_check_aten(args)
    captured = capsys.readouterr()
    assert "ATen Operation 'matmul' is valid." in captured.out
    assert "Inputs:     2" in captured.out


def test_cmd_check_aten_cli_invalid_exits(capsys: Any) -> None:
    """Verify CLI command exits with non-zero code on invalid operator."""
    args = argparse.Namespace(
        op_name="invalid_op",
        inputs_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_aten(args)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "ATen Operation 'invalid_op' is invalid:" in captured.out


def test_get_torch_ops_aten_branches() -> None:
    """Verify module lookup branches for _get_torch_ops_aten."""
    ensure_torch()
    # Case where torch is found
    ops = aten._get_torch_ops_aten()
    assert ops is not None

    # Case where torch lacks ops.aten
    fake_torch = MagicMock()
    del fake_torch.ops
    with patch(
        "ml_ecosystem_snapshots.frameworks.aten._resolve_torch", return_value=fake_torch
    ):
        assert aten._get_torch_ops_aten() is None

    # Case where _resolve_torch returns None
    with patch(
        "ml_ecosystem_snapshots.frameworks.aten._resolve_torch", return_value=None
    ):
        assert aten._get_torch_ops_aten() is None

    # Case where _resolve_torch raises exception
    with patch(
        "ml_ecosystem_snapshots.frameworks.aten._resolve_torch",
        side_effect=ValueError("boom"),
    ):
        assert aten._get_torch_ops_aten() is None

    # Test _resolve_torch direct invocation
    res = aten._resolve_torch()
    assert res is not None

    # Test _resolve_torch exception branch
    with patch.dict("sys.modules", {"torch": None}):
        with patch("builtins.__import__", side_effect=ImportError("missing")):
            assert aten._resolve_torch() is None
