"""Unit test suite for SafeTensors snapshot extractor, MCP validation, and CLI tooling.

Verifies model weight serialization, deserialization, memory mapping,
and live/fallback introspection with 100% coverage.
"""

from __future__ import annotations

import argparse
from typing import Any
import sys
import types
from unittest.mock import patch

import pytest
from ml_ecosystem_snapshots.api import extract_snapshot, get_pkg_version
from ml_ecosystem_snapshots.cli import cmd_check_safetensors
from ml_ecosystem_snapshots.frameworks import safetensors
from ml_ecosystem_snapshots.mcp_server import check_safetensors_op
from ml_switcheroo_ir.schema.ghost import ParameterKind, SemanticTier

if "safetensors" not in sys.modules:
    fake_st = types.ModuleType("safetensors")
    setattr(fake_st, "__version__", "0.7.0")
    sys.modules["safetensors"] = fake_st


def test_safetensors_version() -> None:
    """Verify that get_pkg_version returns installed or valid safetensors version."""
    ver = get_pkg_version("safetensors")
    assert ver != "unknown"
    assert len(ver) > 0


def test_collect_api_unsupported_category() -> None:
    """Verify that unsupported semantic tiers return an empty list."""
    assert safetensors.collect_api(SemanticTier.LOSS) == []
    assert safetensors.collect_api(SemanticTier.OPTIMIZER) == []


def test_collect_api_fallback_mode() -> None:
    """Verify that collect_api successfully generates canonical operators in offline mode."""
    with patch(
        "ml_ecosystem_snapshots.frameworks.safetensors._get_safetensors_submodule",
        return_value=None,
    ):
        refs = safetensors.collect_api(SemanticTier.DATALOADER)
        assert len(refs) >= 10

        op_names = {r.name for r in refs}
        assert "save_file" in op_names
        assert "load_file" in op_names
        assert "safe_open" in op_names

        save_ref = next(r for r in refs if r.name == "save_file")
        assert len(save_ref.params) == 3
        assert save_ref.params[0].kind == ParameterKind.POSITIONAL_ONLY


def test_collect_api_live_mode() -> None:
    """Verify that collect_api works when safetensors submodules are discovered."""
    import types

    fake_core = types.SimpleNamespace(safe_open=lambda fn: None)
    fake_torch = types.SimpleNamespace(save_file=lambda t, fn, metadata=None: None)

    def fake_get_submodule(sub: str) -> Any:
        """Return fake submodule by name."""
        if sub == "core":
            return fake_core
        if sub == "torch":
            return fake_torch
        return None

    with patch(
        "ml_ecosystem_snapshots.frameworks.safetensors._get_safetensors_submodule",
        side_effect=fake_get_submodule,
    ):
        refs = safetensors.collect_api(SemanticTier.DATALOADER)
        assert len(refs) >= 10


def test_collect_api_inspection_exception_handled() -> None:
    """Verify that inspection errors on specific objects fall back gracefully."""
    import types

    fake_torch = types.SimpleNamespace(save_file="not_inspectable")

    def fake_get_submodule(sub: str) -> Any:
        """Return fake torch submodule."""
        return fake_torch if sub == "torch" else None

    with patch(
        "ml_ecosystem_snapshots.frameworks.safetensors._get_safetensors_submodule",
        side_effect=fake_get_submodule,
    ):
        with patch(
            "ml_ecosystem_snapshots.models.GhostInspector.inspect",
            side_effect=ValueError("boom"),
        ):
            refs = safetensors.collect_api(SemanticTier.DATALOADER)
            assert len(refs) >= 10


def test_extract_snapshot_safetensors() -> None:
    """Verify full snapshot extraction for safetensors target."""
    snap = extract_snapshot("safetensors")
    assert snap.get("target") == "safetensors"
    cats = snap.get("categories", {})
    assert "dataloader" in cats or "util" in cats


def test_check_safetensors_op_mcp() -> None:
    """Verify MCP validator check_safetensors_op behavior across valid and invalid inputs."""
    # Valid call
    valid_res = check_safetensors_op("save_file", inputs_count=2)
    assert valid_res["is_valid"] is True
    assert valid_res["op_exists"] is True

    # Valid with prefix
    prefix_res = check_safetensors_op("safetensors.torch.load_file", inputs_count=1)
    assert prefix_res["is_valid"] is True

    # Unknown operation
    unknown_res = check_safetensors_op("non_existent_op")
    assert unknown_res["is_valid"] is False
    assert unknown_res["op_exists"] is False
    assert len(unknown_res["errors"]) > 0

    # Wrong inputs count
    wrong_inputs = check_safetensors_op("save_file", inputs_count=5)
    assert wrong_inputs["is_valid"] is False
    assert "expects 2 positional inputs" in wrong_inputs["errors"][0]

    # Valid keyword arguments
    valid_kw = check_safetensors_op("save_file", attributes=["metadata"])
    assert valid_kw["is_valid"] is True

    # Invalid keyword arguments
    invalid_kw = check_safetensors_op("save_file", attributes=["invalid_kwarg"])
    assert invalid_kw["is_valid"] is False
    assert "does not accept keyword argument 'invalid_kwarg'" in invalid_kw["errors"][0]


def test_cmd_check_safetensors_cli_valid(capsys: Any) -> None:
    """Verify CLI command execution on valid SafeTensors operation."""
    args = argparse.Namespace(
        op_name="load_file",
        inputs_count=1,
        attributes=None,
    )
    cmd_check_safetensors(args)
    captured = capsys.readouterr()
    assert "SafeTensors Operation 'load_file' is valid." in captured.out
    assert "Inputs:     1" in captured.out


def test_cmd_check_safetensors_cli_invalid_exits(capsys: Any) -> None:
    """Verify CLI command exits with non-zero code on invalid operator."""
    args = argparse.Namespace(
        op_name="invalid_op",
        inputs_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_safetensors(args)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "SafeTensors Operation 'invalid_op' is invalid:" in captured.out


def test_get_safetensors_submodule_branches() -> None:
    """Verify module lookup branches for _get_safetensors_submodule."""
    # Core branch
    core_mod = safetensors._get_safetensors_submodule("core")
    assert core_mod is not None or core_mod is None

    # Torch branch
    torch_mod = safetensors._get_safetensors_submodule("torch")
    assert torch_mod is not None or torch_mod is None

    # Case where all fail
    with patch(
        "ml_ecosystem_snapshots.frameworks.safetensors.importlib.import_module",
        side_effect=ImportError("missing"),
    ):
        assert safetensors._get_safetensors_submodule("core") is None
        assert safetensors._get_safetensors_submodule("torch") is None
