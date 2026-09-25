"""Unit test suite for NCCL/RCCL collective communications extractor, MCP validation, and CLI tooling.

Verifies multi-GPU communication primitives, AllReduce, AllGather, ReduceScatter,
and live/fallback introspection with 100% coverage.
"""

from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from ml_ecosystem_snapshots.api import extract_snapshot, get_pkg_version
from ml_ecosystem_snapshots.cli import cmd_check_nccl
from ml_ecosystem_snapshots.frameworks import nccl
from ml_ecosystem_snapshots.mcp_server import check_nccl_op
from ml_switcheroo_ir.schema.ghost import ParameterKind, SemanticTier


def test_nccl_version() -> None:
    """Verify that get_pkg_version returns expected version for nccl and rccl."""
    assert get_pkg_version("nccl") == "2.21"
    assert get_pkg_version("rccl") == "2.21"


def test_collect_api_unsupported_category() -> None:
    """Verify that unsupported semantic tiers return an empty list."""
    assert nccl.collect_api(SemanticTier.LOSS) == []
    assert nccl.collect_api(SemanticTier.OPTIMIZER) == []


def test_collect_api_fallback_mode() -> None:
    """Verify that collect_api successfully generates canonical operators in offline mode."""
    with patch(
        "ml_ecosystem_snapshots.frameworks.nccl._discover_nccl_library",
        return_value=None,
    ):
        refs = nccl.collect_api(SemanticTier.ARRAY_API)
        assert len(refs) >= 8

        op_names = {r.name for r in refs}
        assert "all_reduce" in op_names
        assert "all_gather" in op_names
        assert "reduce_scatter" in op_names
        assert "broadcast" in op_names

        ar_ref = next(r for r in refs if r.name == "all_reduce")
        assert len(ar_ref.params) == 7
        assert ar_ref.params[0].kind == ParameterKind.POSITIONAL_ONLY


def test_collect_api_rccl_discovered() -> None:
    """Verify that collect_api tags rccl when rccl shared library is discovered."""
    fake_lib = MagicMock()
    with patch.object(fake_lib, "__str__", return_value="<CDLL 'librccl.so'>"):
        with patch(
            "ml_ecosystem_snapshots.frameworks.nccl._discover_nccl_library",
            return_value=fake_lib,
        ):
            refs = nccl.collect_api(SemanticTier.ARRAY_API)
            assert len(refs) >= 8
            tags = refs[0].environment_tags
            assert tags is not None and "rccl" in tags


def test_extract_snapshot_nccl() -> None:
    """Verify full snapshot extraction for nccl target."""
    snap = extract_snapshot("nccl")
    assert snap.get("target") == "nccl"
    cats = snap.get("categories", {})
    assert "array" in cats or "util" in cats


def test_check_nccl_op_mcp() -> None:
    """Verify MCP validator check_nccl_op behavior across valid and invalid inputs."""
    # Valid call
    valid_res = check_nccl_op("all_reduce", inputs_count=3)
    assert valid_res["is_valid"] is True
    assert valid_res["op_exists"] is True

    # Valid with prefix nccl.
    prefix_nccl = check_nccl_op("nccl.all_gather", inputs_count=3)
    assert prefix_nccl["is_valid"] is True

    # Valid with prefix rccl.
    prefix_rccl = check_nccl_op("rccl.reduce_scatter", inputs_count=3)
    assert prefix_rccl["is_valid"] is True

    # Unknown operation
    unknown_res = check_nccl_op("non_existent_op")
    assert unknown_res["is_valid"] is False
    assert unknown_res["op_exists"] is False
    assert len(unknown_res["errors"]) > 0

    # Wrong inputs count
    wrong_inputs = check_nccl_op("all_reduce", inputs_count=1)
    assert wrong_inputs["is_valid"] is False
    assert "expects 3 positional inputs" in wrong_inputs["errors"][0]

    # Valid keyword arguments
    valid_kw = check_nccl_op("all_reduce", attributes=["op", "comm"])
    assert valid_kw["is_valid"] is True

    # Invalid keyword arguments
    invalid_kw = check_nccl_op("all_reduce", attributes=["invalid_kwarg"])
    assert invalid_kw["is_valid"] is False
    assert "does not accept keyword argument 'invalid_kwarg'" in invalid_kw["errors"][0]


def test_cmd_check_nccl_cli_valid(capsys: Any) -> None:
    """Verify CLI command execution on valid NCCL collective operation."""
    args = argparse.Namespace(
        op_name="broadcast",
        inputs_count=3,
        attributes=None,
    )
    cmd_check_nccl(args)
    captured = capsys.readouterr()
    assert "NCCL Operation 'broadcast' is valid." in captured.out
    assert "Inputs:     3" in captured.out


def test_cmd_check_nccl_cli_invalid_exits(capsys: Any) -> None:
    """Verify CLI command exits with non-zero code on invalid operator."""
    args = argparse.Namespace(
        op_name="invalid_op",
        inputs_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_nccl(args)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "NCCL Operation 'invalid_op' is invalid:" in captured.out


def test_discover_nccl_library_branches() -> None:
    """Verify library discovery branches for _discover_nccl_library."""
    # Case where library is found
    fake_cdll = MagicMock()
    with patch("ctypes.CDLL", return_value=fake_cdll):
        assert nccl._discover_nccl_library() is fake_cdll

    # Case where loading raises exception
    with patch("ctypes.CDLL", side_effect=OSError("not found")):
        assert nccl._discover_nccl_library() is None
