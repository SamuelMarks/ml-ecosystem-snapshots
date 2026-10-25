"""Unit test suite for Orbax (`orbax.checkpoint` and `orbax.export`) snapshot extractor, MCP validation, and CLI tooling."""

from __future__ import annotations

import argparse
import types
from unittest.mock import patch

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier
import pytest

from ml_ecosystem_snapshots.api import extract_snapshot, get_pkg_version
from ml_ecosystem_snapshots.cli.commands.check import cmd_check_orbax
from ml_ecosystem_snapshots.frameworks import orbax_checkpoint
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.mcp.registry import dispatch_tool_call
from ml_ecosystem_snapshots.mcp.tools.domain_ops import check_orbax_op
from ml_ecosystem_snapshots.mcp.tools.hallucination import (
    check_code_block,
    explain_anti_pattern,
)
from ml_ecosystem_snapshots.utils import get_framework_docstring_parser


def test_orbax_docstring_parser_and_version_fallback() -> None:
    """Verify docstring parser selection and fallback version resolution for orbax."""
    assert get_framework_docstring_parser("orbax") == "google"
    assert get_framework_docstring_parser("orbax_checkpoint") == "google"

    with (
        patch(
            "importlib.metadata.version",
            side_effect=RuntimeError("not installed"),
        ),
        patch(
            "ml_ecosystem_snapshots.api._extract_version_with_griffe",
            return_value=None,
        ),
    ):
        assert get_pkg_version("orbax") == "0.6.4"
        assert get_pkg_version("orbax_checkpoint") == "0.6.4"


def test_get_orbax_export_and_griffe_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_get_orbax_export_module` and `_get_orbax_griffe_modules` across offline and live states.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    assert orbax_checkpoint._get_orbax_export_module() is None
    assert orbax_checkpoint._get_orbax_griffe_modules() == {
        "checkpoint": None,
        "export": None,
    }

    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "0")
    dummy_exp = types.ModuleType("orbax.export")
    with patch(
        "ml_ecosystem_snapshots.frameworks.orbax_checkpoint.importlib.import_module",
        return_value=dummy_exp,
    ):
        assert orbax_checkpoint._get_orbax_export_module() is dummy_exp

    with patch(
        "ml_ecosystem_snapshots.frameworks.orbax_checkpoint.importlib.import_module",
        side_effect=ImportError("no orbax.export"),
    ):
        assert orbax_checkpoint._get_orbax_export_module() is None

    with patch(
        "ml_ecosystem_snapshots.frameworks.orbax_checkpoint.load_griffe_module_safely",
        return_value=None,
    ):
        assert orbax_checkpoint._get_orbax_griffe_modules() == {
            "checkpoint": None,
            "export": None,
        }


def test_scan_and_collect_orbax_api_all_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_scan_orbax_checkpoint` and `collect_orbax_api` across offline, live, Griffe, and error branches.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    offline_refs = orbax_checkpoint.collect_api(SemanticTier.ARRAY_API)
    assert len(offline_refs) == len(orbax_checkpoint.CANONICAL_ORBAX_OPS)
    offline_full = orbax_checkpoint.collect_orbax_api()
    assert len(offline_full) == len(orbax_checkpoint.CANONICAL_ORBAX_OPS)

    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "0")
    fake_exp = types.ModuleType("orbax.export")

    def _private_export_helper(x: int) -> int:
        """Private export helper.

        Args:
            x: Input integer.

        Returns:
            Input integer.
        """
        return x

    def custom_export_fn(x: int) -> int:
        """Public export helper.

        Args:
            x: Input integer.

        Returns:
            Input integer.
        """
        return x

    def failing_export_fn(x: int) -> int:
        """Export helper that fails inspection.

        Args:
            x: Input integer.

        Returns:
            Input integer.
        """
        return x

    fake_exp.__dict__["_private_export_helper"] = _private_export_helper
    fake_exp.__dict__["custom_export_fn"] = custom_export_fn
    fake_exp.__dict__["failing_export_fn"] = failing_export_fn
    fake_exp.__dict__["ExportManager"] = custom_export_fn
    fake_exp.__dict__["CONST_VAL"] = 42

    ocp_ref = GhostRef(
        name="CheckpointManager",
        api_path="orbax.checkpoint.CheckpointManager",
        kind="class",
        params=[],
    )
    griffe_ref = GhostRef(
        name="GriffeHandler",
        api_path="orbax.checkpoint.GriffeHandler",
        kind="class",
        params=[],
    )

    real_inspect = GhostInspector.inspect

    def selective_inspect(
        obj: types.FunctionType | type | types.ModuleType,
        api_path: str,
        is_public: bool = True,
    ) -> GhostRef:
        """Raise on `failing_export_fn` and delegate otherwise.

        Args:
            obj: Target object.
            api_path: API path string.
            is_public: Public visibility flag.

        Returns:
            Inspected `GhostRef`.
        """
        if api_path.endswith("failing_export_fn"):
            raise RuntimeError("boom")
        return real_inspect(obj, api_path, is_public=is_public)

    with (
        patch(
            "ml_ecosystem_snapshots.frameworks.orbax_checkpoint._scan_orbax_checkpoint",
            return_value=[ocp_ref, ocp_ref],
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.orbax_checkpoint._get_orbax_export_module",
            return_value=fake_exp,
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.orbax_checkpoint.GhostInspector.inspect",
            side_effect=selective_inspect,
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.orbax_checkpoint._get_orbax_griffe_modules",
            return_value={"checkpoint": fake_exp, "export": None},
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.orbax_checkpoint.scan_griffe_module_members",
            return_value=[griffe_ref, ocp_ref],
        ),
    ):
        collected = orbax_checkpoint.collect_orbax_api(include_nonpublic=False)
        paths = {r.api_path for r in collected}
        assert "orbax.export.custom_export_fn" in paths
        assert "orbax.export._private_export_helper" not in paths
        assert "orbax.checkpoint.GriffeHandler" in paths
        assert "orbax.checkpoint.CheckpointManager" in paths

    with (
        patch(
            "ml_ecosystem_snapshots.frameworks.orbax_checkpoint._scan_orbax_checkpoint",
            return_value=[],
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.orbax_checkpoint._get_orbax_export_module",
            return_value=None,
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.orbax_checkpoint._get_orbax_griffe_modules",
            return_value={"checkpoint": None, "export": None},
        ),
    ):
        fallback = orbax_checkpoint.collect_orbax_api(include_nonpublic=True)
        assert len(fallback) == len(orbax_checkpoint.CANONICAL_ORBAX_OPS)


def test_check_orbax_op_mcp_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify `check_orbax_op`, MCP dispatch, CLI `cmd_check_orbax`, and anti-patterns.

    Args:
        capsys: Pytest stdout/stderr capture fixture.
    """
    res_valid = check_orbax_op(
        "orbax.checkpoint.CheckpointManager",
        inputs_count=8,
        attributes=["options", "metadata"],
    )
    assert res_valid["is_valid"] is True

    res_export = check_orbax_op(
        "orbax.export.ExportManager",
        inputs_count=2,
    )
    assert res_export["is_valid"] is True

    res_bad = check_orbax_op("orbax.checkpoint.NonExistentCheckpointer")
    assert res_bad["is_valid"] is False

    rpc_res = dispatch_tool_call(
        1,
        "check_orbax_op",
        {"op_name": "CheckpointManagerOptions", "attributes": ["max_to_keep"]},
    )
    assert "result" in rpc_res

    args_ok = argparse.Namespace(
        op_name="CheckpointManagerOptions",
        inputs_count=None,
        attributes="max_to_keep,save_interval_steps",
    )
    cmd_check_orbax(args_ok)
    assert (
        "Orbax Operation 'CheckpointManagerOptions' is valid."
        in capsys.readouterr().out
    )

    args_fail = argparse.Namespace(
        op_name="InvalidOrbaxOp", inputs_count=None, attributes=None
    )
    with pytest.raises(SystemExit):
        cmd_check_orbax(args_fail)

    anti = explain_anti_pattern(
        "orbax", "orbax.checkpoint.CheckpointManagerOptions", "max_checkpoints"
    )
    assert anti["is_known_anti_pattern"] is True
    assert anti["canonical_argument"] == "max_to_keep"

    block_res_orbax = check_code_block(
        """import orbax.checkpoint
mgr = orbax.checkpoint.CheckpointManagerOptions(max_checkpoints=3)"""
    )
    assert block_res_orbax["total_analyzed"] == 1

    block_res_ocp = check_code_block(
        """mgr = ocp.CheckpointManagerOptions(max_checkpoints=3)"""
    )
    assert block_res_ocp["total_analyzed"] == 1


def test_extract_snapshot_orbax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `extract_snapshot('orbax')` and `extract_snapshot('orbax_checkpoint')` in offline mode.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    snap = extract_snapshot("orbax")
    assert snap.get("target") == "orbax"
    snap_cp = extract_snapshot("orbax_checkpoint")
    assert snap_cp.get("target") == "orbax_checkpoint"
