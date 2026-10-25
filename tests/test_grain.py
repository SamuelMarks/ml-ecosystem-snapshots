"""Unit test suite for Grain (`grain` / `grain.python`) snapshot extractor, MCP validation, and CLI tooling."""

from __future__ import annotations

import argparse
import types
from unittest.mock import patch

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier
import pytest

from ml_ecosystem_snapshots.api import extract_snapshot, get_pkg_version
from ml_ecosystem_snapshots.cli.commands.check import cmd_check_grain
from ml_ecosystem_snapshots.frameworks import grain
from ml_ecosystem_snapshots.mcp.registry import dispatch_tool_call
from ml_ecosystem_snapshots.mcp.tools.domain_ops import check_grain_op
from ml_ecosystem_snapshots.mcp.tools.hallucination import (
    check_code_block,
    explain_anti_pattern,
)
from ml_ecosystem_snapshots.mcp.tools.translation import translate_concept_arguments
from ml_ecosystem_snapshots.utils import get_framework_docstring_parser


def test_grain_docstring_parser_and_version() -> None:
    """Verify docstring parser selection and fallback version resolution for grain and pygrain."""
    assert get_framework_docstring_parser("grain") == "google"
    assert get_framework_docstring_parser("pygrain") == "google"

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
        assert get_pkg_version("grain") == "0.2.2"
        assert get_pkg_version("pygrain") == "0.2.2"


def test_get_grain_module_and_griffe_offline_and_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_get_grain_module` and `_get_grain_griffe_module` across offline and live modes.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    mods_offline = grain._get_grain_module()
    assert mods_offline == {"core": None, "python": None, "experimental": None}
    assert grain._get_grain_griffe_module() is None

    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "0")
    dummy_mod = types.ModuleType("grain")
    with patch(
        "ml_ecosystem_snapshots.frameworks.grain.importlib.import_module",
        return_value=dummy_mod,
    ):
        mods_live = grain._get_grain_module()
        assert mods_live["core"] is dummy_mod
        assert mods_live["python"] is dummy_mod
        assert mods_live["experimental"] is dummy_mod

    with patch(
        "ml_ecosystem_snapshots.frameworks.grain.importlib.import_module",
        side_effect=ImportError("no grain"),
    ):
        mods_missing = grain._get_grain_module()
        assert mods_missing == {"core": None, "python": None, "experimental": None}

    with patch(
        "ml_ecosystem_snapshots.frameworks.grain.load_griffe_module_safely",
        return_value=None,
    ):
        assert grain._get_grain_griffe_module() is None


def test_attach_grain_python_alias_branches() -> None:
    """Verify `_attach_grain_python_alias` handles non-grain, already-prefixed, and duplicate aliases."""
    ref_main = GhostRef(
        name="DataLoader",
        api_path="grain.DataLoader",
        kind="class",
        params=[],
        aliases=[],
    )
    grain._attach_grain_python_alias(ref_main)
    assert ref_main.aliases is not None
    assert "grain.python.DataLoader" in ref_main.aliases

    # Calling again does not duplicate the alias
    grain._attach_grain_python_alias(ref_main)
    assert ref_main.aliases is not None
    assert ref_main.aliases.count("grain.python.DataLoader") == 1

    ref_already_py = GhostRef(
        name="DataLoader",
        api_path="grain.python.DataLoader",
        kind="class",
        params=[],
        aliases=[],
    )
    grain._attach_grain_python_alias(ref_already_py)
    assert ref_already_py.aliases == []


def test_collect_grain_api_offline_and_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `collect_grain_api` and `collect_api` across offline, live, and Griffe branches.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    assert grain.collect_api(SemanticTier.OPTIMIZER) == []

    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    offline_refs = grain.collect_api(SemanticTier.ARRAY_API)
    assert len(offline_refs) == len(grain.CANONICAL_GRAIN_OPS)
    dl_ref = next(r for r in offline_refs if r.api_path == "grain.DataLoader")
    assert dl_ref.aliases is not None
    assert "grain.python.DataLoader" in dl_ref.aliases

    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "0")
    fake_core = types.ModuleType("grain")
    fake_py = types.ModuleType("grain.python")

    def custom_loader(source: int) -> int:
        """Custom Grain loader helper.

        Args:
            source: Input source ID.

        Returns:
            Input source ID.
        """
        return source

    fake_core.__dict__["custom_loader"] = custom_loader
    fake_py.__dict__["custom_loader"] = custom_loader
    fake_py.__dict__["DataLoader"] = custom_loader

    griffe_ref = GhostRef(
        name="griffe_sampler",
        api_path="grain.griffe_sampler",
        kind="class",
        params=[],
    )
    griffe_dup = GhostRef(
        name="DataLoader",
        api_path="grain.DataLoader",
        kind="class",
        params=[],
    )

    with (
        patch(
            "ml_ecosystem_snapshots.frameworks.grain._get_grain_module",
            return_value={
                "core": fake_core,
                "python": fake_py,
                "experimental": None,
            },
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.grain._get_grain_griffe_module",
            return_value=fake_core,
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.grain.scan_griffe_module_members",
            return_value=[griffe_ref, griffe_dup],
        ),
    ):
        live_refs = grain.collect_grain_api(include_nonpublic=False)
        paths = {r.api_path for r in live_refs}
        assert "grain.custom_loader" in paths
        assert "grain.griffe_sampler" in paths
        assert "grain.DataLoader" in paths

    with (
        patch(
            "ml_ecosystem_snapshots.frameworks.grain._get_grain_module",
            return_value={"core": None, "python": None, "experimental": None},
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.grain._get_grain_griffe_module",
            return_value=None,
        ),
    ):
        fallback_refs = grain.collect_grain_api(include_nonpublic=False)
        assert len(fallback_refs) == len(grain.CANONICAL_GRAIN_OPS)


def test_check_grain_op_mcp_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify `check_grain_op`, MCP dispatch, CLI `cmd_check_grain`, anti-patterns, and concept translation.

    Args:
        capsys: Pytest stdout/stderr capture fixture.
    """
    res_valid = check_grain_op(
        "grain.python.DataLoader",
        attributes=["data_source", "sampler", "worker_count"],
    )
    assert res_valid["is_valid"] is True

    res_bad = check_grain_op("grain.UnknownDatasetOp")
    assert res_bad["is_valid"] is False

    rpc_res = dispatch_tool_call(
        1,
        "check_grain_op",
        {"op_name": "IndexSampler", "inputs_count": 2},
    )
    assert "result" in rpc_res

    args_ok = argparse.Namespace(
        op_name="DataLoader",
        inputs_count=None,
        attributes="data_source,sampler,worker_count",
    )
    cmd_check_grain(args_ok)
    assert "Grain Operation 'DataLoader' is valid." in capsys.readouterr().out

    args_fail = argparse.Namespace(
        op_name="InvalidGrainOp", inputs_count=None, attributes=None
    )
    with pytest.raises(SystemExit):
        cmd_check_grain(args_fail)

    anti = explain_anti_pattern("grain", "grain.DataLoader", "num_workers")
    assert anti["is_known_anti_pattern"] is True
    assert anti["canonical_argument"] == "worker_count"

    trans = translate_concept_arguments(
        "data_loader",
        "torch",
        "grain",
        {"dataset": "ds", "sampler": "samp", "num_workers": 4},
    )
    assert trans["translated_kwargs"] == {
        "data_source": "ds",
        "sampler": "samp",
        "worker_count": 4,
    }

    block_res = check_code_block(
        """import grain
loader = grain.DataLoader(data_source=ds, sampler=s, num_workers=4)"""
    )
    assert block_res["total_analyzed"] == 1


def test_extract_snapshot_grain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `extract_snapshot('grain')` and `extract_snapshot('pygrain')` in offline mode.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    snap = extract_snapshot("grain")
    assert snap.get("target") == "grain"
    snap_alias = extract_snapshot("pygrain")
    assert snap_alias.get("target") == "pygrain"
