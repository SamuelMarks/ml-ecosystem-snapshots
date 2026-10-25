"""Unit test suite for Optax snapshot extractor, Griffe static scanning, MCP validation, and CLI tooling."""

from __future__ import annotations

import argparse
import types
from unittest.mock import patch

from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier
import pytest

from ml_ecosystem_snapshots.api import extract_snapshot, get_pkg_version
from ml_ecosystem_snapshots.cli.commands.check import cmd_check_optax
from ml_ecosystem_snapshots.frameworks import optax as optax_fw
from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
from ml_ecosystem_snapshots.mcp.registry import dispatch_tool_call
from ml_ecosystem_snapshots.mcp.tools.domain_ops import check_optax_op
from ml_ecosystem_snapshots.mcp.tools.hallucination import (
    check_code_block,
    explain_anti_pattern,
)
from ml_ecosystem_snapshots.mcp.tools.translation import translate_concept_arguments
from ml_ecosystem_snapshots.utils import get_framework_docstring_parser


def test_optax_docstring_parser_and_version_fallback() -> None:
    """Verify docstring parser selection and fallback version resolution for optax."""
    assert get_framework_docstring_parser("optax") == "google"

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
        assert get_pkg_version("optax") == "0.2.3"


def test_get_optax_griffe_module_and_hydrate_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_get_optax_griffe_module` and `_hydrate_canonical_optax_tier` alias attachment.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    assert optax_fw._get_optax_griffe_module() is None

    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "0")
    with patch(
        "ml_ecosystem_snapshots.frameworks.optax.load_griffe_module_safely",
        return_value=None,
    ):
        assert optax_fw._get_optax_griffe_module() is None

    losses = optax_fw._hydrate_canonical_optax_tier(
        optax_fw.CANONICAL_OPTAX_LOSSES, attach_root_alias=True
    )
    ce_ref = next(
        r for r in losses if r.api_path == "optax.losses.softmax_cross_entropy"
    )
    assert ce_ref.aliases is not None
    assert "optax.softmax_cross_entropy" in ce_ref.aliases


def test_optax_scanner_offline_all_tiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `OptaxScanner` and `collect_api` across all semantic tiers in offline mode.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    opts = optax_fw.collect_api(SemanticTier.OPTIMIZER)
    losses = optax_fw.collect_api(SemanticTier.LOSS)
    scheds = optax_fw.collect_api(SemanticTier.SCHEDULER)
    arr_ops = optax_fw.collect_api(SemanticTier.ARRAY_API)
    assert len(opts) == len(optax_fw.CANONICAL_OPTAX_OPTIMIZERS)
    assert len(losses) == len(optax_fw.CANONICAL_OPTAX_LOSSES)
    assert len(scheds) == len(optax_fw.CANONICAL_OPTAX_SCHEDULERS)
    assert len(arr_ops) == len(optax_fw.CANONICAL_OPTAX_ARRAY_API)
    assert optax_fw.collect_api(SemanticTier.LAYER) == []


def test_optax_scan_array_api_live_and_griffe_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `OptaxScanner.scan_array_api` across live, submodule, Griffe, and error branches.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "0")
    fake_optax = types.ModuleType("optax")
    fake_proj = types.ModuleType("optax.projections")

    def apply_updates(params: int, updates: int) -> int:
        """Apply updates to params.

        Args:
            params: Model parameters.
            updates: Gradient updates.

        Returns:
            Updated parameters.
        """
        return params + updates

    def global_norm(updates: int) -> int:
        """Compute global norm (fails inspection in test).

        Args:
            updates: Gradient updates.

        Returns:
            Global norm.
        """
        return updates

    def _private_top(x: int) -> int:
        """Private top-level helper.

        Args:
            x: Input integer.

        Returns:
            Input integer.
        """
        return x

    def projection_non_negative(tree: int) -> int:
        """Project onto non-negative orthant.

        Args:
            tree: Input PyTree.

        Returns:
            Projected PyTree.
        """
        return tree

    def projection_box(tree: int, lower: int, upper: int) -> int:
        """Project onto box constraints (fails inspection in test).

        Args:
            tree: Input PyTree.
            lower: Lower bound.
            upper: Upper bound.

        Returns:
            Projected PyTree.
        """
        return tree

    def _private_proj(tree: int) -> int:
        """Private projection helper.

        Args:
            tree: Input PyTree.

        Returns:
            Input PyTree.
        """
        return tree

    fake_proj.__dict__["projection_non_negative"] = projection_non_negative
    fake_proj.__dict__["projection_box"] = projection_box
    fake_proj.__dict__["_private_proj"] = _private_proj
    fake_proj.__dict__["NOT_A_FN"] = 123

    fake_optax.__dict__["apply_updates"] = apply_updates
    fake_optax.__dict__["global_norm"] = global_norm
    fake_optax.__dict__["_private_top"] = _private_top
    fake_optax.__dict__["projections"] = fake_proj
    fake_optax.__dict__["tree_utils"] = "not_a_module"

    real_inspect = GhostInspector.inspect

    def selective_inspect(
        obj: types.FunctionType | type | types.ModuleType,
        api_path: str,
        is_public: bool = True,
    ) -> GhostRef:
        """Raise on `global_norm` and `projection_box` to test error handling.

        Args:
            obj: Target object.
            api_path: API path string.
            is_public: Public visibility flag.

        Returns:
            Inspected `GhostRef`.
        """
        if api_path.endswith(("global_norm", "projection_box")):
            raise ValueError("inspect error")
        return real_inspect(obj, api_path, is_public=is_public)

    griffe_inc = GhostRef(
        name="incremental_update",
        api_path="optax.incremental_update",
        kind="function",
        params=[],
    )
    griffe_dup = GhostRef(
        name="apply_updates",
        api_path="optax.apply_updates",
        kind="function",
        params=[],
    )
    griffe_ignored = GhostRef(
        name="unknown_helper",
        api_path="optax.unknown_helper",
        kind="function",
        params=[],
    )

    with (
        patch.object(optax_fw, "optax", fake_optax),
        patch(
            "ml_ecosystem_snapshots.frameworks.optax.GhostInspector.inspect",
            side_effect=selective_inspect,
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.optax._get_optax_griffe_module",
            return_value=fake_optax,
        ),
        patch(
            "ml_ecosystem_snapshots.frameworks.optax.scan_griffe_module_members",
            return_value=[griffe_inc, griffe_dup, griffe_ignored],
        ),
    ):
        arr_refs = optax_fw.OptaxScanner.scan_array_api(include_nonpublic=False)
        paths = {r.api_path for r in arr_refs}
        assert "optax.apply_updates" in paths
        assert "optax.incremental_update" in paths
        assert "optax.projections.projection_non_negative" in paths
        assert "optax.projections.projection_box" in paths

    with (
        patch.object(optax_fw, "optax", None),
        patch(
            "ml_ecosystem_snapshots.frameworks.optax._get_optax_griffe_module",
            return_value=None,
        ),
    ):
        fallback_arr = optax_fw.OptaxScanner.scan_array_api(include_nonpublic=True)
        assert len(fallback_arr) == len(optax_fw.CANONICAL_OPTAX_ARRAY_API)


def test_check_optax_op_mcp_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify `check_optax_op`, MCP dispatch, CLI `cmd_check_optax`, anti-patterns, and concept translation.

    Args:
        capsys: Pytest stdout/stderr capture fixture.
    """
    res_valid = check_optax_op(
        "optax.adamw",
        inputs_count=8,
        attributes=["b1", "b2", "weight_decay"],
    )
    assert res_valid["is_valid"] is True

    res_loss = check_optax_op(
        "optax.losses.softmax_cross_entropy",
        inputs_count=2,
    )
    assert res_loss["is_valid"] is True

    res_sched = check_optax_op(
        "optax.schedules.warmup_cosine_decay_schedule",
        inputs_count=6,
    )
    assert res_sched["is_valid"] is True

    res_util = check_optax_op(
        "optax.apply_updates",
        inputs_count=2,
    )
    assert res_util["is_valid"] is True

    res_bad = check_optax_op("optax.non_existent_optimizer")
    assert res_bad["is_valid"] is False

    rpc_res = dispatch_tool_call(
        1,
        "check_optax_op",
        {"op_name": "adam", "attributes": ["learning_rate", "b1", "b2"]},
    )
    assert "result" in rpc_res

    args_ok = argparse.Namespace(
        op_name="adam",
        inputs_count=6,
        attributes="b1,b2",
    )
    cmd_check_optax(args_ok)
    assert "Optax Operation 'adam' is valid." in capsys.readouterr().out

    args_fail = argparse.Namespace(
        op_name="InvalidOptaxOp", inputs_count=None, attributes=None
    )
    with pytest.raises(SystemExit):
        cmd_check_optax(args_fail)

    anti_lr = explain_anti_pattern("optax", "optax.adam", "lr")
    assert anti_lr["is_known_anti_pattern"] is True
    assert anti_lr["canonical_argument"] == "learning_rate"

    anti_betas = explain_anti_pattern("optax", "optax.adam", "betas")
    assert anti_betas["is_known_anti_pattern"] is True
    assert anti_betas["canonical_argument"] == "b1, b2"

    trans = translate_concept_arguments(
        "adam_optimizer",
        "torch",
        "optax",
        {"lr": 1e-3, "eps": 1e-8, "weight_decay": 0.01},
    )
    assert trans["translated_kwargs"] == {
        "learning_rate": 1e-3,
        "eps": 1e-8,
        "weight_decay": 0.01,
    }

    block_res = check_code_block(
        """import optax
tx = optax.adam(lr=1e-3)"""
    )
    assert block_res["total_analyzed"] == 1


def test_extract_snapshot_optax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `extract_snapshot('optax')` in offline mode.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("ML_SNAPSHOTS_OFFLINE", "1")
    snap = extract_snapshot("optax")
    assert snap.get("target") == "optax"
    cats = snap.get("categories", {})
    assert isinstance(cats, dict)
    assert "optimizer" in cats
    assert "loss" in cats
    assert "scheduler" in cats
    assert "array" in cats
