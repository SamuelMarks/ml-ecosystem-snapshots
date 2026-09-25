"""Unit test suite for FlashAttention, PagedAttention, and custom attention kernels extractor.

Verifies attention parameter signatures, causal/paged mask configurations,
and live/fallback introspection with 100% coverage.
"""

from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from ml_ecosystem_snapshots.api import extract_snapshot, get_pkg_version
from ml_ecosystem_snapshots.cli import cmd_check_flash_attention
from ml_ecosystem_snapshots.frameworks import flash_attention
from ml_ecosystem_snapshots.mcp_server import check_flash_attention_op
from ml_switcheroo_ir.schema.ghost import ParameterKind, SemanticTier


def test_flash_attention_version() -> None:
    """Verify that get_pkg_version returns expected version for flash_attention and flash_attn."""
    assert get_pkg_version("flash_attention") == "2.6.3"
    assert get_pkg_version("flash_attn") == "2.6.3"


def test_collect_api_unsupported_category() -> None:
    """Verify that unsupported semantic tiers return an empty list."""
    assert flash_attention.collect_api(SemanticTier.LOSS) == []
    assert flash_attention.collect_api(SemanticTier.OPTIMIZER) == []


def test_collect_api_fallback_mode() -> None:
    """Verify that collect_api successfully generates canonical operators in offline mode."""
    with patch(
        "ml_ecosystem_snapshots.frameworks.flash_attention._discover_flash_attn_module",
        return_value=None,
    ):
        refs = flash_attention.collect_api(SemanticTier.NEURAL_OPS)
        assert len(refs) >= 7

        op_names = {r.name for r in refs}
        assert "flash_attn_func" in op_names
        assert "paged_attention_v1" in op_names
        assert "FlashAttention" in op_names
        assert "ScaledDotProductAttention" in op_names

        fa_ref = next(r for r in refs if r.name == "flash_attn_func")
        assert len(fa_ref.params) == 9
        assert fa_ref.params[0].kind == ParameterKind.POSITIONAL_ONLY


def test_collect_api_live_mode() -> None:
    """Verify that collect_api works when flash_attn module is discovered."""
    import types

    fake_mod = types.SimpleNamespace(flash_attn_func=lambda q, k, v: q)

    with patch(
        "ml_ecosystem_snapshots.frameworks.flash_attention._discover_flash_attn_module",
        return_value=fake_mod,
    ):
        refs = flash_attention.collect_api(SemanticTier.NEURAL_OPS)
        assert len(refs) >= 7


def test_collect_api_inspection_exception_handled() -> None:
    """Verify that inspection errors on specific objects fall back gracefully."""
    import types

    fake_mod = types.SimpleNamespace(flash_attn_func="not_inspectable")

    with patch(
        "ml_ecosystem_snapshots.frameworks.flash_attention._discover_flash_attn_module",
        return_value=fake_mod,
    ):
        with patch(
            "ml_ecosystem_snapshots.models.GhostInspector.inspect",
            side_effect=ValueError("boom"),
        ):
            refs = flash_attention.collect_api(SemanticTier.NEURAL_OPS)
            assert len(refs) >= 7


def test_extract_snapshot_flash_attention() -> None:
    """Verify full snapshot extraction for flash_attention target."""
    snap = extract_snapshot("flash_attention")
    assert snap.get("target") == "flash_attention"
    cats = snap.get("categories", {})
    assert "neural_ops" in cats or "util" in cats


def test_check_flash_attention_op_mcp() -> None:
    """Verify MCP validator check_flash_attention_op behavior across valid and invalid inputs."""
    # Valid call
    valid_res = check_flash_attention_op("flash_attn_func", inputs_count=3)
    assert valid_res["is_valid"] is True
    assert valid_res["op_exists"] is True

    # Valid with prefix flash_attn.
    prefix_fa = check_flash_attention_op("flash_attn.flash_attn_func", inputs_count=3)
    assert prefix_fa["is_valid"] is True

    # Valid with prefix vllm.
    prefix_vllm = check_flash_attention_op("vllm.paged_attention_v1", inputs_count=10)
    assert prefix_vllm["is_valid"] is True

    # Valid with prefix ml.switcheroo.custom.
    prefix_custom = check_flash_attention_op(
        "ml.switcheroo.custom.FlashAttention", inputs_count=3
    )
    assert prefix_custom["is_valid"] is True

    # Unknown operation
    unknown_res = check_flash_attention_op("non_existent_op")
    assert unknown_res["is_valid"] is False
    assert unknown_res["op_exists"] is False
    assert len(unknown_res["errors"]) > 0

    # Wrong inputs count
    wrong_inputs = check_flash_attention_op("flash_attn_func", inputs_count=1)
    assert wrong_inputs["is_valid"] is False
    assert "expects 3 positional inputs" in wrong_inputs["errors"][0]

    # Valid keyword arguments
    valid_kw = check_flash_attention_op(
        "flash_attn_func", attributes=["causal", "dropout_p"]
    )
    assert valid_kw["is_valid"] is True

    # Invalid keyword arguments
    invalid_kw = check_flash_attention_op(
        "flash_attn_func", attributes=["invalid_kwarg"]
    )
    assert invalid_kw["is_valid"] is False
    assert "does not accept keyword argument 'invalid_kwarg'" in invalid_kw["errors"][0]


def test_cmd_check_flash_attention_cli_valid(capsys: Any) -> None:
    """Verify CLI command execution on valid attention operation."""
    args = argparse.Namespace(
        op_name="flash_attn_func",
        inputs_count=3,
        attributes=None,
    )
    cmd_check_flash_attention(args)
    captured = capsys.readouterr()
    assert "Attention Operation 'flash_attn_func' is valid." in captured.out
    assert "Inputs:     3" in captured.out


def test_cmd_check_flash_attention_cli_invalid_exits(capsys: Any) -> None:
    """Verify CLI command exits with non-zero code on invalid operator."""
    args = argparse.Namespace(
        op_name="invalid_op",
        inputs_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_flash_attention(args)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Attention Operation 'invalid_op' is invalid:" in captured.out


def test_discover_flash_attn_module_branches() -> None:
    """Verify module lookup branches for _discover_flash_attn_module."""
    fake_mod = MagicMock()
    with patch("importlib.import_module", return_value=fake_mod):
        assert flash_attention._discover_flash_attn_module() is fake_mod

    # Case where loading raises exception
    with patch("importlib.import_module", side_effect=ImportError("missing")):
        assert flash_attention._discover_flash_attn_module() is None
