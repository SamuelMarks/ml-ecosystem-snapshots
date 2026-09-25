"""Tests for Apple Metal Shading Language (MSL) extractor, MCP verification, and CLI."""

import argparse
import pytest

from ml_ecosystem_snapshots.frameworks.metal import (
    CANONICAL_METAL_OPS,
    _load_metal_ops,
    collect_api,
)
from ml_ecosystem_snapshots.mcp_server import check_metal_op
from ml_ecosystem_snapshots.cli import cmd_check_metal
from ml_switcheroo_ir.schema.ghost import (
    GhostOperationRef,
    OperandDirection,
    SemanticTier,
)


def test_collect_api_metal_categories() -> None:
    """Verify collect_api populates GhostRefs for valid tiers and rejects invalid ones."""
    refs_array = collect_api(SemanticTier.ARRAY_API)
    assert len(refs_array) > 0

    refs_neural = collect_api(SemanticTier.NEURAL)
    assert len(refs_neural) > 0

    refs_util = collect_api(SemanticTier.UTIL)
    assert len(refs_util) > 0

    refs_invalid = collect_api(SemanticTier.ACTIVATION)
    assert refs_invalid == []


def test_metal_op_properties_and_roles() -> None:
    """Verify that extracted MSL operations have valid operands, address spaces, and results."""
    refs = collect_api(SemanticTier.ARRAY_API)
    op_map = {ref.name: ref for ref in refs if isinstance(ref, GhostOperationRef)}

    assert "simdgroup_multiply_accumulate" in op_map
    mma_ref = op_map["simdgroup_multiply_accumulate"]
    assert mma_ref.api_path == "metal.simdgroup_multiply_accumulate"
    assert mma_ref.operands is not None and len(mma_ref.operands) == 4
    assert mma_ref.returns is not None and len(mma_ref.returns) == 1

    # Verify write direction for store ops
    assert "simdgroup_store" in op_map
    store_ref = op_map["simdgroup_store"]
    assert store_ref.returns is not None and len(store_ref.returns) == 0
    assert store_ref.returns_type == "None"
    assert store_ref.operands is not None
    ptr_op = next(p for p in store_ref.operands if p.name == "ptr")
    assert getattr(ptr_op, "direction", None) == OperandDirection.WRITE


def test_check_metal_op_validation() -> None:
    """Test check_metal_op validating existence, operands count, address spaces, and attributes."""
    # 1. Valid invocation
    res_valid = check_metal_op(
        "simdgroup_multiply_accumulate",
        address_space="thread",
        inputs_count=4,
        attributes=["m", "n", "k"],
    )
    assert res_valid["is_valid"] is True
    assert res_valid["op_exists"] is True
    assert len(res_valid["errors"]) == 0

    # 2. Input count mismatch
    res_count_err = check_metal_op(
        "simdgroup_multiply_accumulate",
        inputs_count=2,
    )
    assert res_count_err["is_valid"] is False
    assert any("expects 4 inputs, but got 2" in e for e in res_count_err["errors"])

    # 3. Address space mismatch
    res_space_err = check_metal_op(
        "simdgroup_multiply_accumulate",
        address_space="device",
    )
    assert res_space_err["is_valid"] is False
    assert any(
        "operates in 'thread' address space" in e for e in res_space_err["errors"]
    )

    # 4. Invalid attribute
    res_attr_err = check_metal_op(
        "simdgroup_multiply_accumulate",
        attributes=["nonexistent_dim"],
    )
    assert res_attr_err["is_valid"] is False
    assert any(
        "Attribute 'nonexistent_dim' is not valid" in e for e in res_attr_err["errors"]
    )

    # 5. Non-existent operator
    res_nonexistent = check_metal_op("nonexistent_metal_op")
    assert res_nonexistent["is_valid"] is False
    assert res_nonexistent["op_exists"] is False
    assert any("Unknown Metal MSL operation" in e for e in res_nonexistent["errors"])


def test_cmd_check_metal_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI command check-metal for valid and invalid queries."""
    # 1. Valid invocation
    args_valid = argparse.Namespace(
        op_name="fma",
        address_space="thread",
        inputs_count=3,
        attributes=None,
    )
    cmd_check_metal(args_valid)
    out_valid = capsys.readouterr().out
    assert "Metal Operation 'fma' is valid." in out_valid

    # 2. Attributes list parsing
    args_attrs = argparse.Namespace(
        op_name="simdgroup_load",
        address_space=None,
        inputs_count=None,
        attributes="transpose",
    )
    cmd_check_metal(args_attrs)
    out_attrs = capsys.readouterr().out
    assert "Metal Operation 'simdgroup_load' is valid." in out_attrs

    # 3. Invalid operation exits 1
    args_invalid = argparse.Namespace(
        op_name="nonexistent_metal_op",
        address_space=None,
        inputs_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_metal(args_invalid)
    assert exc_info.value.code == 1
    out_invalid = capsys.readouterr().out
    assert "Metal Operation 'nonexistent_metal_op' Invalid:" in out_invalid


def test_metal_roundtrip_serialization() -> None:
    """Verify roundtrip serialization through GhostOperationRef and Pydantic models."""
    refs = collect_api(SemanticTier.ARRAY_API)
    assert len(refs) > 0
    for ref in refs:
        assert isinstance(ref, GhostOperationRef)
        serialized = ref.model_dump()
        reconstructed = GhostOperationRef.model_validate(serialized)
        assert reconstructed.name == ref.name
        assert reconstructed.api_path == ref.api_path
        assert len(reconstructed.operands or []) == len(ref.operands or [])
        assert len(reconstructed.params or []) == len(ref.params or [])


def test_load_metal_ops_catalog() -> None:
    """Verify _load_metal_ops catalog returns standard canonical definitions."""
    ops = _load_metal_ops()
    assert ops == CANONICAL_METAL_OPS
