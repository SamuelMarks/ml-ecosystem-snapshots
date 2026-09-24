"""Tests for C++17 and PyBind11 tensor runtime snapshot extractor, MCP server, and CLI."""

import argparse
import pytest

from ml_framework_snapshots.frameworks.cpp_runtime import (
    CANONICAL_CPP_RUNTIME_OPS,
    _load_cpp_runtime_ops,
    collect_api,
)
from ml_framework_snapshots.mcp_server import check_cpp_op
from ml_framework_snapshots.cli import cmd_check_cpp
from ml_switcheroo_ir.schema.ghost import (
    GhostOperationRef,
    OperandDirection,
    SemanticTier,
)


def test_collect_api_cpp_runtime_categories() -> None:
    """Verify collect_api populates GhostRefs for valid tiers and rejects invalid ones."""
    refs_array = collect_api(SemanticTier.ARRAY_API)
    assert len(refs_array) > 0

    refs_neural = collect_api(SemanticTier.NEURAL)
    assert len(refs_neural) > 0

    refs_util = collect_api(SemanticTier.UTIL)
    assert len(refs_util) > 0

    refs_invalid = collect_api(SemanticTier.ACTIVATION)
    assert refs_invalid == []


def test_cpp_runtime_op_properties_and_roles() -> None:
    """Verify that extracted C++ operations have valid operands, directions, and results."""
    refs = collect_api(SemanticTier.ARRAY_API)
    op_map = {ref.name: ref for ref in refs if isinstance(ref, GhostOperationRef)}

    assert "clamp" in op_map
    clamp_ref = op_map["clamp"]
    assert clamp_ref.api_path == "cpp.clamp"
    assert clamp_ref.operands is not None and len(clamp_ref.operands) == 3
    assert clamp_ref.returns is not None and len(clamp_ref.returns) == 1

    # Verify write direction for from_blob
    assert "from_blob" in op_map
    blob_ref = op_map["from_blob"]
    assert blob_ref.operands is not None
    data_op = next(p for p in blob_ref.operands if p.name == "data")
    assert getattr(data_op, "direction", None) == OperandDirection.WRITE


def test_check_cpp_op_validation() -> None:
    """Test check_cpp_op validating existence, operands count, and attributes."""
    # 1. Valid operation
    res_valid = check_cpp_op("clamp", inputs_count=3)
    assert res_valid["is_valid"] is True
    assert res_valid["op_exists"] is True
    assert len(res_valid["errors"]) == 0

    # 2. Prefix stripping (cpp.clamp)
    res_prefix = check_cpp_op("cpp.clamp", inputs_count=3)
    assert res_prefix["is_valid"] is True

    # 3. Operands count mismatch
    res_count_err = check_cpp_op("clamp", inputs_count=1)
    assert res_count_err["is_valid"] is False
    assert any("expects 3 inputs, but got 1" in e for e in res_count_err["errors"])

    # 4. Valid attributes
    res_attr = check_cpp_op("data_ptr", attributes=["dtype"])
    assert res_attr["is_valid"] is True

    # 5. Invalid attribute
    res_bad_attr = check_cpp_op("data_ptr", attributes=["invalid_attr"])
    assert res_bad_attr["is_valid"] is False
    assert any(
        "Attribute 'invalid_attr' is not valid" in e for e in res_bad_attr["errors"]
    )

    # 6. Non-existent operator
    res_nonexistent = check_cpp_op("nonexistent_cpp_op")
    assert res_nonexistent["is_valid"] is False
    assert res_nonexistent["op_exists"] is False
    assert any("Unknown C++ operation" in e for e in res_nonexistent["errors"])


def test_cmd_check_cpp_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI command check-cpp for valid and invalid queries."""
    # 1. Valid invocation
    args_valid = argparse.Namespace(
        op_name="fma",
        inputs_count=3,
        attributes=None,
    )
    cmd_check_cpp(args_valid)
    out_valid = capsys.readouterr().out
    assert "C++ Operation 'fma' is valid." in out_valid

    # 2. Attributes filtering
    args_attrs = argparse.Namespace(
        op_name="data_ptr",
        inputs_count=None,
        attributes="dtype",
    )
    cmd_check_cpp(args_attrs)
    out_attrs = capsys.readouterr().out
    assert "C++ Operation 'data_ptr' is valid." in out_attrs

    # 3. Invalid operation exits 1
    args_invalid = argparse.Namespace(
        op_name="nonexistent_cpp_op",
        inputs_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_cpp(args_invalid)
    assert exc_info.value.code == 1
    out_invalid = capsys.readouterr().out
    assert "C++ Operation 'nonexistent_cpp_op' Invalid:" in out_invalid


def test_cpp_runtime_roundtrip_serialization() -> None:
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


def test_load_cpp_runtime_ops_catalog() -> None:
    """Verify _load_cpp_runtime_ops catalog returns standard canonical definitions."""
    ops = _load_cpp_runtime_ops()
    assert ops == CANONICAL_CPP_RUNTIME_OPS
