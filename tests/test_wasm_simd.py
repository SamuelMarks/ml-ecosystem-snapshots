"""Tests for WebAssembly 2.0 / Fixed-Width SIMD snapshot extractor, MCP server, and CLI."""

import argparse
import pytest

from ml_framework_snapshots.frameworks.wasm_simd import (
    CANONICAL_WASM_SIMD_OPS,
    _load_wasm_simd_ops,
    collect_api,
)
from ml_framework_snapshots.mcp_server import check_wasm_instruction
from ml_framework_snapshots.cli import cmd_check_wasm
from ml_switcheroo_ir.schema.ghost import (
    GhostOperationRef,
    OperandDirection,
    SemanticTier,
)


def test_collect_api_wasm_simd_categories() -> None:
    """Verify collect_api populates GhostRefs for valid tiers and rejects invalid ones."""
    refs_array = collect_api(SemanticTier.ARRAY_API)
    assert len(refs_array) > 0

    refs_neural = collect_api(SemanticTier.NEURAL)
    assert len(refs_neural) > 0

    refs_util = collect_api(SemanticTier.UTIL)
    assert len(refs_util) > 0

    refs_invalid = collect_api(SemanticTier.ACTIVATION)
    assert refs_invalid == []


def test_wasm_simd_op_properties_and_roles() -> None:
    """Verify that extracted WASM SIMD instructions have valid operands, results, and directions."""
    refs = collect_api(SemanticTier.ARRAY_API)
    op_map = {ref.name: ref for ref in refs if isinstance(ref, GhostOperationRef)}

    assert "f32x4.add" in op_map
    add_ref = op_map["f32x4.add"]
    assert add_ref.api_path == "wasm.f32x4.add"
    assert add_ref.operands is not None and len(add_ref.operands) == 2
    assert add_ref.returns is not None and len(add_ref.returns) == 1

    # Verify write direction for store instruction
    assert "v128.store" in op_map
    store_ref = op_map["v128.store"]
    assert store_ref.returns is not None and len(store_ref.returns) == 0
    assert store_ref.returns_type == "None"
    assert store_ref.operands is not None
    addr_op = next(p for p in store_ref.operands if p.name == "addr")
    assert getattr(addr_op, "direction", None) == OperandDirection.WRITE


def test_check_wasm_instruction_validation() -> None:
    """Test check_wasm_instruction validating existence, operands count, and immediate attributes."""
    # 1. Valid instruction
    res_valid = check_wasm_instruction("f32x4.add", operands_count=2)
    assert res_valid["is_valid"] is True
    assert res_valid["op_exists"] is True
    assert len(res_valid["errors"]) == 0

    # 2. Prefix stripping (wasm.f32x4.add)
    res_prefix = check_wasm_instruction("wasm.f32x4.add", operands_count=2)
    assert res_prefix["is_valid"] is True

    # 3. Operands count mismatch
    res_count_err = check_wasm_instruction("f32x4.add", operands_count=4)
    assert res_count_err["is_valid"] is False
    assert any("expects 2 inputs, but got 4" in e for e in res_count_err["errors"])

    # 4. Valid attributes
    res_attr = check_wasm_instruction("v128.load", attributes=["offset", "align"])
    assert res_attr["is_valid"] is True

    # 5. Invalid attribute
    res_bad_attr = check_wasm_instruction("v128.load", attributes=["invalid_attr"])
    assert res_bad_attr["is_valid"] is False
    assert any(
        "Attribute 'invalid_attr' is not valid" in e for e in res_bad_attr["errors"]
    )

    # 6. Non-existent instruction
    res_nonexistent = check_wasm_instruction("nonexistent.opcode")
    assert res_nonexistent["is_valid"] is False
    assert res_nonexistent["op_exists"] is False
    assert any(
        "Unknown WebAssembly SIMD instruction" in e for e in res_nonexistent["errors"]
    )


def test_cmd_check_wasm_cli(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI command check-wasm for valid and invalid queries."""
    # 1. Valid invocation
    args_valid = argparse.Namespace(
        mnemonic="f32x4.mul",
        operands_count=2,
        attributes=None,
    )
    cmd_check_wasm(args_valid)
    out_valid = capsys.readouterr().out
    assert "WASM Instruction 'f32x4.mul' is valid." in out_valid

    # 2. Immediate attributes list
    args_attrs = argparse.Namespace(
        mnemonic="v128.store",
        operands_count=None,
        attributes="offset, align",
    )
    cmd_check_wasm(args_attrs)
    out_attrs = capsys.readouterr().out
    assert "WASM Instruction 'v128.store' is valid." in out_attrs

    # 3. Invalid instruction exits 1
    args_invalid = argparse.Namespace(
        mnemonic="invalid.inst",
        operands_count=None,
        attributes=None,
    )
    with pytest.raises(SystemExit) as exc_info:
        cmd_check_wasm(args_invalid)
    assert exc_info.value.code == 1
    out_invalid = capsys.readouterr().out
    assert "WASM Instruction 'invalid.inst' Invalid:" in out_invalid


def test_wasm_simd_roundtrip_serialization() -> None:
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


def test_load_wasm_simd_ops_catalog() -> None:
    """Verify _load_wasm_simd_ops catalog returns standard canonical definitions."""
    ops = _load_wasm_simd_ops()
    assert ops == CANONICAL_WASM_SIMD_OPS
